#!/usr/bin/env python3
"""Runner de la requalification temporelle V3 (intégration pipeline contrôlée).

Position dans le pipeline :
    6 recherches Perplexica
    aggregate_search_results
    temporal_validation Python
    temporal_requalification Gemma V3 ciblée   <-- ce module
    merge Python sécurisé                       <-- ce module
    prepare_editorial_input
    Gemma éditorial
    mail_builder

La validation Python reste la dernière autorité sur les statuts temporels :
Gemma propose, Python valide (structure, reason_code, dates, transitions,
pré-positionnement déterministe) puis fusionne. Les faits extraits
(source_date, claimed_dates, access_status...) restent immuables.

Configuration (rétrocompatible) :
    "temporal": {
        "enabled": true,
        "requalification": {
            "enabled": true,
            "model": "local-gemma-4",
            "batch_size": 4,
            "timeout": 600
        }
    }

Si la clé requalification est absente ou enabled=false, le pipeline conserve
exactement le comportement Python-only (aucun appel réseau, aucun changement
de statut).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from editorial_rewrite import (
    ENV_ADAPTER_API_KEY,
    ENV_EDITORIAL_API_KEY,
    ENV_EDITORIAL_BASE_URL,
    env_value,
)
from mail_sender import DEFAULT_ENV_FILE, parse_env_file
from temporal_requalification import (
    DEFAULT_WINDOW_MODE,
    REQUALIFICATION_SYSTEM_PROMPT,
    STATUS_CONTEXT,
    STATUS_CURRENT,
    STATUS_UNKNOWN,
    WINDOW_MODES,
    final_status_with_pre_positioning,
    requalification_plan,
    validate_gemma_response_v1,
)

DEFAULT_REQUALIFICATION_MODEL = "local-gemma-4"
DEFAULT_BATCH_SIZE = 4
MAX_BATCH_SIZE = 4
DEFAULT_TIMEOUT_SECONDS = 600
MAX_TIMEOUT_SECONDS = 1800

CONTEXT_NOTE = (
    "Source de contexte ou de référence ; ne pas la présenter comme nouveauté "
    "de la période."
)
UNKNOWN_NOTE = (
    "Temporalité non suffisamment vérifiée ; ne pas affirmer une date de "
    "publication comme certaine."
)

USER_PROMPT_TEMPLATE = """Classe temporellement chacune des sources ci-dessous.

Réponds EXCLUSIVEMENT par un objet JSON de la forme :
{
  "sources": [
    {
      "source_number": <int>,
      "recommended_status": "current|context|unknown",
      "confidence": "high|medium|low",
      "reason_code": "<une des valeurs autorisées>",
      "reason": "<justification courte>"
    }
  ]
}

reason_code autorisés (enum fermée) :
role_context_legal_text
role_context_decision_old
role_context_explicit
role_current_update_claim
role_current_title_date
role_current_publication_context
role_current_recent_context
role_current_legal_update
no_signal

Règles :
- Les champs claim_context, neighbor_context et recent_context_signals du payload
  sont des faits déjà extraits par Python, pas des suggestions.
- Pour choisir current, privilégie le signal réellement présent :
  * title_date dans la fenêtre -> role_current_title_date ;
  * mise à jour / actualisation récente explicite -> role_current_recent_context ;
  * mise à jour juridique récente (Code, Légifrance, version en vigueur) ->
    role_current_legal_update ;
  * signal recent_context_signals avec date dans la fenêtre de veille ->
    role_current_recent_context (ou role_current_legal_update si juridique).
- Ne détourne pas un reason_code dont les préconditions ne sont pas dans le payload.
- reason doit être une justification courte, sans nouvelle date, sans citation [n],
  sans fait nouveau, sans URL.
- Aucun texte avant/après le JSON.

SOURCES À CLASSER :
__PAYLOADS__"""


def requalification_enabled(job: dict[str, Any]) -> bool:
    temporal = job.get("temporal") if isinstance(job.get("temporal"), dict) else {}
    requalification = (
        temporal.get("requalification")
        if isinstance(temporal.get("requalification"), dict)
        else {}
    )
    return bool(requalification.get("enabled", False))


def requalification_config_from_job(job: dict[str, Any]) -> dict[str, Any]:
    temporal = job.get("temporal") if isinstance(job.get("temporal"), dict) else {}
    requalification = (
        temporal.get("requalification")
        if isinstance(temporal.get("requalification"), dict)
        else {}
    )
    enabled = bool(requalification.get("enabled", False))
    model = str(requalification.get("model") or DEFAULT_REQUALIFICATION_MODEL)
    try:
        batch_size = max(
            1,
            min(
                int(requalification.get("batch_size", DEFAULT_BATCH_SIZE)),
                MAX_BATCH_SIZE,
            ),
        )
    except (TypeError, ValueError):
        batch_size = DEFAULT_BATCH_SIZE
    try:
        timeout = max(
            1,
            min(
                int(requalification.get("timeout", DEFAULT_TIMEOUT_SECONDS)),
                MAX_TIMEOUT_SECONDS,
            ),
        )
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS
    window_mode = requalification.get("window_mode") or DEFAULT_WINDOW_MODE
    if window_mode not in WINDOW_MODES:
        window_mode = DEFAULT_WINDOW_MODE
    file_values = parse_env_file(DEFAULT_ENV_FILE)
    base_url = env_value(ENV_EDITORIAL_BASE_URL, file_values).strip()
    api_key = (
        env_value(ENV_EDITORIAL_API_KEY, file_values).strip()
        or env_value(ENV_ADAPTER_API_KEY, file_values).strip()
        or None
    )
    return {
        "enabled": enabled,
        "model": model,
        "batch_size": batch_size,
        "timeout": timeout,
        "window_mode": window_mode,
        "run_date": temporal.get("run_date") or None,
        "base_url": base_url,
        "api_key": api_key,
    }


def load_local_answers_from_run(run_dir: Path | str) -> dict[str, str]:
    """Collecte les answer_markdown locaux des recherches du run."""
    answers: dict[str, str] = {}
    searches_dir = Path(run_dir) / "searches"
    if not searches_dir.is_dir():
        return answers
    for search_dir in sorted(item for item in searches_dir.iterdir() if item.is_dir()):
        result_file = search_dir / "result.json"
        if not result_file.is_file():
            continue
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        answer = payload.get("answer_markdown") if isinstance(payload, dict) else None
        if isinstance(answer, str) and answer.strip():
            answers[search_dir.name] = answer
    return answers


def build_requalification_plans(
    result: dict[str, Any],
    local_answers: dict[str, str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ne conserve que les sources éligibles (règles V3) parmi cited_sources."""
    plans: list[dict[str, Any]] = []
    for source in result.get("cited_sources") or []:
        if not isinstance(source, dict):
            continue
        temporal = source.get("temporal")
        if not isinstance(temporal, dict) or not temporal.get("temporal_status"):
            continue
        try:
            plan = requalification_plan(
                source,
                temporal,
                local_answers,
                run_date=config.get("run_date"),
                window_mode=config.get("window_mode"),
            )
        except Exception:
            # Une source mal formée ne doit jamais bloquer le job.
            continue
        if plan.get("eligible"):
            plan["_source"] = source
            plan["_temporal"] = temporal
            plans.append(plan)
    return plans


def _json_candidates(text: str):
    yield text
    for match in re.finditer(
        r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
    ):
        yield match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        yield text[start : end + 1]


def parse_gemma_batch(raw_text: str) -> list[dict[str, Any]]:
    """Extrait la liste d'objets JSON de la réponse Gemma (robuste)."""
    text = (raw_text or "").strip()
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            sources = parsed.get("sources")
            if isinstance(sources, list):
                return [item for item in sources if isinstance(item, dict)]
            if "source_number" in parsed:
                return [parsed]
    return []


def call_gemma_batch(
    config: dict[str, Any], messages: list[dict[str, str]]
) -> tuple[dict[str, Any] | None, str | None]:
    """Un seul appel Gemma, jamais bloquant : (réponse, erreur)."""
    base_url = config.get("base_url") or ""
    if not base_url:
        return None, "URL du serveur de requalification non configurée"
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(
        {
            "model": config.get("model") or DEFAULT_REQUALIFICATION_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = config.get("api_key")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = Request(endpoint, data=body, method="POST", headers=headers)
    timeout = config.get("timeout") or DEFAULT_TIMEOUT_SECONDS
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            raw = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return None, "{0}: {1}".format(exc.__class__.__name__, exc)
    if status < 200 or status >= 300:
        return None, "statut HTTP inattendu : {0}".format(status)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, "{0}: {1}".format(exc.__class__.__name__, exc)
    if not isinstance(payload, dict):
        return None, "réponse non objet JSON"
    choices = payload.get("choices") or []
    content = ""
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
    if not content:
        content = payload.get("content") or payload.get("text") or ""
    return {"content": content, "usage": payload.get("usage") or {}}, None


def _call_all_batches(
    config: dict[str, Any], plans: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Batchs de taille limitée ; un batch en échec ne fait pas tomber le job."""
    outcomes: dict[int, dict[str, Any]] = {}
    batch_size = config.get("batch_size") or DEFAULT_BATCH_SIZE
    for start in range(0, len(plans), batch_size):
        chunk = plans[start : start + batch_size]
        user_prompt = USER_PROMPT_TEMPLATE.replace(
            "__PAYLOADS__",
            json.dumps(
                [plan["payload"] for plan in chunk], ensure_ascii=False, indent=2
            ),
        )
        messages = [
            {"role": "system", "content": REQUALIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response, error = call_gemma_batch(config, messages)
        if response is None:
            for plan in chunk:
                outcomes[plan["index"]] = {"entry": None, "error": error or "batch en échec"}
            continue
        entries = parse_gemma_batch(response.get("content") or "")
        by_number = {
            entry.get("source_number"): entry
            for entry in entries
            if isinstance(entry.get("source_number"), int)
        }
        for plan in chunk:
            entry = by_number.get(plan["index"])
            outcomes[plan["index"]] = {
                "entry": entry,
                "error": None if entry is not None else "réponse absente du batch",
            }
    return outcomes


def merge_requalification(
    result: dict[str, Any],
    plans: list[dict[str, Any]],
    outcomes: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fusion Python sécurisée : met à jour temporal_status + traçabilité."""
    summary: dict[str, Any] = {
        "status": "completed",
        "temporal_requalification_eligible_count": len(plans),
        "temporal_requalification_processed_count": 0,
        "temporal_requalification_accepted_count": 0,
        "temporal_requalification_rejected_count": 0,
        "temporal_requalification_error_count": 0,
        "temporal_requalification_current_count": 0,
        "temporal_requalification_context_count": 0,
        "temporal_requalification_unknown_count": 0,
        "temporal_requalification_duration_seconds": 0.0,
    }
    for plan in plans:
        index = plan["index"]
        source = plan["_source"]
        temporal = plan["_temporal"]
        python_status = plan["python_status"]
        pre_position = plan["pre_position"]
        outcome = outcomes.get(index) or {}
        entry = outcome.get("entry")
        error = outcome.get("error")
        payload = plan["payload"]
        summary["temporal_requalification_processed_count"] += 1

        gemma_recommended = None
        gemma_confidence = None
        gemma_reason_code = None
        gemma_reason = None
        gemma_validation = "error"
        gemma_error = error
        final_status = python_status
        merge_note = None

        if entry is not None:
            ok, why, normalized = validate_gemma_response_v1(
                payload,
                entry,
                context_to_current_ok=plan["transitions"].get("_ok_ctx", False),
                run_date=config.get("run_date"),
                window_mode=config.get("window_mode"),
            )
            gemma_recommended = entry.get("recommended_status")
            gemma_confidence = entry.get("confidence")
            gemma_reason_code = entry.get("reason_code")
            gemma_reason = entry.get("reason")
            if ok and normalized is not None:
                gemma_validation = "accepted"
                if normalized.get("reason_code") != entry.get("reason_code"):
                    gemma_reason_code = normalized.get("reason_code")
                final_status, merge_note = final_status_with_pre_positioning(
                    python_status, normalized, True, pre_position
                )
                summary["temporal_requalification_accepted_count"] += 1
            else:
                gemma_validation = "rejected"
                gemma_error = why
                final_status, merge_note = final_status_with_pre_positioning(
                    python_status, None, False, pre_position
                )
                summary["temporal_requalification_rejected_count"] += 1
        else:
            summary["temporal_requalification_error_count"] += 1
            gemma_error = gemma_error or "aucune réponse exploitable"

        # Traçabilité séparée (les faits extraits restent immuables).
        temporal["python_status"] = python_status
        temporal["gemma_recommended_status"] = gemma_recommended
        temporal["gemma_confidence"] = gemma_confidence
        temporal["gemma_reason_code"] = gemma_reason_code
        temporal["gemma_reason"] = gemma_reason
        temporal["gemma_validation"] = gemma_validation
        if gemma_error:
            temporal["gemma_error"] = gemma_error
        temporal["final_status"] = final_status
        temporal["temporal_status"] = final_status

        if final_status == STATUS_CONTEXT:
            temporal["note"] = CONTEXT_NOTE
            summary["temporal_requalification_context_count"] += 1
        elif final_status == STATUS_UNKNOWN:
            temporal["note"] = UNKNOWN_NOTE
            summary["temporal_requalification_unknown_count"] += 1
        elif final_status == STATUS_CURRENT:
            temporal.pop("note", None)
            summary["temporal_requalification_current_count"] += 1
        if merge_note:
            temporal["requalification_note"] = merge_note
        if error:
            temporal["requalification_error"] = error
    return summary


def run_temporal_requalification(
    run_dir: Path | str,
    result: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    """Point d'entrée pipeline. Ne lève jamais : fallback Python systématique."""
    started = time.monotonic()
    empty: dict[str, Any] = {
        "status": "disabled",
        "temporal_requalification_eligible_count": 0,
        "temporal_requalification_processed_count": 0,
        "temporal_requalification_accepted_count": 0,
        "temporal_requalification_rejected_count": 0,
        "temporal_requalification_error_count": 0,
        "temporal_requalification_current_count": 0,
        "temporal_requalification_context_count": 0,
        "temporal_requalification_unknown_count": 0,
        "temporal_requalification_duration_seconds": 0.0,
    }
    try:
        config = requalification_config_from_job(job)
        if not config["enabled"]:
            result["temporal_requalification"] = empty
            return empty
        local_answers = load_local_answers_from_run(run_dir)
        plans = build_requalification_plans(result, local_answers, config)
        outcomes = _call_all_batches(config, plans) if plans else {}
        summary = merge_requalification(result, plans, outcomes, config)
        summary["temporal_requalification_duration_seconds"] = round(
            time.monotonic() - started, 2
        )
        result["temporal_requalification"] = summary
        return summary
    except Exception as exc:
        failed = dict(empty)
        failed["status"] = "failed"
        failed["error"] = "{0}: {1}".format(exc.__class__.__name__, exc)
        failed["temporal_requalification_duration_seconds"] = round(
            time.monotonic() - started, 2
        )
        result["temporal_requalification"] = failed
        return failed