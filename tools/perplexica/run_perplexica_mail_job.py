#!/usr/bin/env python3
"""Run configured Perplexica searches, build mail variants, and optionally send them."""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from editorial_rewrite import DEFAULT_MODEL as DEFAULT_EDITORIAL_MODEL
from editorial_rewrite import DEFAULT_TIMEOUT_SECONDS as EDITORIAL_TIMEOUT_SECONDS
from editorial_rewrite import MAX_TIMEOUT_SECONDS as EDITORIAL_MAX_TIMEOUT_SECONDS
from editorial_rewrite import EditorialConfig, EditorialRewriteError, EditorialTemporalViolationError
from editorial_rewrite import editorial_config_from_env, rewrite_editorial, temporal_safe_raw_markdown
from mail_builder import MailBuildError, build_mail
from mail_sender import DEFAULT_ENV_FILE, DEFAULT_TIMEOUT_SECONDS as SMTP_TIMEOUT_SECONDS
from mail_sender import MailSendError, send_mail, smtp_config_from_env_and_args
from perplexica_client import DEFAULT_TIMEOUT_SECONDS, PerplexicaClient, PerplexicaClientError


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = TOOLS_DIR / "output" / "jobs"
SUCCESS_STATUSES = {"completed", "completed_no_mail"}
SAFE_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MAIL_MODES = {"raw", "editorial", "both"}
CITATION_BLOCK_RE = re.compile(r"\[([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]")
GENERIC_EMPTY_RE = re.compile(
    r"(could not find|no relevant information|aucune information pertinente|je n'ai pas trouv)",
    re.IGNORECASE,
)
AGGREGATE_ALL_SOURCES_SEMANTICS = (
    "Aggregated result: all_sources contains the global deduplicated cited sources only. "
    "The total consulted source count is stored in source_count."
)
AXIS_TITLES = {
    "expertise_justice": "Expertise de justice",
    "expertise_construction": "Expertise construction",
    "mediation": "Médiation",
    "mard_textes": "MARD / textes",
    "jurisprudence": "Jurisprudence",
    "institutionnelle": "Actualité institutionnelle",
}


class JobRunError(Exception):
    """User-facing job orchestration failure."""


class PromptReadError(JobRunError):
    """Prompt file could not be read."""


class JobConfigError(JobRunError):
    """Job JSON is invalid."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def compact_timestamp(moment: datetime | None = None) -> str:
    return (moment or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_editorial_job_config(editorial: Any) -> None:
    if not isinstance(editorial, dict):
        raise JobConfigError("Job field 'editorial' must be an object when provided.")
    if "enabled" in editorial and not isinstance(editorial["enabled"], bool):
        raise JobConfigError("Job field 'editorial.enabled' must be a boolean when provided.")
    if "prompt_file" in editorial and not isinstance(editorial["prompt_file"], str):
        raise JobConfigError("Job field 'editorial.prompt_file' must be a string when provided.")
    if "model" in editorial and not isinstance(editorial["model"], str):
        raise JobConfigError("Job field 'editorial.model' must be a string when provided.")
    if "timeout" in editorial and not isinstance(editorial["timeout"], int):
        raise JobConfigError("Job field 'editorial.timeout' must be an integer when provided.")
    if isinstance(editorial.get("timeout"), int) and editorial["timeout"] <= 0:
        raise JobConfigError("Job field 'editorial.timeout' must be positive.")
    if isinstance(editorial.get("timeout"), int) and editorial["timeout"] > EDITORIAL_MAX_TIMEOUT_SECONDS:
        raise JobConfigError(f"Job field 'editorial.timeout' must not exceed {EDITORIAL_MAX_TIMEOUT_SECONDS}.")
    if "min_body_chars" in editorial and not isinstance(editorial["min_body_chars"], int):
        raise JobConfigError("Job field 'editorial.min_body_chars' must be an integer when provided.")
    if isinstance(editorial.get("min_body_chars"), int) and editorial["min_body_chars"] <= 0:
        raise JobConfigError("Job field 'editorial.min_body_chars' must be positive.")


def validate_searches(searches: Any) -> None:
    if not isinstance(searches, list) or not searches:
        raise JobConfigError("Job field 'searches' must be a non-empty list when provided.")
    seen: set[str] = set()
    for index, search in enumerate(searches):
        if not isinstance(search, dict):
            raise JobConfigError(f"Job search #{index + 1} must be an object.")
        name = search.get("name")
        prompt_file = search.get("prompt_file")
        if not isinstance(name, str) or not name.strip():
            raise JobConfigError(f"Job search #{index + 1} field 'name' is required and must be a string.")
        if not SAFE_JOB_NAME_RE.match(name):
            raise JobConfigError(f"Job search '{name}' may contain only letters, digits, dot, dash, and underscore.")
        if name in seen:
            raise JobConfigError(f"Duplicate job search name: {name}")
        seen.add(name)
        if not isinstance(prompt_file, str) or not prompt_file.strip():
            raise JobConfigError(f"Job search '{name}' field 'prompt_file' is required and must be a string.")


def load_job(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise JobConfigError(f"Cannot read job file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise JobConfigError(f"Invalid job JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise JobConfigError("Invalid job JSON: top-level value must be an object.")

    name = payload.get("name")
    prompt_file = payload.get("prompt_file")
    searches = payload.get("searches")
    base_url = payload.get("base_url")
    mail_mode = payload.get("mail_mode", "editorial")
    if not isinstance(name, str) or not name.strip():
        raise JobConfigError("Job field 'name' is required and must be a string.")
    if not SAFE_JOB_NAME_RE.match(name):
        raise JobConfigError("Job field 'name' may contain only letters, digits, dot, dash, and underscore.")
    if searches is None and (not isinstance(prompt_file, str) or not prompt_file.strip()):
        raise JobConfigError("Job field 'prompt_file' is required for simple jobs and must be a string.")
    if searches is not None:
        validate_searches(searches)
    if not isinstance(base_url, str) or not base_url.strip():
        raise JobConfigError("Job field 'base_url' is required and must be a string.")
    if not isinstance(mail_mode, str) or mail_mode not in MAIL_MODES:
        raise JobConfigError("Job field 'mail_mode' must be one of: raw, editorial, both.")
    if "send_mail" in payload and not isinstance(payload["send_mail"], bool):
        raise JobConfigError("Job field 'send_mail' must be a boolean when provided.")
    if "subject" in payload and payload["subject"] is not None and not isinstance(payload["subject"], str):
        raise JobConfigError("Job field 'subject' must be a string when provided.")
    if "display_title" in payload and payload["display_title"] is not None and not isinstance(payload["display_title"], str):
        raise JobConfigError("Job field 'display_title' must be a string when provided.")
    if "timeout" in payload and not isinstance(payload["timeout"], int):
        raise JobConfigError("Job field 'timeout' must be an integer when provided.")
    if isinstance(payload.get("timeout"), int) and payload["timeout"] <= 0:
        raise JobConfigError("Job field 'timeout' must be positive.")
    if "perplexica_options" in payload and not isinstance(payload["perplexica_options"], dict):
        raise JobConfigError("Job field 'perplexica_options' must be an object when provided.")
    if "temporal" in payload:
        temporal = payload["temporal"]
        if not isinstance(temporal, dict):
            raise JobConfigError("Job field 'temporal' must be an object when provided.")
        if "enabled" in temporal and not isinstance(temporal["enabled"], bool):
            raise JobConfigError("Job field 'temporal.enabled' must be a boolean when provided.")
    if "editorial" in payload:
        validate_editorial_job_config(payload["editorial"])
    return payload


def resolve_job_relative_path(job_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = job_path.resolve().parent / path
    return path


def resolve_prompt_path(job_path: Path, job: dict[str, Any]) -> Path:
    return resolve_job_relative_path(job_path, job["prompt_file"])


def read_prompt(prompt_path: Path) -> str:
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptReadError(f"Cannot read prompt file: {prompt_path}") from exc
    if not prompt.strip():
        raise PromptReadError(f"Prompt file is empty: {prompt_path}")
    return prompt


def create_run_dir(output_root: Path, job_name: str) -> Path:
    run_id = f"{compact_timestamp()}_{uuid.uuid4().hex[:10]}"
    run_dir = output_root / job_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def job_mail_mode(job: dict[str, Any]) -> str:
    return job.get("mail_mode", "editorial")


def is_multi_search_job(job: dict[str, Any]) -> bool:
    return isinstance(job.get("searches"), list)


def editorial_enabled(job: dict[str, Any]) -> bool:
    editorial = job.get("editorial")
    return isinstance(editorial, dict) and bool(editorial.get("enabled", False))


def editorial_needed(job: dict[str, Any]) -> bool:
    return job_mail_mode(job) in {"editorial", "both"} and editorial_enabled(job)


def editorial_config_from_job(job_path: Path, job: dict[str, Any]) -> EditorialConfig:
    editorial = job.get("editorial") if isinstance(job.get("editorial"), dict) else {}
    prompt_file = editorial.get("prompt_file") or "../prompts/prompt_editorial_veille.md"
    return editorial_config_from_env(
        model=editorial.get("model") or DEFAULT_EDITORIAL_MODEL,
        prompt_file=resolve_job_relative_path(job_path, prompt_file),
        timeout=editorial["timeout"] if "timeout" in editorial else EDITORIAL_TIMEOUT_SECONDS,
        min_body_chars=editorial.get("min_body_chars"),
    )


def subject_for_variant(base_subject: str | None, variant: str, mail_mode: str) -> str | None:
    if mail_mode != "both" or not base_subject:
        return base_subject
    suffix = "Source Perplexica" if variant == "raw" else "Synthèse éditoriale"
    return f"{base_subject} — {suffix}"


def source_count(result: dict[str, Any] | None) -> int:
    if not isinstance(result, dict):
        return 0
    value = result.get("source_count")
    if isinstance(value, int) and value >= 0:
        return value
    sources = result.get("all_sources")
    return len(sources) if isinstance(sources, list) else 0


def cited_source_count(result: dict[str, Any] | None) -> int:
    if not isinstance(result, dict):
        return 0
    value = result.get("cited_source_count")
    if isinstance(value, int) and value >= 0:
        return value
    sources = result.get("cited_sources")
    return len(sources) if isinstance(sources, list) else 0


def is_empty_result(result: dict[str, Any]) -> bool:
    if result.get("status") != "completed":
        return False
    if source_count(result) == 0 and cited_source_count(result) == 0:
        return True
    answer = result.get("answer_markdown") or ""
    return source_count(result) == 0 and bool(GENERIC_EMPTY_RE.search(str(answer)))


def classify_search_result(result: dict[str, Any]) -> str:
    return "empty" if is_empty_result(result) else "completed"


def normalize_source_url(url: Any) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def axis_title(name: str, search: dict[str, Any] | None = None) -> str:
    if isinstance(search, dict) and isinstance(search.get("display_title"), str) and search["display_title"].strip():
        return " ".join(search["display_title"].split())
    return AXIS_TITLES.get(name, name.replace("_", " ").title())


def source_lookup(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for collection_name in ("all_sources", "cited_sources"):
        collection = result.get(collection_name)
        if not isinstance(collection, list):
            continue
        for source in collection:
            if not isinstance(source, dict):
                continue
            index = source.get("index")
            if isinstance(index, int) and index > 0 and index not in lookup:
                lookup[index] = source
    return lookup


def rewrite_citations(markdown: str, local_to_global: dict[int, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        numbers: list[str] = []
        for part in match.group(1).split(","):
            local = int(part.strip())
            numbers.append(str(local_to_global.get(local, local)))
        return "[" + ", ".join(numbers) + "]"

    return CITATION_BLOCK_RE.sub(replace, markdown or "")


BEST_TITLE_MAX_LENGTH = 160


def better_source_title(current: Any, candidate: Any) -> Any:
    """Pick the most informative of two Perplexica-provided titles.

    Non-empty beats empty; a non-truncated title beats one ending with "...";
    otherwise the longest title wins, capped at a reasonable length. Titles are
    only ever selected from what Perplexica already returned.
    """
    current_ok = isinstance(current, str) and bool(current.strip())
    candidate_ok = isinstance(candidate, str) and bool(candidate.strip())
    if not current_ok:
        return candidate if candidate_ok else current
    if not candidate_ok:
        return current
    current = " ".join(current.split())
    candidate = " ".join(candidate.split())
    current_truncated = current.endswith("...")
    candidate_truncated = candidate.endswith("...")
    if current_truncated != candidate_truncated:
        return candidate if not candidate_truncated else current
    current_score = min(len(current), BEST_TITLE_MAX_LENGTH)
    candidate_score = min(len(candidate), BEST_TITLE_MAX_LENGTH)
    if candidate_score > current_score:
        return candidate
    return current


def register_global_cited_sources(
    search_name: str,
    result: dict[str, Any],
    global_sources: list[dict[str, Any]],
    source_keys: dict[str, int],
) -> dict[int, int]:
    lookup = source_lookup(result)
    local_to_global: dict[int, int] = {}
    cited = result.get("cited_sources") if isinstance(result.get("cited_sources"), list) else []
    for source in cited:
        if not isinstance(source, dict):
            continue
        local_index = source.get("index")
        if not isinstance(local_index, int) or local_index <= 0:
            continue
        url_key = normalize_source_url(source.get("url"))
        key = f"url:{url_key}" if url_key else f"local:{search_name}:{local_index}"
        if key not in source_keys:
            global_index = len(global_sources) + 1
            source_keys[key] = global_index
            source_for_output = dict(source)
            source_for_output.pop("content", None)
            source_for_output["index"] = global_index
            source_for_output["source_searches"] = [search_name]
            source_for_output["original_indices"] = {search_name: local_index}
            global_sources.append(source_for_output)
        else:
            global_index = source_keys[key]
            existing = global_sources[global_index - 1]
            existing["title"] = better_source_title(existing.get("title"), source.get("title"))
            existing.setdefault("source_searches", [])
            if search_name not in existing["source_searches"]:
                existing["source_searches"].append(search_name)
            existing.setdefault("original_indices", {})
            existing["original_indices"][search_name] = local_index
        local_to_global[local_index] = global_index

    for local_index in CITATION_BLOCK_RE.findall(result.get("answer_markdown", "") or ""):
        for part in local_index.split(","):
            number = int(part.strip())
            if number in local_to_global or number not in lookup:
                continue
            source = lookup[number]
            url_key = normalize_source_url(source.get("url"))
            if url_key and f"url:{url_key}" in source_keys:
                local_to_global[number] = source_keys[f"url:{url_key}"]
    return local_to_global


def build_search_summary(name: str, status: str, result: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    result = result or {}
    return {
        "status": status,
        "source_count": source_count(result),
        "cited_source_count": cited_source_count(result),
        "chat_id": result.get("chat_id"),
        "message_id": result.get("message_id"),
        "error": error,
    }


def aggregate_search_results(
    job: dict[str, Any],
    search_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    global_sources: list[dict[str, Any]] = []
    source_keys: dict[str, int] = {}
    summaries: dict[str, dict[str, Any]] = {}
    raw_sections: list[str] = []
    editorial_sections: list[str] = []
    all_citation_numbers: list[int] = []
    total_source_count = 0
    completed = empty = failed = 0

    for run in search_runs:
        name = run["name"]
        title = run["title"]
        status = run["status"]
        result = run.get("result")
        error = run.get("error")
        if isinstance(result, dict):
            total_source_count += source_count(result)
        summaries[name] = build_search_summary(name, status, result, error)

        if status == "completed" and isinstance(result, dict):
            completed += 1
            local_to_global = register_global_cited_sources(name, result, global_sources, source_keys)
            section_body = rewrite_citations(str(result.get("answer_markdown") or "").strip(), local_to_global).strip()
            if not section_body:
                section_body = "Aucune actualité significative identifiée pour cette période."
            raw_sections.append(f"## {title}\n\n{section_body}")
            editorial_sections.append(f"## {title}\n\n{section_body}")
            for number in CITATION_BLOCK_RE.findall(section_body):
                for part in number.split(","):
                    global_number = int(part.strip())
                    if global_number not in all_citation_numbers:
                        all_citation_numbers.append(global_number)
        elif status == "empty":
            empty += 1
            raw_sections.append(f"## {title}\n\nAucune actualité significative identifiée pour cette période.")
        else:
            failed += 1
            raw_sections.append(f"## {title}\n\nRecherche indisponible pour cet axe.")

    aggregate_status = "completed" if completed else "no_results"
    display_title = job.get("display_title") or job.get("subject") or job["name"]
    return {
        "status": aggregate_status,
        "question": display_title,
        "answer_markdown": "\n\n".join(raw_sections).strip(),
        "editorial_answer_markdown": "\n\n".join(editorial_sections).strip(),
        "all_sources": global_sources,
        "all_sources_semantics": AGGREGATE_ALL_SOURCES_SEMANTICS,
        "cited_sources": global_sources,
        "citation_numbers": all_citation_numbers,
        "unresolved_citations": [],
        "searches": summaries,
        "source_count": total_source_count,
        "cited_source_count": len(global_sources),
        "search_count": len(search_runs),
        "completed_search_count": completed,
        "empty_search_count": empty,
        "failed_search_count": failed,
        "created_at": iso_now(),
    }


def execute_single_prompt(
    prompt: str,
    job: dict[str, Any],
    client_factory: Callable[..., Any],
) -> dict[str, Any]:
    options = dict(job.get("perplexica_options") or {})
    client = client_factory(job["base_url"].rstrip("/"), timeout=job.get("timeout", DEFAULT_TIMEOUT_SECONDS))
    return client.ask(prompt, **options)


def execute_multi_searches(
    job_path: Path,
    job: dict[str, Any],
    run_dir: Path,
    client_factory: Callable[..., Any],
) -> dict[str, Any]:
    search_runs: list[dict[str, Any]] = []
    for search in job["searches"]:
        name = search["name"]
        title = axis_title(name, search)
        search_dir = run_dir / "searches" / name
        try:
            prompt = read_prompt(resolve_job_relative_path(job_path, search["prompt_file"]))
            search_job = dict(job)
            if isinstance(search.get("perplexica_options"), dict):
                options = dict(job.get("perplexica_options") or {})
                options.update(search["perplexica_options"])
                search_job["perplexica_options"] = options
            result = execute_single_prompt(prompt, search_job, client_factory)
            status = classify_search_result(result)
            write_json(search_dir / "result.json", result)
            search_runs.append({"name": name, "title": title, "status": status, "result": result})
        except (PromptReadError, PerplexicaClientError) as exc:
            error = str(exc)
            write_json(search_dir / "error.json", {"status": "failed", "error": error})
            search_runs.append({"name": name, "title": title, "status": "failed", "error": error})
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            write_json(search_dir / "error.json", {"status": "failed", "error": error})
            search_runs.append({"name": name, "title": title, "status": "failed", "error": error})
    return aggregate_search_results(job, search_runs)


def temporal_safe_result_copy(result: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a canonical result with a temporal-safe raw answer.

    Used only for the editorial fallback_raw path so the fallback mail never
    reintroduces invalid claimed dates or freshness markers associated with a
    temporal mismatch source.
    """
    safe = dict(result or {})
    safe["answer_markdown"] = temporal_safe_raw_markdown(
        (result or {}).get("answer_markdown") or "",
        (result or {}).get("cited_sources") or [],
    )
    return safe


def build_run_metadata(
    job_name: str,
    started_at: str,
    status: str,
    result: dict[str, Any] | None,
    mail_requested: bool,
    error: str | None,
    *,
    mail_mode: str,
    raw_mail_built: bool = False,
    raw_mail_sent: bool = False,
    editorial_requested: bool = False,
    editorial_status: str = "disabled",
    editorial_model: str | None = None,
    editorial_fallback_reason: str | None = None,
    editorial_temporal_violation_count: int | None = None,
    editorial_retry_count: int | None = None,
    editorial_output_validation_status: str | None = None,
    editorial_output_invalid_reason: str | None = None,
    editorial_citation_numbers_normalized: bool | None = None,
    editorial_declared_citation_count: int | None = None,
    editorial_actual_citation_count: int | None = None,
    editorial_mail_built: bool = False,
    editorial_mail_sent: bool = False,
) -> dict[str, Any]:
    result = result or {}
    mail_sent = raw_mail_sent or editorial_mail_sent
    metadata = {
        "job_name": job_name,
        "started_at": started_at,
        "finished_at": iso_now(),
        "status": status,
        "chat_id": result.get("chat_id"),
        "message_id": result.get("message_id"),
        "source_count": result.get("source_count", source_count(result)),
        "cited_source_count": result.get("cited_source_count", cited_source_count(result)),
        "mail_requested": mail_requested,
        "mail_sent": mail_sent,
        "mail_mode": mail_mode,
        "raw_mail_built": raw_mail_built,
        "raw_mail_sent": raw_mail_sent,
        "editorial_requested": editorial_requested,
        "editorial_status": editorial_status,
        "editorial_model": editorial_model,
        "editorial_fallback_reason": editorial_fallback_reason,
        "editorial_mail_built": editorial_mail_built,
        "editorial_mail_sent": editorial_mail_sent,
        "error": error,
    }
    if editorial_temporal_violation_count is not None:
        metadata["editorial_temporal_violation_count"] = editorial_temporal_violation_count
    if editorial_requested:
        metadata["editorial_retry_count"] = editorial_retry_count
        metadata["editorial_output_validation_status"] = editorial_output_validation_status
        metadata["editorial_output_invalid_reason"] = editorial_output_invalid_reason
        metadata["editorial_citation_numbers_normalized"] = editorial_citation_numbers_normalized
        metadata["editorial_declared_citation_count"] = editorial_declared_citation_count
        metadata["editorial_actual_citation_count"] = editorial_actual_citation_count
    for key in ("searches", "search_count", "completed_search_count", "empty_search_count", "failed_search_count"):
        if key in result:
            metadata[key] = result[key]
    temporal = result.get("temporal_validation")
    if isinstance(temporal, dict):
        for key in (
            "temporal_validation_count",
            "current_count",
            "context_count",
            "mismatch_count",
            "unknown_count",
            "direct_date_count",
            "indirect_date_count",
            "unknown_date_count",
        ):
            if key in temporal:
                metadata[key] = temporal[key]
        metadata["temporal_validation_status"] = temporal.get("status", "completed")
    requalification = result.get("temporal_requalification")
    if isinstance(requalification, dict):
        for key in (
            "temporal_requalification_eligible_count",
            "temporal_requalification_processed_count",
            "temporal_requalification_accepted_count",
            "temporal_requalification_rejected_count",
            "temporal_requalification_error_count",
            "temporal_requalification_current_count",
            "temporal_requalification_context_count",
            "temporal_requalification_unknown_count",
            "temporal_requalification_duration_seconds",
        ):
            if key in requalification:
                metadata[key] = requalification[key]
        metadata["temporal_requalification_status"] = requalification.get(
            "status", "disabled"
        )
    if editorial_temporal_violation_count is not None:
        metadata["editorial_temporal_violation_count"] = editorial_temporal_violation_count
    return metadata


def write_mail_outputs(run_dir: Path, prefix: str, mail: dict[str, Any]) -> None:
    (run_dir / f"{prefix}_mail.html").write_text(mail["html"], encoding="utf-8")
    (run_dir / f"{prefix}_mail.txt").write_text(mail["text"], encoding="utf-8")
    write_json(run_dir / f"{prefix}_mail.json", mail["metadata"])


def write_editorial_outputs(run_dir: Path, editorial: dict[str, Any], raw_text: str) -> None:
    (run_dir / "editorial_raw.txt").write_text(raw_text, encoding="utf-8")
    write_json(run_dir / "editorial.json", editorial)


def temporal_validation_enabled(job: dict[str, Any]) -> bool:
    temporal = job.get("temporal")
    return isinstance(temporal, dict) and bool(temporal.get("enabled", False))


def load_local_answers_from_run(run_dir: Path) -> dict[str, str]:
    """Collect the local answer_markdown of every completed search."""
    local_answers: dict[str, str] = {}
    searches_dir = run_dir / "searches"
    if not searches_dir.is_dir():
        return local_answers
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
            local_answers[search_dir.name] = answer
    return local_answers


def run_temporal_validation(run_dir: Path, result: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Validate the aggregated cited sources and persist the temporal data.

    Local answers are read from <run_dir>/searches/<name>/result.json before the
    global renumbering, so claimed dates stay attached to the original local
    citation numbers and are then mapped to global indices by
    validate_cited_sources via source_searches / original_indices.
    """
    from temporal_validation import validate_cited_sources

    temporal_config = job.get("temporal") if isinstance(job.get("temporal"), dict) else {}
    local_answers = load_local_answers_from_run(run_dir)
    validated, summary = validate_cited_sources(
        result.get("cited_sources") or [],
        local_answers=local_answers,
        run_date=temporal_config.get("run_date"),
    )
    result["temporal_validation"] = summary
    result["temporal_validation_by_source"] = [
        {
            "index": source.get("index"),
            "title": source.get("title"),
            "url": source.get("url"),
            "temporal": source.get("temporal"),
        }
        for source in validated
        if isinstance(source, dict)
    ]
    result["cited_sources"] = validated
    return summary


def default_smtp_config():
    args = argparse.Namespace(
        env_file=DEFAULT_ENV_FILE,
        smtp_host=None,
        smtp_port=None,
        smtp_username=None,
        smtp_password=None,
        from_address=None,
        to=None,
        smtp_security=None,
        smtp_starttls=None,
        timeout=SMTP_TIMEOUT_SECONDS,
    )
    return smtp_config_from_env_and_args(args)


def run_job(
    job_path: Path,
    *,
    dry_run: bool = False,
    no_mail: bool = False,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    client_factory: Callable[..., Any] = PerplexicaClient,
    build_mail_func: Callable[..., dict[str, Any]] = build_mail,
    smtp_config_factory: Callable[[], Any] = default_smtp_config,
    send_mail_func: Callable[..., dict[str, Any]] = send_mail,
    editorial_rewrite_func: Callable[..., tuple[dict[str, Any], str]] = rewrite_editorial,
) -> tuple[Path, dict[str, Any]]:
    job_path = Path(job_path)
    job = load_job(job_path)
    started_at = iso_now()
    job_name = job["name"]
    mail_mode = job_mail_mode(job)
    mail_requested = bool(job.get("send_mail", False)) and not no_mail
    display_title = job.get("display_title")
    raw_mail_built = False
    raw_mail_sent = False
    editorial_requested = editorial_needed(job)
    editorial_status = "disabled" if not editorial_requested else "failed"
    editorial_model = None
    editorial_fallback_reason = None
    editorial_temporal_violation_count = 0
    editorial_retry_count = 0
    editorial_output_validation_status = None
    editorial_output_invalid_reason = None
    editorial_citation_numbers_normalized = None
    editorial_declared_citation_count = None
    editorial_actual_citation_count = None
    editorial_mail_built = False
    editorial_mail_sent = False
    result: dict[str, Any] | None = None
    raw_mail: dict[str, Any] | None = None
    editorial_mail: dict[str, Any] | None = None
    editorial_result: dict[str, Any] | None = None
    run_dir = create_run_dir(output_root, job_name)

    def finish(status: str, error: str | None = None) -> dict[str, Any]:
        metadata = build_run_metadata(
            job_name,
            started_at,
            status,
            result,
            mail_requested,
            error,
            mail_mode=mail_mode,
            raw_mail_built=raw_mail_built,
            raw_mail_sent=raw_mail_sent,
            editorial_requested=editorial_requested,
            editorial_status=editorial_status,
            editorial_model=editorial_model,
            editorial_fallback_reason=editorial_fallback_reason,
            editorial_temporal_violation_count=editorial_temporal_violation_count,
            editorial_retry_count=editorial_retry_count,
            editorial_output_validation_status=editorial_output_validation_status,
            editorial_output_invalid_reason=editorial_output_invalid_reason,
            editorial_citation_numbers_normalized=editorial_citation_numbers_normalized,
            editorial_declared_citation_count=editorial_declared_citation_count,
            editorial_actual_citation_count=editorial_actual_citation_count,
            editorial_mail_built=editorial_mail_built,
            editorial_mail_sent=editorial_mail_sent,
        )
        write_json(run_dir / "run.json", metadata)
        return metadata

    if is_multi_search_job(job):
        result = execute_multi_searches(job_path, job, run_dir, client_factory)
        write_json(run_dir / "result.json", result)
        if result["status"] == "no_results":
            return run_dir, finish("no_results", "No useful Perplexica results across configured searches.")
    else:
        try:
            prompt = read_prompt(resolve_prompt_path(job_path, job))
        except PromptReadError as exc:
            return run_dir, finish("failed", str(exc))
        try:
            result = execute_single_prompt(prompt, job, client_factory)
            write_json(run_dir / "result.json", result)
        except PerplexicaClientError as exc:
            return run_dir, finish("perplexica_failed", str(exc))
        except Exception as exc:
            return run_dir, finish("perplexica_failed", f"{exc.__class__.__name__}: {exc}")

    if temporal_validation_enabled(job) and result is not None and result.get("status") == "completed" and is_multi_search_job(job):
        try:
            run_temporal_validation(run_dir, result, job)
        except Exception as exc:
            result["temporal_validation"] = {
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        try:
            from temporal_requalification_runner import run_temporal_requalification
            run_temporal_requalification(run_dir, result, job)
        except Exception as exc:
            result["temporal_requalification"] = {
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        write_json(run_dir / "result.json", result)

    try:
        if mail_mode in {"raw", "both"}:
            raw_mail = build_mail_func(
                result,
                subject=subject_for_variant(job.get("subject"), "raw", mail_mode),
                display_title=display_title,
            )
            write_mail_outputs(run_dir, "raw", raw_mail)
            raw_mail_built = True
    except MailBuildError as exc:
        return run_dir, finish("build_failed", str(exc))
    except Exception as exc:
        return run_dir, finish("build_failed", f"{exc.__class__.__name__}: {exc}")

    if editorial_requested:
        try:
            editorial_config = editorial_config_from_job(job_path, job)
            editorial_model = editorial_config.model
            editorial_result, editorial_raw = editorial_rewrite_func(result, editorial_config)
            write_editorial_outputs(run_dir, editorial_result, editorial_raw)
            editorial_status = "completed"
            editorial_temporal_violation_count = (editorial_result or {}).get(
                "editorial_temporal_violation_count",
                (editorial_result or {}).get("temporal_violation_count", 0),
            )
            editorial_retry_count = (editorial_result or {}).get("editorial_retry_count", 0)
            editorial_output_validation_status = (editorial_result or {}).get(
                "editorial_output_validation_status"
            )
            editorial_output_invalid_reason = (editorial_result or {}).get(
                "editorial_output_invalid_reason"
            )
            editorial_citation_numbers_normalized = (editorial_result or {}).get(
                "editorial_citation_numbers_normalized"
            )
            editorial_declared_citation_count = (editorial_result or {}).get(
                "editorial_declared_citation_count"
            )
            editorial_actual_citation_count = (editorial_result or {}).get(
                "editorial_actual_citation_count"
            )
        except EditorialTemporalViolationError as exc:
            editorial_status = "fallback_raw"
            editorial_fallback_reason = str(exc)
            editorial_temporal_violation_count = exc.violation_count
            editorial_retry_count = getattr(exc, "retry_count", 0) or 0
            editorial_output_validation_status = "invalid"
            editorial_output_invalid_reason = getattr(exc, "reason", "temporal_violation")
            editorial_result = None
        except EditorialRewriteError as exc:
            editorial_status = "fallback_raw"
            editorial_fallback_reason = str(exc)
            editorial_retry_count = getattr(exc, "retry_count", 0) or 0
            editorial_output_validation_status = "invalid"
            editorial_output_invalid_reason = getattr(exc, "reason", None)
            editorial_result = None
        except Exception as exc:
            editorial_status = "fallback_raw"
            editorial_fallback_reason = f"{exc.__class__.__name__}: {exc}"
            editorial_retry_count = 0
            editorial_output_validation_status = "invalid"
            editorial_output_invalid_reason = None
            editorial_result = None

    try:
        if mail_mode == "editorial":
            if editorial_result is not None:
                editorial_mail = build_mail_func(result, subject=job.get("subject"), editorial=editorial_result, display_title=display_title)
                write_mail_outputs(run_dir, "editorial", editorial_mail)
                editorial_mail_built = True
            else:
                raw_mail = build_mail_func(
                    temporal_safe_result_copy(result),
                    subject=job.get("subject"),
                    display_title=display_title,
                )
                write_mail_outputs(run_dir, "raw", raw_mail)
                raw_mail_built = True
        elif mail_mode == "both":
            if editorial_result is not None:
                editorial_mail = build_mail_func(
                    result,
                    subject=subject_for_variant(job.get("subject"), "editorial", mail_mode),
                    editorial=editorial_result,
                    display_title=display_title,
                )
                write_mail_outputs(run_dir, "editorial", editorial_mail)
                editorial_mail_built = True
            else:
                raw_mail = build_mail_func(
                    temporal_safe_result_copy(result),
                    subject=subject_for_variant(job.get("subject"), "raw", mail_mode),
                    display_title=display_title,
                )
                write_mail_outputs(run_dir, "raw", raw_mail)
                raw_mail_built = True
    except MailBuildError as exc:
        return run_dir, finish("build_failed", str(exc))
    except Exception as exc:
        return run_dir, finish("build_failed", f"{exc.__class__.__name__}: {exc}")

    if not mail_requested or dry_run:
        return run_dir, finish("completed_no_mail")

    try:
        smtp_config = smtp_config_factory()
        if mail_mode == "raw":
            raw_mail_sent = bool(send_mail_func(raw_mail, smtp_config, dry_run=False).get("sent"))
        elif mail_mode == "editorial":
            if editorial_mail is not None:
                editorial_mail_sent = bool(send_mail_func(editorial_mail, smtp_config, dry_run=False).get("sent"))
            elif raw_mail is not None:
                raw_mail_sent = bool(send_mail_func(raw_mail, smtp_config, dry_run=False).get("sent"))
        elif mail_mode == "both":
            raw_mail_sent = bool(send_mail_func(raw_mail, smtp_config, dry_run=False).get("sent"))
            if editorial_mail is not None:
                editorial_mail_sent = bool(send_mail_func(editorial_mail, smtp_config, dry_run=False).get("sent"))
    except MailSendError as exc:
        return run_dir, finish("mail_failed", str(exc))
    except Exception as exc:
        return run_dir, finish("mail_failed", f"{exc.__class__.__name__}: {exc}")

    expected_sent = raw_mail_sent if mail_mode == "raw" else editorial_mail_sent or raw_mail_sent
    if mail_mode == "both" and editorial_mail_built:
        expected_sent = raw_mail_sent and editorial_mail_sent
    return run_dir, finish("completed" if expected_sent else "mail_failed", None if expected_sent else "SMTP did not confirm send.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a configured Perplexica mail job.")
    parser.add_argument("--job", type=Path, required=True, help="Path to the job JSON file.")
    parser.add_argument("--dry-run", action="store_true", help="Run Perplexica and build mail, but do not send SMTP.")
    parser.add_argument("--no-mail", action="store_true", help="Override job send_mail=false.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_dir, metadata = run_job(args.job, dry_run=args.dry_run, no_mail=args.no_mail)
    except JobConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error: cannot create run directory: {exc}", file=sys.stderr)
        return 1

    print(f"status: {metadata['status']}")
    print(f"job_name: {metadata['job_name']}")
    print(f"run_dir: {run_dir}")
    print(f"chat_id: {metadata['chat_id']}")
    print(f"message_id: {metadata['message_id']}")
    print(f"sources: {metadata['source_count']}")
    print(f"cited_sources: {metadata['cited_source_count']}")
    if "search_count" in metadata:
        print(f"search_count: {metadata['search_count']}")
        print(f"completed_searches: {metadata['completed_search_count']}")
        print(f"empty_searches: {metadata['empty_search_count']}")
        print(f"failed_searches: {metadata['failed_search_count']}")
    if "temporal_validation_count" in metadata:
        print(f"temporal_validation: {metadata['temporal_validation_count']} "
              f"(current={metadata.get('current_count', 0)}, "
              f"context={metadata.get('context_count', 0)}, "
              f"mismatch={metadata.get('mismatch_count', 0)}, "
              f"unknown={metadata.get('unknown_count', 0)})")
    if "temporal_requalification_status" in metadata:
        print(f"temporal_requalification: {metadata['temporal_requalification_status']} "
              f"(eligible={metadata.get('temporal_requalification_eligible_count', 0)}, "
              f"accepted={metadata.get('temporal_requalification_accepted_count', 0)}, "
              f"rejected={metadata.get('temporal_requalification_rejected_count', 0)}, "
              f"current={metadata.get('temporal_requalification_current_count', 0)}, "
              f"context={metadata.get('temporal_requalification_context_count', 0)}, "
              f"unknown={metadata.get('temporal_requalification_unknown_count', 0)})")
    print(f"mail_requested: {metadata['mail_requested']}")
    print(f"mail_sent: {metadata['mail_sent']}")
    print(f"mail_mode: {metadata['mail_mode']}")
    print(f"raw_mail_built: {metadata['raw_mail_built']}")
    print(f"editorial_mail_built: {metadata['editorial_mail_built']}")
    print(f"editorial_requested: {metadata['editorial_requested']}")
    print(f"editorial_status: {metadata['editorial_status']}")
    if metadata.get("error"):
        print(f"error: {metadata['error']}", file=sys.stderr)
    return 0 if metadata["status"] in SUCCESS_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
