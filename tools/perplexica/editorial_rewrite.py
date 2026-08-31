#!/usr/bin/env python3
"""Best-effort editorial rewrite for Perplexica canonical results.

The module sends only the answer and cited sources to an OpenAI-compatible local
endpoint. It never sends the complete all_sources list.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mail_sender import DEFAULT_ENV_FILE, parse_env_file
from temporal_validation import DEFAULT_MISMATCH_GAP_DAYS


ENV_EDITORIAL_BASE_URL = "PERPLEXICA_EDITORIAL_BASE_URL"
ENV_EDITORIAL_API_KEY = "PERPLEXICA_EDITORIAL_API_KEY"
ENV_ADAPTER_API_KEY = "ADAPTER_API_KEY"
DEFAULT_MODEL = "local-gemma-4"
DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 1800
DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "prompt_editorial_veille.md"
CITATION_RE = re.compile(r"\[([1-9]\d*)\]")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

EDITORIAL_STATUS_CURRENT = "current"
EDITORIAL_STATUS_CONTEXT = "context"
EDITORIAL_STATUS_UNKNOWN = "unknown"
EDITORIAL_STATUS_MISMATCH = "mismatch"

NOTE_MISMATCH_EDITORIAL = (
    "Ne pas présenter cette source comme actualité récente "
    "et ne pas reprendre la date invalidée."
)
NOTE_CONTEXT_EDITORIAL = (
    "Source de contexte/référence ; ne pas présenter comme nouveauté de la période."
)
NOTE_UNKNOWN_EDITORIAL = (
    "Temporalité non vérifiée ; ne pas affirmer une date certaine."
)

_MONTHS_FR: dict[int, tuple[str, ...]] = {
    1: ("janvier",),
    2: ("février", "fevrier"),
    3: ("mars",),
    4: ("avril",),
    5: ("mai",),
    6: ("juin",),
    7: ("juillet",),
    8: ("août", "aout"),
    9: ("septembre",),
    10: ("octobre",),
    11: ("novembre",),
    12: ("décembre", "decembre"),
}

_TEMPORAL_VERB_PREFIXES = (
    r"a\s+été\s+publiée?\s+le\s+",
    r"ont\s+été\s+publiées?\s+le\s+",
    r"est\s+publiée?\s+le\s+",
    r"sont\s+publiées?\s+le\s+",
    r"publiée?\s+le\s+",
    r"publiées?\s+le\s+",
    r"publication\s+(?:du|le|au)\s+",
    r"mise(?:s)?\s+(?:à|a)\s+jour\s+(?:le\s+)?",
    r"mis(?:es)?\s+en\s+ligne\s+(?:le\s+)?",
    r"actualisée?s?\s+(?:le\s+)?",
    r"datée?s?\s+du\s+",
)
_TEMPORAL_VERB_RE = re.compile("|".join(_TEMPORAL_VERB_PREFIXES), re.IGNORECASE)

FRESHNESS_MARKER_RE = re.compile(
    r"\brécent(?:es?|s)?\b"
    r"|\brécemment\b"
    r"|\bnouveau(?:x)?\b"
    r"|\bnouvelles?\b"
    r"|\bcette\s+semaine\b"
    r"|\bces\s+derniers?\s+jours\b"
    r"|\bvien(?:t|nent)\s+(?:de\s+publier|d['\u2019]être\s+publi)"
    r"|\bpublié(?:e|es)?\s+récemment\b"
    r"|\bactualité\s+récente\b",
    re.IGNORECASE,
)



def invalid_date_forms(iso_date: str) -> list[str]:
    """Literal French/ISO representations of an ISO date (for matching)."""
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(iso_date or "").strip())
    if not match:
        return []
    year, month, day = match.group(1), int(match.group(2)), int(match.group(3))
    forms = [
        f"{year}-{match.group(2)}-{match.group(3)}",
        f"{day}/{month:02d}/{year}",
        f"{day}/{month}/{year}",
        f"{day:02d}/{month:02d}/{year}",
        f"{day}-{month:02d}-{year}",
    ]
    for name in _MONTHS_FR.get(month, ()):
        forms.append(f"{day} {name} {year}")
    seen: set[str] = set()
    unique: list[str] = []
    for form in forms:
        if form not in seen:
            seen.add(form)
            unique.append(form)
    return unique


def _invalid_date_pattern(iso_date: str):
    forms = invalid_date_forms(iso_date)
    if not forms:
        return None
    forms = sorted(forms, key=len, reverse=True)
    return re.compile("|".join(re.escape(form) for form in forms), re.IGNORECASE)


def _neutralize_line(line: str, mismatch_numbers: dict[int, list[str]]) -> str:
    numbers_in_line = extract_citation_numbers(line)
    dates: list[str] = []
    for number in numbers_in_line:
        if number in mismatch_numbers:
            dates.extend(mismatch_numbers[number])
    if not dates:
        return line
    date_patterns = [
        pattern
        for pattern in (_invalid_date_pattern(value) for value in dates)
        if pattern is not None
    ]
    if not date_patterns:
        return line
    if not any(pattern.search(line) for pattern in date_patterns):
        return line
    for pattern in date_patterns:
        verb = _TEMPORAL_VERB_RE.pattern
        date_pattern = pattern.pattern
        # "Une publication du <date> ..." en début de phrase -> "Cette source ..."
        lead = re.compile(
            r"(?:^|([\n.:;(*\-]\s*))([Uu]ne|[Uu]n)\s+(?:" + verb + r")\s*(?:"
            + date_pattern + r")",
            re.IGNORECASE | re.MULTILINE,
        )
        line = lead.sub(lambda match: (match.group(1) or "") + "Cette source", line)
        # "une publication du <date>" en milieu de phrase -> "une source"
        article = re.compile(
            r"([Uu]ne|[Uu]n)\s+(?:" + verb + r")\s*(?:" + date_pattern + r")",
            re.IGNORECASE,
        )
        line = article.sub("une source", line)
        # span temporel nu (sans article) -> suppression
        span = re.compile(
            r"(?:" + verb + r")\s*(?:" + date_pattern + r")"
            + r"(?:\s+(?:et|qui)\b)?",
            re.IGNORECASE,
        )
        line = span.sub("", line)
    for pattern in date_patterns:
        line = re.sub(
            r"(?:(?:du|de|le|au)\s+)?" + pattern.pattern, "", line, flags=re.IGNORECASE
        )
    line = re.sub(r"[ \t]{2,}", " ", line)
    line = re.sub(r"\s+([,.;:!?])", r"\1", line)
    line = re.sub(r",\s*,", ",", line)
    line = re.sub(r"\s+»", "»", line)
    line = re.sub(r"«\s+", "«", line)
    line = re.sub(r"\(\s+", "(", line)
    line = re.sub(r"\s+\)", ")", line)
    return line.strip()


def invalid_claimed_dates_for_source(temporal: dict[str, Any]) -> list[str]:
    """Claimed publication dates contradicted by the verified source date.

    A claim is invalidated only when it diverges from the verified source date
    beyond the mismatch gap (same rule as temporal_validation). This keeps the
    neutralization deterministic and limited to actually invalid claims.
    """
    if not isinstance(temporal, dict):
        return []
    claimed = [
        value
        for value in (temporal.get("claimed_dates") or [])
        if isinstance(value, str)
    ]
    source_date = temporal.get("source_date")
    if not claimed or not isinstance(source_date, str):
        return []
    try:
        parsed_source = date.fromisoformat(source_date)
    except ValueError:
        return []
    invalid: list[str] = []
    for iso in claimed:
        try:
            parsed = date.fromisoformat(iso)
        except ValueError:
            continue
        if abs((parsed - parsed_source).days) > DEFAULT_MISMATCH_GAP_DAYS:
            invalid.append(iso)
    return sorted(set(invalid))


def mismatch_dates_by_number(cited_sources: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Map each mismatch source index to its invalid claimed dates."""
    mismatch_numbers: dict[int, list[str]] = {}
    for source in cited_sources or []:
        if not isinstance(source, dict):
            continue
        temporal = source.get("temporal")
        if not isinstance(temporal, dict):
            continue
        if temporal.get("temporal_status") != EDITORIAL_STATUS_MISMATCH:
            continue
        number = source.get("index")
        if not isinstance(number, int):
            continue
        invalid = invalid_claimed_dates_for_source(temporal)
        if invalid:
            mismatch_numbers[number] = invalid
    return mismatch_numbers


def _nearest_citation_number(line: str, marker_start: int) -> int | None:
    """Return the citation number closest to a marker position within a line."""
    best_number: int | None = None
    best_distance: int | None = None
    for match in CITATION_RE.finditer(line):
        distance = min(
            abs(match.start() - marker_start),
            abs(match.end() - 1 - marker_start),
        )
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_number = int(match.group(1))
    return best_number


_V_PUBLIER_RE = re.compile(r"^vien(?:t|nent)\s+de\s+publier$", re.IGNORECASE)
_V_ETRE_PUBLIE_RE = re.compile(r"^vien(?:t|nent)\s+d['\u2019]\u00eatre\s+(publi\w*)$", re.IGNORECASE)
_PUBLIE_RECENT_RE = re.compile(r"^(publi\w*)\s+r\u00e9cemment$", re.IGNORECASE)


def _freshness_marker_replacement(matched: str) -> str:
    """Return the deterministic replacement for a freshness marker span.

    Plain markers (récents, cette semaine, ...) are removed. Verb phrases are
    rewritten without their freshness claim so the surrounding grammar stays
    readable in the temporal-safe raw fallback.
    """
    if _V_PUBLIER_RE.fullmatch(matched.strip()):
        return "ont publié" if matched.strip().startswith(("viennent", "Viennent")) else "a publié"
    etre = _V_ETRE_PUBLIE_RE.fullmatch(matched.strip())
    if etre:
        prefix = "ont été " if matched.strip().startswith(("viennent", "Viennent")) else "a été "
        return prefix + etre.group(1)
    publie = _PUBLIE_RECENT_RE.fullmatch(matched.strip())
    if publie:
        return publie.group(1)
    return ""


def neutralize_mismatch_freshness(
    response_markdown: str,
    cited_sources: list[dict[str, Any]],
) -> str:
    """Neutralize freshness markers attributed to temporal mismatch sources.

    Only markers whose nearest citation in the same line belongs to a
    temporal.status=mismatch source are removed. Markers that qualify another
    citation (e.g. "Une publication récente [11] compare une ancienne
    référence [10]") are kept. Non-temporal facts and citations are never
    removed and no new date is ever invented.
    """
    if not isinstance(response_markdown, str) or not response_markdown:
        return response_markdown or ""
    mismatch_numbers: set[int] = set()
    for source in cited_sources or []:
        if not isinstance(source, dict):
            continue
        temporal = source.get("temporal")
        if not isinstance(temporal, dict):
            continue
        if temporal.get("temporal_status") != EDITORIAL_STATUS_MISMATCH:
            continue
        number = source.get("index")
        if isinstance(number, int):
            mismatch_numbers.add(number)
    if not mismatch_numbers:
        return response_markdown

    out_lines: list[str] = []
    for line in response_markdown.split("\n"):
        if not CITATION_RE.search(line):
            out_lines.append(line)
            continue
        markers = list(FRESHNESS_MARKER_RE.finditer(line))
        if not markers:
            out_lines.append(line)
            continue
        replacements: list[tuple[int, int, str]] = []
        for marker in markers:
            number = _nearest_citation_number(line, marker.start())
            if number not in mismatch_numbers:
                continue
            matched = marker.group(0)
            replacement = _freshness_marker_replacement(matched)
            replacements.append((marker.start(), marker.end(), replacement))
        if not replacements:
            out_lines.append(line)
            continue
        for start, end, replacement in sorted(replacements, reverse=True):
            line = line[:start] + replacement + line[end:]
        line = re.sub(r"[ \t]{2,}", " ", line)
        line = re.sub(r"^[,;:\s]+", "", line)
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r",\s*,", ",", line)
        line = re.sub(r"\s+»", "»", line)
        line = re.sub(r"«\s+", "«", line)
        line = re.sub(r"\(\s+", "(", line)
        line = re.sub(r"\s+\)", ")", line)
        out_lines.append(line.strip())
    return "\n".join(out_lines)


def temporal_safe_raw_markdown(
    response_markdown: str,
    cited_sources: list[dict[str, Any]],
) -> str:
    """Neutralize invalid dates and freshness markers near mismatch sources.

    Reusable for the material sent to Gemma and for the temporal-safe raw
    fallback mail, so a fallback never reintroduces invalid temporal claims.
    """
    neutralized = neutralize_mismatch_claims(response_markdown, cited_sources)
    return neutralize_mismatch_freshness(neutralized, cited_sources)


def neutralize_mismatch_claims(
    response_markdown: str,
    cited_sources: list[dict[str, Any]],
) -> str:
    """Neutralize invalid claimed dates near mismatch sources (deterministic).

    Only lines containing both a mismatch citation and one of its invalid
    claimed dates are modified. Citations, non-temporal facts and other
    sources are left untouched. No new date is ever invented.
    """
    if not isinstance(response_markdown, str) or not response_markdown:
        return response_markdown or ""
    mismatch_numbers = mismatch_dates_by_number(cited_sources)
    if not mismatch_numbers:
        return response_markdown
    lines = response_markdown.split("\n")
    return "\n".join(_neutralize_line(line, mismatch_numbers) for line in lines)


_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")


def _citation_spans(body: str) -> list[tuple[int, int, int]]:
    return [
        (int(match.group(1)), match.start(), match.end())
        for match in CITATION_RE.finditer(body)
    ]


def _freshness_violations(
    body: str,
    mismatch_numbers: set[int],
    proximity_chars: int = 350,
) -> list[dict[str, Any]]:
    """Detect explicit freshness markers attributed to mismatch citations.

    Each citation gets a local segment bounded by neighbouring citations and
    sentence boundaries so a marker attached to another source in the same
    paragraph is never misattributed to the mismatch source.
    """
    if not mismatch_numbers or not body:
        return []
    spans = _citation_spans(body)
    violations: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for index, (number, start, end) in enumerate(spans):
        if number not in mismatch_numbers:
            continue
        left = start - proximity_chars
        right = end + proximity_chars
        if index > 0:
            left = max(left, spans[index - 1][2])
        if index < len(spans) - 1:
            right = min(right, spans[index + 1][1])
        before = list(_SENTENCE_BOUNDARY_RE.finditer(body, max(0, left), start))
        if before:
            left = max(left, before[-1].end())
        after = list(_SENTENCE_BOUNDARY_RE.finditer(body, end, min(len(body), right)))
        if after:
            right = min(right, after[0].start())
        segment = body[max(0, left):right]
        for marker in FRESHNESS_MARKER_RE.finditer(segment):
            matched = marker.group(0).strip()
            key = (number, matched.lower())
            if key in seen:
                continue
            seen.add(key)
            violations.append(
                {
                    "source_number": number,
                    "type": "freshness_marker",
                    "matched_marker": matched,
                }
            )
    return violations


def editorial_temporal_violations(
    body_markdown: str,
    cited_sources: list[dict[str, Any]],
    proximity_chars: int = 350,
) -> list[dict[str, Any]]:
    """Detect temporal violations near mismatch citations.

    Two deterministic checks run locally around each mismatch citation:
    reuse of an invalid claimed date and explicit freshness markers
    (récents, nouvelle, cette semaine, ...).
    """
    body = body_markdown or ""
    violations: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    mismatch_numbers: set[int] = set()
    for source in cited_sources or []:
        if not isinstance(source, dict):
            continue
        temporal = source.get("temporal")
        if not isinstance(temporal, dict):
            continue
        if temporal.get("temporal_status") != EDITORIAL_STATUS_MISMATCH:
            continue
        number = source.get("index")
        if not isinstance(number, int):
            continue
        mismatch_numbers.add(number)
        citation_re = re.compile(r"\[" + str(number) + r"\]")
        for iso in invalid_claimed_dates_for_source(temporal):
            if not isinstance(iso, str):
                continue
            key = (number, iso)
            if key in seen:
                continue
            found = False
            for form in invalid_date_forms(iso):
                for match in re.finditer(re.escape(form), body, re.IGNORECASE):
                    start = match.start()
                    window = body[
                        max(0, start - proximity_chars): start + proximity_chars
                    ]
                    if citation_re.search(window):
                        violations.append(
                            {
                                "source_number": number,
                                "type": "invalid_date",
                                "invalid_claimed_date": iso,
                                "matched_form": form,
                            }
                        )
                        found = True
                        break
                if found:
                    break
            if found:
                seen.add(key)
    violations.extend(_freshness_violations(body, mismatch_numbers, proximity_chars))
    return violations




class EditorialRewriteError(Exception):
    """User-facing editorial rewrite failure."""


class EditorialOutputValidationError(EditorialRewriteError):
    """Raised when the parsed editorial JSON is structurally invalid.

    ``reason`` carries a stable machine-readable code (e.g.
    body_too_short, unexpected_sectioned_output, citations_inconsistent,
    invented_citation, body_truncated, invalid_json) so the caller can trace
    why the output was rejected and decide to retry once.
    """

    def __init__(
        self,
        message: str,
        reason: str | None = None,
        retry_count: int = 0,
    ):
        super().__init__(message)
        self.reason = reason
        self.retry_count = retry_count


class EditorialTemporalViolationError(EditorialRewriteError):
    """Raised when the editorial output reuses an invalid claimed date."""

    def __init__(self, message: str, violations: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.violations = list(violations or [])
        self.violation_count = len(self.violations)
        self.reason = "temporal_violation"
        self.retry_count = 0


@dataclass(frozen=True)
class EditorialConfig:
    base_url: str
    model: str = DEFAULT_MODEL
    prompt_file: Path = DEFAULT_PROMPT_FILE
    api_key: str | None = None
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    min_body_chars: int | None = None


def extract_citation_numbers(text: str) -> list[int]:
    seen: set[int] = set()
    numbers: list[int] = []
    for match in CITATION_RE.finditer(text or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def citation_numbers_from_result(canonical_result: dict[str, Any]) -> list[int]:
    declared = canonical_result.get("citation_numbers")
    if isinstance(declared, list):
        numbers = [item for item in declared if isinstance(item, int) and item > 0]
        if numbers:
            return list(dict.fromkeys(numbers))
    return extract_citation_numbers(canonical_result.get("answer_markdown", ""))


def prepare_editorial_input(canonical_result: dict[str, Any]) -> dict[str, Any]:
    allowed_numbers = set(citation_numbers_from_result(canonical_result))
    cited_sources = []
    for source in canonical_result.get("cited_sources", []) or []:
        if not isinstance(source, dict):
            continue
        number = source.get("index")
        if not isinstance(number, int) or number not in allowed_numbers:
            continue
        entry: dict[str, Any] = {
            "number": number,
            "title": str(source.get("title") or ""),
        }
        temporal = source.get("temporal")
        if isinstance(temporal, dict):
            compact: dict[str, Any] = {
                "status": temporal.get("temporal_status"),
                "role": temporal.get("temporal_role"),
            }
            source_date = temporal.get("source_date")
            if isinstance(source_date, str) and source_date:
                compact["source_date"] = source_date
            status = temporal.get("temporal_status")
            note = temporal.get("note")
            existing_note = note.strip() if isinstance(note, str) else ""
            if status == EDITORIAL_STATUS_MISMATCH:
                invalid = invalid_claimed_dates_for_source(temporal)
                if invalid:
                    compact["invalid_claimed_dates"] = invalid
                compact["note"] = NOTE_MISMATCH_EDITORIAL
            elif status == EDITORIAL_STATUS_CONTEXT:
                compact["note"] = existing_note or NOTE_CONTEXT_EDITORIAL
            elif status == EDITORIAL_STATUS_UNKNOWN:
                compact["note"] = existing_note or NOTE_UNKNOWN_EDITORIAL
            else:
                if existing_note:
                    compact["note"] = existing_note
            entry["temporal"] = compact
        cited_sources.append(entry)
    raw_response = canonical_result.get("editorial_answer_markdown") or canonical_result.get("answer_markdown") or ""
    editorial_response = temporal_safe_raw_markdown(
        raw_response,
        canonical_result.get("cited_sources") or [],
    )
    payload = {
        "question": canonical_result.get("question") or "",
        "response": editorial_response,
        "citation_numbers": sorted(allowed_numbers),
        "cited_sources": cited_sources,
    }
    temporal_summary = canonical_result.get("temporal_validation")
    if isinstance(temporal_summary, dict) and temporal_summary.get("status") == "completed":
        payload["temporal_validation"] = {
            "validated_count": temporal_summary.get("temporal_validation_count"),
            "current_count": temporal_summary.get("current_count"),
            "context_count": temporal_summary.get("context_count"),
            "mismatch_count": temporal_summary.get("mismatch_count"),
            "unknown_count": temporal_summary.get("unknown_count"),
        }
    searches = canonical_result.get("searches")
    if isinstance(searches, dict):
        payload["searches"] = {
            name: {
                "status": summary.get("status"),
                "source_count": summary.get("source_count"),
                "cited_source_count": summary.get("cited_source_count"),
            }
            for name, summary in searches.items()
            if isinstance(summary, dict) and summary.get("status") == "completed"
        }
    return payload


def read_editorial_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EditorialRewriteError(f"Cannot read editorial prompt: {path}") from exc
    if not prompt.strip():
        raise EditorialRewriteError(f"Editorial prompt is empty: {path}")
    return prompt


def build_messages(system_prompt: str, editorial_input: dict[str, Any]) -> list[dict[str, str]]:
    user_payload = json.dumps(editorial_input, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Réécris la veille à partir de ce JSON source strictement limité :\n" + user_payload,
        },
    ]


def chat_completions_payload(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
        "response_format": {"type": "json_object"},
    }


def extract_response_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content
    for key in ("content", "text", "response"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, str):
            return value
    raise EditorialRewriteError("Editorial endpoint response did not contain text content.")


def call_editorial_llm(config: EditorialConfig, messages: list[dict[str, str]]) -> str:
    if not config.base_url:
        raise EditorialRewriteError(f"Missing editorial base URL. Set {ENV_EDITORIAL_BASE_URL}.")
    if config.timeout <= 0 or config.timeout > MAX_TIMEOUT_SECONDS:
        raise EditorialRewriteError(
            f"Editorial timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds."
        )

    endpoint = config.base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(chat_completions_payload(config.model, messages), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = Request(endpoint, data=body, method="POST", headers=headers)
    try:
        with urlopen(request, timeout=config.timeout) as response:
            status = response.getcode()
            raw_body = response.read()
    except HTTPError as exc:
        raise EditorialRewriteError(f"Editorial HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise EditorialRewriteError(f"Editorial network error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise EditorialRewriteError(f"Editorial timeout after {config.timeout} seconds.") from exc

    if status < 200 or status >= 300:
        raise EditorialRewriteError(f"Editorial unexpected HTTP status: {status}")
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise EditorialRewriteError("Editorial endpoint returned non UTF-8 data.") from exc
    except json.JSONDecodeError as exc:
        raise EditorialRewriteError("Editorial endpoint returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise EditorialRewriteError("Editorial endpoint returned a non-object JSON response.")
    return extract_response_content(payload)


RETRY_STRUCTURE_INSTRUCTION = (
    "Retourne exactement et uniquement les clés :\n"
    "title\nbody_markdown\ncitation_numbers\n\n"
    "Tout le contenu éditorial doit être dans body_markdown.\n"
    "Ne crée aucune autre section ou clé."
)


def reinforce_editorial_prompt(system_prompt: str) -> str:
    """Append the strict-output instruction used for the single retry pass."""
    prompt = (system_prompt or "").rstrip()
    if prompt and not prompt.endswith("\n"):
        prompt += "\n"
    return prompt + "\n" + RETRY_STRUCTURE_INSTRUCTION + "\n"


SECTIONED_KEY_MIN_CHARS = 200
SECTIONED_EXCLUDED_KEYS = frozenset({"title", "body_markdown", "citation_numbers"})


def _sectioned_output(
    editorial: dict[str, Any],
    body: str,
    min_body_chars: int | None,
) -> bool:
    """True when body_markdown is short and long content sits in other keys."""
    if not min_body_chars:
        return False
    if len(body) >= min_body_chars:
        return False
    for key, value in editorial.items():
        if key in SECTIONED_EXCLUDED_KEYS:
            continue
        if isinstance(value, str) and len(value) >= SECTIONED_KEY_MIN_CHARS:
            return True
        if isinstance(value, list):
            total = sum(len(item) if isinstance(item, str) else 0 for item in value)
            if total >= SECTIONED_KEY_MIN_CHARS:
                return True
    return False


def _body_looks_truncated(body: str) -> bool:
    stripped = (body or "").rstrip()
    if not stripped:
        return True
    if stripped.count("```") % 2 != 0:
        return True
    if re.search(r"(^|\n)#{1,6}\s*[^\n]*$", stripped):
        return True
    return False


def validate_editorial_output(
    editorial: dict[str, Any],
    allowed_numbers: list[int],
    min_body_chars: int | None = None,
) -> None:
    """Validate the parsed editorial JSON structure and citations.

    Raises EditorialOutputValidationError with a precise reason when the
    output is structurally degraded so the pipeline can retry once.
    """
    title = editorial.get("title")
    if not isinstance(title, str) or not title.strip():
        raise EditorialOutputValidationError(
            "Editorial JSON missing string field 'title'.", "invalid_json"
        )
    body = editorial.get("body_markdown")
    if not isinstance(body, str):
        raise EditorialOutputValidationError(
            "Editorial JSON field 'body_markdown' must be a string.", "invalid_json"
        )
    if _sectioned_output(editorial, body, min_body_chars):
        raise EditorialOutputValidationError(
            "Editorial JSON puts long content in unexpected keys while "
            "body_markdown is short.",
            "unexpected_sectioned_output",
        )
    if min_body_chars and len(body) < min_body_chars:
        raise EditorialOutputValidationError(
            "Editorial body_markdown too short ({} chars < {}).".format(
                len(body), min_body_chars
            ),
            "body_too_short",
        )
    if _body_looks_truncated(body):
        raise EditorialOutputValidationError(
            "Editorial body_markdown appears truncated.", "body_truncated"
        )

    allowed = set(allowed_numbers)
    detected = extract_citation_numbers(body)
    unknown = [number for number in detected if number not in allowed]
    if unknown:
        raise EditorialOutputValidationError(
            "Editorial output contains unknown citations: {}".format(unknown),
            "invented_citation",
        )
    declared = [
        number
        for number in (editorial.get("citation_numbers") or [])
        if isinstance(number, int) and number > 0
    ]
    declared_order = list(dict.fromkeys(declared))
    normalized = detected != declared_order
    editorial["citation_numbers"] = detected
    return {
        "normalized": normalized,
        "declared_count": len(declared),
        "actual_count": len(detected),
        "declared_unused_count": len(set(declared) - set(detected)),
    }


def parse_editorial_output(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise EditorialRewriteError("Editorial output is empty.")
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in JSON_BLOCK_RE.finditer(text))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            last_error = EditorialRewriteError("Editorial JSON must be an object.")
            continue
        title = parsed.get("title")
        body = parsed.get("body_markdown")
        numbers = parsed.get("citation_numbers", [])
        if not isinstance(title, str) or not title.strip():
            raise EditorialRewriteError("Editorial JSON missing string field 'title'.")
        if not isinstance(body, str) or not body.strip():
            raise EditorialRewriteError("Editorial JSON missing string field 'body_markdown'.")
        if not isinstance(numbers, list) or not all(isinstance(item, int) and item > 0 for item in numbers):
            raise EditorialRewriteError("Editorial JSON field 'citation_numbers' must be a list of positive integers.")
        parsed["title"] = title.strip()
        parsed["body_markdown"] = body.strip()
        parsed["citation_numbers"] = list(dict.fromkeys(numbers))
        return parsed
    raise EditorialRewriteError("Editorial output did not contain valid JSON.") from last_error


def validate_editorial_citations(editorial: dict[str, Any], allowed_numbers: list[int]) -> None:
    """Backward-compatible wrapper over validate_editorial_output."""
    validate_editorial_output(editorial, allowed_numbers)


def build_editorial_result(
    canonical_result: dict[str, Any],
    editorial: dict[str, Any],
    raw_text: str,
    model: str,
    temporal_violation_count: int = 0,
    retry_count: int = 0,
    output_validation_status: str = "ok",
    output_invalid_reason: str | None = None,
    citation_numbers_normalized: bool | None = None,
    declared_citation_count: int | None = None,
    actual_citation_count: int | None = None,
    declared_unused_citation_count: int | None = None,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "model": model,
        "title": editorial["title"],
        "body_markdown": editorial["body_markdown"],
        "citation_numbers": editorial["citation_numbers"],
        "raw_length": len(raw_text),
        "source_chat_id": canonical_result.get("chat_id"),
        "source_message_id": canonical_result.get("message_id"),
        "editorial_temporal_violation_count": temporal_violation_count,
        "temporal_violation_count": temporal_violation_count,
        "editorial_retry_count": retry_count,
        "editorial_output_validation_status": output_validation_status,
        "editorial_output_invalid_reason": output_invalid_reason,
        "editorial_citation_numbers_normalized": citation_numbers_normalized,
        "editorial_declared_citation_count": declared_citation_count,
        "editorial_actual_citation_count": actual_citation_count,
        "editorial_declared_unused_citation_count": declared_unused_citation_count,
    }


def _editorial_attempt_diagnostic(
    *,
    attempt_number: int,
    raw_text: str,
    reason: str,
    error: str,
    editorial: dict[str, Any] | None,
    temporal_violation_count: int | None = None,
) -> dict[str, Any]:
    body = editorial.get("body_markdown") if isinstance(editorial, dict) else None
    declared = []
    if isinstance(editorial, dict):
        declared = [
            number
            for number in (editorial.get("citation_numbers") or [])
            if isinstance(number, int) and number > 0
        ]
    actual = extract_citation_numbers(body or "")
    return {
        "attempt": attempt_number,
        "reason": reason,
        "error": error,
        "declared_citations": declared,
        "actual_citations": actual,
        "body_length": len(body) if isinstance(body, str) else None,
        "temporal_violation_count": temporal_violation_count,
        "raw_length": len(raw_text or ""),
    }


def _write_editorial_attempt_diagnostic(
    diagnostics_dir: Path | None,
    attempt_number: int,
    raw_text: str,
    payload: dict[str, Any],
) -> None:
    if diagnostics_dir is None:
        return
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / f"editorial_attempt_{attempt_number}_raw.txt").write_text(
        raw_text or "",
        encoding="utf-8",
    )
    (diagnostics_dir / f"editorial_attempt_{attempt_number}_error.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def rewrite_editorial(
    canonical_result: dict[str, Any],
    config: EditorialConfig,
    *,
    llm_func=call_editorial_llm,
    diagnostics_dir: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Rewrite the editorial answer with a single controlled retry.

    Transport/config errors (raised by llm_func) are never retried. A
    structurally degraded or temporally violating output triggers exactly one
    new Gemma pass with a reinforced structure instruction. If the retry also
    fails, the exception propagates and the caller falls back to the raw mail.
    """
    editorial_input = prepare_editorial_input(canonical_result)
    system_prompt = read_editorial_prompt(config.prompt_file)
    retry_count = 0
    while True:
        prompt = reinforce_editorial_prompt(system_prompt) if retry_count else system_prompt
        messages = build_messages(prompt, editorial_input)
        raw_text = llm_func(config, messages)
        attempt_number = retry_count + 1
        citation_stats: dict[str, Any] | None = None
        editorial: dict[str, Any] | None = None
        try:
            editorial = parse_editorial_output(raw_text)
            citation_stats = validate_editorial_output(
                editorial,
                editorial_input["citation_numbers"],
                min_body_chars=config.min_body_chars,
            )
            violations = editorial_temporal_violations(
                editorial["body_markdown"],
                canonical_result.get("cited_sources") or [],
            )
            if violations:
                raise EditorialTemporalViolationError(
                    "Editorial output contains temporal violations "
                    "(editorial_temporal_violation_count={}): {}".format(
                        len(violations),
                        [violation["source_number"] for violation in violations],
                    ),
                    violations,
                )
        except EditorialTemporalViolationError as exc:
            _write_editorial_attempt_diagnostic(
                diagnostics_dir,
                attempt_number,
                raw_text,
                _editorial_attempt_diagnostic(
                    attempt_number=attempt_number,
                    raw_text=raw_text,
                    reason=exc.reason,
                    error=str(exc),
                    editorial=editorial,
                    temporal_violation_count=exc.violation_count,
                ),
            )
            if retry_count >= 1:
                exc.retry_count = retry_count
                raise
            retry_count += 1
            continue
        except EditorialOutputValidationError as exc:
            _write_editorial_attempt_diagnostic(
                diagnostics_dir,
                attempt_number,
                raw_text,
                _editorial_attempt_diagnostic(
                    attempt_number=attempt_number,
                    raw_text=raw_text,
                    reason=exc.reason,
                    error=str(exc),
                    editorial=editorial,
                ),
            )
            if retry_count >= 1:
                exc.retry_count = retry_count
                raise
            retry_count += 1
            continue
        except EditorialRewriteError as exc:
            wrapped = EditorialOutputValidationError(
                str(exc), reason="invalid_json", retry_count=retry_count
            )
            _write_editorial_attempt_diagnostic(
                diagnostics_dir,
                attempt_number,
                raw_text,
                _editorial_attempt_diagnostic(
                    attempt_number=attempt_number,
                    raw_text=raw_text,
                    reason=wrapped.reason,
                    error=str(wrapped),
                    editorial=editorial,
                ),
            )
            if retry_count >= 1:
                raise wrapped
            retry_count += 1
            continue
        return (
            build_editorial_result(
                canonical_result,
                editorial,
                raw_text,
                config.model,
                temporal_violation_count=0,
                retry_count=retry_count,
                output_validation_status="ok",
                output_invalid_reason=None,
                citation_numbers_normalized=bool(
                    citation_stats and citation_stats["normalized"]
                ),
                declared_citation_count=(
                    citation_stats or {}
                ).get("declared_count"),
                actual_citation_count=(
                    citation_stats or {}
                ).get("actual_count"),
                declared_unused_citation_count=(
                    citation_stats or {}
                ).get("declared_unused_count"),
            ),
            raw_text,
        )


def env_value(name: str, file_values: dict[str, str]) -> str:
    if name in os.environ:
        return os.environ.get(name, "")
    return file_values.get(name, "")


def editorial_config_from_env(
    model: str | None = None,
    prompt_file: Path | None = None,
    timeout: int | None = None,
    env_file: Path = DEFAULT_ENV_FILE,
    min_body_chars: int | None = None,
) -> EditorialConfig:
    file_values = parse_env_file(env_file)
    base_url = env_value(ENV_EDITORIAL_BASE_URL, file_values).strip()
    api_key = (
        env_value(ENV_EDITORIAL_API_KEY, file_values).strip()
        or env_value(ENV_ADAPTER_API_KEY, file_values).strip()
        or None
    )
    return EditorialConfig(
        base_url=base_url,
        model=model or DEFAULT_MODEL,
        prompt_file=prompt_file or DEFAULT_PROMPT_FILE,
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout,
        min_body_chars=min_body_chars,
    )


def load_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EditorialRewriteError(f"Cannot read input result: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EditorialRewriteError(f"Invalid input JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise EditorialRewriteError("Input JSON must be an object.")
    return payload.get("result") if isinstance(payload.get("result"), dict) else payload


def write_editorial_files(output_dir: Path, editorial: dict[str, Any], raw_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "editorial_raw.txt").write_text(raw_text, encoding="utf-8")
    (output_dir / "editorial.json").write_text(
        json.dumps(editorial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the optional editorial rewrite for a Perplexica result.")
    parser.add_argument("--input", type=Path, required=True, help="Canonical Perplexica result JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for editorial.json and editorial_raw.txt.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = load_result(args.input)
        config = editorial_config_from_env(
            model=args.model,
            prompt_file=args.prompt_file,
            timeout=args.timeout,
            env_file=args.env_file,
        )
        editorial, raw_text = rewrite_editorial(result, config)
        write_editorial_files(args.output_dir, editorial, raw_text)
        print(f"status: {editorial['status']}")
        print(f"model: {editorial['model']}")
        print(f"citations: {editorial['citation_numbers']}")
        print(f"editorial_json: {args.output_dir / 'editorial.json'}")
        print(f"editorial_raw: {args.output_dir / 'editorial_raw.txt'}")
        return 0
    except EditorialRewriteError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
