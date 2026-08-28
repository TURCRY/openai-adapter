#!/usr/bin/env python3
"""Temporal validation of cited Perplexica sources (V1).

Runs between the multi-search aggregation and the Gemma editorial pass. For
each cited source it tries to determine the real publication date of the page
and compares it with the date(s) claimed in the Perplexica answers.

Stdlib only (urllib + concurrent.futures). No external scraping dependency and
no hard coupling to the fixed PC; an optional browser fallback can be added
later through the same fetch_page() hook.
"""

from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

ACCESS_ACCESSIBLE = "accessible"
ACCESS_PAYWALLED = "paywalled"
ACCESS_BLOCKED = "blocked"
ACCESS_JS_REQUIRED = "js_required"
ACCESS_NOT_FOUND = "not_found"
ACCESS_TIMEOUT = "timeout"
ACCESS_UNKNOWN = "unknown"
ACCESS_STATUSES = frozenset(
    {
        ACCESS_ACCESSIBLE,
        ACCESS_PAYWALLED,
        ACCESS_BLOCKED,
        ACCESS_JS_REQUIRED,
        ACCESS_NOT_FOUND,
        ACCESS_TIMEOUT,
        ACCESS_UNKNOWN,
    }
)

STATUS_CURRENT = "current"
STATUS_CONTEXT = "context"
STATUS_MISMATCH = "mismatch"
STATUS_UNKNOWN = "unknown"
TEMPORAL_STATUSES = frozenset({STATUS_CURRENT, STATUS_CONTEXT, STATUS_MISMATCH, STATUS_UNKNOWN})

ROLE_CURRENT = "current"
ROLE_CONTEXT = "context"
ROLE_UNKNOWN = "unknown"
TEMPORAL_ROLES = frozenset({ROLE_CURRENT, ROLE_CONTEXT, ROLE_UNKNOWN})

VERIF_DIRECT = "direct"
VERIF_INDIRECT = "indirect"
VERIF_UNKNOWN = "unknown"

CONF_HIGH = "high"
CONF_MEDIUM = "medium"
CONF_LOW = "low"

DEFAULT_TIMEOUT_SECONDS = 6.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_CONCURRENCY = 4
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 "
    "PerplexicaTemporalValidator/1.0"
)
DEFAULT_WINDOW_DAYS = 7
DEFAULT_RECENT_TOLERANCE_DAYS = 7
DEFAULT_MISMATCH_GAP_DAYS = 30
ADJACENCY_CHARS = 60
MAX_HTML_BYTES = 250000

FRENCH_MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}
MONTH_ALT = "|".join(FRENCH_MONTHS.keys())

NOTE_CONTEXT = "Source utilisable comme cadre ou contexte, pas comme nouveaut\u00e9 de la p\u00e9riode."
NOTE_MISMATCH = "Ne pas pr\u00e9senter cette source comme actualit\u00e9 r\u00e9cente."
NOTE_UNKNOWN = "Date de publication non v\u00e9rifi\u00e9e."
NOTE_UNKNOWN_WITH_CLAIM = (
    "Date de publication non v\u00e9rifi\u00e9e ; ne pas pr\u00e9senter une date comme certaine."
)
NOTE_OLD_CONTEXT = "Source ant\u00e9rieure \u00e0 la fen\u00eatre d'actualit\u00e9 ; utilisable comme contexte."
NOTE_LOW_CONFIDENCE = "Date r\u00e9elle de faible confiance ; ne pas pr\u00e9senter une date comme certaine."
NOTE_VISIBLE_DIVERGENCE = "Publication visible r\u00e9cente divergente du JSON-LD."

CLAIM_PUBLICATION = "publication_claim"
CLAIM_UPDATE = "update_claim"
CLAIM_DECISION = "decision_date"
CLAIM_LEGAL_TEXT = "legal_text_date"
CLAIM_GENERIC = "generic_date"
CLAIM_CATEGORIES = frozenset(
    {CLAIM_PUBLICATION, CLAIM_UPDATE, CLAIM_DECISION, CLAIM_LEGAL_TEXT, CLAIM_GENERIC}
)

_CITATION_RE = re.compile(r"\[([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]")
_CLAIMED_DATE_RE = re.compile(
    r"(\d{1,2})\s+(" + MONTH_ALT + r")\s+(\d{4})"
    r"|(\d{4})-(\d{2})-(\d{2})"
    r"|(\d{1,2})/(\d{1,2})/(\d{4})",
    re.IGNORECASE,
)
_DATE_TEXT_RE = re.compile(
    r"(?:publiee?\s+(?:le\s+)?|publication\s+(?:du\s+|le\s+)?"
    r"|mise?\s+a\s+jour\s+(?:le\s+)?|actualisee?\s+(?:le\s+)?"
    r"|datee?\s+(?:du\s+|de\s+publication\s+)?|du\s+)"
    r"(\d{1,2})\s+(" + MONTH_ALT + r")\s+(\d{4})",
    re.IGNORECASE,
)
_DOC_SIGNATURE_RE = re.compile(
    r"\b(?:decret|arrete|loi|ordonnance|decision|avis|deliberation|circulaire)"
    r"\s+n\s*[\w.\-]+\s+du\s+(\d{1,2})\s+(" + MONTH_ALT + r")\s+(\d{4})",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(
    r"version\s+en\s+vigueur\s+(?:depuis\s+le|au)\s+(\d{1,2})\s+(" + MONTH_ALT + r")\s+(\d{4})",
    re.IGNORECASE,
)

_CURRENT_MARKERS = (
    "publiee le",
    "publie le",
    "a ete publiee",
    "a ete publie",
    "vient d etre publie",
    "mise a jour le",
    "mis a jour le",
    "actualisee",
    "actualise",
    "actualite",
    "actualites",
    "recente",
    "recent",
    "nouvelle",
    "nouveau",
    "cette semaine",
    "derniers jours",
    "aujourd hui",
    "annonce",
    "publiees",
    "publies",
    "mis en ligne",
)
_CONTEXT_MARKERS = (
    "rappel",
    "rappelle",
    "cadre",
    "contexte",
    "reference",
    "references",
    "en vigueur",
    "version en vigueur",
    "fond",
    "anterieure",
    "anterieur",
    "ancienne",
    "ancien",
    "anciens",
    "historique",
    "retrospective",
    "reste une reference",
    "texte de reference",
)

_PAYWALL_MARKERS = (
    "article reserve aux abonnes",
    "article reserve a nos abonnes",
    "reserve aux abonnes",
    "cet article est reserve aux abonnes",
    "votre abonnement n'autorise pas la lecture",
    "votre abonnement ne permet pas de lire",
    "lecture restreinte",
    "contenu reserve aux abonnes",
    "acces reserve aux abonnes",
    "paywall",
)
_JS_MARKERS = (
    'id="root"',
    "id='root'",
    'id="__next"',
    "__NEXT_DATA__",
    "__NUXT__",
    'id="app"',
    "data-server-rendered",
    'id="facebook"',
)
def _norm(value: Any) -> str:
    """Strip accents and normalize to lowercase ASCII."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFD", str(value))
    return text.encode("ascii", "ignore").decode("ascii")


def parse_date(value: Any) -> date | None:
    """Parse a date from ISO, dd/mm/yyyy or French text formats."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        year, month, day = (int(part) for part in match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if match:
        day, month, year = (int(part) for part in match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None
    norm = _norm(text).lower()
    match = re.match(r"^(\d{1,2})\s+([a-z]+)\s+(\d{4})$", norm)
    if match:
        day = int(match.group(1))
        month = FRENCH_MONTHS.get(match.group(2))
        year = int(match.group(3))
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


def _iter_json_ld_blocks(html: str):
    pattern = re.compile(
        r"<script[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(html or ""):
        yield match.group(1)


def _collect_dates(obj: Any, published: list[str], modified: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = str(key).strip()
            if name in ("datePublished", "dateCreated") and isinstance(value, str):
                published.append(value)
            elif name == "dateModified" and isinstance(value, str):
                modified.append(value)
            else:
                _collect_dates(value, published, modified)
    elif isinstance(obj, list):
        for item in obj:
            _collect_dates(item, published, modified)


def extract_json_ld_dates(html: str) -> tuple[list[date], list[date]]:
    published: list[date] = []
    modified: list[date] = []
    for block in _iter_json_ld_blocks(html):
        try:
            data = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            continue
        raw_published: list[str] = []
        raw_modified: list[str] = []
        _collect_dates(data, raw_published, raw_modified)
        published.extend(parsed for parsed in (parse_date(value) for value in raw_published) if parsed)
        modified.extend(parsed for parsed in (parse_date(value) for value in raw_modified) if parsed)
    return published, modified


def _parse_meta_tags(html: str) -> list[dict[str, str]]:
    tags: list[dict[str, str]] = []
    tag_pattern = re.compile(r"<meta\b([^>]*)>", re.IGNORECASE)
    attr_pattern = re.compile(r"([\w:.-]+)\s*=\s*[\"']([^\"']*)[\"']")
    for match in tag_pattern.finditer(html or ""):
        attrs = dict(attr_pattern.findall(match.group(1)))
        if attrs:
            tags.append(attrs)
    return tags


_OG_PUBLISHED_KEYS = {"article:published_time", "og:article:published_time"}
_GENERIC_PUBLISHED_KEYS = {"datepublished", "dc.date", "dcterms.date", "date"}
_MODIFIED_KEYS = {"article:modified_time", "og:article:modified_time", "datemodified"}


def extract_meta_dates(html: str) -> tuple[list[date], list[date], list[date]]:
    """Return (og_published, generic_published, modified) dates from <meta>."""
    og_published: list[date] = []
    generic_published: list[date] = []
    modified: list[date] = []
    for attrs in _parse_meta_tags(html):
        key = None
        for candidate in ("property", "name", "itemprop"):
            value = attrs.get(candidate)
            if value:
                key = _norm(value).lower()
                break
        if not key:
            continue
        parsed = parse_date(attrs.get("content"))
        if not parsed:
            continue
        if key in _OG_PUBLISHED_KEYS:
            og_published.append(parsed)
        elif key in _GENERIC_PUBLISHED_KEYS:
            generic_published.append(parsed)
        elif key in _MODIFIED_KEYS:
            modified.append(parsed)
    return og_published, generic_published, modified


def extract_time_dates(html: str) -> list[date]:
    result: list[date] = []
    tag_pattern = re.compile(r"<time\b([^>]*)>(.*?)</time>", re.IGNORECASE | re.DOTALL)
    attr_pattern = re.compile(r"([\w:.-]+)\s*=\s*[\"']([^\"']*)[\"']")
    for match in tag_pattern.finditer(html or ""):
        attrs = dict(attr_pattern.findall(match.group(1)))
        parsed = parse_date(attrs.get("datetime") or match.group(2))
        if parsed:
            result.append(parsed)
    return result


def _html_to_text(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return _norm(unescape(text))


def extract_text_dates(html: str) -> list[date]:
    text = _html_to_text(html)
    result: list[date] = []
    for match in _DATE_TEXT_RE.finditer(text):
        parsed = parse_date("{0} {1} {2}".format(match.group(1), match.group(2), match.group(3)))
        if parsed:
            result.append(parsed)
    return result


_VISIBLE_PUB_PREFIX = re.compile(
    r"\b(?:publi(?:e|ee|es|ees)?\s+le\s+|publication\s*:\s*|"
    r"mis(?:e)?\s+en\s+ligne\s+(?:le\s+)?)",
    re.IGNORECASE,
)
_VISIBLE_UPDATE_PREFIX = re.compile(
    r"\b(?:mis(?:e)?\s+a\s+jour\s+(?:le\s+)?|actualis(?:e|ee|es|ees)?\s+(?:le\s+)?|"
    r"derniere\s+mise\s+a\s+jour\s*:?\s*|modifi(?:e|ee)?\s+(?:le\s+)?)",
    re.IGNORECASE,
)


def extract_visible_dates(html: str, marker_pattern: Any) -> list[date]:
    """Extract dates that immediately follow an explicit visible marker."""
    text = _html_to_text(html)
    result: list[date] = []
    for match in marker_pattern.finditer(text):
        tail = text[match.end(): match.end() + 60]
        date_match = _CLAIMED_DATE_RE.search(tail)
        if date_match:
            candidate = _date_from_claim_groups(date_match.groups())
            if candidate is not None and candidate not in result:
                result.append(candidate)
    return result


def extract_visible_publication_dates(html: str) -> list[date]:
    """Dates shown as "Publié le ..." / "Publication : ..." on the page."""
    return extract_visible_dates(html, _VISIBLE_PUB_PREFIX)


def extract_visible_update_dates(html: str) -> list[date]:
    """Dates shown as "Mis à jour le ..." / "Actualisé le ..." on the page."""
    return extract_visible_dates(html, _VISIBLE_UPDATE_PREFIX)


def extract_document_signature_dates(html: str) -> list[date]:
    """Extract the date of an official text signature, e.g. 'decret n 2025-660 du 18 juillet 2025'."""
    text = _html_to_text(html)
    result: list[date] = []
    for match in _DOC_SIGNATURE_RE.finditer(text):
        parsed = parse_date("{0} {1} {2}".format(match.group(1), match.group(2), match.group(3)))
        if parsed:
            result.append(parsed)
    return result


def legifrance_url_dates(url: str | None) -> tuple[date | None, str | None, str | None]:
    """Specific rules for Legifrance URLs (JORF issue pages, listings, codes)."""
    if not url:
        return None, None, None
    parts = urlsplit(url)
    host = (parts.netloc or "").lower()
    if "legifrance" not in host:
        return None, None, None
    path = parts.path or ""
    match = re.search(r"/jorf/jo/(\d{4})/(\d{2})/(\d{2})", path)
    if match:
        year, month, day = (int(part) for part in match.groups())
        try:
            return date(year, month, day), "legifrance_url", CONF_HIGH
        except ValueError:
            pass
    query = parse_qs(parts.query)
    if "datePubli" in query:
        parsed = parse_date(query["datePubli"][0])
        if parsed:
            return parsed, "legifrance_listing", CONF_LOW
    if "/codes/" in path:
        return None, "legifrance_codes", None
    return None, None, None


def institutional_url_dates(url: str | None) -> tuple[date | None, str | None, str | None]:
    """URL date rules for institutional sites and article-style URL paths."""
    if not url:
        return None, None, None
    parts = urlsplit(url)
    host = (parts.netloc or "").lower()
    path = parts.path or ""
    if "conseil-etat.fr" in host:
        match = re.search(r"/decision/(\d{4})-(\d{2})-(\d{2})/", path)
        if match:
            year, month, day = (int(part) for part in match.groups())
            try:
                return date(year, month, day), "institutional_url", CONF_HIGH
            except ValueError:
                pass
    if "legifrance" not in host:
        match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", path)
        if match:
            year, month, day = (int(part) for part in match.groups())
            try:
                return date(year, month, day), "url", CONF_MEDIUM
            except ValueError:
                pass
    return None, None, None


def _extract_legifrance_version_date(html: str) -> date | None:
    text = _html_to_text(html)
    match = _VERSION_RE.search(text)
    if match:
        return parse_date("{0} {1} {2}".format(match.group(1), match.group(2), match.group(3)))
    return None


def extract_source_date(html: str, url: str | None = None) -> dict[str, Any]:
    """Extract the best real publication date with its evidence and confidence."""
    info: dict[str, Any] = {
        "source_date": None,
        "modified_date": None,
        "date_evidence": None,
        "date_confidence": None,
        "visible_publication_date": None,
        "visible_update_date": None,
        "visible_publication_dates": None,
        "visible_update_dates": None,
    }
    json_published, json_modified = extract_json_ld_dates(html)
    og_published, generic_published, meta_modified = extract_meta_dates(html)
    time_dates = extract_time_dates(html)
    text_dates = extract_text_dates(html)
    signature_dates = extract_document_signature_dates(html)
    visible_pub = extract_visible_publication_dates(html)
    visible_upd = extract_visible_update_dates(html)
    url_date, url_evidence, url_confidence = legifrance_url_dates(url)
    inst_date, inst_evidence, inst_confidence = institutional_url_dates(url)

    if visible_pub:
        info["visible_publication_date"] = visible_pub[0]
        info["visible_publication_dates"] = visible_pub
    if visible_upd:
        info["visible_update_date"] = visible_upd[0]
        info["visible_update_dates"] = visible_upd

    ranked: list[tuple[int, date, str, str]] = []

    def add(rank: int, value: date | None, evidence: str, confidence: str) -> None:
        if value is not None:
            ranked.append((rank, value, evidence, confidence))

    add(10, json_published[0] if json_published else None, "json_ld", CONF_HIGH)
    add(20, og_published[0] if og_published else None, "meta", CONF_HIGH)
    if url_date is not None and url_confidence == CONF_HIGH:
        add(30, url_date, str(url_evidence), CONF_HIGH)
    if inst_date is not None and inst_confidence == CONF_HIGH:
        add(30, inst_date, str(inst_evidence), CONF_HIGH)
    add(40, signature_dates[0] if signature_dates else None, "doc_signature", CONF_MEDIUM)
    if inst_date is not None and inst_confidence == CONF_MEDIUM:
        add(50, inst_date, str(inst_evidence), CONF_MEDIUM)
    add(60, time_dates[0] if time_dates else None, "time", CONF_MEDIUM)
    add(70, generic_published[0] if generic_published else None, "meta_date", CONF_MEDIUM)
    if url_date is not None and url_confidence == CONF_LOW:
        add(80, url_date, str(url_evidence), CONF_LOW)
    add(90, text_dates[0] if text_dates else None, "text", CONF_LOW)

    ranked.sort(key=lambda item: item[0])
    if ranked:
        _, chosen, evidence, confidence = ranked[0]
        info["source_date"] = chosen
        info["date_evidence"] = evidence
        info["date_confidence"] = confidence

    modified = json_modified or meta_modified
    if modified:
        info["modified_date"] = modified[0]
        if not info["source_date"]:
            info["date_evidence"] = "modified_time"
            info["date_confidence"] = CONF_LOW

    if not info["source_date"] and url_evidence == "legifrance_codes":
        version = _extract_legifrance_version_date(html)
        if version is not None:
            info["modified_date"] = version
            info["date_evidence"] = "legifrance_version"
            info["date_confidence"] = CONF_LOW
    return info
def is_paywalled(html: str) -> bool:
    norm = _norm(html).lower()
    return any(marker in norm for marker in _PAYWALL_MARKERS)


def _html_word_count(html: str) -> int:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html or "", flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return len(_norm(text).split())


def is_js_heavy(html: str) -> bool:
    if not html:
        return False
    if _html_word_count(html) > 300:
        return False
    return any(marker in html for marker in _JS_MARKERS)


def detect_access_status(status_code: int | None, html: str = "", url: str | None = None) -> str:
    if status_code == 200:
        if is_paywalled(html):
            return ACCESS_PAYWALLED
        if is_js_heavy(html):
            return ACCESS_JS_REQUIRED
        return ACCESS_ACCESSIBLE
    if status_code is None:
        return ACCESS_TIMEOUT
    if status_code in (401, 403, 429, 451):
        return ACCESS_BLOCKED
    if status_code in (404, 410):
        return ACCESS_NOT_FOUND
    if 500 <= status_code < 600:
        return ACCESS_UNKNOWN
    return ACCESS_UNKNOWN


def fetch_page(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache: dict[str, Any] | None = None,
) -> tuple[int | None, str, str]:
    """Fetch one page. Returns (status, final_url, html); never raises."""
    if cache is not None and url in cache:
        return cache[url]
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        },
    )
    attempt = 0
    retryable = (429, 500, 502, 503, 504)
    while True:
        result: tuple[int | None, str, str]
        try:
            with urlopen(request, timeout=timeout) as response:
                status = response.getcode() or 200
                final_url = response.geturl() or url
                html = response.read(MAX_HTML_BYTES).decode("utf-8", errors="replace")
                if status in retryable and attempt < max_retries:
                    attempt += 1
                    continue
                result = (status, final_url, html)
        except HTTPError as exc:
            if exc.code in retryable and attempt < max_retries:
                attempt += 1
                continue
            result = (exc.code, url, "")
        except (URLError, OSError):
            if attempt < max_retries:
                attempt += 1
                continue
            result = (None, url, "")
        if cache is not None:
            cache[url] = result
        return result


def fetch_many(
    urls: list[str],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    cache: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, tuple[int | None, str, str]]:
    cache = cache if cache is not None else {}
    results: dict[str, tuple[int | None, str, str]] = {}
    unique = list(dict.fromkeys(url for url in urls if url))
    if not unique:
        return results
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {pool.submit(fetch_page, url, cache=cache, **kwargs): url for url in unique}
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = (None, url, "")
    return results
def _append_claimed(claimed: dict[int, list[str]], number: int, value: date) -> None:
    iso = value.isoformat()
    bucket = claimed.setdefault(number, [])
    if iso not in bucket:
        bucket.append(iso)


_CLAIM_UPDATE_MARKERS = (
    "mis a jour",
    "mise a jour",
    "mises a jour",
    "actualise",
    "actualisee",
    "actualisation",
    "a ete actualise",
    "a ete actualisee",
)
_CLAIM_PUBLICATION_MARKERS = (
    "publie",
    "publiee",
    "publication",
    "paru",
    "parue",
    "vient d etre publie",
    "date de publication",
    "mis en ligne",
    "mise en ligne",
    "publies",
    "publiees",
)
_CLAIM_LEGAL_TEXT_MARKERS = (
    "decret",
    "arrete",
    "loi",
    "ordonnance",
    "circulaire",
    "instruction",
    "deliberation",
    "reglement",
    "texte",
    "jorf",
    "journal officiel",
    "jo du",
)
_CLAIM_DECISION_MARKERS = (
    "decision",
    "jugement",
    "arret",
    "rendu",
    "audience",
    "delibere",
    "sentence",
    "prononce",
    "refere",
)


def _window_has_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(re.search(r"\b" + re.escape(marker) + r"\b", text) for marker in markers)


def _claim_category(line: str, position: int) -> str:
    """Classify a claimed date into publication / update / legal / decision."""
    before = line[max(0, position - 25): position]
    if _window_has_marker(before, _CLAIM_LEGAL_TEXT_MARKERS):
        return CLAIM_LEGAL_TEXT
    if _window_has_marker(before, _CLAIM_DECISION_MARKERS):
        return CLAIM_DECISION
    window = line[max(0, position - 90): position + 40]
    has_update = _window_has_marker(window, _CLAIM_UPDATE_MARKERS)
    has_publication = _window_has_marker(window, _CLAIM_PUBLICATION_MARKERS)
    if has_update and not has_publication:
        return CLAIM_UPDATE
    if has_publication:
        return CLAIM_PUBLICATION
    return CLAIM_GENERIC


def _date_from_claim_groups(groups: tuple[str | None, ...]) -> date | None:
    if groups[0]:
        return parse_date("{0} {1} {2}".format(groups[0], groups[1], groups[2]))
    if groups[3]:
        return parse_date("{0}-{1}-{2}".format(groups[3], groups[4], groups[5]))
    if groups[6]:
        return parse_date("{0}/{1}/{2}".format(groups[6], groups[7], groups[8]))
    return None


def _append_claimed_detailed(
    claimed: dict[int, dict[str, list[str]]],
    number: int,
    category: str,
    value: date,
) -> None:
    iso = value.isoformat()
    bucket = claimed.setdefault(number, {}).setdefault(category, [])
    if iso not in bucket:
        bucket.append(iso)


def _extract_claimed_dates_detailed(
    answer_markdown: str,
    citation_numbers: list[int] | None = None,
) -> dict[int, dict[str, list[str]]]:
    """Return {citation: {category: [...]}} with the claim categories.

    A date is attached to a citation only when the association is reliable:
    - exactly one citation in the line, or
    - the date is adjacent to exactly one citation.
    Dates are classified as publication_claim, update_claim, decision_date,
    legal_text_date or generic_date; only publication/update claims drive the
    temporal mismatch decision.
    """
    claimed: dict[int, dict[str, list[str]]] = {}
    allowed = set(citation_numbers) if citation_numbers else None
    for raw_line in re.split(r"\r?\n", answer_markdown or ""):
        line = _norm(raw_line).lower()
        citations = []
        for match in _CITATION_RE.finditer(line):
            numbers = [int(part.strip()) for part in match.group(1).split(",")]
            numbers = [number for number in numbers if number > 0]
            if allowed is not None:
                numbers = [number for number in numbers if number in allowed]
            if numbers:
                citations.append((match.start(), match.end(), numbers))
        if not citations:
            continue
        dates = []
        for match in _CLAIMED_DATE_RE.finditer(line):
            candidate = _date_from_claim_groups(match.groups())
            if candidate is not None:
                dates.append((match.start(), candidate))
        if not dates:
            continue
        if len(citations) == 1:
            _, _, numbers = citations[0]
            for position, value in dates:
                category = _claim_category(line, position)
                for number in numbers:
                    _append_claimed_detailed(claimed, number, category, value)
        else:
            for position, value in dates:
                category = _claim_category(line, position)
                nearby = [
                    numbers
                    for start, end, numbers in citations
                    if min(abs(position - start), abs(position - end)) <= ADJACENCY_CHARS
                ]
                if len(nearby) == 1:
                    for number in nearby[0]:
                        _append_claimed_detailed(claimed, number, category, value)
    return claimed


def extract_claimed_dates(
    answer_markdown: str,
    citation_numbers: list[int] | None = None,
) -> dict[int, list[str]]:
    """Map each local citation to the explicit dates claimed in the answer.

    A date is attached to a citation only when the association is reliable:
    - exactly one citation in the line, or
    - the date is adjacent to exactly one citation.
    Ambiguous multi-citation sentences do not produce claimed dates.
    """
    merged: dict[int, list[str]] = {}
    for number, buckets in _extract_claimed_dates_detailed(answer_markdown, citation_numbers).items():
        values = sorted(
            set(
                (buckets.get(CLAIM_PUBLICATION) or [])
                + (buckets.get(CLAIM_UPDATE) or [])
            )
        )
        if values:
            merged[number] = values
    return merged


def extract_claimed_dates_with_modes(
    answer_markdown: str,
    citation_numbers: list[int] | None = None,
) -> dict[int, dict[str, list[str]]]:
    """Same as extract_claimed_dates but split by claim mode."""
    return _extract_claimed_dates_detailed(answer_markdown, citation_numbers)


def infer_temporal_role(answer_markdown: str, citation_number: int, url: str | None = None) -> str:
    text = _norm(answer_markdown or "").lower()
    needle = re.compile(r"\[" + re.escape(str(citation_number)) + r"(?:\]|,|\s)")
    relevant = [line for line in text.splitlines() if needle.search(line)]
    role = ROLE_UNKNOWN
    if relevant:
        blob = " ".join(relevant)
        current_score = sum(1 for marker in _CURRENT_MARKERS if marker in blob)
        context_score = sum(1 for marker in _CONTEXT_MARKERS if marker in blob)
        if context_score > current_score:
            role = ROLE_CONTEXT
        elif current_score > context_score:
            role = ROLE_CURRENT
    if role == ROLE_UNKNOWN and url and "/codes/" in (urlsplit(url).path or ""):
        role = ROLE_CONTEXT
    return role


def classify_temporal(
    source_date: date | None,
    date_confidence: str | None,
    date_verification: str,
    access_status: str,
    claimed_dates: list[str],
    temporal_role: str,
    *,
    run_date: str | date | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    recent_tolerance_days: int = DEFAULT_RECENT_TOLERANCE_DAYS,
    mismatch_gap_days: int = DEFAULT_MISMATCH_GAP_DAYS,
    claimed_updates: list[str] | None = None,
    modified_date: date | None = None,
    claimed_decisions: list[str] | None = None,
    source_is_decision: bool = False,
    visible_publication_date: date | None = None,
    visible_update_date: date | None = None,
    visible_publication_dates: list[date] | None = None,
) -> tuple[str, str]:
    """Return (temporal_status, note) for one source.

    Publication claims are checked against the real publication date. Update
    claims are compared with modified_date / visible_update_date only. When a
    page explicitly shows a recent "Publié le ..." that diverges from a stale
    JSON-LD datePublished, the most recent visible publication date drives the
    recency decision and the source is not flagged as a mismatch.
    """
    if source_date is None:
        if temporal_role == ROLE_CONTEXT:
            return STATUS_CONTEXT, NOTE_CONTEXT
        if temporal_role == ROLE_CURRENT and claimed_dates:
            return STATUS_UNKNOWN, NOTE_UNKNOWN_WITH_CLAIM
        return STATUS_UNKNOWN, NOTE_UNKNOWN

    run_date_value = parse_date(run_date) or date.today()
    window_start = run_date_value - timedelta(days=window_days)
    recent_cutoff = window_start - timedelta(days=recent_tolerance_days)
    horizon = run_date_value + timedelta(days=3)
    strong = date_confidence in (CONF_HIGH, CONF_MEDIUM)

    effective_date = source_date
    note = ""
    visible_candidates = list(visible_publication_dates or [])
    if visible_publication_date is not None and visible_publication_date not in visible_candidates:
        visible_candidates.append(visible_publication_date)
    if strong and visible_candidates:
        most_recent_visible = max(visible_candidates)
        if (most_recent_visible - source_date).days > mismatch_gap_days:
            effective_date = most_recent_visible
            note = NOTE_VISIBLE_DIVERGENCE

    publication_claims = sorted(
        {parsed for parsed in (parse_date(value) for value in claimed_dates or []) if parsed}
    )
    if strong and publication_claims and effective_date is not None:
        gap = max(abs((claim - effective_date).days) for claim in publication_claims)
        if gap > mismatch_gap_days:
            return STATUS_MISMATCH, NOTE_MISMATCH

    update_claims = sorted(
        {parsed for parsed in (parse_date(value) for value in claimed_updates or []) if parsed}
    )
    comparable_update = modified_date or visible_update_date
    if strong and update_claims and comparable_update is not None:
        if any(abs(claim - comparable_update).days <= mismatch_gap_days for claim in update_claims):
            if recent_cutoff <= comparable_update <= horizon:
                return STATUS_CURRENT, note
        elif not publication_claims and abs((comparable_update - source_date).days) <= mismatch_gap_days:
            return STATUS_MISMATCH, NOTE_MISMATCH

    decision_claims = sorted(
        {parsed for parsed in (parse_date(value) for value in claimed_decisions or []) if parsed}
    )
    if (
        strong
        and source_is_decision
        and decision_claims
        and not publication_claims
        and effective_date is not None
    ):
        gap = max(abs((claim - effective_date).days) for claim in decision_claims)
        if gap > mismatch_gap_days:
            return STATUS_MISMATCH, NOTE_MISMATCH

    if recent_cutoff <= effective_date <= horizon:
        if temporal_role == ROLE_CONTEXT:
            return STATUS_CONTEXT, note or NOTE_CONTEXT
        return STATUS_CURRENT, note
    if temporal_role == ROLE_CONTEXT:
        return STATUS_CONTEXT, note or NOTE_CONTEXT
    if not strong:
        return STATUS_UNKNOWN, note or NOTE_LOW_CONFIDENCE
    return STATUS_CONTEXT, note or NOTE_OLD_CONTEXT
def _local_citation_numbers(answer_markdown: str) -> list[int]:
    numbers: list[int] = []
    for match in _CITATION_RE.finditer(answer_markdown or ""):
        for part in match.group(1).split(","):
            number = int(part.strip())
            if number > 0 and number not in numbers:
                numbers.append(number)
    return numbers


def _merge_role(role_votes: list[str], url: str | None) -> str:
    if not role_votes:
        role = ROLE_UNKNOWN
    elif ROLE_CONTEXT in role_votes and ROLE_CURRENT in role_votes:
        role = ROLE_UNKNOWN
    elif ROLE_CONTEXT in role_votes:
        role = ROLE_CONTEXT
    elif ROLE_CURRENT in role_votes:
        role = ROLE_CURRENT
    else:
        role = ROLE_UNKNOWN
    if role == ROLE_UNKNOWN and url and "/codes/" in (urlsplit(url).path or ""):
        role = ROLE_CONTEXT
    return role


_DECISION_URL_HINTS = (
    "courdecassation",
    "conseil-etat",
    "tribunal",
    "jurinet",
    "jurisprudence",
    "/juri/",
    "decision",
)


def looks_like_decision_source(url: str | None, title: Any = None) -> bool:
    if url:
        parts = ((urlsplit(url).netloc or "") + (urlsplit(url).path or "")).lower()
        if any(hint in parts for hint in _DECISION_URL_HINTS):
            return True
    if title:
        text = _norm(title).lower()
        if re.search(r"\b(decision|arret|jugement)\b", text) or re.search(r"\brg\s*n", text):
            return True
    return False


def _build_temporal(
    source: dict[str, Any],
    fetched: dict[str, tuple[int | None, str, str]],
    claims_by_search: dict[str, dict[int, dict[str, list[str]]]],
    roles_by_search: dict[str, dict[int, str]],
    **classify_kwargs: Any,
) -> dict[str, Any]:
    url = source.get("url")
    claimed: list[str] = []
    claimed_updates: list[str] = []
    claimed_decisions: list[str] = []
    claimed_legal: list[str] = []
    claimed_generic: list[str] = []
    claimed_from: list[str] = []
    role_votes: list[str] = []
    source_searches = source.get("source_searches") or []
    original = source.get("original_indices") or {}
    if source_searches:
        pairs = []
        for name in source_searches:
            local_index = original.get(name)
            if local_index is not None:
                pairs.append((name, local_index))
    else:
        pairs = [(name, source.get("index")) for name in claims_by_search]
    for name, local_index in pairs:
        local_claims = claims_by_search.get(name, {}).get(local_index)
        if local_claims:
            for iso in local_claims.get(CLAIM_PUBLICATION) or []:
                if iso not in claimed:
                    claimed.append(iso)
            for iso in local_claims.get(CLAIM_UPDATE) or []:
                if iso not in claimed_updates:
                    claimed_updates.append(iso)
            for iso in local_claims.get(CLAIM_DECISION) or []:
                if iso not in claimed_decisions:
                    claimed_decisions.append(iso)
            for iso in local_claims.get(CLAIM_LEGAL_TEXT) or []:
                if iso not in claimed_legal:
                    claimed_legal.append(iso)
            for iso in local_claims.get(CLAIM_GENERIC) or []:
                if iso not in claimed_generic:
                    claimed_generic.append(iso)
            if name not in claimed_from:
                claimed_from.append(name)
        role = roles_by_search.get(name, {}).get(local_index)
        if role:
            role_votes.append(role)
    role = _merge_role(role_votes, url)

    if not url:
        access = ACCESS_UNKNOWN
        info = extract_source_date("", None)
    else:
        status, final_url, html = fetched.get(url, (None, url, ""))
        access = detect_access_status(status, html, url)
        info = extract_source_date(html, url or final_url)

    source_date = info["source_date"]
    confidence = info["date_confidence"]
    verification = VERIF_DIRECT if source_date is not None else VERIF_UNKNOWN
    is_decision = looks_like_decision_source(url, source.get("title"))
    temporal_status, note = classify_temporal(
        source_date,
        confidence,
        verification,
        access,
        claimed,
        role,
        claimed_updates=claimed_updates,
        modified_date=info.get("modified_date"),
        claimed_decisions=claimed_decisions,
        source_is_decision=is_decision,
        visible_publication_date=info.get("visible_publication_date"),
        visible_update_date=info.get("visible_update_date"),
        visible_publication_dates=info.get("visible_publication_dates"),
        **classify_kwargs,
    )
    temporal: dict[str, Any] = {
        "access_status": access,
        "source_date": source_date.isoformat() if source_date is not None else None,
        "date_evidence": info["date_evidence"],
        "date_confidence": confidence,
        "date_verification": verification,
        "claimed_dates": sorted(claimed),
        "claimed_from_searches": claimed_from,
        "temporal_role": role,
        "temporal_status": temporal_status,
        "note": note,
    }
    if info.get("modified_date") is not None:
        temporal["modified_date"] = info["modified_date"].isoformat()
    if info.get("visible_publication_date") is not None:
        temporal["visible_publication_date"] = info["visible_publication_date"].isoformat()
    if info.get("visible_update_date") is not None:
        temporal["visible_update_date"] = info["visible_update_date"].isoformat()
    if info.get("visible_publication_dates"):
        temporal["visible_publication_dates"] = [
            value.isoformat() for value in info["visible_publication_dates"]
        ]
    if info.get("visible_update_dates"):
        temporal["visible_update_dates"] = [
            value.isoformat() for value in info["visible_update_dates"]
        ]
    if claimed_updates:
        temporal["claimed_updates"] = sorted(claimed_updates)
    if claimed_decisions:
        temporal["claimed_decision_dates"] = sorted(claimed_decisions)
    if claimed_legal:
        temporal["claimed_legal_text_dates"] = sorted(claimed_legal)
    if claimed_generic:
        temporal["claimed_generic_dates"] = sorted(claimed_generic)
    return temporal


def validate_cited_sources(
    cited_sources: list[dict[str, Any]],
    *,
    local_answers: dict[str, str] | None = None,
    fetch_fn: Any = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    concurrency: int = DEFAULT_CONCURRENCY,
    user_agent: str = DEFAULT_USER_AGENT,
    run_date: str | date | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    recent_tolerance_days: int = DEFAULT_RECENT_TOLERANCE_DAYS,
    mismatch_gap_days: int = DEFAULT_MISMATCH_GAP_DAYS,
    cache: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every cited source and return (validated_sources, summary)."""
    local_answers = local_answers or {}
    claims_by_search: dict[str, dict[int, dict[str, list[str]]]] = {}
    roles_by_search: dict[str, dict[int, str]] = {}
    for name, answer in local_answers.items():
        if not isinstance(answer, str):
            continue
        claims_by_search[name] = extract_claimed_dates_with_modes(answer)
        roles_by_search[name] = {}
        for number in _local_citation_numbers(answer):
            roles_by_search[name][number] = infer_temporal_role(answer, number)

    urls = [source["url"] for source in cited_sources if isinstance(source, dict) and source.get("url")]
    if fetch_fn is not None:
        fetched: dict[str, tuple[int | None, str, str]] = {}
        for url in dict.fromkeys(urls):
            try:
                fetched[url] = fetch_fn(url)
            except Exception:
                fetched[url] = (None, url, "")
    else:
        fetched = fetch_many(
            urls,
            timeout=timeout,
            concurrency=concurrency,
            user_agent=user_agent,
            cache=cache,
        )

    classify_kwargs = {
        "run_date": run_date,
        "window_days": window_days,
        "recent_tolerance_days": recent_tolerance_days,
        "mismatch_gap_days": mismatch_gap_days,
    }
    validated: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {
        STATUS_CURRENT: 0,
        STATUS_CONTEXT: 0,
        STATUS_MISMATCH: 0,
        STATUS_UNKNOWN: 0,
    }
    verif_counts: dict[str, int] = {
        VERIF_DIRECT: 0,
        VERIF_INDIRECT: 0,
        VERIF_UNKNOWN: 0,
    }
    for source in cited_sources:
        entry = dict(source)
        temporal = _build_temporal(
            entry,
            fetched,
            claims_by_search,
            roles_by_search,
            **classify_kwargs,
        )
        entry["temporal"] = temporal
        validated.append(entry)
        status_counts[temporal["temporal_status"]] = status_counts.get(temporal["temporal_status"], 0) + 1
        verif_counts[temporal["date_verification"]] = verif_counts.get(temporal["date_verification"], 0) + 1

    summary = {
        "status": "completed",
        "temporal_validation_count": len(validated),
        "current_count": status_counts[STATUS_CURRENT],
        "context_count": status_counts[STATUS_CONTEXT],
        "mismatch_count": status_counts[STATUS_MISMATCH],
        "unknown_count": status_counts[STATUS_UNKNOWN],
        "direct_date_count": verif_counts[VERIF_DIRECT],
        "indirect_date_count": verif_counts[VERIF_INDIRECT],
        "unknown_date_count": verif_counts[VERIF_UNKNOWN],
    }
    return validated, summary
