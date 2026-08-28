#!/usr/bin/env python3
"""Requalification temporelle ciblée par Gemma (V3, dry-run).

S'exécute après la validation temporelle Python et avant la passe éditoriale
finale. Seules les sources ambiguës sont éligibles : Python conserve la
priorité sur les faits objectifs et Gemma ne peut que interpréter le rôle
temporel.

Aucun appel LLM n'est effectué ici : le rapport est en dry-run et toutes les
règles (éligibilité, transitions, garde-fous, validation de réponse, fusion)
sont testables unitairement, sans réseau ni modèle.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlsplit

from temporal_validation import (
    STATUS_CONTEXT,
    STATUS_CURRENT,
    STATUS_MISMATCH,
    STATUS_UNKNOWN,
    extract_claimed_dates_with_modes,
    parse_date,
)

# --- Enums fermées ---
REQUALIFICATION_STATUSES = frozenset(
    {STATUS_CURRENT, STATUS_CONTEXT, STATUS_MISMATCH, STATUS_UNKNOWN}
)
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
REASON_CODES = frozenset(
    {
        "role_context_legal_text",
        "role_context_decision_old",
        "role_context_explicit",
        "role_current_update_claim",
        "role_current_title_date",
        "role_current_publication_context",
        "role_current_recent_context",
        "role_current_legal_update",
        "no_signal",
        "mismatch_confirmed",
        "mismatch_not_justified",
    }
)

MAX_REASON_CHARS = 240
MAX_CONTEXT_CHARS = 500
MIN_CONTEXT_CHARS = 12
DEFAULT_WINDOW_CHARS = 260

# --- Fenêtre temporelle configurable (V3) ---
WINDOW_STRICT_7D = "strict_7d"
WINDOW_EXTENDED_30D = "extended_30d"
WINDOW_MODES: dict[str, dict[str, int]] = {
    WINDOW_STRICT_7D: {"window_days": 7, "recent_tolerance_days": 7},
    WINDOW_EXTENDED_30D: {"window_days": 30, "recent_tolerance_days": 7},
}
DEFAULT_WINDOW_MODE = WINDOW_STRICT_7D


def resolve_window_params(
    window_mode: str | None = None,
    *,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
) -> tuple[int, int]:
    """Résout (window_days, recent_tolerance_days) à partir d'un mode nommé.

    Un mode explicite (strict_7d / extended_30d) prime sur les valeurs brutes ;
    sans mode, le comportement historique (7 j + 7 j de tolérance) est conservé
    pour éviter toute régression implicite.
    """
    if window_mode is not None:
        params = WINDOW_MODES.get(window_mode)
        if params is not None:
            return params["window_days"], params["recent_tolerance_days"]
    return window_days, recent_tolerance_days

REQUALIFICATION_SYSTEM_PROMPT = """Tu es un classificateur temporel.
Tu classes le rôle d'une source dans une VEILLE HEBDOMADAIRE.

Tu ne recherches aucune information nouvelle.
Tu ne modifies aucun fait fourni.
Tu dois seulement interpréter le rôle temporel des sources à partir
des données fournies.

Une source juridique ou institutionnelle n'est pas automatiquement context.

Choisir current si la matière fournie indique qu'un document,
une décision, une publication, une mise à jour ou une information
est effectivement nouvelle dans la fenêtre de veille.

Choisir context si la source est principalement utilisée comme cadre,
référence antérieure, rappel du droit applicable ou élément historique.

Exemples conceptuels :
- JORF daté du jour ou de la semaine => probablement current
- décret ancien cité comme fondement juridique => context
- décision ancienne citée pour expliquer une règle => context
- annuaire mis à jour cette semaine => current
- circulaire datée hors fenêtre mais utilisée comme cadre => context

Ne privilégie pas context simplement parce que la source est juridique.

Une mise à jour juridique récente (Code, Légifrance, version en vigueur) peut
justifier current ; utilise le reason_code role_current_legal_update et non
un simple événement récent.

Les champs claim_context, neighbor_context et recent_context_signals sont
des faits déjà extraits par Python : tu peux t'appuyer sur un signal
recent_context_signals explicite et récent pour choisir current, sans en
inventer un nouveau.

Tu ne peux produire que :
current
context
unknown

Tu ne peux jamais produire mismatch dans cette V1.

Ne crée :
- aucune date ;
- aucune citation ;
- aucun fait ;
- aucune URL.

En cas d'incertitude réelle, choisir unknown."""

# --- Signaux récents (faits déjà extraits par Python) ---
SIG_UPDATE_CLAIM = "update_claim"
SIG_VISIBLE_UPDATE = "visible_update_date"
SIG_VISIBLE_PUB = "visible_publication_date"
SIG_MODIFIED = "modified_date"
SIG_TITLE_DATE = "title_date"
SIG_UPDATE_FORMULATION = "update_formulation"

# --- Signaux temporels voisins (V2) ---
SIGNAL_RECENT_EVENT = "recent_event"
SIGNAL_RECENT_UPDATE = "recent_update"
SIGNAL_RECENT_PUBLICATION = "recent_publication"

_RECENT_EVENT_RE = re.compile(
    r"nouvelle\s+[ÉéE]tape|nouveau\s+[ÉéE]tape|nouveaut[ée]\s+r[ée]cente|"
    r"d[ée]sormais|\bannonc[ée]\b|mise\s+en\s+d[ée]lib[ée]r[ée]|"
    r"(?:a\s+mis|mis)\s+l[’']affaire\s+en\s+d[ée]lib[ée]r[ée]|"
    r"audience\s+tenue|r[ée]cent(?:e|es)?\s+suite|suite\s+proc[ée]durale|"
    r"suites?\s+proc[ée]durales?\s+r[ée]centes?|"
    r"r[ée]centes?\s+suites?\s+proc[ée]durales?|"
    r"d[ée]signation\s+d[’']un\s+expert|"
    r"r[ée]cent(?:e|es)?\s+d[ée]cisions?",
    re.IGNORECASE,
)
_RECENT_UPDATE_RE = re.compile(
    r"mis(?:e)?\s*[\-]?\s*[àa]\s*jour|actualis\w*|r[ée]actualis\w*|"
    r"actualisation|derni[èe]re\s+mise\s+[àa]\s+jour|\bmaj\b",
    re.IGNORECASE,
)
_RECENT_UPDATE_STOP_RE = re.compile(
    r"mise\s+[àa]\s+jour\s+en\s+continu|pour\s+mise\s+[àa]\s+jour|"
    r"mettre\s+[àa]\s+jour|mise\s+[àa]\s+jour\s+des\s+bases|"
    r"afin\s+d[’']en\s+assurer\s+la\s+mise\s+[àa]\s+jour",
    re.IGNORECASE,
)
_RECENT_PUBLICATION_RE = re.compile(
    r"publi[ée]\s+(?:cette\s+semaine|r[ée]cemment|le\b)|"
    r"mise\s+en\s+ligne|mise\s+en\s+ligne\s+r[ée]cente|"
    r"mise\s+en\s+ligne\s+sur\s+le\s+site|actualit[ée]s?\s+publi[ée]es",
    re.IGNORECASE,
)
_RECENT_RANGE_RE = re.compile(
    r"(?:du\s+|de\s+)?(\d{1,2})\s+au\s+(\d{1,2})\s+"
    r"(janvier|f[ée]vrier|fevrier|mars|avril|mai|juin|juillet|ao[ûu]t|aout|"
    r"septembre|octobre|novembre|d[ée]cembre|decembre)\s+(\d{4})",
    re.IGNORECASE,
)
_MARKDOWN_BOUNDARY_RE = re.compile(
    r"^\s*(?:#+\s*|\|\s*|>\s*|[-*+]\s+)",
)

_CITATION_RE = re.compile(r"\[[1-9]\d*\]")
_URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])\s+-\s+|(?<=[.!?])\s+[–—]\s+|(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9«\"'([])",
)
_SENTENCE_START_RE = re.compile(r"^[A-ZÀ-ÖØ-Þ0-9«\"'([]")
_FRENCH_DATE_RE = re.compile(
    r"(\d{1,2})\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+(\d{4})"
    r"|(\d{4})-(\d{2})-(\d{2})"
    r"|(\d{1,2})/(\d{1,2})/(\d{4})",
    re.IGNORECASE,
)
_TITLE_DATE_RE = re.compile(
    r"(?:du|au|de|le)\s+(\d{1,2})\s+"
    r"(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+(\d{4})",
    re.IGNORECASE,
)
_TITLE_DATE_MARKER_RE = re.compile(r"(?:du|le|au|de)\s*$", re.IGNORECASE)
_LEGAL_MARKER_RE = re.compile(
    r"décret|decret|loi\b|code\b|JORF|journal officiel|arrêté|arrete|ordonnance|"
    r"circulaire|règlement|reglement|Légifrance|legifrance|cour de cassation|"
    r"conseil d[’']état|conseil d[’']etat|texte législatif|texte juridique|"
    r"juridiction",
    re.IGNORECASE,
)
_UPDATE_FORMULATION_RE = re.compile(
    r"mise?s?\s+à\s+jour|actualis\w*|actualisation|réactualis\w*|"
    r"dernière\s+mise\s+à\s+jour|\bmaj\b",
    re.IGNORECASE,
)
_LEGAL_CONTEXT_RE = re.compile(
    r"(?:guide|fiche)\s+(?:administratif|juridique|r[ée]glementaire|pratique)|"
    r"documents?\s+(?:administratif|officiel|de\s+r[ée]f[ée]rence)|"
    r"textes?\s+(?:administratif|officiel|de\s+r[ée]f[ée]rence|en\s+vigueur)|"
    r"r[ée]glementation\b|normes?\s+applicables|"
    r"experts?\s+de\s+justice|experts?\s+judiciaires?|"
    r"compagnies?\s+d[’']experts|justice\b|juridiction\b|"
    r"d[ée]mat[ée]rialisation\b",
    re.IGNORECASE,
)
_LEGAL_DOMAIN_RE = re.compile(
    r"(?:^|\.)(?:legifrance|justice|courdecassation|cour-appel|juridiction|"
    r"gouv|vie-publique|senat|assemblee-nationale|defenseurdesdroits|"
    r"service-public|bnds)\.?",
    re.IGNORECASE,
)
_LEGAL_UPDATE_RE = re.compile(
    r"code\b|L[ée]gifrance|version\s+en\s+vigueur|r[ée]glement|d[ée]cret|"
    r"loi\b|article\b|texte\s+(?:juridique|l[ée]gislatif)",
    re.IGNORECASE,
)
_OLD_DECISION_RE = re.compile(
    r"pas\s+d[ée]cision\s+nouvelle|aucune\s+d[ée]cision\s+nouvelle|"
    r"n[’']est\s+pas\s+nouvelle|pas\s+nouvelle\s+dans\s+la\s+fen[êe]tre|"
    r"d[ée]cision\s+ancienne|dates?\s+ant[ée]rieures?\s+[àa]\s+cette\s+fen[êe]tre|"
    r"ant[ée]rieures?\s+[àa]\s+cette\s+fen[êe]tre|"
    r"hors\s+de\s+la\s+fen[êe]tre|en\s+dehors\s+de\s+la\s+fen[êe]tre",
    re.IGNORECASE,
)

_GENERIC_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"je\s+(?:fournis|présente|vous\s+fournis|vous\s+présente|vais|donne)|"
    r"voici\b|bonjour\b|en\s+résumé|pour\s+résumer|en\s+conclusion|pour\s+conclure|"
    r"introduction|conclusion\b|si\s+vous\s+le\s+souhaitez|"
    r"la\s+veille\b|les\s+actualités\b|le\s+présent\s+rapport|ma\s+synthèse|cette\s+synthèse"
    r")",
    re.IGNORECASE,
)


def hostname(url: Any) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    netloc = urlsplit(url.strip()).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


# --- Contextes de citation ---

def _contains_citation(text: str, citation_number: int) -> bool:
    return bool(re.search(r"\[" + str(citation_number) + r"\]", text or ""))


def _split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    for chunk in _SENTENCE_END_RE.split(paragraph):
        chunk = chunk.strip()
        if not chunk:
            continue
        if sentences and not _SENTENCE_START_RE.match(chunk):
            sentences[-1] = sentences[-1] + " " + chunk
        else:
            sentences.append(chunk)
    return sentences


def _paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        if lines:
            paragraphs.append(" ".join(lines))
    return paragraphs


def _trim_context(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for marker in (". ", "; ", " — ", " - "):
        position = cut.rfind(marker)
        if position > int(max_chars * 0.6):
            return cut[:position + 1].strip() + " (...)"
    return cut.strip() + " (...)"


def _paragraph_units(text: str) -> list[list[str]]:
    """Unités de sens au niveau des lignes, sans franchir de section.

    Un bloc est un ensemble de lignes non vides séparées par une ligne vide.
    À l'intérieur d'un bloc, un titre markdown, une entrée de tableau, une
    citation ou une puce démarre une nouvelle unité : le voisinage d'une
    citation ne franchit donc ni un titre de section ni un élément de liste.
    """
    units: list[list[str]] = []
    current: list[str] = []
    for raw_line in (text or "").split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            if current:
                units.append(current)
                current = []
            continue
        if _MARKDOWN_BOUNDARY_RE.match(stripped):
            if current:
                units.append(current)
                current = []
            units.append([stripped])
            continue
        current.append(stripped)
    if current:
        units.append(current)
    return units


def _range_end_date(text: str) -> date | None:
    """Date de fin d'une plage explicite 'du X au Y mois année'."""
    match = _RECENT_RANGE_RE.search(text or "")
    if not match:
        return None
    month = FRENCH_MONTHS.get(match.group(3).lower())
    if month is None:
        return None
    try:
        return date(int(match.group(4)), month, int(match.group(2)))
    except ValueError:
        return None


def _window(run_date: str | date | None, window_days: int, recent_tolerance_days: int, horizon_days: int = 3) -> tuple[date, date]:
    run_value = parse_date(run_date) or date.today()
    cutoff = run_value - timedelta(days=window_days + recent_tolerance_days)
    horizon = run_value + timedelta(days=horizon_days)
    return cutoff, horizon


def extract_recent_context_signals(
    sentences: list[str],
    citation_number: int,
    *,
    claim_index: int | None = None,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Signaux temporels récents présents dans les phrases voisines.

    Un signal n'est produit que si la phrase contient un marqueur explicite
    (mise à jour, publication/mise en ligne récente, nouvelle étape, événement
    procédural) ET une date dans la fenêtre de veille, OU un marqueur de
    récence sans date chiffrée (ex. "publié cette semaine"). Une date isolée
    de décret/arrêté/décision ne devient jamais un signal.
    """
    window_days, recent_tolerance_days = resolve_window_params(
        window_mode, window_days=window_days, recent_tolerance_days=recent_tolerance_days
    )
    cutoff, horizon = _window(run_date, window_days, recent_tolerance_days)
    if claim_index is None:
        claim_index = next(
            (
                i
                for i, sentence in enumerate(sentences)
                if _contains_citation(sentence, citation_number)
            ),
            -1,
        )
    signals: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        if claim_index == -1:
            if not _contains_citation(sentence, citation_number):
                continue
            proximity = "same_sentence"
        elif index == claim_index:
            proximity = "same_sentence"
        elif index == claim_index - 1:
            proximity = "previous_sentence"
        elif index == claim_index + 1:
            proximity = "next_sentence"
        else:
            continue
        date_in_window = None
        for parsed in _extract_dates(sentence):
            if cutoff <= parsed <= horizon:
                if date_in_window is None or parsed > parse_date(date_in_window):
                    date_in_window = parsed.isoformat()
        range_end = _range_end_date(sentence)
        if (
            date_in_window is None
            and range_end is not None
            and cutoff <= range_end <= horizon
        ):
            date_in_window = range_end.isoformat()
        explicit_recent = (
            bool(_RECENT_UPDATE_RE.search(sentence) and not _RECENT_UPDATE_STOP_RE.search(sentence))
            or bool(_RECENT_PUBLICATION_RE.search(sentence))
            or bool(_RECENT_EVENT_RE.search(sentence))
        )
        if not explicit_recent:
            continue
        if date_in_window is None and not _RECENT_PUBLICATION_RE.search(sentence):
            continue
        if _RECENT_EVENT_RE.search(sentence):
            signal_type = SIGNAL_RECENT_EVENT
        elif _RECENT_UPDATE_RE.search(sentence) and not _RECENT_UPDATE_STOP_RE.search(sentence):
            signal_type = SIGNAL_RECENT_UPDATE
        else:
            signal_type = SIGNAL_RECENT_PUBLICATION
        signals.append(
            {
                "type": signal_type,
                "date": date_in_window,
                "text": _trim_context(sentence, 240),
                "proximity": proximity,
            }
        )
    return signals


def extract_claim_and_neighbor_contexts(
    answer_markdown: str,
    citation_number: int,
    *,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    max_chars: int = MAX_CONTEXT_CHARS,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
) -> dict[str, Any]:
    """Phrase de citation + phrase(s) voisine(s), sans franchir de section.

    Retourne claim_context (phrase contenant la citation), neighbor_context
    (au plus la phrase précédente et la suivante du même bloc) et
    recent_context_signals (signaux temporels récents extraits par Python).
    """
    text = (answer_markdown or "").replace("\r\n", "\n")
    units = _paragraph_units(text)
    candidates: list[dict[str, Any]] = []
    for line in units:
        sentences = _split_sentences(line[0] if len(line) == 1 else " ".join(line))
        citation_indices = [
            index
            for index, sentence in enumerate(sentences)
            if _contains_citation(sentence, citation_number)
        ]
        if not citation_indices:
            continue
        first_index = citation_indices[0]
        for index in citation_indices:
            sentence = sentences[index]
            signals = extract_recent_context_signals(
                sentences,
                citation_number,
                claim_index=index,
                run_date=run_date,
                window_days=window_days,
                recent_tolerance_days=recent_tolerance_days,
                window_mode=window_mode,
            )
            neighbors: list[str] = []
            if index > 0:
                neighbor = sentences[index - 1]
                if not _MARKDOWN_BOUNDARY_RE.match(neighbor):
                    neighbors.append(neighbor)
            if index + 1 < len(sentences):
                neighbor = sentences[index + 1]
                if not _MARKDOWN_BOUNDARY_RE.match(neighbor):
                    neighbors.append(neighbor)
            candidates.append(
                {
                    "claim": _trim_context(sentence, max_chars),
                    "neighbors": [_trim_context(value, max_chars) for value in neighbors],
                    "score": _context_score(sentence, citation_number),
                    "signals": signals,
                    "claim_signals": [
                        signal
                        for signal in signals
                        if signal.get("proximity") == "same_sentence"
                    ],
                    "index": index,
                    "first_index": first_index,
                }
            )
    if not candidates:
        window_text = _window_context(text, citation_number, window_chars, max_chars)
        if window_text is None:
            return {"claim_context": None, "neighbor_context": None, "recent_context_signals": []}
        return {
            "claim_context": window_text,
            "neighbor_context": None,
            "recent_context_signals": [],
        }
    def _candidate_key(candidate: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            1 if candidate["index"] == candidate["first_index"] else 0,
            candidate.get("score", 0),
            len(candidate.get("claim_signals") or []),
            -candidate.get("first_index", 0),
        )

    best = max(candidates, key=_candidate_key)
    neighbors = best["neighbors"][:2]
    return {
        "claim_context": best["claim"],
        "neighbor_context": " ".join(neighbors) if neighbors else None,
        "recent_context_signals": best["signals"],
    }


def _context_score(candidate: str, citation_number: int) -> int:
    """Score de pertinence d'une phrase candidate comme contexte de citation."""
    score = 0
    if _FRENCH_DATE_RE.search(candidate):
        score += 10
    others = {
        int(match.group(0)[1:-1])
        for match in _CITATION_RE.finditer(candidate)
    } - {citation_number}
    if not others:
        score += 8
    elif len(others) == 1:
        score += 3
    else:
        score -= 5
    if _GENERIC_PREAMBLE_RE.match(candidate):
        score -= 6
    return score


def _window_context(
    text: str, citation_number: int, window_chars: int, max_chars: int
) -> str | None:
    match = re.search(r"\[" + str(citation_number) + r"\]", text or "")
    if not match:
        return None
    start = max(0, match.start() - window_chars)
    end = min(len(text), match.end() + window_chars)
    return _trim_context(text[start:end].strip(), max_chars)


def extract_claim_context(
    answer_markdown: str,
    citation_number: int,
    *,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str | None:
    """Phrase complète contenant la citation locale (contexte le plus précis).

    Préfère une phrase datée avec la citation isolée et écarte les préambules
    génériques ou les phrases à forte densité de citations, qui mélangent
    plusieurs informations temporelles indépendantes.
    """
    text = (answer_markdown or "").replace("\r\n", "\n")
    candidates: list[str] = []
    for paragraph in _paragraphs(text):
        for sentence in _split_sentences(paragraph):
            if _contains_citation(sentence, citation_number):
                candidates.append(_trim_context(sentence, max_chars))
    if not candidates:
        return _window_context(text, citation_number, window_chars, max_chars)
    return max(
        candidates,
        key=lambda candidate: _context_score(candidate, citation_number),
    )


def build_claim_contexts_for_source(
    source: dict[str, Any],
    local_answers: dict[str, str],
    *,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Contextes de citation enrichis (V2) avec provenance et signaux voisins.

    Chaque entrée contient claim_context (phrase de citation),
    neighbor_context (phrases voisines du même bloc) et
    recent_context_signals (signaux temporels récents extraits par Python).
    """
    contexts: list[dict[str, Any]] = []
    global_index = source.get("index")
    source_searches = source.get("source_searches") or []
    original = source.get("original_indices") or {}
    if source_searches:
        pairs = [(name, original.get(name)) for name in source_searches]
    else:
        pairs = [(name, global_index) for name in local_answers]
    claims_cache: dict[str, dict[int, dict[str, list[str]]]] = {}
    for name, local_index in pairs:
        if local_index is None:
            continue
        answer = local_answers.get(name, "")
        enriched = extract_claim_and_neighbor_contexts(
            answer,
            local_index,
            run_date=run_date,
            window_days=window_days,
            recent_tolerance_days=recent_tolerance_days,
            window_mode=window_mode,
        )
        if not enriched["claim_context"]:
            continue
        claims = claims_cache.get(name)
        if claims is None:
            claims = extract_claimed_dates_with_modes(answer)
            claims_cache[name] = claims
        local_claims = claims.get(local_index) or {}
        claim_types = sorted({key for key, values in local_claims.items() if values})
        entry = {
            "search_name": name,
            "local_citation": local_index,
            "global_index": global_index,
            "claim_context": enriched["claim_context"],
            "neighbor_context": enriched["neighbor_context"],
            "recent_context_signals": enriched["recent_context_signals"],
            "claim_types": claim_types,
        }
        contexts.append(entry)
    return contexts


# --- Signaux récents et garde-fous ---

def _is_recent(value: Any, cutoff: date, horizon: date) -> bool:
    parsed = parse_date(value)
    return parsed is not None and cutoff <= parsed <= horizon


def recent_signals(
    temporal: dict[str, Any],
    contexts: list[dict[str, Any]],
    title: str | None,
    *,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    horizon_days: int = 3,
    window_mode: str | None = None,
) -> list[str]:
    window_days, recent_tolerance_days = resolve_window_params(
        window_mode, window_days=window_days, recent_tolerance_days=recent_tolerance_days
    )
    run_value = parse_date(run_date) or date.today()
    cutoff = run_value - timedelta(days=window_days + recent_tolerance_days)
    horizon = run_value + timedelta(days=horizon_days)
    signals: list[str] = []
    if any(
        _is_recent(value, cutoff, horizon) for value in temporal.get("claimed_updates") or []
    ):
        signals.append(SIG_UPDATE_CLAIM)
    if _is_recent(temporal.get("visible_update_date"), cutoff, horizon):
        signals.append(SIG_VISIBLE_UPDATE)
    if _is_recent(temporal.get("visible_publication_date"), cutoff, horizon):
        signals.append(SIG_VISIBLE_PUB)
    if _is_recent(temporal.get("modified_date"), cutoff, horizon):
        signals.append(SIG_MODIFIED)
    if title_date_in_window(title, cutoff, horizon) is not None:
        signals.append(SIG_TITLE_DATE)
    if any(
        _UPDATE_FORMULATION_RE.search(context.get("claim_context") or "")
        for context in contexts
    ):
        signals.append(SIG_UPDATE_FORMULATION)
    return signals


def _title_dates(title: str | None) -> list[date]:
    result: list[date] = []
    for match in _TITLE_DATE_RE.finditer(title or ""):
        parsed = parse_date(
            "{0} {1} {2}".format(match.group(1), match.group(2), match.group(3))
        )
        if parsed is not None:
            result.append(parsed)
    return result


def title_date_in_window(title: str | None, cutoff: date, horizon: date) -> date | None:
    for parsed in _title_dates(title):
        if cutoff <= parsed <= horizon:
            return parsed
    return None


def extract_title_date(title: str | None) -> tuple[str | None, str | None]:
    """Date explicite du titre (extraction Python, jamais inventée par Gemma).

    Retourne (title_date ISO, confiance) ou (None, None). Une date précédée
    d'un marqueur (du/le/au/de) ou au format structuré est "high" ; une date
    française nue est "medium".
    """
    text = (title or "").strip()
    if not text:
        return None, None
    candidates: list[tuple[date, str, int]] = []
    for match in _FRENCH_DATE_RE.finditer(text):
        groups = match.groups()
        if groups[0]:
            parsed = parse_date(
                "{0} {1} {2}".format(groups[0], groups[1], groups[2])
            )
        elif groups[3]:
            parsed = parse_date("{0}-{1}-{2}".format(groups[3], groups[4], groups[5]))
        elif groups[6]:
            parsed = parse_date("{0}/{1}/{2}".format(groups[6], groups[7], groups[8]))
        else:
            parsed = None
        if parsed is None:
            continue
        prefix = text[max(0, match.start() - 10) : match.start()]
        marked = bool(_TITLE_DATE_MARKER_RE.search(prefix))
        structured = groups[3] is not None or groups[6] is not None
        confidence = "high" if (marked or structured) else "medium"
        candidates.append((parsed, confidence, match.start()))
    if not candidates:
        return None, None
    marked_candidates = [
        candidate for candidate in candidates if candidate[1] == "high"
    ]
    pool = marked_candidates or candidates
    best = min(pool, key=lambda candidate: candidate[2])
    return best[0].isoformat(), best[1]


def context_to_current_allowed(
    temporal: dict[str, Any],
    contexts: list[dict[str, Any]],
    title: str | None,
    *,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
) -> tuple[bool, str]:
    """Garde-fou V1 : context->current exige un signal sémantique récent."""
    signals = recent_signals(
        temporal,
        contexts,
        title,
        run_date=run_date,
        window_days=window_days,
        recent_tolerance_days=recent_tolerance_days,
        window_mode=window_mode,
    )
    if SIG_UPDATE_CLAIM in signals:
        return True, "context->current autorisé : update_claim récent déjà extrait"
    if SIG_VISIBLE_UPDATE in signals:
        return True, "context->current autorisé : visible_update_date récente déjà extraite"
    if SIG_VISIBLE_PUB in signals:
        return True, "context->current autorisé : visible_publication_date récente déjà extraite"
    if SIG_TITLE_DATE in signals:
        return True, "context->current autorisé : titre daté dans la fenêtre"
    if SIG_MODIFIED in signals and SIG_UPDATE_FORMULATION in signals:
        return True, (
            "context->current autorisé : modified_date récente + "
            "formulation d'actualisation dans claim_context"
        )
    if SIG_MODIFIED in signals:
        return False, (
            "context->current bloqué : modified_date seule insuffisante "
            "sans signal sémantique complémentaire"
        )
    return False, "context->current bloqué : aucun signal récent exploitable"


# --- Éligibilité et transitions ---

def eligibility_reason(
    source: dict[str, Any],
    temporal: dict[str, Any],
    contexts: list[dict[str, Any]],
    *,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
) -> tuple[bool, str]:
    status = temporal.get("temporal_status")
    title = str(source.get("title") or "")
    if status == STATUS_CURRENT:
        return False, "python_status=current : statut sûr, non éligible"
    if status == STATUS_MISMATCH:
        return False, (
            "python_status=mismatch : conservé en V1 (signal contradictoire "
            "explicite réservé à la V2)"
        )
    if status == STATUS_UNKNOWN:
        clues: list[str] = []
        if contexts:
            clues.append("contexte de citation exploitable")
        if any(
            temporal.get(key)
            for key in (
                "source_date",
                "modified_date",
                "claimed_dates",
                "claimed_updates",
                "visible_publication_date",
                "visible_update_date",
            )
        ):
            clues.append("date(s) déjà extraite(s)")
        if _title_dates(title):
            clues.append("titre daté")
        if clues:
            return True, "unknown avec matière exploitable ({0})".format(" ; ".join(clues))
        return False, "unknown sans matière exploitable"
    if status == STATUS_CONTEXT:
        signals = recent_signals(
            temporal,
            contexts,
            title,
            run_date=run_date,
            window_days=window_days,
            recent_tolerance_days=recent_tolerance_days,
            window_mode=window_mode,
        )
        if signals:
            return True, "context avec signal récent exploitable ({0})".format(
                " ; ".join(sorted(set(signals)))
            )
        return False, "context sans signal récent exploitable"
    return False, "statut Python non reconnu"


def allowed_transitions(
    python_status: str,
    *,
    context_to_current_ok: bool = False,
    current_allowed: bool = True,
) -> dict[str, list[str]]:
    if python_status == STATUS_UNKNOWN:
        allowed = [STATUS_CURRENT, STATUS_CONTEXT]
        if not current_allowed:
            allowed = [STATUS_CONTEXT]
        return {"allowed": allowed, "forbidden": [STATUS_MISMATCH]}
    if python_status == STATUS_CONTEXT:
        allowed = [STATUS_CONTEXT]
        if context_to_current_ok and current_allowed:
            allowed.append(STATUS_CURRENT)
        return {"allowed": allowed, "forbidden": [STATUS_MISMATCH]}
    if python_status == STATUS_CURRENT:
        return {"allowed": [STATUS_CURRENT], "forbidden": [STATUS_CONTEXT, STATUS_MISMATCH]}
    if python_status == STATUS_MISMATCH:
        return {"allowed": [STATUS_MISMATCH], "forbidden": [STATUS_CURRENT, STATUS_CONTEXT]}
    return {"allowed": [], "forbidden": []}


# --- Payload envoyé à Gemma ---

def build_requalification_payload(
    source: dict[str, Any],
    temporal: dict[str, Any],
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    title = str(source.get("title") or "")
    title_date, title_date_confidence = extract_title_date(title)
    return {
        "source_number": source.get("index"),
        "title": title,
        "title_date": title_date,
        "title_date_confidence": title_date_confidence,
        "domain": hostname(source.get("url")),
        "python_status": temporal.get("temporal_status"),
        "source_date": temporal.get("source_date"),
        "modified_date": temporal.get("modified_date"),
        "visible_publication_date": temporal.get("visible_publication_date"),
        "visible_update_date": temporal.get("visible_update_date"),
        "date_confidence": temporal.get("date_confidence"),
        "claimed_dates": temporal.get("claimed_dates") or [],
        "claimed_updates": temporal.get("claimed_updates") or [],
        "access_status": temporal.get("access_status"),
        "claim_types": sorted(
            {
                item
                for context in contexts
                for item in (context.get("claim_types") or [])
            }
        ),
        "claim_contexts": contexts,
        "recent_context_signals": [
            signal
            for context in contexts
            for signal in (context.get("recent_context_signals") or [])
        ],
    }


# --- Validation de la réponse Gemma ---

def _extract_dates(text: str) -> set[date]:
    found: set[date] = set()
    for match in _FRENCH_DATE_RE.finditer(text or ""):
        groups = match.groups()
        if groups[0]:
            parsed = parse_date(
                "{0} {1} {2}".format(groups[0], groups[1], groups[2])
            )
        elif groups[3]:
            parsed = parse_date("{0}-{1}-{2}".format(groups[3], groups[4], groups[5]))
        elif groups[6]:
            parsed = parse_date("{0}/{1}/{2}".format(groups[6], groups[7], groups[8]))
        else:
            parsed = None
        if parsed is not None:
            found.add(parsed)
    return found


def _allowed_dates_from_payload(payload: dict[str, Any]) -> set[date]:
    allowed: set[date] = set()
    for key in (
        "source_date",
        "modified_date",
        "visible_publication_date",
        "visible_update_date",
        "title_date",
    ):
        parsed = parse_date(payload.get(key))
        if parsed is not None:
            allowed.add(parsed)
    for key in ("claimed_dates", "claimed_updates"):
        for value in payload.get(key) or []:
            parsed = parse_date(value)
            if parsed is not None:
                allowed.add(parsed)
    for context in payload.get("claim_contexts") or []:
        allowed |= _extract_dates(context.get("claim_context"))
    title = payload.get("title")
    if title:
        allowed |= set(_title_dates(title))
    return allowed


def validate_gemma_response(
    payload: dict[str, Any],
    response: dict[str, Any],
    *,
    context_to_current_ok: bool = False,
    allowed_dates: set[date] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Valide une réponse Gemma sans jamais appeler de modèle."""
    if not isinstance(response, dict):
        return False, "réponse non objet JSON", None
    if response.get("source_number") != payload.get("source_number"):
        return False, "source_number différent du payload", None
    status = response.get("recommended_status")
    if status not in REQUALIFICATION_STATUSES:
        return False, "recommended_status hors enum", None
    transitions = allowed_transitions(
        payload.get("python_status"), context_to_current_ok=context_to_current_ok
    )
    if status not in transitions["allowed"]:
        return False, "transition non autorisée pour le statut Python", None
    confidence = response.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        return False, "confidence hors enum", None
    reason_code = response.get("reason_code")
    if reason_code not in REASON_CODES:
        return False, "reason_code hors enum", None
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False, "reason manquante ou vide", None
    if len(reason) > MAX_REASON_CHARS:
        return False, "reason trop longue", None
    if _CITATION_RE.search(reason):
        return False, "citation interdite dans reason", None
    if _URL_RE.search(reason):
        return False, "URL interdite dans reason", None
    allowed = allowed_dates if allowed_dates is not None else _allowed_dates_from_payload(payload)
    for parsed in _extract_dates(reason):
        if parsed not in allowed:
            return False, "date nouvelle dans reason", None
    normalized = {
        "source_number": payload.get("source_number"),
        "recommended_status": status,
        "confidence": confidence,
        "reason_code": reason_code,
        "reason": reason,
        "applied": confidence != "low",
    }
    return True, "", normalized


def _recent_publication_signals(
    payload: dict[str, Any],
    contexts: list[dict[str, Any]],
    cutoff: date,
    horizon: date,
) -> list[str]:
    signals: list[str] = []
    if any(
        _is_recent(value, cutoff, horizon) for value in payload.get("claimed_updates") or []
    ):
        signals.append(SIG_UPDATE_CLAIM)
    if _is_recent(payload.get("visible_update_date"), cutoff, horizon):
        signals.append(SIG_VISIBLE_UPDATE)
    if _is_recent(payload.get("visible_publication_date"), cutoff, horizon):
        signals.append(SIG_VISIBLE_PUB)
    context_dates: set[date] = set()
    embedded_signals: list[dict[str, Any]] = []
    for context in contexts or []:
        context_dates |= _extract_dates(context.get("claim_context"))
        context_dates |= _extract_dates(context.get("neighbor_context"))
        embedded_signals.extend(context.get("recent_context_signals") or [])
    recent_context_signals = payload.get("recent_context_signals") or []
    signal_dates: set[date] = set()
    signal_types: set[str] = set()
    for signal in list(recent_context_signals) + embedded_signals:
        parsed = parse_date(signal.get("date"))
        if parsed is not None:
            signal_dates.add(parsed)
        signal_type = signal.get("type")
        if signal_type:
            signal_types.add(signal_type)
    explicit_marker = any(
        _UPDATE_FORMULATION_RE.search(context.get("claim_context") or "")
        or (
            _RECENT_UPDATE_RE.search(context.get("claim_context") or "")
            and not _RECENT_UPDATE_STOP_RE.search(context.get("claim_context") or "")
        )
        or _RECENT_PUBLICATION_RE.search(context.get("claim_context") or "")
        or _RECENT_EVENT_RE.search(context.get("claim_context") or "")
        for context in contexts or []
    )
    if any(cutoff <= value <= horizon for value in signal_dates):
        signals.append("event_in_window")
        if SIGNAL_RECENT_UPDATE in signal_types:
            signals.append(SIG_UPDATE_CLAIM)
        if SIGNAL_RECENT_PUBLICATION in signal_types:
            signals.append(SIG_VISIBLE_PUB)
    elif explicit_marker and any(cutoff <= value <= horizon for value in context_dates):
        signals.append("event_in_window")
    if any(
        _UPDATE_FORMULATION_RE.search(context.get("claim_context") or "")
        for context in contexts or []
    ) and (
        any(cutoff <= value <= horizon for value in context_dates)
        or _is_recent(payload.get("visible_update_date"), cutoff, horizon)
        or _is_recent(payload.get("modified_date"), cutoff, horizon)
    ):
        signals.append(SIG_UPDATE_FORMULATION)
    return signals


def _strict_recent_signals(
    payload: dict[str, Any],
    contexts: list[dict[str, Any]],
    cutoff: date,
    horizon: date,
) -> list[str]:
    """Signaux récents STRICTS : uniquement les preuves explicites datées
    attribuables à la source (claims de mise à jour, dates visibles, signaux
    de contexte avec date chiffrée dans la fenêtre).

    N'inclut PAS les marqueurs génériques dérivés du texte brut du contexte
    (event_in_window / update_formulation calculés par _recent_publication_signals) :
    ceux-ci peuvent être déclenchés par une phrase méta de l'answer (ex. négation
    « aucune décision nouvelle entre le 20/08 et le 27/08 ») et neutraliseraient
    le garde-fou « title_date hors fenêtre sans signal récent explicite ».
    """
    signals: list[str] = []
    if any(
        _is_recent(value, cutoff, horizon) for value in payload.get("claimed_updates") or []
    ):
        signals.append(SIG_UPDATE_CLAIM)
    if _is_recent(payload.get("visible_update_date"), cutoff, horizon):
        signals.append(SIG_VISIBLE_UPDATE)
    if _is_recent(payload.get("visible_publication_date"), cutoff, horizon):
        signals.append(SIG_VISIBLE_PUB)
    embedded_signals: list[dict[str, Any]] = []
    for context in contexts or []:
        embedded_signals.extend(context.get("recent_context_signals") or [])
    recent_context_signals = payload.get("recent_context_signals") or []
    for signal in list(recent_context_signals) + embedded_signals:
        parsed = parse_date(signal.get("date"))
        if parsed is None or not (cutoff <= parsed <= horizon):
            continue
        signals.append("event_in_window")
        if signal.get("type") == SIGNAL_RECENT_UPDATE:
            signals.append(SIG_UPDATE_CLAIM)
        elif signal.get("type") == SIGNAL_RECENT_PUBLICATION:
            signals.append(SIG_VISIBLE_PUB)
    return sorted(set(signals))


def reason_code_coherent(
    payload: dict[str, Any],
    reason_code: str | None,
    *,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
) -> tuple[bool, str | None, str]:
    """Cohérence d'un reason_code avec les données du payload.

    Retourne (ok, reason_code effectif, message). Normalise uniquement les cas
    purement déterministes : role_current_title_date sans title_date mais avec
    signal récent explicite -> role_current_publication_context ; ou mise à
    jour juridique récente -> role_current_legal_update. Sinon rejet.
    """
    if reason_code not in REASON_CODES:
        return False, None, "reason_code hors enum"
    window_days, recent_tolerance_days = resolve_window_params(
        window_mode, window_days=window_days, recent_tolerance_days=recent_tolerance_days
    )
    run_value = parse_date(run_date) or date.today()
    cutoff = run_value - timedelta(days=window_days + recent_tolerance_days)
    horizon = run_value + timedelta(days=3)
    if reason_code == "role_current_title_date":
        title_date = parse_date(payload.get("title_date"))
        if title_date is None:
            signals = _recent_publication_signals(
                payload, payload.get("claim_contexts") or [], cutoff, horizon
            )
            if signals:
                return True, "role_current_publication_context", (
                    "normalisé : title_date absent mais signal récent ({0})".format(
                        "; ".join(sorted(signals))
                    )
                )
            return False, None, "role_current_title_date sans title_date et sans signal récent"
        if not (cutoff <= title_date <= horizon):
            return False, None, "role_current_title_date hors fenêtre de veille"
        return True, reason_code, "title_date dans la fenêtre"
    if reason_code == "role_current_update_claim":
        if payload.get("claimed_updates") or payload.get("visible_update_date"):
            return True, reason_code, "signal de mise à jour présent"
        return False, None, "role_current_update_claim sans claimed_updates ni visible_update_date"
    if reason_code == "role_current_publication_context":
        strict_signals = _strict_recent_signals(
            payload, payload.get("claim_contexts") or [], cutoff, horizon
        )
        if not strict_signals:
            return False, None, (
                "role_current_publication_context sans signal récent explicite daté"
            )
        return True, reason_code, (
            "signal récent explicite ({0})".format("; ".join(sorted(strict_signals)))
        )
    if reason_code == "role_current_recent_context":
        recent_context_signals = payload.get("recent_context_signals") or []
        if not recent_context_signals:
            return False, None, "role_current_recent_context sans recent_context_signals"
        in_window = False
        for signal in recent_context_signals:
            parsed = parse_date(signal.get("date"))
            if parsed is not None and cutoff <= parsed <= horizon:
                in_window = True
            if signal.get("type") == SIGNAL_RECENT_PUBLICATION and not parsed:
                return True, reason_code, "publication récente sans date chiffrée dans le contexte"
        if not in_window:
            return False, None, "role_current_recent_context sans signal récent exploitable"
        if _all_legal_updates(payload):
            return True, "role_current_legal_update", (
                "normalisé : mise à jour juridique récente (signaux recent_update + texte juridique)"
            )
        return True, reason_code, "signal récent du contexte"
    if reason_code == "role_current_legal_update":
        if not _legal_update_evidence(payload, cutoff, horizon):
            return False, None, (
                "role_current_legal_update sans mise à jour juridique récente "
                "(texte juridique ou update récent manquant)"
            )
        return True, reason_code, "mise à jour juridique récente"
    if reason_code == "role_context_legal_text":
        if "legal_text_date" in (payload.get("claim_types") or []):
            return True, reason_code, "legal_text_date présent"
        haystack = " ".join(
            [str(payload.get("title") or "")]
            + [
                str(context.get("claim_context") or "")
                for context in payload.get("claim_contexts") or []
            ]
        )
        if _LEGAL_MARKER_RE.search(haystack):
            return True, reason_code, "contexte juridique explicite"
        if _LEGAL_CONTEXT_RE.search(haystack):
            return True, reason_code, "contexte juridique implicite"
        domain = str(payload.get("domain") or "")
        if _LEGAL_DOMAIN_RE.search(domain):
            return True, reason_code, "domaine institutionnel juridique"
        return False, None, "role_context_legal_text sans signal juridique"
    return True, reason_code, ""


def current_recent_signal_required(
    payload: dict[str, Any],
    *,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
) -> tuple[bool, str]:
    """Règle V1 : une date de document hors fenêtre ne suffit jamais à current."""
    window_days, recent_tolerance_days = resolve_window_params(
        window_mode, window_days=window_days, recent_tolerance_days=recent_tolerance_days
    )
    run_value = parse_date(run_date) or date.today()
    cutoff = run_value - timedelta(days=window_days + recent_tolerance_days)
    horizon = run_value + timedelta(days=3)
    for key in ("title_date", "source_date", "visible_publication_date", "visible_update_date"):
        if _is_recent(payload.get(key), cutoff, horizon):
            return True, "date de document récente ({0}={1})".format(key, payload.get(key))
    signals = _recent_publication_signals(
        payload, payload.get("claim_contexts") or [], cutoff, horizon
    )
    if signals:
        return True, "signal récent explicite ({0})".format(" ; ".join(sorted(signals)))
    return False, "current sans date ni signal récent : reclasser context ou unknown"


def _all_legal_updates(payload: dict[str, Any]) -> bool:
    """True si tous les signaux de contexte sont des mises à jour récentes
    d'un texte juridique (Code, Légifrance, version en vigueur...)."""
    signals = payload.get("recent_context_signals") or []
    if not signals or any(signal.get("type") != SIGNAL_RECENT_UPDATE for signal in signals):
        return False
    haystack = " ".join(
        [str(payload.get("title") or "")]
        + [
            str(context.get("claim_context") or "")
            for context in payload.get("claim_contexts") or []
        ]
    )
    return bool(_LEGAL_UPDATE_RE.search(haystack))


def _legal_update_evidence(
    payload: dict[str, Any], cutoff: date, horizon: date
) -> bool:
    """Préconditions du reason_code role_current_legal_update."""
    haystack = " ".join(
        [str(payload.get("title") or "")]
        + [
            str(context.get("claim_context") or "")
            for context in payload.get("claim_contexts") or []
        ]
    )
    if not (_LEGAL_UPDATE_RE.search(haystack) or _LEGAL_MARKER_RE.search(haystack)):
        return False
    if any(
        _is_recent(value, cutoff, horizon) for value in payload.get("claimed_updates") or []
    ):
        return True
    return any(
        signal.get("type") == SIGNAL_RECENT_UPDATE
        and parse_date(signal.get("date")) is not None
        and cutoff <= parse_date(signal.get("date")) <= horizon
        for signal in payload.get("recent_context_signals") or []
    )


def _old_decision_context(payload: dict[str, Any]) -> bool:
    """Un contexte de citation présentant explicitement une décision comme
    ancienne / hors fenêtre (pas de décision nouvelle, dates antérieures...)."""
    for context in payload.get("claim_contexts") or []:
        if _OLD_DECISION_RE.search(context.get("claim_context") or ""):
            return True
    return False


def deterministic_pre_positioning(
    payload: dict[str, Any],
    *,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
) -> dict[str, Any]:
    """Pré-positionnement déterministe avant décision Gemma (V3).

    Règles :
    a. title_date manifestement hors fenêtre + aucun signal récent explicite
       => current interdit ; reclassement context proposé (évite le cas [48]) ;
    b. décision présentée comme ancienne sans événement récent associé
       => current interdit ;
    c. date récente explicite + signal récent fiable => current autorisé.

    Ne modifie pas les faits extraits ; sert de garde-fou complémentaire.
    """
    window_days, recent_tolerance_days = resolve_window_params(
        window_mode, window_days=window_days, recent_tolerance_days=recent_tolerance_days
    )
    run_value = parse_date(run_date) or date.today()
    cutoff, horizon = _window(run_value, window_days, recent_tolerance_days)
    signals = _strict_recent_signals(
        payload, payload.get("claim_contexts") or [], cutoff, horizon
    )
    title_date = parse_date(payload.get("title_date"))
    current_allowed = True
    forced_context = False
    reasons: list[str] = []
    if title_date is not None and not (cutoff <= title_date <= horizon):
        if not signals:
            current_allowed = False
            forced_context = True
            reasons.append(
                "title_date {0} hors fenêtre ({1}..{2}) sans signal récent explicite".format(
                    title_date.isoformat(), cutoff.isoformat(), horizon.isoformat()
                )
            )
    if _old_decision_context(payload) and not signals:
        current_allowed = False
        reasons.append("décision présentée comme ancienne sans événement récent associé")
    return {
        "current_allowed": current_allowed,
        "forced_status": STATUS_CONTEXT if forced_context else None,
        "reason": " ; ".join(reasons) if reasons else None,
        "recent_signals": sorted(set(signals)),
        "window_cutoff": cutoff.isoformat(),
        "window_horizon": horizon.isoformat(),
    }


def final_status_with_pre_positioning(
    python_status: str,
    recommendation: dict[str, Any] | None,
    validation_ok: bool,
    pre_position: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Statut final V3 : Python conserve la priorité sur les faits.

    En cas de rejet de validation, un pré-positionnement déterministe
    (ex. décision ancienne manifeste) peut fixer context pour une source
    unknown ; sinon le statut Python est conservé.
    """
    if validation_ok and recommendation is not None:
        applied = (
            bool(recommendation.get("applied"))
            and recommendation.get("recommended_status") != python_status
        )
        if applied:
            return recommendation.get("recommended_status"), ""
        return python_status, ""
    forced = (pre_position or {}).get("forced_status")
    if forced and python_status == STATUS_UNKNOWN:
        reason = (pre_position or {}).get("reason") or "source ancienne sans signal récent"
        return forced, "pré-positionnement déterministe : {0}".format(reason)
    return python_status, "statut Python conservé"


def validate_gemma_response_v1(
    payload: dict[str, Any],
    response: dict[str, Any],
    *,
    context_to_current_ok: bool = False,
    allowed_dates: set[date] | None = None,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
    pre_position: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Validation V1 : validations existantes + cohérence reason_code + fenêtre
    + pré-positionnement déterministe (current interdit pour les sources
    anciennes sans signal récent)."""
    ok, why, normalized = validate_gemma_response(
        payload,
        response,
        context_to_current_ok=context_to_current_ok,
        allowed_dates=allowed_dates,
    )
    if not ok or normalized is None:
        return ok, why, normalized
    ok_code, effective_code, code_message = reason_code_coherent(
        payload,
        normalized["reason_code"],
        run_date=run_date,
        window_days=window_days,
        recent_tolerance_days=recent_tolerance_days,
        window_mode=window_mode,
    )
    if not ok_code:
        return False, "reason_code incohérent : {0}".format(code_message), None
    if effective_code != normalized["reason_code"]:
        normalized["reason_code"] = effective_code
    if normalized["recommended_status"] == STATUS_CURRENT:
        ok_window, window_message = current_recent_signal_required(
            payload,
            run_date=run_date,
            window_days=window_days,
            recent_tolerance_days=recent_tolerance_days,
            window_mode=window_mode,
        )
        if not ok_window:
            return False, window_message, None
        pre = pre_position or deterministic_pre_positioning(
            payload,
            run_date=run_date,
            window_days=window_days,
            recent_tolerance_days=recent_tolerance_days,
            window_mode=window_mode,
        )
        if not pre.get("current_allowed"):
            return False, (
                "current interdit par pré-positionnement déterministe : {0}".format(
                    pre.get("reason") or "source ancienne sans signal récent"
                )
            ), None
    return True, "", normalized


# --- Fusion (Python garde la priorité sur les faits) ---

def apply_requalification(
    temporal: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(temporal)
    python_status = temporal.get("temporal_status")
    recommended_status = recommendation.get("recommended_status")
    confidence = recommendation.get("confidence")
    applied = bool(recommendation.get("applied")) and recommended_status != python_status
    updated["requalification"] = {
        "python_status": python_status,
        "recommended_status": recommended_status,
        "confidence": confidence,
        "reason_code": recommendation.get("reason_code"),
        "reason": recommendation.get("reason"),
        "applied": applied,
    }
    if applied:
        updated["temporal_status"] = recommended_status
        updated["note"] = "Requalifié Gemma ({0}) : {1}".format(
            recommendation.get("reason_code"), recommendation.get("reason")
        )
    return updated


# --- Plan dry-run complet ---

def requalification_plan(
    source: dict[str, Any],
    temporal: dict[str, Any],
    local_answers: dict[str, str],
    *,
    run_date: str | date | None = None,
    window_days: int = 7,
    recent_tolerance_days: int = 7,
    window_mode: str | None = None,
) -> dict[str, Any]:
    contexts = build_claim_contexts_for_source(
        source,
        local_answers,
        run_date=run_date,
        window_days=window_days,
        recent_tolerance_days=recent_tolerance_days,
        window_mode=window_mode,
    )
    eligible, elig_reason = eligibility_reason(
        source,
        temporal,
        contexts,
        run_date=run_date,
        window_days=window_days,
        recent_tolerance_days=recent_tolerance_days,
        window_mode=window_mode,
    )
    ok_ctx, ctx_reason = context_to_current_allowed(
        temporal,
        contexts,
        str(source.get("title") or ""),
        run_date=run_date,
        window_days=window_days,
        recent_tolerance_days=recent_tolerance_days,
        window_mode=window_mode,
    )
    payload = build_requalification_payload(source, temporal, contexts)
    pre_position = deterministic_pre_positioning(
        payload,
        run_date=run_date,
        window_days=window_days,
        recent_tolerance_days=recent_tolerance_days,
        window_mode=window_mode,
    )
    transitions = allowed_transitions(
        temporal.get("temporal_status"),
        context_to_current_ok=ok_ctx,
        current_allowed=pre_position.get("current_allowed", True),
    )
    guardrails: list[str] = []
    if temporal.get("temporal_status") == STATUS_CONTEXT:
        guardrails.append(ctx_reason)
    if not pre_position.get("current_allowed"):
        guardrails.append(
            "current interdit par pré-positionnement déterministe : {0}".format(
                pre_position.get("reason")
            )
        )
    return {
        "index": source.get("index"),
        "domain": hostname(source.get("url")),
        "title": str(source.get("title") or ""),
        "eligible": eligible,
        "eligibility_reason": elig_reason,
        "python_status": temporal.get("temporal_status"),
        "python_role": temporal.get("temporal_role"),
        "transitions": transitions,
        "guardrails": guardrails,
        "pre_position": pre_position,
        "payload": payload,
    }
