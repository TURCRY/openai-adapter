#!/usr/bin/env python3
"""CLI pour appliquer le validateur temporel a un result.json existant.

Sans Perplexica, sans Gemma, sans SMTP. Lit les reponses locales depuis
<run_dir>/searches/<nom>/result.json, revalide les cited_sources du result.json
et affiche un rapport texte (ou JSON avec --json).

Exemples :
    python temporal_report.py --run-dir <run> --indices 2,6,10,36
    python temporal_report.py --run-dir <run> --sample 16
    python temporal_report.py --run-dir <run> --all --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from temporal_validation import validate_cited_sources

SUMMARY_KEYS = (
    "temporal_validation_count",
    "current_count",
    "context_count",
    "mismatch_count",
    "unknown_count",
    "direct_date_count",
    "indirect_date_count",
    "unknown_date_count",
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"error: cannot read {path}: {exc}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: invalid JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        raise SystemExit(f"error: {path} must contain a JSON object.")
    return payload


def load_local_answers(run_dir: Path) -> dict[str, str]:
    local_answers: dict[str, str] = {}
    searches_dir = run_dir / "searches"
    if not searches_dir.is_dir():
        return local_answers
    for search_dir in sorted(item for item in searches_dir.iterdir() if item.is_dir()):
        result_file = search_dir / "result.json"
        if not result_file.is_file():
            continue
        payload = load_json(result_file)
        answer = payload.get("answer_markdown")
        if isinstance(answer, str) and answer.strip():
            local_answers[search_dir.name] = answer
    return local_answers


def hostname(url: Any) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    netloc = urlsplit(url.strip()).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def parse_indices(raw: str) -> list[int]:
    indices: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            indices.append(int(part))
        except ValueError as exc:
            raise SystemExit(f"error: invalid index '{part}'") from exc
    return indices


def select_sources(result: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    sources = [source for source in result.get("cited_sources") or [] if isinstance(source, dict)]
    if args.indices:
        wanted = set(parse_indices(args.indices))
        sources = [source for source in sources if isinstance(source.get("index"), int) and source["index"] in wanted]
    elif args.sample is not None:
        sources = sources[: args.sample]
    return sources


def build_rows(validated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in validated:
        temporal = source.get("temporal") if isinstance(source.get("temporal"), dict) else {}
        rows.append(
            {
                "index": source.get("index"),
                "domain": hostname(source.get("url")),
                "access_status": temporal.get("access_status"),
                "source_date": temporal.get("source_date"),
                "date_evidence": temporal.get("date_evidence"),
                "date_confidence": temporal.get("date_confidence"),
                "date_verification": temporal.get("date_verification"),
                "claimed_dates": temporal.get("claimed_dates") or [],
                "temporal_role": temporal.get("temporal_role"),
                "temporal_status": temporal.get("temporal_status"),
                "note": temporal.get("note") or "",
            }
        )
    return rows


def render_text(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    for row in rows:
        claimed = ", ".join(row["claimed_dates"]) if row["claimed_dates"] else "-"
        lines.append(
            "[{index}] {domain} | access={access_status} | date={source_date} | "
            "evidence={date_evidence} | conf={date_confidence} | verif={date_verification}".format(**row)
        )
        lines.append(
            "    claimed_dates={claimed} | role={temporal_role} | status={temporal_status}".format(claimed=claimed, **row)
        )
        if row["note"]:
            lines.append("    note: {note}".format(**row))
    lines.append("")
    lines.append("Summary:")
    for key in SUMMARY_KEYS:
        lines.append(f"  {key}: {summary.get(key, 0)}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Valider temporellement les cited_sources d'un result.json.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Repertoire du run (contient result.json et searches/).")
    parser.add_argument("--indices", help="Liste d'indices globaux separes par des virgules (ex: 2,6,10,36).")
    parser.add_argument("--sample", type=int, help="Valider les N premieres cited_sources.")
    parser.add_argument("--all", action="store_true", help="Valider toutes les cited_sources.")
    parser.add_argument("--json", action="store_true", help="Sortie JSON.")
    parser.add_argument("--run-date", help="Date de reference du run (YYYY-MM-DD), sinon date du jour.")
    parser.add_argument("--timeout", type=float, default=6.0, help="Timeout HTTP par source (defaut 6 s).")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrence HTTP (defaut 4).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.indices or args.sample is not None or args.all):
        print("error: specify --indices, --sample or --all", file=sys.stderr)
        return 2
    run_dir = args.run_dir
    result_file = run_dir / "result.json"
    if not result_file.is_file():
        print(f"error: no result.json in {run_dir}", file=sys.stderr)
        return 2
    result = load_json(result_file)
    local_answers = load_local_answers(run_dir)
    selected = select_sources(result, args)
    if not selected:
        print("error: no cited source matched the selection", file=sys.stderr)
        return 2
    validated, summary = validate_cited_sources(
        selected,
        local_answers=local_answers,
        timeout=args.timeout,
        concurrency=args.concurrency,
        run_date=args.run_date,
    )
    rows = build_rows(validated)
    if args.json:
        print(json.dumps({"sources": rows, "summary": summary}, ensure_ascii=False, indent=2))
    else:
        print(render_text(rows, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
