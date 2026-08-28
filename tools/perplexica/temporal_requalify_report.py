#!/usr/bin/env python3
"""Rapport dry-run de la requalification temporelle ciblée (V1).

Ne relance ni Perplexica ni Gemma : réutilise la validation Python puis
affiche, pour chaque source sélectionnée, l'éligibilité, le payload exact qui
serait envoyé à Gemma, les transitions autorisées et les garde-fous.

Exemples :
    python temporal_requalify_report.py --run-dir <run> --indices 38,33,6,11,12,23,4,48,10,1 --run-date 2026-08-27
    python temporal_requalify_report.py --run-dir <run> --all --run-date 2026-08-27 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from temporal_validation import validate_cited_sources
from temporal_requalification import (
    WINDOW_EXTENDED_30D,
    WINDOW_MODES,
    WINDOW_STRICT_7D,
    requalification_plan,
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
        sources = [
            source
            for source in sources
            if isinstance(source.get("index"), int) and source["index"] in wanted
        ]
    elif args.sample is not None:
        sources = sources[: args.sample]
    return sources


def render_text(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    eligible_count = 0
    for row in rows:
        eligible_count += 1 if row["eligible"] else 0
        allowed = ", ".join(row["transitions"]["allowed"]) if row["transitions"]["allowed"] else "-"
        forbidden = ", ".join(row["transitions"]["forbidden"]) if row["transitions"]["forbidden"] else "-"
        lines.append("[{index}] {domain} — {title}".format(**row))
        lines.append(
            "    python_status={python_status} | eligible={oui_non}".format(
                oui_non="oui" if row["eligible"] else "non", **row
            )
        )
        lines.append("    raison d'éligibilité: {eligibility_reason}".format(**row))
        lines.append(
            "    transitions autorisées: [{allowed}] | interdites: [{forbidden}]".format(
                allowed=allowed, forbidden=forbidden
            )
        )
        for guardrail in row["guardrails"]:
            lines.append("    garde-fou: {0}".format(guardrail))
        lines.append("    payload (envoyé à Gemma en dry-run):")
        payload_text = json.dumps(row["payload"], ensure_ascii=False, indent=2)
        for payload_line in payload_text.split("\n"):
            lines.append("      " + payload_line)
        lines.append("")
    lines.append("Résumé dry-run : {0}/{1} sources éligibles".format(eligible_count, len(rows)))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rapport dry-run de la requalification temporelle (V1)."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Répertoire du run (contient result.json et searches/).",
    )
    parser.add_argument(
        "--indices", help="Liste d'indices globaux séparés par des virgules (ex: 38,33,6)."
    )
    parser.add_argument("--sample", type=int, help="Analyser les N premières cited_sources.")
    parser.add_argument("--all", action="store_true", help="Analyser toutes les cited_sources.")
    parser.add_argument("--run-date", help="Date de référence du run (YYYY-MM-DD).")
    parser.add_argument("--window-days", type=int, default=7, help="Fenêtre d'actualité (défaut 7).")
    parser.add_argument(
        "--recent-tolerance-days", type=int, default=7, help="Tolérance récence (défaut 7)."
    )
    parser.add_argument(
        "--window-mode",
        choices=[WINDOW_STRICT_7D, WINDOW_EXTENDED_30D],
        default=None,
        help="Mode de fenêtre temporelle (strict_7d / extended_30d).",
    )
    parser.add_argument("--timeout", type=float, default=6.0, help="Timeout HTTP par source (défaut 6 s).")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrence HTTP (défaut 4).")
    parser.add_argument("--json", action="store_true", help="Sortie JSON.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode dry-run (toujours actif en V1 : aucun appel Gemma).",
    )
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
    if args.window_mode:
        params = WINDOW_MODES[args.window_mode]
        args.window_days = params["window_days"]
        args.recent_tolerance_days = params["recent_tolerance_days"]
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
        window_days=args.window_days,
        recent_tolerance_days=args.recent_tolerance_days,
    )
    rows: list[dict[str, Any]] = []
    for source in validated:
        temporal = source.get("temporal") if isinstance(source.get("temporal"), dict) else {}
        rows.append(
            requalification_plan(
                source,
                temporal,
                local_answers,
                run_date=args.run_date,
                window_days=args.window_days,
                recent_tolerance_days=args.recent_tolerance_days,
                window_mode=args.window_mode,
            )
        )
    if args.json:
        print(json.dumps({"sources": rows, "temporal_summary": summary}, ensure_ascii=False, indent=2))
    else:
        print(render_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

