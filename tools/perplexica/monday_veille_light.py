#!/usr/bin/env python3
"""Lightweight helpers for the Monday veille wrapper (tools/perplexica).

Deliberately small scope: SearXNG availability check, daily marker read/write,
run.json summary for marker decisions, and construction of the light failure
mail. SMTP delivery is delegated to mail_sender.py (config from .env).
No cron, no secrets, no heavy job logic here.
"""

from __future__ import annotations

import argparse
import html as _html
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

MARKER_KINDS = ("ok", "partial", "failed")


def _print(name: str, value: Any) -> None:
    print(f"{name}={value}")


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def cmd_searxng_check(url: str, timeout: float) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        _print("searxng_error", f"{exc.__class__.__name__}: {exc}")
        return 1
    results = payload.get("results") or []
    unresponsive = payload.get("unresponsive_engines") or []
    _print("results", len(results))
    _print("unresponsive_engines", len(unresponsive))
    return 0 if len(results) > 0 else 1


def cmd_marker_status(marker_dir: Path, date_str: str) -> int:
    for kind in MARKER_KINDS:
        if (marker_dir / f"{date_str}.{kind}").is_file():
            print(kind)
            return 0
    print("none")
    return 0


def cmd_write_marker(
    marker_dir: Path,
    date_str: str,
    kind: str,
    slot: str,
    run_id: str,
    run_status: str,
    raw_mail_sent: bool,
    editorial_mail_sent: bool,
    mail_sent: bool,
    source_count: int,
    cited_source_count: int,
) -> int:
    if kind not in MARKER_KINDS:
        print(f"invalid marker kind: {kind}", file=sys.stderr)
        return 1
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "date": date_str,
        "slot": slot,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": kind,
        "run_id": run_id,
        "run_status": run_status,
        "raw_mail_sent": raw_mail_sent,
        "editorial_mail_sent": editorial_mail_sent,
        "mail_sent": mail_sent,
        "source_count": source_count,
        "cited_source_count": cited_source_count,
    }
    marker_file = marker_dir / f"{date_str}.{kind}"
    marker_file.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print("marker", str(marker_file))
    return 0


def cmd_run_summary(run_dir: Path) -> int:
    run_file = run_dir / "run.json"
    if not run_file.is_file():
        print(f"run.json missing: {run_file}", file=sys.stderr)
        return 1
    data = json.loads(run_file.read_text(encoding="utf-8"))
    fields = (
        "status",
        "raw_mail_sent",
        "editorial_mail_sent",
        "mail_sent",
        "source_count",
        "cited_source_count",
    )
    _print("run_id", run_dir.name)
    for field in fields:
        _print(field, data.get(field, ""))
    return 0


def _html_from_text(text: str) -> str:
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    body = "\n".join(f"<p>{_html.escape(p)}</p>" for p in paragraphs)
    return f"<!DOCTYPE html><html><body>{body}</body></html>"


def cmd_failure_mail(out_json: Path, subject: str, body: str) -> int:
    payload = {"subject": subject, "text": body, "html": _html_from_text(body)}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print("mail_json", str(out_json))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Light helpers for the Monday veille wrapper.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("searxng-check")
    p.add_argument("url")
    p.add_argument("--timeout", type=float, default=20.0)
    p.set_defaults(func=lambda a: cmd_searxng_check(a.url, a.timeout))

    p = sub.add_parser("marker-status")
    p.add_argument("marker_dir")
    p.add_argument("date")
    p.set_defaults(func=lambda a: cmd_marker_status(Path(a.marker_dir), a.date))

    p = sub.add_parser("write-marker")
    p.add_argument("marker_dir")
    p.add_argument("date")
    p.add_argument("kind", choices=MARKER_KINDS)
    p.add_argument("--slot", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--run-status", default="")
    p.add_argument("--raw-mail-sent", default="false")
    p.add_argument("--editorial-mail-sent", default="false")
    p.add_argument("--mail-sent", default="false")
    p.add_argument("--source-count", type=int, default=0)
    p.add_argument("--cited-source-count", type=int, default=0)
    p.set_defaults(func=lambda a: cmd_write_marker(
        Path(a.marker_dir), a.date, a.kind, a.slot, a.run_id, a.run_status,
        _parse_bool(a.raw_mail_sent), _parse_bool(a.editorial_mail_sent),
        _parse_bool(a.mail_sent), a.source_count, a.cited_source_count,
    ))

    p = sub.add_parser("run-summary")
    p.add_argument("run_dir")
    p.set_defaults(func=lambda a: cmd_run_summary(Path(a.run_dir)))

    p = sub.add_parser("failure-mail")
    p.add_argument("out_json")
    p.add_argument("--subject", required=True)
    p.add_argument("--body", required=True)
    p.set_defaults(func=lambda a: cmd_failure_mail(Path(a.out_json), a.subject, a.body))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())