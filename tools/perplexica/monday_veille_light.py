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
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

MARKER_KINDS = ("ok", "partial", "failed")
SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


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


def cmd_resume_candidate(marker_dir: Path, run_root: Path, date_str: str) -> int:
    marker_path = marker_dir / f"{date_str}.partial"
    if not marker_path.is_file():
        _print("action", "none")
        return 0
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _print("action", "unsafe")
        _print("reason", "marker_unreadable")
        return 0
    if not isinstance(marker, dict):
        _print("action", "unsafe")
        _print("reason", "marker_not_object")
        return 0
    if marker.get("date") != date_str:
        _print("action", "unsafe")
        _print("reason", "marker_date_mismatch")
        return 0
    run_id = str(marker.get("run_id") or "")
    if not SAFE_RUN_ID_RE.fullmatch(run_id):
        _print("action", "unsafe")
        _print("reason", "invalid_run_id")
        return 0
    run_root = Path(run_root).resolve()
    run_dir = (run_root / run_id).resolve()
    if run_dir.parent != run_root or run_dir.name != run_id:
        _print("action", "unsafe")
        _print("reason", "run_dir_outside_root")
        _print("run_id", run_id)
        return 0
    run_file = run_dir / "run.json"
    if not run_file.is_file():
        _print("action", "unsafe")
        _print("reason", "run_json_missing")
        _print("run_id", run_id)
        return 0
    try:
        metadata = json.loads(run_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _print("action", "unsafe")
        _print("reason", "run_json_unreadable")
        _print("run_id", run_id)
        return 0
    if not isinstance(metadata, dict):
        _print("action", "unsafe")
        _print("reason", "run_json_not_object")
        _print("run_id", run_id)
        return 0
    started_at = str(metadata.get("started_at") or "")
    try:
        business_date = datetime.fromisoformat(started_at.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        _print("action", "unsafe")
        _print("reason", "started_at_invalid")
        _print("run_id", run_id)
        return 0
    if business_date != date_str:
        _print("action", "unsafe")
        _print("reason", "run_date_mismatch")
        _print("run_id", run_id)
        return 0
    if not (run_dir / "result.json").is_file():
        _print("action", "unsafe")
        _print("reason", "result_json_missing")
        _print("run_id", run_id)
        return 0
    if metadata.get("mail_mode") != "both":
        _print("action", "unsafe")
        _print("reason", "mail_mode_not_both")
        _print("run_id", run_id)
        return 0
    if not metadata.get("editorial_requested"):
        _print("action", "unsafe")
        _print("reason", "editorial_not_requested")
        _print("run_id", run_id)
        return 0
    if not metadata.get("raw_mail_sent"):
        _print("action", "unsafe")
        _print("reason", "raw_not_sent")
        _print("run_id", run_id)
        return 0
    if marker.get("raw_mail_sent") is False:
        _print("action", "unsafe")
        _print("reason", "marker_raw_not_sent")
        _print("run_id", run_id)
        return 0
    if marker.get("editorial_mail_sent") is True:
        _print("action", "already_sent")
        _print("reason", "marker_editorial_mail_sent_true")
        _print("run_id", run_id)
        _print("run_dir", str(run_dir))
        return 0
    send_state = metadata.get("editorial_send_state")
    if metadata.get("editorial_mail_sent") or send_state == "sent":
        if metadata.get("editorial_mail_sent"):
            _print("action", "already_sent")
            _print("reason", "editorial_mail_sent_true")
        else:
            _print("action", "smtp_outcome_unknown")
            _print("reason", "send_state_sent_without_sent_flag")
        _print("run_id", run_id)
        _print("run_dir", str(run_dir))
        return 0
    if send_state == "sending":
        _print("action", "smtp_outcome_unknown")
        _print("reason", "send_state_sending")
        _print("run_id", run_id)
        _print("run_dir", str(run_dir))
        return 0
    if send_state not in (None, "", "prepared"):
        _print("action", "unsafe")
        _print("reason", "unknown_send_state")
        _print("run_id", run_id)
        _print("run_dir", str(run_dir))
        return 0
    _print("action", "resume")
    _print("reason", "eligible")
    _print("run_id", run_id)
    _print("run_dir", str(run_dir))
    return 0


def compute_business_context(today_str: str, weekday: int, hour: int) -> dict[str, Any]:
    try:
        today = date.fromisoformat(today_str)
    except ValueError as exc:
        raise ValueError(f"invalid today: {today_str!r}") from exc
    if weekday < 1 or weekday > 7:
        raise ValueError(f"invalid weekday: {weekday}")
    if hour < 0 or hour > 23:
        raise ValueError(f"invalid hour: {hour}")
    if today.isoweekday() != weekday:
        raise ValueError(
            f"weekday {weekday} does not match today {today_str} ({today.isoweekday()})"
        )
    if weekday == 2:
        business_date = today - timedelta(days=1)
        mode = "recovery" if 3 <= hour <= 6 else "recovery_closed"
    else:
        mode = "normal"
        business_date = today
    return {
        "mode": mode,
        "business_date": business_date.isoformat(),
        "today": today.isoformat(),
        "weekday": weekday,
        "hour": hour,
    }


def cmd_business_context(today_str: str, weekday_str: str, hour_str: str) -> int:
    try:
        context = compute_business_context(today_str, int(weekday_str), int(hour_str))
    except (TypeError, ValueError) as exc:
        print(f"business_context_error={exc}", file=sys.stderr)
        return 1
    for key in ("mode", "business_date", "today", "weekday", "hour"):
        _print(key, context[key])
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

    p = sub.add_parser("business-context")
    p.add_argument("--today", required=True)
    p.add_argument("--weekday", required=True)
    p.add_argument("--hour", required=True)
    p.set_defaults(func=lambda a: cmd_business_context(a.today, a.weekday, a.hour))

    p = sub.add_parser("resume-candidate")
    p.add_argument("marker_dir")
    p.add_argument("run_root")
    p.add_argument("date")
    p.set_defaults(func=lambda a: cmd_resume_candidate(
        Path(a.marker_dir), Path(a.run_root), a.date
    ))

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
