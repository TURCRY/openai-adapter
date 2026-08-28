#!/usr/bin/env python3
"""Build readable email content from a Perplexica canonical JSON result.

This module only builds files and strings. It does not send email, configure SMTP,
or store recipients.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "mail"
BRAND_TITLE = "Veille Perplexica"
INLINE_RE = re.compile(
    r"\[([^\]]+)\]\(([^)]+)\)|"
    r"\[([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]|"
    r"\*\*([^*]+)\*\*|"
    r"\*([^*]+)\*"
)
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
TEXT_WRAP_STYLE = "white-space:normal;word-wrap:break-word;overflow-wrap:break-word;"
URL_WRAP_STYLE = "word-break:break-word;word-wrap:break-word;overflow-wrap:anywhere;"


class MailBuildError(Exception):
    """User-facing mail build failure."""


def unwrap_job_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "result" in payload and isinstance(payload.get("result"), dict):
        return payload["result"], {
            "job_id": payload.get("job_id"),
            "job_created_at": payload.get("created_at"),
        }
    return payload, {"job_id": payload.get("job_id"), "job_created_at": None}


def load_input_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MailBuildError(f"Cannot read input JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MailBuildError(f"Invalid JSON input: {path}") from exc
    if not isinstance(payload, dict):
        raise MailBuildError("Invalid JSON input: top-level value must be an object.")
    return unwrap_job_payload(payload)


def source_map(canonical_result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    sources = canonical_result.get("all_sources", [])
    if not isinstance(sources, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        index = source.get("index")
        if isinstance(index, int) and index > 0:
            result[index] = source
    return result


def render_citation_html(raw_numbers: str, sources_by_index: dict[int, dict[str, Any]]) -> str:
    rendered = []
    for part in raw_numbers.split(","):
        number_text = part.strip()
        number = int(number_text)
        label = f"[{number_text}]"
        source = sources_by_index.get(number)
        url = source.get("url") if isinstance(source, dict) else None
        style = (
            "font-size:11px;line-height:1;color:#4f6f9f;text-decoration:none;"
            f"vertical-align:super;margin-left:2px;{TEXT_WRAP_STYLE}"
        )
        if isinstance(url, str) and url:
            rendered.append(
                '<a href="{}" style="{}">{}</a>'.format(
                    html.escape(url, quote=True), style, html.escape(label)
                )
            )
        else:
            rendered.append(f'<span style="{style}color:#6b7280;">{html.escape(label)}</span>')
    return "<span>" + "".join(rendered) + "</span>"


def render_inline_html(text: str, sources_by_index: dict[int, dict[str, Any]]) -> str:
    output: list[str] = []
    pos = 0
    for match in INLINE_RE.finditer(text):
        output.append(html.escape(text[pos : match.start()]))
        link_text, link_url, citation_numbers, bold_text, italic_text = match.groups()
        if link_text is not None and link_url is not None:
            output.append(
                f'<a href="{{}}" style="color:#2454a6;text-decoration:underline;{TEXT_WRAP_STYLE}">{{}}</a>'.format(
                    html.escape(link_url, quote=True),
                    render_inline_html(link_text, sources_by_index),
                )
            )
        elif citation_numbers is not None:
            output.append(render_citation_html(citation_numbers, sources_by_index))
        elif bold_text is not None:
            output.append(f"<strong>{render_inline_html(bold_text, sources_by_index)}</strong>")
        elif italic_text is not None:
            output.append(f"<em>{render_inline_html(italic_text, sources_by_index)}</em>")
        pos = match.end()
    output.append(html.escape(text[pos:]))
    return "".join(output)


def markdown_to_html(markdown: str, sources_by_index: dict[int, dict[str, Any]]) -> str:
    lines = (markdown or "").splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    in_list = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(line.strip() for line in paragraph if line.strip())
            parts.append(
                f'<p style="margin:0 0 14px 0;font-size:15px;line-height:1.58;color:#1f2933;{TEXT_WRAP_STYLE}">'
                f"{render_inline_html(text, sources_by_index)}</p>"
            )
            paragraph = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for line in lines:
        if not line.strip():
            flush_paragraph()
            close_list()
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            size = "18px" if level == 1 else "16px"
            margin_top = "20px" if parts else "0"
            parts.append(
                f'<h2 style="margin:{margin_top} 0 8px 0;font-size:{size};line-height:1.35;'
                f'font-weight:700;color:#172033;{TEXT_WRAP_STYLE}">{render_inline_html(heading.group(2), sources_by_index)}</h2>'
            )
            continue

        bullet = BULLET_RE.match(line)
        if bullet:
            flush_paragraph()
            if not in_list:
                parts.append(f'<ul style="margin:0 0 14px 20px;padding:0;color:#1f2933;{TEXT_WRAP_STYLE}">')
                in_list = True
            parts.append(
                f'<li style="margin:0 0 7px 0;font-size:15px;line-height:1.55;color:#1f2933;{TEXT_WRAP_STYLE}">'
                f"{render_inline_html(bullet.group(1), sources_by_index)}</li>"
            )
            continue

        close_list()
        paragraph.append(line)

    flush_paragraph()
    close_list()
    return "\n".join(parts)


def markdown_to_text(markdown: str) -> str:
    text = markdown or ""
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = text.replace("**", "").replace("*", "")
    return text.strip()


def make_subject(question: str, subject: str | None = None) -> str:
    clean = " ".join((subject or question or "Conversation Perplexica").split())
    if subject:
        return clean
    return f"{BRAND_TITLE} — {clean}"


def cited_sources(canonical_result: dict[str, Any]) -> list[dict[str, Any]]:
    sources = canonical_result.get("cited_sources", [])
    return [source for source in sources if isinstance(source, dict)] if isinstance(sources, list) else []


def validate_canonical_result(canonical_result: dict[str, Any]) -> None:
    if not isinstance(canonical_result, dict):
        raise MailBuildError("Canonical result must be an object.")
    status = canonical_result.get("status")
    if status != "completed":
        raise MailBuildError(f"Cannot build mail for status={status!r}; expected 'completed'.")
    if not isinstance(canonical_result.get("question"), str):
        raise MailBuildError("Canonical result is missing string field 'question'.")
    if not isinstance(canonical_result.get("answer_markdown"), str):
        raise MailBuildError("Canonical result is missing string field 'answer_markdown'.")


def build_html_body(
    mail_subject: str,
    question: str,
    generated_at: str,
    answer_html: str,
    html_sources: list[str],
    footer: str,
) -> str:
    source_rows = "".join(html_sources) or (
        '<tr><td style="padding:0 0 10px 0;font-size:14px;line-height:1.5;color:#4b5563;">'
        "Aucune source citée.</td></tr>"
    )
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>{html.escape(mail_subject)}</title>
</head>
<body style="margin:0;padding:0;background:#f3f5f7;font-family:Arial,'Segoe UI',sans-serif;color:#1f2933;{TEXT_WRAP_STYLE}">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#f3f5f7;margin:0;padding:24px 0;border-collapse:collapse;">
    <tr>
      <td align="center" style="padding:0 12px;{TEXT_WRAP_STYLE}">
        <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0" style="width:600px;border-collapse:collapse;background:#ffffff;border:1px solid #e2e8f0;">
          <tr>
            <td width="100%" style="width:100%;padding:26px 30px 18px 30px;border-bottom:1px solid #e5e7eb;{TEXT_WRAP_STYLE}">
              <div style="font-size:13px;line-height:1.4;font-weight:700;color:#2454a6;text-transform:uppercase;{TEXT_WRAP_STYLE}">{BRAND_TITLE}</div>
              <div style="margin-top:5px;font-size:12px;line-height:1.4;color:#6b7280;{TEXT_WRAP_STYLE}">Date de génération : {html.escape(generated_at)}</div>
              <h1 style="margin:18px 0 0 0;font-size:24px;line-height:1.28;font-weight:700;color:#111827;{TEXT_WRAP_STYLE}">{html.escape(question)}</h1>
            </td>
          </tr>
          <tr>
            <td width="100%" style="width:100%;padding:26px 30px 8px 30px;{TEXT_WRAP_STYLE}">
              {answer_html}
            </td>
          </tr>
          <tr>
            <td width="100%" style="width:100%;padding:8px 30px 0 30px;{TEXT_WRAP_STYLE}">
              <div style="border-top:1px solid #e5e7eb;font-size:1px;line-height:1px;{TEXT_WRAP_STYLE}">&nbsp;</div>
            </td>
          </tr>
          <tr>
            <td width="100%" style="width:100%;padding:22px 30px 8px 30px;{TEXT_WRAP_STYLE}">
              <h2 style="margin:0 0 14px 0;font-size:18px;line-height:1.35;font-weight:700;color:#172033;{TEXT_WRAP_STYLE}">Sources principales</h2>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="width:100%;border-collapse:collapse;{TEXT_WRAP_STYLE}">
                {source_rows}
              </table>
            </td>
          </tr>
          <tr>
            <td width="100%" style="width:100%;padding:14px 30px 26px 30px;{TEXT_WRAP_STYLE}">
              <div style="font-size:12px;line-height:1.45;color:#6b7280;{TEXT_WRAP_STYLE}">{html.escape(footer)}</div>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def valid_editorial_payload(editorial: dict[str, Any] | None) -> bool:
    return (
        isinstance(editorial, dict)
        and editorial.get("status") == "completed"
        and isinstance(editorial.get("body_markdown"), str)
        and bool(editorial.get("body_markdown", "").strip())
    )


def clean_display_title(display_title: str | None) -> str | None:
    if not isinstance(display_title, str):
        return None
    clean = " ".join(display_title.split())
    return clean or None


def display_url(url: Any, limit: int = 40) -> str:
    """Build a short visible preview of a URL for the sources list.

    Only the visible text is shortened: the scheme, a leading "www.", the query
    string and the fragment are never displayed. The caller must keep the
    original URL for the href attribute.
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    raw = url.strip()
    parts = urlsplit(raw)
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path
    if not host:
        host, _, rest = path.lstrip("/").partition("/")
        host = host.lower()
        path = "/" + rest if rest else ""
    if not path or path == "/":
        full = host
    else:
        full = host + path.rstrip("/")
    if len(full) <= limit:
        return full
    segments = [segment for segment in path.strip("/").split("/") if segment]
    base = host
    if segments:
        candidate = host + "/" + segments[0] + "/"
        if len(candidate) <= limit:
            base = candidate
        else:
            base = host + "/"
    if len(base) > limit:
        base = base[:limit]
    return base + "(...)"


def build_mail(
    canonical_result: dict[str, Any],
    subject: str | None = None,
    editorial: dict[str, Any] | None = None,
    display_title: str | None = None,
) -> dict[str, Any]:
    validate_canonical_result(canonical_result)
    use_editorial = valid_editorial_payload(editorial)
    question = canonical_result["question"]
    configured_display_title = clean_display_title(display_title)
    editorial_title = clean_display_title(editorial.get("title")) if use_editorial else None
    visible_title = editorial_title or configured_display_title or question
    answer_markdown = editorial["body_markdown"].strip() if use_editorial else canonical_result["answer_markdown"]
    sources_by_index = source_map(canonical_result)
    cited = cited_sources(canonical_result)
    generated_at = datetime.now(timezone.utc).isoformat()
    mail_subject = make_subject(question, subject)
    answer_html = markdown_to_html(answer_markdown, sources_by_index)
    answer_text = markdown_to_text(answer_markdown)

    html_sources = []
    text_sources = []
    for source in cited:
        index = source.get("index")
        url = source.get("url") or ""
        title = source.get("title")
        title = " ".join(str(title).split()) if isinstance(title, str) and title.strip() else ""
        if not title:
            title = display_url(url)
        if not title:
            title = "Source sans titre"
        title_label = f"[{html.escape(str(index))}] {html.escape(title)}"
        if url:
            href = html.escape(str(url), quote=True)
            title_link = (
                f'<a href="{href}" style="font-weight:700;color:#111827;text-decoration:underline;{TEXT_WRAP_STYLE}">'
                f"{title_label}</a>"
            )
            url_link = (
                f'<div style="margin-top:2px;{TEXT_WRAP_STYLE}">'
                f'<a href="{href}" style="color:#6b7280;text-decoration:none;font-size:12px;{URL_WRAP_STYLE}">'
                f"{html.escape(display_url(url))}</a></div>"
            )
        else:
            title_link = f'<span style="font-weight:700;color:#111827;{TEXT_WRAP_STYLE}">{title_label}</span>'
            url_link = ""
        html_sources.append(
            f'<tr><td style="padding:0 0 12px 0;font-size:14px;line-height:1.45;color:#1f2933;{TEXT_WRAP_STYLE}">'
            f'<div style="{TEXT_WRAP_STYLE}">{title_link}</div>{url_link}'
            "</td></tr>"
        )
        text_sources.append(f"[{index}] {title}\n{url}".rstrip())

    all_source_count = len(canonical_result.get("all_sources", []) or [])
    cited_source_count = len(cited)
    footer = f"{all_source_count} sources consultées · {cited_source_count} sources citées"
    html_body = build_html_body(mail_subject, visible_title, generated_at, answer_html, html_sources, footer)

    text_body = "\n\n".join(
        [
            BRAND_TITLE,
            f"Date : {generated_at}",
            "QUESTION\n" + visible_title,
            "SYNTHÈSE\n" + answer_text,
            "SOURCES PRINCIPALES\n" + ("\n\n".join(text_sources) if text_sources else "Aucune source citée."),
            footer,
        ]
    )

    metadata = {
        "subject": mail_subject,
        "generated_at": generated_at,
        "chat_id": canonical_result.get("chat_id"),
        "message_id": canonical_result.get("message_id"),
        "question": question,
        "display_title": configured_display_title,
        "visible_title": visible_title,
        "status": canonical_result.get("status"),
        "all_source_count": all_source_count,
        "cited_source_count": cited_source_count,
        "citation_numbers": canonical_result.get("citation_numbers", []),
        "unresolved_citations": canonical_result.get("unresolved_citations", []),
        "editorial_used": use_editorial,
        "editorial_model": editorial.get("model") if use_editorial else None,
    }

    return {
        "subject": mail_subject,
        "text": text_body,
        "html": html_body,
        "metadata": metadata,
    }


def output_stem(input_path: Path, job_metadata: dict[str, Any], canonical_result: dict[str, Any]) -> str:
    job_id = job_metadata.get("job_id")
    if isinstance(job_id, str) and job_id:
        return f"mail_{job_id}"
    message_id = canonical_result.get("message_id")
    if isinstance(message_id, str) and message_id:
        return f"mail_{message_id}"
    return f"mail_{input_path.stem}"


def write_mail_files(mail: dict[str, Any], output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{stem}.html"
    text_path = output_dir / f"{stem}.txt"
    metadata_path = output_dir / f"{stem}.json"
    html_path.write_text(mail["html"], encoding="utf-8")
    text_path.write_text(mail["text"], encoding="utf-8")
    metadata_path.write_text(json.dumps(mail["metadata"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"html": html_path, "text": text_path, "metadata": metadata_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build HTML/text email content from a Perplexica JSON result.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subject", help="Custom subject text. When provided, it is used as-is.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        canonical_result, job_metadata = load_input_json(args.input_json)
        mail = build_mail(canonical_result, subject=args.subject)
        stem = output_stem(args.input_json, job_metadata, canonical_result)
        paths = write_mail_files(mail, args.output_dir, stem)
        print(f"subject: {mail['subject']}")
        print(f"cited_sources: {mail['metadata']['cited_source_count']}")
        print(f"html: {paths['html']}")
        print(f"text: {paths['text']}")
        print(f"metadata: {paths['metadata']}")
        return 0
    except MailBuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
