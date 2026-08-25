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


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "mail"
INLINE_RE = re.compile(
    r"\[([^\]]+)\]\(([^)]+)\)|"
    r"\[([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]|"
    r"\*\*([^*]+)\*\*|"
    r"\*([^*]+)\*"
)
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")


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
        if isinstance(url, str) and url:
            rendered.append(
                '<a class="citation" href="{}">{}</a>'.format(
                    html.escape(url, quote=True), html.escape(label)
                )
            )
        else:
            rendered.append(html.escape(label))
    return "".join(rendered)


def render_inline_html(text: str, sources_by_index: dict[int, dict[str, Any]]) -> str:
    output: list[str] = []
    pos = 0
    for match in INLINE_RE.finditer(text):
        output.append(html.escape(text[pos : match.start()]))
        link_text, link_url, citation_numbers, bold_text, italic_text = match.groups()
        if link_text is not None and link_url is not None:
            output.append(
                '<a href="{}">{}</a>'.format(
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
            parts.append(f"<p>{render_inline_html(text, sources_by_index)}</p>")
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
            parts.append(f"<h{level}>{render_inline_html(heading.group(2), sources_by_index)}</h{level}>")
            continue

        bullet = BULLET_RE.match(line)
        if bullet:
            flush_paragraph()
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{render_inline_html(bullet.group(1), sources_by_index)}</li>")
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
    return f"[Perplexica] {clean}"


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


def build_mail(canonical_result: dict[str, Any], subject: str | None = None) -> dict[str, Any]:
    validate_canonical_result(canonical_result)
    question = canonical_result["question"]
    answer_markdown = canonical_result["answer_markdown"]
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
        title = source.get("title") or "Source sans titre"
        url = source.get("url") or ""
        html_sources.append(
            "<li><strong>[{}] {}</strong><br><a href=\"{}\">{}</a></li>".format(
                html.escape(str(index)),
                html.escape(str(title)),
                html.escape(str(url), quote=True),
                html.escape(str(url)),
            )
        )
        text_sources.append(f"[{index}] {title}\n{url}".rstrip())

    all_source_count = len(canonical_result.get("all_sources", []) or [])
    cited_source_count = len(cited)
    footer = f"Sources consultées : {all_source_count} — Sources citées : {cited_source_count}"

    html_body = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>{html.escape(mail_subject)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.55; color: #202124; }}
    main {{ max-width: 760px; margin: 0 auto; }}
    h1, h2, h3 {{ line-height: 1.25; }}
    blockquote {{ border-left: 3px solid #d0d7de; margin-left: 0; padding-left: 1rem; color: #57606a; }}
    .citation {{ font-size: 0.9em; text-decoration: none; margin-left: 0.12rem; }}
    .meta {{ color: #57606a; font-size: 0.95em; }}
    .sources li {{ margin-bottom: 0.75rem; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(mail_subject)}</h1>
  <p class="meta">Date de génération : {html.escape(generated_at)}</p>
  <h2>Question</h2>
  <p>{html.escape(question)}</p>
  <h2>Réponse</h2>
  {answer_html}
  <h2>Sources citées</h2>
  <ol class="sources">
    {' '.join(html_sources)}
  </ol>
  <p class="meta">{html.escape(footer)}</p>
</main>
</body>
</html>
"""

    text_body = "\n\n".join(
        [
            mail_subject,
            f"Date de génération : {generated_at}",
            "Question\n" + question,
            "Réponse\n" + answer_text,
            "Sources citées\n" + ("\n\n".join(text_sources) if text_sources else "Aucune source citée."),
            footer,
        ]
    )

    metadata = {
        "subject": mail_subject,
        "generated_at": generated_at,
        "chat_id": canonical_result.get("chat_id"),
        "message_id": canonical_result.get("message_id"),
        "question": question,
        "status": canonical_result.get("status"),
        "all_source_count": all_source_count,
        "cited_source_count": cited_source_count,
        "citation_numbers": canonical_result.get("citation_numbers", []),
        "unresolved_citations": canonical_result.get("unresolved_citations", []),
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
    parser.add_argument("--subject", help="Custom subject text without the [Perplexica] prefix.")
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
