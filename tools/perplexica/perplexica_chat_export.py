#!/usr/bin/env python3
"""Read-only Perplexica chat exporter.

Fetches GET /api/chats/<chatId> and converts the saved Library payload into a
mail-friendly JSON structure without modifying Perplexica.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SECONDS = 15
ENV_BASE_URL = "PERPLEXICA_URL"
HTTP_USER_AGENT = "Mozilla/5.0 PerplexicaAutomation/1.0"
GET_JSON_HEADERS = {
    "User-Agent": HTTP_USER_AGENT,
    "Accept": "application/json",
}
POST_CHAT_HEADERS = {
    "User-Agent": HTTP_USER_AGENT,
    "Content-Type": "application/json",
    "Accept": "text/event-stream, application/json",
}
CITATION_RE = re.compile(r"\[([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]")


class ExportError(Exception):
    """User-facing export failure."""


def get_json_headers() -> dict[str, str]:
    return dict(GET_JSON_HEADERS)


def post_chat_headers() -> dict[str, str]:
    return dict(POST_CHAT_HEADERS)


def ordered_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_citation_numbers(answer_markdown: str) -> list[int]:
    """Extract positive numeric citations from Markdown answer text."""
    numbers: list[int] = []
    for match in CITATION_RE.finditer(answer_markdown or ""):
        for part in match.group(1).split(","):
            try:
                number = int(part.strip())
            except ValueError:
                continue
            if number > 0:
                numbers.append(number)
    return numbers


def normalize_source(source: Any, index: int, citation_counts: Counter[int]) -> dict[str, Any]:
    metadata = source.get("metadata", {}) if isinstance(source, dict) else {}
    return {
        "index": index,
        "title": metadata.get("title") if isinstance(metadata, dict) else None,
        "url": metadata.get("url") if isinstance(metadata, dict) else None,
        "content": source.get("content") if isinstance(source, dict) else None,
        "cited": citation_counts[index] > 0,
        "citation_count": citation_counts[index],
    }


def extract_text_blocks(response_blocks: list[Any]) -> str:
    text_parts = [
        block.get("data", "")
        for block in response_blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n\n".join(str(part) for part in text_parts if part is not None)


def extract_source_items(response_blocks: list[Any]) -> list[Any]:
    sources: list[Any] = []
    for block in response_blocks:
        if not isinstance(block, dict) or block.get("type") != "source":
            continue
        data = block.get("data", [])
        if isinstance(data, list):
            sources.extend(data)
    return sources


def transform_message(message: dict[str, Any]) -> dict[str, Any]:
    response_blocks = message.get("responseBlocks", [])
    if not isinstance(response_blocks, list):
        response_blocks = []

    answer_markdown = extract_text_blocks(response_blocks)
    raw_sources = extract_source_items(response_blocks)
    citation_numbers_all = extract_citation_numbers(answer_markdown)
    citation_counts: Counter[int] = Counter(citation_numbers_all)

    all_sources = [
        normalize_source(source, index, citation_counts)
        for index, source in enumerate(raw_sources, start=1)
    ]

    cited_sources = [
        {
            "index": source["index"],
            "title": source["title"],
            "url": source["url"],
            "citation_count": source["citation_count"],
        }
        for source in all_sources
        if source["cited"]
    ]

    unresolved = [
        {"number": number, "citation_count": count}
        for number, count in sorted(citation_counts.items())
        if number > len(all_sources)
    ]

    return {
        "id": message.get("id"),
        "message_id": message.get("messageId"),
        "chat_id": message.get("chatId"),
        "backend_id": message.get("backendId"),
        "question": message.get("query"),
        "answer_markdown": answer_markdown,
        "created_at": message.get("createdAt"),
        "status": message.get("status"),
        "all_sources": all_sources,
        "cited_sources": cited_sources,
        "citation_numbers": ordered_unique(citation_numbers_all),
        "unresolved_citations": unresolved,
    }


def transform_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    chat = payload.get("chat")
    messages = payload.get("messages")
    if not isinstance(chat, dict):
        raise ExportError("Invalid API response: missing object 'chat'.")
    if not isinstance(messages, list):
        raise ExportError("Invalid API response: missing array 'messages'.")

    return {
        "chat": {
            "id": chat.get("id"),
            "title": chat.get("title"),
            "created_at": chat.get("createdAt"),
            "sources": chat.get("sources") if isinstance(chat.get("sources"), list) else [],
            "files": chat.get("files") if isinstance(chat.get("files"), list) else [],
        },
        "messages": [
            transform_message(message)
            for message in messages
            if isinstance(message, dict)
        ],
    }


def resolve_base_url(cli_base_url: str | None) -> str:
    base_url = cli_base_url or os.environ.get(ENV_BASE_URL)
    if not base_url:
        raise ExportError(
            f"Missing Perplexica base URL. Use --base-url or set {ENV_BASE_URL}."
        )
    return base_url.rstrip("/")


def fetch_chat_payload(base_url: str, chat_id: str, timeout: int) -> dict[str, Any]:
    url = f"{base_url}/api/chats/{quote(chat_id, safe='')}"
    request = Request(url, method="GET", headers=get_json_headers())
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            body = response.read()
    except HTTPError as exc:
        raise ExportError(f"HTTP error while fetching chat: {exc.code}") from exc
    except URLError as exc:
        raise ExportError(f"Network error while fetching chat: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ExportError(f"Network timeout after {timeout} seconds.") from exc

    if status < 200 or status >= 300:
        raise ExportError(f"Unexpected HTTP status while fetching chat: {status}")

    try:
        payload = json.loads(body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ExportError("Invalid API response: response is not UTF-8.") from exc
    except json.JSONDecodeError as exc:
        raise ExportError("Invalid API response: response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ExportError("Invalid API response: top-level JSON is not an object.")
    return payload


def default_output_path(chat_id: str) -> Path:
    safe_chat_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", chat_id).strip("._")
    safe_chat_id = safe_chat_id or "chat"
    return Path.cwd() / f"perplexica_chat_{safe_chat_id}.json"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Perplexica Library chat through its read-only API."
    )
    parser.add_argument("chat_id", help="Perplexica chat id")
    parser.add_argument("--base-url", help=f"Perplexica base URL. Overrides {ENV_BASE_URL}.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file. Defaults to perplexica_chat_<chatId>.json in cwd.",
    )
    parser.add_argument("--stdout", action="store_true", help="Also print JSON to stdout.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Network timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ExportError("--timeout must be a positive integer.")
        base_url = resolve_base_url(args.base_url)
        payload = fetch_chat_payload(base_url, args.chat_id, args.timeout)
        exported = transform_chat_payload(payload)
        output_path = args.output or default_output_path(args.chat_id)
        write_json(output_path, exported)
        if args.stdout:
            print(json.dumps(exported, ensure_ascii=False, indent=2))
        print(f"Export written to: {output_path}", file=sys.stderr)
        return 0
    except ExportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
