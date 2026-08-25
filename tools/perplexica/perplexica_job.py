#!/usr/bin/env python3
"""Minimal Perplexica job runner.

Creates one Perplexica chat from a prompt and stores the canonical JSON result.
No cron, no mail sending, no domain-specific prompt handling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from perplexica_chat_export import ENV_BASE_URL, write_json
from perplexica_client import (
    DEFAULT_OPTIMIZATION_MODE,
    DEFAULT_SOURCES,
    DEFAULT_TIMEOUT_SECONDS,
    PerplexicaClient,
    PerplexicaClientError,
)


TOOLS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TOOLS_DIR / "output"


def resolve_base_url(cli_base_url: str | None) -> str:
    base_url = cli_base_url or os.environ.get(ENV_BASE_URL)
    if not base_url:
        raise PerplexicaClientError(
            f"Missing Perplexica base URL. Use --base-url or set {ENV_BASE_URL}."
        )
    return base_url.rstrip("/")


def read_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if bool(prompt) == bool(prompt_file):
        raise PerplexicaClientError("Use exactly one of --prompt or --prompt-file.")
    if prompt_file:
        try:
            return prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise PerplexicaClientError(f"Cannot read prompt file: {prompt_file}") from exc
    return prompt or ""


def parse_sources(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_SOURCES)
    sources = [item.strip() for item in value.split(",") if item.strip()]
    if not sources:
        raise PerplexicaClientError("--sources must contain at least one source.")
    return sources


def build_job_result(job_id: str, prompt: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "result": result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Perplexica prompt and save JSON output.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text to send to Perplexica.")
    prompt_group.add_argument("--prompt-file", type=Path, help="UTF-8 text file containing the prompt.")
    parser.add_argument("--base-url", help=f"Perplexica base URL. Overrides {ENV_BASE_URL}.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--optimization-mode",
        default=DEFAULT_OPTIMIZATION_MODE,
        choices=["speed", "balanced", "quality"],
    )
    parser.add_argument("--sources", help="Comma-separated sources. Default: web.")
    parser.add_argument("--system-instructions", help="Optional Perplexica system instructions.")
    parser.add_argument("--chat-model-provider-id", help="Override chat model providerId.")
    parser.add_argument("--chat-model-key", help="Override chat model key.")
    parser.add_argument("--embedding-model-provider-id", help="Override embedding model providerId.")
    parser.add_argument("--embedding-model-key", help="Override embedding model key.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prompt = read_prompt(args.prompt, args.prompt_file).strip()
        if not prompt:
            raise PerplexicaClientError("Prompt is empty.")
        if args.timeout <= 0:
            raise PerplexicaClientError("--timeout must be a positive integer.")

        base_url = resolve_base_url(args.base_url)
        options: dict[str, Any] = {
            "optimization_mode": args.optimization_mode,
            "sources": parse_sources(args.sources),
            "system_instructions": args.system_instructions,
        }

        if args.chat_model_provider_id or args.chat_model_key:
            if not args.chat_model_provider_id or not args.chat_model_key:
                raise PerplexicaClientError(
                    "Both --chat-model-provider-id and --chat-model-key are required together."
                )
            options["chat_model"] = {
                "providerId": args.chat_model_provider_id,
                "key": args.chat_model_key,
            }

        if args.embedding_model_provider_id or args.embedding_model_key:
            if not args.embedding_model_provider_id or not args.embedding_model_key:
                raise PerplexicaClientError(
                    "Both --embedding-model-provider-id and --embedding-model-key are required together."
                )
            options["embedding_model"] = {
                "providerId": args.embedding_model_provider_id,
                "key": args.embedding_model_key,
            }

        client = PerplexicaClient(base_url, timeout=args.timeout)
        result = client.ask(prompt, **options)
        job_id = uuid.uuid4().hex
        output_path = OUTPUT_DIR / f"perplexica_job_{job_id}.json"
        write_json(output_path, build_job_result(job_id, prompt, result))

        print(f"chat_id: {result.get('chat_id')}")
        print(f"message_id: {result.get('message_id')}")
        print(f"sources: {len(result.get('all_sources', []))}")
        print(f"cited_sources: {len(result.get('cited_sources', []))}")
        print(f"json: {output_path}")
        return 0
    except PerplexicaClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
