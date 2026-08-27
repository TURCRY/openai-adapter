#!/usr/bin/env python3
"""Best-effort editorial rewrite for Perplexica canonical results.

The module sends only the answer and cited sources to an OpenAI-compatible local
endpoint. It never sends the complete all_sources list.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mail_sender import DEFAULT_ENV_FILE, parse_env_file


ENV_EDITORIAL_BASE_URL = "PERPLEXICA_EDITORIAL_BASE_URL"
ENV_EDITORIAL_API_KEY = "PERPLEXICA_EDITORIAL_API_KEY"
ENV_ADAPTER_API_KEY = "ADAPTER_API_KEY"
DEFAULT_MODEL = "local-gemma-4"
DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 1800
DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "prompt_editorial_veille.md"
CITATION_RE = re.compile(r"\[([1-9]\d*)\]")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class EditorialRewriteError(Exception):
    """User-facing editorial rewrite failure."""


@dataclass(frozen=True)
class EditorialConfig:
    base_url: str
    model: str = DEFAULT_MODEL
    prompt_file: Path = DEFAULT_PROMPT_FILE
    api_key: str | None = None
    timeout: int = DEFAULT_TIMEOUT_SECONDS


def extract_citation_numbers(text: str) -> list[int]:
    seen: set[int] = set()
    numbers: list[int] = []
    for match in CITATION_RE.finditer(text or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def citation_numbers_from_result(canonical_result: dict[str, Any]) -> list[int]:
    declared = canonical_result.get("citation_numbers")
    if isinstance(declared, list):
        numbers = [item for item in declared if isinstance(item, int) and item > 0]
        if numbers:
            return list(dict.fromkeys(numbers))
    return extract_citation_numbers(canonical_result.get("answer_markdown", ""))


def prepare_editorial_input(canonical_result: dict[str, Any]) -> dict[str, Any]:
    allowed_numbers = set(citation_numbers_from_result(canonical_result))
    cited_sources = []
    for source in canonical_result.get("cited_sources", []) or []:
        if not isinstance(source, dict):
            continue
        number = source.get("index")
        if not isinstance(number, int) or number not in allowed_numbers:
            continue
        cited_sources.append(
            {
                "number": number,
                "title": str(source.get("title") or ""),
            }
        )
    editorial_response = canonical_result.get("editorial_answer_markdown") or canonical_result.get("answer_markdown") or ""
    payload = {
        "question": canonical_result.get("question") or "",
        "response": editorial_response,
        "citation_numbers": sorted(allowed_numbers),
        "cited_sources": cited_sources,
    }
    searches = canonical_result.get("searches")
    if isinstance(searches, dict):
        payload["searches"] = {
            name: {
                "status": summary.get("status"),
                "source_count": summary.get("source_count"),
                "cited_source_count": summary.get("cited_source_count"),
            }
            for name, summary in searches.items()
            if isinstance(summary, dict) and summary.get("status") == "completed"
        }
    return payload


def read_editorial_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EditorialRewriteError(f"Cannot read editorial prompt: {path}") from exc
    if not prompt.strip():
        raise EditorialRewriteError(f"Editorial prompt is empty: {path}")
    return prompt


def build_messages(system_prompt: str, editorial_input: dict[str, Any]) -> list[dict[str, str]]:
    user_payload = json.dumps(editorial_input, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Réécris la veille à partir de ce JSON source strictement limité :\n" + user_payload,
        },
    ]


def chat_completions_payload(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
        "response_format": {"type": "json_object"},
    }


def extract_response_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content
    for key in ("content", "text", "response"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, str):
            return value
    raise EditorialRewriteError("Editorial endpoint response did not contain text content.")


def call_editorial_llm(config: EditorialConfig, messages: list[dict[str, str]]) -> str:
    if not config.base_url:
        raise EditorialRewriteError(f"Missing editorial base URL. Set {ENV_EDITORIAL_BASE_URL}.")
    if config.timeout <= 0 or config.timeout > MAX_TIMEOUT_SECONDS:
        raise EditorialRewriteError(
            f"Editorial timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds."
        )

    endpoint = config.base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(chat_completions_payload(config.model, messages), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = Request(endpoint, data=body, method="POST", headers=headers)
    try:
        with urlopen(request, timeout=config.timeout) as response:
            status = response.getcode()
            raw_body = response.read()
    except HTTPError as exc:
        raise EditorialRewriteError(f"Editorial HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise EditorialRewriteError(f"Editorial network error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise EditorialRewriteError(f"Editorial timeout after {config.timeout} seconds.") from exc

    if status < 200 or status >= 300:
        raise EditorialRewriteError(f"Editorial unexpected HTTP status: {status}")
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise EditorialRewriteError("Editorial endpoint returned non UTF-8 data.") from exc
    except json.JSONDecodeError as exc:
        raise EditorialRewriteError("Editorial endpoint returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise EditorialRewriteError("Editorial endpoint returned a non-object JSON response.")
    return extract_response_content(payload)


def parse_editorial_output(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise EditorialRewriteError("Editorial output is empty.")
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in JSON_BLOCK_RE.finditer(text))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            last_error = EditorialRewriteError("Editorial JSON must be an object.")
            continue
        title = parsed.get("title")
        body = parsed.get("body_markdown")
        numbers = parsed.get("citation_numbers", [])
        if not isinstance(title, str) or not title.strip():
            raise EditorialRewriteError("Editorial JSON missing string field 'title'.")
        if not isinstance(body, str) or not body.strip():
            raise EditorialRewriteError("Editorial JSON missing string field 'body_markdown'.")
        if not isinstance(numbers, list) or not all(isinstance(item, int) and item > 0 for item in numbers):
            raise EditorialRewriteError("Editorial JSON field 'citation_numbers' must be a list of positive integers.")
        return {"title": title.strip(), "body_markdown": body.strip(), "citation_numbers": list(dict.fromkeys(numbers))}
    raise EditorialRewriteError("Editorial output did not contain valid JSON.") from last_error


def validate_editorial_citations(editorial: dict[str, Any], allowed_numbers: list[int]) -> None:
    allowed = set(allowed_numbers)
    detected = extract_citation_numbers(editorial.get("body_markdown", ""))
    unknown = [number for number in detected if number not in allowed]
    if unknown:
        raise EditorialRewriteError(f"Editorial output contains unknown citations: {unknown}")
    declared_unknown = [number for number in editorial.get("citation_numbers", []) if number not in allowed]
    if declared_unknown:
        raise EditorialRewriteError(f"Editorial JSON declares unknown citations: {declared_unknown}")
    editorial["citation_numbers"] = detected


def build_editorial_result(canonical_result: dict[str, Any], editorial: dict[str, Any], raw_text: str, model: str) -> dict[str, Any]:
    return {
        "status": "completed",
        "model": model,
        "title": editorial["title"],
        "body_markdown": editorial["body_markdown"],
        "citation_numbers": editorial["citation_numbers"],
        "raw_length": len(raw_text),
        "source_chat_id": canonical_result.get("chat_id"),
        "source_message_id": canonical_result.get("message_id"),
    }


def rewrite_editorial(
    canonical_result: dict[str, Any],
    config: EditorialConfig,
    *,
    llm_func=call_editorial_llm,
) -> tuple[dict[str, Any], str]:
    editorial_input = prepare_editorial_input(canonical_result)
    system_prompt = read_editorial_prompt(config.prompt_file)
    messages = build_messages(system_prompt, editorial_input)
    raw_text = llm_func(config, messages)
    editorial = parse_editorial_output(raw_text)
    validate_editorial_citations(editorial, editorial_input["citation_numbers"])
    return build_editorial_result(canonical_result, editorial, raw_text, config.model), raw_text


def env_value(name: str, file_values: dict[str, str]) -> str:
    if name in os.environ:
        return os.environ.get(name, "")
    return file_values.get(name, "")


def editorial_config_from_env(
    model: str | None = None,
    prompt_file: Path | None = None,
    timeout: int | None = None,
    env_file: Path = DEFAULT_ENV_FILE,
) -> EditorialConfig:
    file_values = parse_env_file(env_file)
    base_url = env_value(ENV_EDITORIAL_BASE_URL, file_values).strip()
    api_key = (
        env_value(ENV_EDITORIAL_API_KEY, file_values).strip()
        or env_value(ENV_ADAPTER_API_KEY, file_values).strip()
        or None
    )
    return EditorialConfig(
        base_url=base_url,
        model=model or DEFAULT_MODEL,
        prompt_file=prompt_file or DEFAULT_PROMPT_FILE,
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout,
    )


def load_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EditorialRewriteError(f"Cannot read input result: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EditorialRewriteError(f"Invalid input JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise EditorialRewriteError("Input JSON must be an object.")
    return payload.get("result") if isinstance(payload.get("result"), dict) else payload


def write_editorial_files(output_dir: Path, editorial: dict[str, Any], raw_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "editorial_raw.txt").write_text(raw_text, encoding="utf-8")
    (output_dir / "editorial.json").write_text(
        json.dumps(editorial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the optional editorial rewrite for a Perplexica result.")
    parser.add_argument("--input", type=Path, required=True, help="Canonical Perplexica result JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for editorial.json and editorial_raw.txt.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = load_result(args.input)
        config = editorial_config_from_env(
            model=args.model,
            prompt_file=args.prompt_file,
            timeout=args.timeout,
            env_file=args.env_file,
        )
        editorial, raw_text = rewrite_editorial(result, config)
        write_editorial_files(args.output_dir, editorial, raw_text)
        print(f"status: {editorial['status']}")
        print(f"model: {editorial['model']}")
        print(f"citations: {editorial['citation_numbers']}")
        print(f"editorial_json: {args.output_dir / 'editorial.json'}")
        print(f"editorial_raw: {args.output_dir / 'editorial_raw.txt'}")
        return 0
    except EditorialRewriteError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
