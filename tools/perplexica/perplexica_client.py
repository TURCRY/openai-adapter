#!/usr/bin/env python3
"""Generic Perplexica 1.12.1 client.

Implements the frontend protocol used by POST /api/chat and GET /api/chats/<id>.
Only GET and POST requests are performed; no destructive endpoint is used.
"""

from __future__ import annotations

import json
import secrets
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from perplexica_chat_export import ExportError, get_json_headers, post_chat_headers, transform_chat_payload


DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_OPTIMIZATION_MODE = "speed"
DEFAULT_SOURCES = ["web"]
STREAM_END_EVENT = "messageEnd"
STREAM_ERROR_EVENT = "error"


class PerplexicaClientError(ExportError):
    """User-facing Perplexica client failure."""


def generate_chat_id() -> str:
    """Mirror crypto.randomBytes(20).toString('hex') from useChat.tsx."""
    return secrets.token_hex(20)


def generate_message_id() -> str:
    """Mirror crypto.randomBytes(7).toString('hex') from useChat.tsx."""
    return secrets.token_hex(7)


def iter_json_lines(chunks: Iterable[bytes]) -> Iterable[dict[str, Any]]:
    """Parse Perplexica stream chunks as newline-delimited JSON objects."""
    pending = ""
    for chunk in chunks:
        if not chunk:
            continue
        try:
            pending += chunk.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PerplexicaClientError("Invalid stream response: non UTF-8 data.") from exc

        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PerplexicaClientError("Invalid stream response: malformed JSON line.") from exc
            if not isinstance(event, dict):
                raise PerplexicaClientError("Invalid stream response: event is not an object.")
            yield event

    if pending.strip():
        try:
            event = json.loads(pending)
        except json.JSONDecodeError as exc:
            raise PerplexicaClientError("Invalid stream response: trailing malformed JSON.") from exc
        if not isinstance(event, dict):
            raise PerplexicaClientError("Invalid stream response: trailing event is not an object.")
        yield event


def canonical_message(chat_json: dict[str, Any], message_id: str | None = None) -> dict[str, Any]:
    transformed = transform_chat_payload(chat_json)
    messages = transformed["messages"]
    if not messages:
        raise PerplexicaClientError("Chat is incomplete: no messages returned.")

    if message_id:
        selected = next((msg for msg in messages if msg.get("message_id") == message_id), None)
        if selected is None:
            raise PerplexicaClientError(f"Chat is incomplete: messageId not found: {message_id}")
    else:
        selected = messages[-1]

    return {
        "chat_id": selected.get("chat_id") or transformed["chat"].get("id"),
        "message_id": selected.get("message_id"),
        "question": selected.get("question"),
        "answer_markdown": selected.get("answer_markdown"),
        "all_sources": selected.get("all_sources", []),
        "cited_sources": selected.get("cited_sources", []),
        "citation_numbers": selected.get("citation_numbers", []),
        "unresolved_citations": selected.get("unresolved_citations", []),
        "created_at": selected.get("created_at"),
        "status": selected.get("status"),
    }


class PerplexicaClient:
    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        if not base_url:
            raise PerplexicaClientError("Missing Perplexica base URL.")
        if timeout <= 0:
            raise PerplexicaClientError("timeout must be a positive integer.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request_json(self, path: str) -> dict[str, Any]:
        request = Request(self._url(path), method="GET", headers=get_json_headers())
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = response.getcode()
                body = response.read()
        except HTTPError as exc:
            raise PerplexicaClientError(f"HTTP error on GET {path}: {exc.code}") from exc
        except URLError as exc:
            raise PerplexicaClientError(f"Network error on GET {path}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PerplexicaClientError(f"Network timeout on GET {path} after {self.timeout} seconds.") from exc

        if status < 200 or status >= 300:
            raise PerplexicaClientError(f"Unexpected HTTP status on GET {path}: {status}")
        try:
            payload = json.loads(body.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise PerplexicaClientError(f"Invalid JSON response on GET {path}: non UTF-8 data.") from exc
        except json.JSONDecodeError as exc:
            raise PerplexicaClientError(f"Invalid JSON response on GET {path}.") from exc
        if not isinstance(payload, dict):
            raise PerplexicaClientError(f"Invalid JSON response on GET {path}: top-level is not an object.")
        return payload

    def get_providers(self) -> list[dict[str, Any]]:
        payload = self._request_json("/api/providers")
        providers = payload.get("providers")
        if not isinstance(providers, list):
            raise PerplexicaClientError("Invalid /api/providers response: missing providers array.")
        return [provider for provider in providers if isinstance(provider, dict)]

    def _resolve_models(self, options: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
        chat_model = options.get("chat_model")
        embedding_model = options.get("embedding_model")
        if chat_model and embedding_model:
            return self._validate_model(chat_model, "chat_model"), self._validate_model(
                embedding_model, "embedding_model"
            )

        providers = self.get_providers()
        if not chat_model:
            provider = next((p for p in providers if p.get("chatModels")), None)
            if not provider:
                raise PerplexicaClientError("No chat model provider found in /api/providers.")
            chat_model = {
                "providerId": provider.get("id"),
                "key": provider["chatModels"][0].get("key"),
            }
        if not embedding_model:
            provider = next((p for p in providers if p.get("embeddingModels")), None)
            if not provider:
                raise PerplexicaClientError("No embedding model provider found in /api/providers.")
            embedding_model = {
                "providerId": provider.get("id"),
                "key": provider["embeddingModels"][0].get("key"),
            }
        return self._validate_model(chat_model, "chat_model"), self._validate_model(
            embedding_model, "embedding_model"
        )

    @staticmethod
    def _validate_model(model: Any, label: str) -> dict[str, str]:
        if not isinstance(model, dict):
            raise PerplexicaClientError(f"{label} must be an object with providerId and key.")
        provider_id = model.get("providerId")
        key = model.get("key")
        if not isinstance(provider_id, str) or not provider_id:
            raise PerplexicaClientError(f"{label}.providerId is required.")
        if not isinstance(key, str) or not key:
            raise PerplexicaClientError(f"{label}.key is required.")
        return {"providerId": provider_id, "key": key}

    def _build_ask_payload(self, prompt: str, options: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        if not prompt:
            raise PerplexicaClientError("prompt must not be empty.")

        chat_id = options.get("chat_id") or generate_chat_id()
        message_id = options.get("message_id") or generate_message_id()
        chat_model, embedding_model = self._resolve_models(options)

        payload = {
            "content": prompt,
            "message": {
                "messageId": message_id,
                "chatId": chat_id,
                "content": prompt,
            },
            "chatId": chat_id,
            "files": options.get("files", []),
            "sources": options.get("sources", DEFAULT_SOURCES),
            "optimizationMode": options.get("optimization_mode", DEFAULT_OPTIMIZATION_MODE),
            "history": options.get("history", []),
            "chatModel": chat_model,
            "embeddingModel": embedding_model,
            "systemInstructions": options.get("system_instructions"),
        }
        return chat_id, message_id, payload

    def _post_chat_stream(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self._url("/api/chat"),
            data=body,
            method="POST",
            headers=post_chat_headers(),
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = response.getcode()
                if status < 200 or status >= 300:
                    raise PerplexicaClientError(f"Unexpected HTTP status on POST /api/chat: {status}")
                events = list(iter_json_lines(response))
        except HTTPError as exc:
            detail = ""
            try:
                error_payload = json.loads(exc.read().decode("utf-8"))
                detail = f": {error_payload.get('message') or error_payload}"
            except Exception:
                detail = ""
            raise PerplexicaClientError(f"HTTP error on POST /api/chat: {exc.code}{detail}") from exc
        except URLError as exc:
            raise PerplexicaClientError(f"Network error on POST /api/chat: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PerplexicaClientError(
                f"Network timeout on POST /api/chat after {self.timeout} seconds."
            ) from exc

        self._validate_stream_events(events)
        return events

    @staticmethod
    def _validate_stream_events(events: list[dict[str, Any]]) -> None:
        if not events:
            raise PerplexicaClientError("Perplexica stream ended without events.")
        for event in events:
            event_type = event.get("type")
            if event_type == STREAM_ERROR_EVENT:
                raise PerplexicaClientError(f"Perplexica stream error: {event.get('data')}")
            if event_type not in {"block", "updateBlock", "researchComplete", STREAM_END_EVENT}:
                raise PerplexicaClientError(f"Unknown Perplexica stream event: {event_type}")
        if events[-1].get("type") != STREAM_END_EVENT:
            raise PerplexicaClientError("Perplexica stream ended before messageEnd.")

    def ask(self, prompt: str, **options: Any) -> dict[str, Any]:
        chat_id, message_id, payload = self._build_ask_payload(prompt, options)
        self._post_chat_stream(payload)
        chat_json = self.get_chat(chat_id)
        result = self.extract_message(chat_json, message_id)
        if result.get("status") == "error":
            raise PerplexicaClientError(f"Perplexica message ended with status=error: {message_id}")
        if result.get("status") != "completed":
            raise PerplexicaClientError(
                f"Chat is incomplete: message status is {result.get('status')!r}."
            )
        return result

    def get_chat(self, chat_id: str) -> dict[str, Any]:
        if not chat_id:
            raise PerplexicaClientError("chat_id must not be empty.")
        return self._request_json(f"/api/chats/{quote(chat_id, safe='')}")

    def extract_message(self, chat_json: dict[str, Any], message_id: str | None = None) -> dict[str, Any]:
        return canonical_message(chat_json, message_id)


__all__ = [
    "DEFAULT_OPTIMIZATION_MODE",
    "DEFAULT_SOURCES",
    "DEFAULT_TIMEOUT_SECONDS",
    "PerplexicaClient",
    "PerplexicaClientError",
    "canonical_message",
    "generate_chat_id",
    "generate_message_id",
    "iter_json_lines",
]

