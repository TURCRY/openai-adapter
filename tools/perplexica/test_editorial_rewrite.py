import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from editorial_rewrite import (
    DEFAULT_TIMEOUT_SECONDS,
    ENV_ADAPTER_API_KEY,
    ENV_EDITORIAL_API_KEY,
    ENV_EDITORIAL_BASE_URL,
    MAX_TIMEOUT_SECONDS,
    EditorialConfig,
    EditorialRewriteError,
    build_messages,
    call_editorial_llm,
    editorial_config_from_env,
    extract_citation_numbers,
    parse_editorial_output,
    prepare_editorial_input,
    rewrite_editorial,
    validate_editorial_citations,
)


def canonical_result():
    return {
        "chat_id": "chat123",
        "message_id": "msg123",
        "question": "Sujet ?",
        "answer_markdown": "Texte Perplexica [3][1]. Invitation \u00e0 poursuivre.",
        "all_sources": [
            {"index": 1, "title": "Cit\u00e9e 1", "url": "https://example.com/1", "content": "secret long content"},
            {"index": 2, "title": "Non cit\u00e9e", "url": "https://example.com/2", "content": "must not be sent"},
            {"index": 3, "title": "Cit\u00e9e 3", "url": "https://example.com/3"},
        ],
        "cited_sources": [
            {"index": 3, "title": "Cit\u00e9e 3", "url": "https://example.com/3"},
            {"index": 1, "title": "Cit\u00e9e 1", "url": "https://example.com/1"},
        ],
        "citation_numbers": [3, 1],
        "status": "completed",
    }


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self.body


class EditorialRewriteTests(unittest.TestCase):
    def clear_editorial_env(self):
        old = {
            ENV_EDITORIAL_BASE_URL: os.environ.pop(ENV_EDITORIAL_BASE_URL, None),
            ENV_EDITORIAL_API_KEY: os.environ.pop(ENV_EDITORIAL_API_KEY, None),
            ENV_ADAPTER_API_KEY: os.environ.pop(ENV_ADAPTER_API_KEY, None),
        }
        return old

    def restore_env(self, old):
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_extract_citation_numbers_preserves_first_seen_order(self):
        self.assertEqual(extract_citation_numbers("A [3][1][3] [42]"), [3, 1, 42])

    def test_default_timeout_is_300(self):
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, 300)
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(f"{ENV_EDITORIAL_BASE_URL}=http://adapter.local\n", encoding="utf-8")
            old = self.clear_editorial_env()
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.timeout, 300)

    def test_prepare_input_sends_only_cited_sources(self):
        result = canonical_result()
        payload = prepare_editorial_input(result)
        self.assertEqual(payload["citation_numbers"], [1, 3])
        self.assertEqual(
            payload["cited_sources"],
            [
                {"number": 3, "title": "Cit\u00e9e 3"},
                {"number": 1, "title": "Cit\u00e9e 1"},
            ],
        )
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("all_sources", raw)
        self.assertNotIn("Non cit\u00e9e", raw)
        self.assertNotIn("must not be sent", raw)
        self.assertNotIn("https://example.com/1", raw)
        self.assertNotIn("https://example.com/3", raw)
        self.assertEqual(result["cited_sources"][0]["url"], "https://example.com/3")

    def test_prepare_aggregated_input_uses_editorial_answer_and_completed_searches_only(self):
        result = canonical_result()
        result["answer_markdown"] = "## A\n\nTexte brut avec empty."
        result["editorial_answer_markdown"] = "## A\n\nTexte utile [1]."
        result["searches"] = {
            "a": {"status": "completed", "source_count": 2, "cited_source_count": 1},
            "b": {"status": "empty", "source_count": 0, "cited_source_count": 0},
            "c": {"status": "failed", "source_count": 0, "cited_source_count": 0},
        }
        payload = prepare_editorial_input(result)
        self.assertEqual(payload["response"], "## A\n\nTexte utile [1].")
        self.assertEqual(list(payload["searches"].keys()), ["a"])
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("all_sources", raw)
        self.assertNotIn("empty", raw)

    def test_build_messages_contains_prompt_and_limited_payload(self):
        payload = prepare_editorial_input(canonical_result())
        messages = build_messages("SYSTEM", payload)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("SYSTEM", messages[0]["content"])
        self.assertIn("cited_sources", messages[1]["content"])
        self.assertNotIn("all_sources", messages[1]["content"])

    def test_parse_strict_json_output(self):
        parsed = parse_editorial_output('{"title":"Titre","body_markdown":"Corps [1]","citation_numbers":[1]}')
        self.assertEqual(parsed["title"], "Titre")
        self.assertEqual(parsed["citation_numbers"], [1])

    def test_parse_json_inside_text_or_fence(self):
        raw = 'Voici:\n```json\n{"title":"Titre","body_markdown":"Corps [3]","citation_numbers":[3]}\n```'
        parsed = parse_editorial_output(raw)
        self.assertEqual(parsed["body_markdown"], "Corps [3]")

    def test_invalid_json_output_raises(self):
        with self.assertRaises(EditorialRewriteError):
            parse_editorial_output("pas du json")

    def test_citations_conserved(self):
        editorial = {"title": "Titre", "body_markdown": "Texte [3][1]", "citation_numbers": [3, 1]}
        validate_editorial_citations(editorial, [1, 3])
        self.assertEqual(editorial["citation_numbers"], [3, 1])

    def test_unknown_citation_raises(self):
        editorial = {"title": "Titre", "body_markdown": "Texte [99]", "citation_numbers": [99]}
        with self.assertRaises(EditorialRewriteError):
            validate_editorial_citations(editorial, [1, 3])

    def test_rewrite_editorial_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text("Prompt \u00e9ditorial", encoding="utf-8")
            config = EditorialConfig(base_url="http://local", model="local-gemma-4", prompt_file=prompt)
            raw = '{"title":"Note","body_markdown":"Synth\u00e8se [3]","citation_numbers":[3]}'
            editorial, raw_text = rewrite_editorial(canonical_result(), config, llm_func=lambda cfg, msgs: raw)
            self.assertEqual(editorial["status"], "completed")
            self.assertEqual(editorial["model"], "local-gemma-4")
            self.assertEqual(editorial["citation_numbers"], [3])
            self.assertEqual(raw_text, raw)

    def test_call_editorial_llm_uses_openai_compatible_endpoint(self):
        captured = {}
        response_body = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(response_body)

        config = EditorialConfig(
            base_url="http://adapter.local",
            model="local-gemma-4",
            prompt_file=Path("prompt"),
            api_key="secret",
            timeout=12,
        )
        with patch("editorial_rewrite.urlopen", fake_urlopen):
            content = call_editorial_llm(config, [{"role": "user", "content": "Bonjour"}])
        self.assertEqual(content, "{}")
        self.assertEqual(captured["url"], "http://adapter.local/v1/chat/completions")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["body"]["model"], "local-gemma-4")
        self.assertFalse(captured["body"]["stream"])
        self.assertNotIn("max_tokens", captured["body"])
        self.assertIn("Authorization", captured["headers"])

    def test_call_editorial_llm_http_error(self):
        def fake_urlopen(request, timeout):
            raise HTTPError(request.full_url, 500, "boom", {}, None)

        config = EditorialConfig(base_url="http://adapter.local", prompt_file=Path("prompt"))
        with patch("editorial_rewrite.urlopen", fake_urlopen):
            with self.assertRaises(EditorialRewriteError):
                call_editorial_llm(config, [])

    def test_call_editorial_llm_network_error(self):
        def fake_urlopen(request, timeout):
            raise URLError("offline")

        config = EditorialConfig(base_url="http://adapter.local", prompt_file=Path("prompt"))
        with patch("editorial_rewrite.urlopen", fake_urlopen):
            with self.assertRaises(EditorialRewriteError):
                call_editorial_llm(config, [])

    def test_invalid_timeout_is_rejected_before_http_call(self):
        for timeout in (0, -1, MAX_TIMEOUT_SECONDS + 1):
            config = EditorialConfig(base_url="http://adapter.local", prompt_file=Path("prompt"), timeout=timeout)
            with self.subTest(timeout=timeout):
                with patch("editorial_rewrite.urlopen") as fake_urlopen:
                    with self.assertRaises(EditorialRewriteError):
                        call_editorial_llm(config, [])
                fake_urlopen.assert_not_called()

    def test_editorial_config_reads_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                f"{ENV_EDITORIAL_BASE_URL}=http://adapter.local\n"
                f"{ENV_EDITORIAL_API_KEY}='secret-from-file'\n",
                encoding="utf-8",
            )
            old = self.clear_editorial_env()
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.base_url, "http://adapter.local")
        self.assertEqual(config.api_key, "secret-from-file")

    def test_system_env_has_priority_over_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(f"{ENV_EDITORIAL_BASE_URL}=http://from-file\n", encoding="utf-8")
            old = self.clear_editorial_env()
            os.environ[ENV_EDITORIAL_BASE_URL] = "http://from-env"
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.base_url, "http://from-env")

    def test_adapter_api_key_is_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                f"{ENV_EDITORIAL_BASE_URL}=http://adapter.local\n"
                f"{ENV_ADAPTER_API_KEY}='adapter-secret'\n",
                encoding="utf-8",
            )
            old = self.clear_editorial_env()
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.api_key, "adapter-secret")

    def test_editorial_api_key_has_priority_over_adapter_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                f"{ENV_EDITORIAL_BASE_URL}=http://adapter.local\n"
                f"{ENV_EDITORIAL_API_KEY}=editorial-secret\n"
                f"{ENV_ADAPTER_API_KEY}=adapter-secret\n",
                encoding="utf-8",
            )
            old = self.clear_editorial_env()
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.api_key, "editorial-secret")


if __name__ == "__main__":
    unittest.main()