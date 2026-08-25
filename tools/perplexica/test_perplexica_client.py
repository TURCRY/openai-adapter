import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from perplexica_client import (
    PerplexicaClient,
    PerplexicaClientError,
    canonical_message,
    generate_chat_id,
    generate_message_id,
    iter_json_lines,
)


class FakeResponse:
    def __init__(self, status=200, body=b"", chunks=None):
        self.status = status
        self.body = body
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self.body

    def __iter__(self):
        return iter(self.chunks if self.chunks is not None else [self.body])


class PerplexicaClientTests(unittest.TestCase):
    def sample_chat_payload(self, chat_id="chat123", message_id="msg123"):
        return {
            "chat": {
                "id": chat_id,
                "title": "Question",
                "createdAt": "2026-08-26T00:00:00.000Z",
                "sources": ["web"],
                "files": [],
            },
            "messages": [
                {
                    "id": 1,
                    "messageId": message_id,
                    "chatId": chat_id,
                    "backendId": "backend123",
                    "query": "Question",
                    "createdAt": "2026-08-26T00:00:01.000Z",
                    "status": "completed",
                    "responseBlocks": [
                        {"type": "text", "data": "Answer [1]."},
                        {
                            "type": "source",
                            "data": [
                                {
                                    "content": "Source content",
                                    "metadata": {"title": "Source", "url": "https://example.com"},
                                },
                                {
                                    "content": "Uncited",
                                    "metadata": {"title": "Uncited", "url": "https://example.org"},
                                },
                            ],
                        },
                    ],
                }
            ],
        }

    def test_iter_json_lines_handles_fragmented_stream(self):
        chunks = [b'{"type":"blo', b'ck","block":{}}\n{"type":"messageEnd"}\n']
        self.assertEqual(
            list(iter_json_lines(chunks)),
            [{"type": "block", "block": {}}, {"type": "messageEnd"}],
        )

    def test_stream_must_end_with_message_end(self):
        client = PerplexicaClient("http://perplexica")
        with self.assertRaisesRegex(PerplexicaClientError, "messageEnd"):
            client._validate_stream_events([{"type": "researchComplete"}])

    def test_stream_error_event_raises(self):
        client = PerplexicaClient("http://perplexica")
        with self.assertRaisesRegex(PerplexicaClientError, "stream error"):
            client._validate_stream_events([{"type": "error", "data": "boom"}])

    def test_http_error_is_clear(self):
        client = PerplexicaClient("http://perplexica")
        error = HTTPError("http://perplexica/api/providers", 500, "Server Error", {}, None)
        with patch("perplexica_client.urlopen", side_effect=error):
            with self.assertRaisesRegex(PerplexicaClientError, "HTTP error on GET /api/providers: 500"):
                client.get_providers()

    def test_ask_recovers_chat_and_message_ids(self):
        client = PerplexicaClient("http://perplexica")
        events = b'{"type":"researchComplete"}\n{"type":"messageEnd"}\n'
        chat_payload = json.dumps(self.sample_chat_payload()).encode("utf-8")

        with patch.object(client, "_resolve_models", return_value=({"providerId": "p", "key": "c"}, {"providerId": "p", "key": "e"})):
            with patch(
                "perplexica_client.urlopen",
                side_effect=[FakeResponse(chunks=[events]), FakeResponse(body=chat_payload)],
            ):
                result = client.ask("Question", chat_id="chat123", message_id="msg123")

        self.assertEqual(result["chat_id"], "chat123")
        self.assertEqual(result["message_id"], "msg123")
        self.assertEqual(result["answer_markdown"], "Answer [1].")
        self.assertEqual(len(result["all_sources"]), 2)
        self.assertEqual(len(result["cited_sources"]), 1)

    def test_extract_message_selects_requested_message(self):
        result = canonical_message(self.sample_chat_payload(), "msg123")
        self.assertEqual(result["question"], "Question")
        self.assertEqual(result["citation_numbers"], [1])
        self.assertFalse(result["all_sources"][1]["cited"])

    def test_extract_message_reports_missing_message(self):
        with self.assertRaisesRegex(PerplexicaClientError, "messageId not found"):
            canonical_message(self.sample_chat_payload(), "missing")

    def test_chat_id_and_message_id_match_frontend_lengths(self):
        self.assertRegex(generate_chat_id(), r"^[0-9a-f]{40}$")
        self.assertRegex(generate_message_id(), r"^[0-9a-f]{14}$")


if __name__ == "__main__":
    unittest.main()
