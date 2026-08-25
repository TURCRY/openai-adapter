import json
import tempfile
import unittest
from pathlib import Path

from mail_builder import MailBuildError, build_mail, load_input_json, markdown_to_html


class MailBuilderTests(unittest.TestCase):
    def canonical(self):
        return {
            "chat_id": "chat123",
            "message_id": "msg123",
            "question": "Quelle est la capitale de la France ?",
            "answer_markdown": "Paris est la capitale [1].",
            "all_sources": [
                {
                    "index": 1,
                    "title": "Paris — Wikipédia",
                    "url": "https://example.com/paris",
                    "content": "full content excluded",
                    "cited": True,
                    "citation_count": 1,
                },
                {
                    "index": 2,
                    "title": "Source non citée",
                    "url": "https://example.com/uncited",
                    "content": "must not appear",
                    "cited": False,
                    "citation_count": 0,
                },
            ],
            "cited_sources": [
                {"index": 1, "title": "Paris — Wikipédia", "url": "https://example.com/paris", "citation_count": 1}
            ],
            "citation_numbers": [1],
            "unresolved_citations": [],
            "created_at": "2026-08-26T00:00:00Z",
            "status": "completed",
        }

    def test_answer_simple(self):
        mail = build_mail(self.canonical())
        self.assertIn("Paris est la capitale", mail["text"])
        self.assertIn("Paris est la capitale", mail["html"])

    def test_markdown_headings_lists_bold_italic_and_links(self):
        result = self.canonical()
        result["answer_markdown"] = "# Titre\n\n## Section\n\n- **Fort**\n- *Italique*\n- [Lien](https://example.com)"
        html = build_mail(result)["html"]
        self.assertIn("<h1>Titre</h1>", html)
        self.assertIn("<h2>Section</h2>", html)
        self.assertIn("<ul>", html)
        self.assertIn("<strong>Fort</strong>", html)
        self.assertIn("<em>Italique</em>", html)
        self.assertIn('<a href="https://example.com">Lien</a>', html)

    def test_citations_adjacent_preserve_real_indices(self):
        result = self.canonical()
        result["answer_markdown"] = "Sources [1][42][59]."
        result["all_sources"] = [
            {"index": 1, "title": "A", "url": "https://a"},
            {"index": 42, "title": "B", "url": "https://b"},
            {"index": 59, "title": "C", "url": "https://c"},
        ]
        result["cited_sources"] = [
            {"index": 1, "title": "A", "url": "https://a", "citation_count": 1},
            {"index": 42, "title": "B", "url": "https://b", "citation_count": 1},
            {"index": 59, "title": "C", "url": "https://c", "citation_count": 1},
        ]
        html = build_mail(result)["html"]
        self.assertIn(">[1]</a>", html)
        self.assertIn(">[42]</a>", html)
        self.assertIn(">[59]</a>", html)
        self.assertIn("[42] B", html)

    def test_citations_comma_format(self):
        result = self.canonical()
        result["answer_markdown"] = "Sources croisées [1, 3]."
        result["all_sources"] = [
            {"index": 1, "title": "A", "url": "https://a"},
            {"index": 3, "title": "C", "url": "https://c"},
        ]
        html = build_mail(result)["html"]
        self.assertIn('href="https://a"', html)
        self.assertIn('href="https://c"', html)

    def test_cited_source_is_rendered(self):
        mail = build_mail(self.canonical())
        self.assertIn("[1] Paris — Wikipédia", mail["text"])
        self.assertIn("https://example.com/paris", mail["html"])

    def test_uncited_source_is_excluded_from_mail_sections(self):
        mail = build_mail(self.canonical())
        self.assertNotIn("Source non citée", mail["text"])
        self.assertNotIn("https://example.com/uncited", mail["html"])
        self.assertNotIn("must not appear", mail["html"])

    def test_unresolved_citation_is_plain_text(self):
        result = self.canonical()
        result["answer_markdown"] = "Citation absente [99]."
        result["citation_numbers"] = [99]
        result["unresolved_citations"] = [{"number": 99, "citation_count": 1}]
        html = build_mail(result)["html"]
        self.assertIn("[99]", html)
        self.assertNotIn('href="[99]"', html)

    def test_french_utf8_characters(self):
        result = self.canonical()
        result["answer_markdown"] = "Réponse à propos d’Évreux et de l’été [1]."
        mail = build_mail(result)
        self.assertIn("Évreux", mail["text"])
        self.assertIn("été", mail["html"])

    def test_html_is_escaped(self):
        result = self.canonical()
        result["question"] = "<script>alert(1)</script>"
        result["answer_markdown"] = "**<b>attaque</b>** [1]"
        result["cited_sources"][0]["title"] = "<img src=x onerror=alert(1)>"
        html = build_mail(result)["html"]
        self.assertNotIn("<script>alert", html)
        self.assertNotIn("<b>attaque</b>", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;b&gt;attaque&lt;/b&gt;", html)

    def test_invalid_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")
            with self.assertRaises(MailBuildError):
                load_input_json(path)

    def test_status_must_be_completed(self):
        result = self.canonical()
        result["status"] = "answering"
        with self.assertRaisesRegex(MailBuildError, "status"):
            build_mail(result)

    def test_job_payload_is_unwrapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.json"
            payload = {"job_id": "job123", "created_at": "now", "result": self.canonical()}
            path.write_text(json.dumps(payload), encoding="utf-8")
            canonical, metadata = load_input_json(path)
            self.assertEqual(canonical["message_id"], "msg123")
            self.assertEqual(metadata["job_id"], "job123")


if __name__ == "__main__":
    unittest.main()
