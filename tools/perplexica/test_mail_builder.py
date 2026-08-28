import json
import re
import tempfile
import unittest
from pathlib import Path

from mail_builder import MailBuildError, build_mail, display_url, load_input_json, markdown_to_html


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
                    "title": "Paris - Wikipedia",
                    "url": "https://example.com/paris",
                    "content": "full content excluded",
                    "cited": True,
                    "citation_count": 1,
                },
                {
                    "index": 2,
                    "title": "Source non citee",
                    "url": "https://example.com/uncited",
                    "content": "must not appear",
                    "cited": False,
                    "citation_count": 0,
                },
            ],
            "cited_sources": [
                {"index": 1, "title": "Paris - Wikipedia", "url": "https://example.com/paris", "citation_count": 1}
            ],
            "citation_numbers": [1],
            "unresolved_citations": [],
            "created_at": "2026-08-26T00:00:00Z",
            "status": "completed",
        }

    def test_default_subject_is_watch_note(self):
        mail = build_mail(self.canonical())
        self.assertEqual(mail["subject"], "Veille Perplexica — Quelle est la capitale de la France ?")

    def test_custom_subject_is_used_as_is(self):
        mail = build_mail(self.canonical(), subject="Sujet personnalise")
        self.assertEqual(mail["subject"], "Sujet personnalise")

    def test_display_title_overrides_raw_visible_title(self):
        result = self.canonical()
        result["question"] = "Prompt complet ligne 1.\n\nConsignes longues internes."
        title = "Veille hebdomadaire \u2014 Expertise de justice et m\u00e9diation"
        mail = build_mail(result, subject="Objet SMTP", display_title=title)
        self.assertIn(title, mail["text"])
        self.assertIn(title, mail["html"])
        self.assertNotIn("QUESTION\nPrompt complet ligne 1.", mail["text"])
        self.assertEqual(mail["metadata"]["question"], result["question"])
        self.assertEqual(result["question"], "Prompt complet ligne 1.\n\nConsignes longues internes.")

    def test_without_display_title_keeps_historical_visible_question(self):
        mail = build_mail(self.canonical())
        self.assertIn("QUESTION\nQuelle est la capitale de la France ?", mail["text"])
        self.assertIn("Quelle est la capitale de la France ?", mail["html"])
        self.assertIsNone(mail["metadata"]["display_title"])

    def test_editorial_title_has_priority_over_display_title(self):
        editorial = {
            "status": "completed",
            "title": "Titre \u00e9ditorial prioritaire",
            "body_markdown": "Synth\u00e8se [1].",
            "model": "local-gemma-4",
        }
        mail = build_mail(self.canonical(), editorial=editorial, display_title="Titre job")
        self.assertIn("QUESTION\nTitre \u00e9ditorial prioritaire", mail["text"])
        self.assertIn("Titre \u00e9ditorial prioritaire", mail["html"])
        self.assertNotIn("QUESTION\nTitre job", mail["text"])
        self.assertEqual(mail["metadata"]["display_title"], "Titre job")
        self.assertEqual(mail["metadata"]["visible_title"], "Titre \u00e9ditorial prioritaire")

    def test_editorial_empty_title_falls_back_to_display_title(self):
        editorial = {
            "status": "completed",
            "title": "  ",
            "body_markdown": "Synth\u00e8se [1].",
            "model": "local-gemma-4",
        }
        title = "Veille hebdomadaire \u2014 Expertise de justice et m\u00e9diation"
        mail = build_mail(self.canonical(), editorial=editorial, display_title=title)
        self.assertIn("QUESTION\n" + title, mail["text"])
        self.assertIn(title, mail["html"])
        self.assertTrue(mail["metadata"]["editorial_used"])

    def test_display_title_utf8_is_preserved(self):
        title = "Veille hebdomadaire \u2014 Expertise de justice et m\u00e9diation, \u0153uvre et l\u2019\u00e9t\u00e9"
        mail = build_mail(self.canonical(), display_title=title)
        self.assertEqual(mail["metadata"]["display_title"], title)
        self.assertIn("\u0153uvre", mail["text"])
        self.assertIn("l\u2019\u00e9t\u00e9", mail["html"])

    def test_answer_simple(self):
        mail = build_mail(self.canonical())
        self.assertIn("Paris est la capitale", mail["text"])
        self.assertIn("Paris est la capitale", mail["html"])

    def test_html_structure_is_professional_email_layout(self):
        html = build_mail(self.canonical())["html"]
        self.assertIn("Veille Perplexica", html)
        self.assertIn("Date de génération", html)
        self.assertIn("Sources principales", html)
        self.assertIn('role="presentation"', html)
        self.assertIn("width:600px", html)
        self.assertIn("background:#f3f5f7", html)
        self.assertIn("background:#ffffff", html)

    def test_markdown_headings_lists_bold_italic_and_links(self):
        result = self.canonical()
        result["answer_markdown"] = "# Titre\n\n## Section\n\n- **Fort**\n- *Italique*\n- [Lien](https://example.com)"
        html = build_mail(result)["html"]
        self.assertNotIn("<h1>Titre</h1>", html)
        self.assertIn("font-size:18px", html)
        self.assertIn("font-size:16px", html)
        self.assertIn("Titre", html)
        self.assertIn("Section", html)
        self.assertIn("<ul style=", html)
        self.assertIn("<strong>Fort</strong>", html)
        self.assertIn("<em>Italique</em>", html)
        self.assertIn('href="https://example.com"', html)
        self.assertIn("text-decoration:underline", html)

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
        self.assertIn("vertical-align:super", html)
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
        self.assertIn(">[1]</a>", html)
        self.assertIn(">[3]</a>", html)

    def test_cited_source_is_rendered(self):
        mail = build_mail(self.canonical())
        self.assertIn("[1] Paris - Wikipedia", mail["text"])
        self.assertIn("https://example.com/paris", mail["html"])

    def test_uncited_source_is_excluded_from_mail_sections(self):
        mail = build_mail(self.canonical())
        self.assertNotIn("Source non citee", mail["text"])
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
        self.assertNotIn('href="https://example.com/99"', html)

    def test_french_utf8_characters(self):
        result = self.canonical()
        result["answer_markdown"] = "Réponse à propos d’Évreux et de l’été [1]."
        result["cited_sources"][0]["title"] = "Paris — Wikipédia"
        mail = build_mail(result)
        self.assertIn("Évreux", mail["text"])
        self.assertIn("été", mail["html"])
        self.assertIn("Paris — Wikipédia", mail["text"])

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

    def test_text_body_is_readable_watch_note(self):
        text = build_mail(self.canonical())["text"]
        self.assertIn("Veille Perplexica\n\nDate :", text)
        self.assertIn("QUESTION\nQuelle est la capitale", text)
        self.assertIn("SYNTHÈSE\nParis est la capitale [1].", text)
        self.assertIn("SOURCES PRINCIPALES\n[1] Paris - Wikipedia", text)
        self.assertIn("2 sources consultées · 1 sources citées", text)

    def test_chat_and_message_ids_are_metadata_only(self):
        mail = build_mail(self.canonical())
        self.assertNotIn("chat123", mail["html"])
        self.assertNotIn("msg123", mail["html"])
        self.assertNotIn("chat123", mail["text"])
        self.assertNotIn("msg123", mail["text"])
        self.assertEqual(mail["metadata"]["chat_id"], "chat123")
        self.assertEqual(mail["metadata"]["message_id"], "msg123")

    def test_outlook_safe_html_stays_simple(self):
        html = build_mail(self.canonical())["html"]
        self.assertNotIn("<style", html.lower())
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("<main", html.lower())
        self.assertNotIn("class=", html.lower())
        self.assertIn("<table", html.lower())
        self.assertIn("style=", html.lower())

    def test_outlook_print_layout_uses_600px_table_and_text_wrap(self):
        html = build_mail(self.canonical())["html"]
        self.assertIn('width="600"', html)
        self.assertIn("width:600px", html)
        self.assertNotIn('width="760"', html)
        self.assertNotIn("max-width:760px", html)
        self.assertNotIn("table-layout:fixed", html)
        self.assertNotIn("white-space:nowrap", html)
        self.assertIn("border-collapse:collapse", html)
        self.assertIn("white-space:normal", html)
        self.assertIn("word-wrap:break-word", html)
        self.assertIn("overflow-wrap:break-word", html)
        self.assertIn("overflow-wrap:anywhere", html)
        self.assertIn("https://example.com/paris", html)
        self.assertIn(">[1]</a>", html)

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

    def test_display_url_strips_scheme_and_www(self):
        self.assertEqual(display_url("https://www.cncej.com/"), "cncej.com")
        self.assertEqual(display_url("http://www.example.com"), "example.com")

    def test_display_url_keeps_domain_and_first_path_segment(self):
        self.assertEqual(
            display_url("https://www.village-justice.com/articles/expertise-judiciaire-transmission-electronique-entre-expert-juridiction-les,26458.html"),
            "village-justice.com/articles/(...)",
        )
        self.assertEqual(
            display_url("https://www.economie.gouv.fr/daj/publication-dune-circulaire-relative-lexecution-des-contrats-de-la-commande-publique-affectes-par-les-incendies-de-foret-de-lete-2026"),
            "economie.gouv.fr/daj/(...)",
        )
        self.assertEqual(
            display_url("https://www.bnds.fr/edition-numerique/read/eyJpdiI6Im9qRWFINHNvbnFCZVUxTkwwZHlucHc9PSIsInZhbHVlIjoiSHUwSk14eGpGbGVlUmV6a2N3TGUvM2RIWWF1SFZDbjFITnNYRUw2RTN6TXFVZDFTWHdzRWhYd0xaREJITFdSdnhvUEMvaG5mSEY4QnlaZU93WFhuOWc9PSIsIm1hYyI6IjM1OGRjOGFlNmEwOWYwYTQzYmIyNTdmNzdlYWNlMmE0ODEzMDMwODlmYTUzNjk5YWI3MmI1NmRhN2ZmOThhNDYiLCJ0YWciOiIifQ==/10429"),
            "bnds.fr/edition-numerique/(...)",
        )

    def test_display_url_preview_respects_limit_and_suffix(self):
        url = "https://www.village-justice.com/articles/expertise-judiciaire-transmission-electronique-entre-expert-juridiction-les,26458.html"
        preview = display_url(url, limit=40)
        self.assertLessEqual(len(preview), 40 + len("(...)"))
        self.assertTrue(preview.endswith("(...)"))

    def test_display_url_never_shows_long_query_string(self):
        url = "https://www.example.com/path?token=" + "x" * 200
        preview = display_url(url, limit=40)
        self.assertNotIn("?", preview)
        self.assertNotIn("token", preview)
        self.assertEqual(preview, "example.com/path")
        long_url = "https://www.example.com/" + "a" * 50 + "?token=" + "x" * 200
        preview = display_url(long_url, limit=40)
        self.assertNotIn("?", preview)
        self.assertNotIn("token", preview)
        self.assertTrue(preview.endswith("(...)"))
        self.assertEqual(preview, "example.com/(...)")

    def test_source_row_keeps_full_url_in_href_and_short_preview_visible(self):
        mail = build_mail(self.canonical())
        html = mail["html"]
        self.assertIn('href="https://example.com/paris"', html)
        visible = re.sub(r'href="[^"]*"', "", html)
        self.assertNotIn("https://example.com/paris", visible)
        self.assertIn("example.com/paris</a>", html)

    def test_source_title_and_number_are_clickable(self):
        mail = build_mail(self.canonical())
        html = mail["html"]
        self.assertIn("[1] Paris - Wikipedia</a>", html)
        self.assertIn('href="https://example.com/paris"', html)

    def test_missing_source_title_falls_back_to_hostname(self):
        result = self.canonical()
        result["cited_sources"][0]["title"] = None
        result["cited_sources"][0]["url"] = "https://www.example.com/"
        result["all_sources"][0]["title"] = None
        result["all_sources"][0]["url"] = "https://www.example.com/"
        mail = build_mail(result)
        self.assertIn("[1] example.com</a>", mail["html"])
        self.assertIn("example.com</a>", mail["html"])


if __name__ == "__main__":
    unittest.main()
