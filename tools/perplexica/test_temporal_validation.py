#!/usr/bin/env python3
"""Unit tests for temporal_validation (no real network)."""

import json
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_validation import (
    ACCESS_ACCESSIBLE,
    ACCESS_BLOCKED,
    ACCESS_PAYWALLED,
    ACCESS_TIMEOUT,
    STATUS_CONTEXT,
    STATUS_CURRENT,
    STATUS_MISMATCH,
    STATUS_UNKNOWN,
    ROLE_CONTEXT,
    classify_temporal,
    detect_access_status,
    extract_claimed_dates,
    extract_claimed_dates_with_modes,
    extract_source_date,
    fetch_page,
    infer_temporal_role,
    is_js_heavy,
    is_paywalled,
    parse_date,
    validate_cited_sources,
)
from editorial_rewrite import build_messages, prepare_editorial_input


def json_ld_page(published="2017-11-15", modified="2026-08-20"):
    return (
        '<html><head><script type="application/ld+json">'
        '{"@type": "Article", "datePublished": "' + published
        + '", "dateModified": "' + modified + '"}'
        "</script></head><body>Article test</body></html>"
    )


class DateExtractionTests(unittest.TestCase):
    def test_json_ld_datePublished(self):
        info = extract_source_date(json_ld_page(), None)
        self.assertEqual(info["source_date"], date(2017, 11, 15))
        self.assertEqual(info["date_evidence"], "json_ld")
        self.assertEqual(info["date_confidence"], "high")

    def test_article_published_time(self):
        html = '<meta property="article:published_time" content="2026-08-20T08:00:00+02:00">'
        info = extract_source_date(html, None)
        self.assertEqual(info["source_date"], date(2026, 8, 20))
        self.assertEqual(info["date_evidence"], "meta")

    def test_time_datetime(self):
        html = "<article><time datetime=\"2020-01-02\">2 janvier 2020</time></article>"
        info = extract_source_date(html, None)
        self.assertEqual(info["source_date"], date(2020, 1, 2))
        self.assertEqual(info["date_evidence"], "time")
        self.assertEqual(info["date_confidence"], "medium")

    def test_meta_date_generic(self):
        html = '<meta name="DC.date" content="2019-05-04">'
        info = extract_source_date(html, None)
        self.assertEqual(info["source_date"], date(2019, 5, 4))
        self.assertEqual(info["date_evidence"], "meta_date")
        self.assertEqual(info["date_confidence"], "medium")

    def test_page_without_date(self):
        info = extract_source_date("<html><body>No date here</body></html>", None)
        self.assertIsNone(info["source_date"])
        self.assertIsNone(info["date_evidence"])
        self.assertIsNone(info["date_confidence"])

    def test_datePublished_priority_over_dateModified(self):
        info = extract_source_date(json_ld_page(published="2017-11-15", modified="2026-08-20"), None)
        self.assertEqual(info["source_date"], date(2017, 11, 15))
        self.assertEqual(info["modified_date"], date(2026, 8, 20))
        info_only_modified = extract_source_date(
            '<script type="application/ld+json">{"dateModified":"2026-08-20"}</script>', None
        )
        self.assertIsNone(info_only_modified["source_date"])
        self.assertEqual(info_only_modified["modified_date"], date(2026, 8, 20))
        self.assertEqual(info_only_modified["date_evidence"], "modified_time")

    def test_document_signature_date(self):
        html = (
            "<html><title>D\u00e9cret n\u00b0 2025-660</title><body>"
            "D\u00e9cret n\u00b0 2025-660 du 18 juillet 2025"
            "</body></html>"
        )
        info = extract_source_date(html, "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000051919659")
        self.assertEqual(info["source_date"], date(2025, 7, 18))
        self.assertIn(info["date_evidence"], ("doc_signature", "json_ld", "meta", "meta_date"))

    def test_legifrance_jorf_url(self):
        info = extract_source_date("", "https://www.legifrance.gouv.fr/jorf/jo/2026/08/15/0190")
        self.assertEqual(info["source_date"], date(2026, 8, 15))
        self.assertEqual(info["date_evidence"], "legifrance_url")
        self.assertEqual(info["date_confidence"], "high")

    def test_legifrance_codes_no_date_and_context_role(self):
        url = "https://www.legifrance.gouv.fr/codes/section_lc/LEGITEXT000006070716/LEGISCTA000006165192/"
        info = extract_source_date("", url)
        self.assertIsNone(info["source_date"])
        self.assertEqual(infer_temporal_role("", 5, url), "context")

    def test_parse_date_formats(self):
        self.assertEqual(parse_date("2026-08-20"), date(2026, 8, 20))
        self.assertEqual(parse_date("26/08/2026"), date(2026, 8, 26))
        self.assertEqual(parse_date("20 ao\u00fbt 2026"), date(2026, 8, 20))
        self.assertIsNone(parse_date("not a date"))
class ClaimedDateTests(unittest.TestCase):
    def test_claimed_date_mono_citation(self):
        claims = extract_claimed_dates(
            "Une mise au point a \u00e9t\u00e9 publi\u00e9e le 20 ao\u00fbt 2026 "
            "et aborde les enjeux pratiques [72]."
        )
        self.assertEqual(claims.get(72), ["2026-08-20"])

    def test_claimed_date_iso_and_slash(self):
        claims = extract_claimed_dates("Mise \u00e0 jour au 26/08/2026 [82].")
        self.assertEqual(claims.get(82), ["2026-08-26"])
        claims_iso = extract_claimed_dates("Publication 2026-08-15 [9].")
        self.assertEqual(claims_iso.get(9), ["2026-08-15"])

    def test_multi_citation_ambiguous_not_attributed(self):
        claims = extract_claimed_dates(
            "Plusieurs d\u00e9cisions ont \u00e9t\u00e9 publi\u00e9es en ao\u00fbt 2026 [1][2][3]."
        )
        self.assertEqual(claims, {})
        claims_two = extract_claimed_dates("Textes cit\u00e9s [1] et [2] datent du 15/08/2026.")
        self.assertEqual(claims_two, {})

    def test_multi_citation_same_date_not_attributed(self):
        claims = extract_claimed_dates("publi\u00e9e le 20 ao\u00fbt 2026[72][19]")
        self.assertEqual(claims, {})


class ClassificationTests(unittest.TestCase):
    def test_mismatch_2026_vs_2017(self):
        status, note = classify_temporal(
            date(2017, 11, 15), "high", "direct", "accessible", ["2026-08-20"], "current",
            run_date="2026-08-27",
        )
        self.assertEqual(status, STATUS_MISMATCH)
        self.assertIn("actualit\u00e9", note)

    def test_old_context_source(self):
        status, note = classify_temporal(
            date(2025, 7, 18), "medium", "direct", "accessible", [], "context",
            run_date="2026-08-27",
        )
        self.assertEqual(status, STATUS_CONTEXT)

    def test_old_decree_legitimate(self):
        html = (
            "<html><body>D\u00e9cret n\u00b0 2025-660 du 18 juillet 2025 "
            "reste une r\u00e9f\u00e9rence op\u00e9rationnelle</body></html>"
        )
        info = extract_source_date(html, "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000051919659")
        status, _ = classify_temporal(
            info["source_date"], info["date_confidence"], "direct", "accessible", [], "context",
            run_date="2026-08-27",
        )
        self.assertEqual(status, STATUS_CONTEXT)

    def test_old_source_recent_section_not_automatic_mismatch(self):
        status, _ = classify_temporal(
            date(2017, 11, 15), "high", "direct", "accessible", [], "current",
            run_date="2026-08-27",
        )
        self.assertNotEqual(status, STATUS_MISMATCH)
        self.assertEqual(status, STATUS_CONTEXT)

    def test_recent_source_current(self):
        status, note = classify_temporal(
            date(2026, 8, 25), "high", "direct", "accessible", [], "current",
            run_date="2026-08-27",
        )
        self.assertEqual(status, STATUS_CURRENT)
        self.assertEqual(note, "")

    def test_paywall_without_date_unknown(self):
        def fetch(url):
            return (200, url, "<html><body>Article r\u00e9serv\u00e9 aux abonn\u00e9s</body></html>")

        validated, _ = validate_cited_sources(
            [{"index": 1, "title": "t", "url": "https://paywall.fr/a"}],
            local_answers={"s": "texte [1]"},
            fetch_fn=fetch,
            run_date="2026-08-27",
        )
        temporal = validated[0]["temporal"]
        self.assertEqual(temporal["access_status"], ACCESS_PAYWALLED)
        self.assertEqual(temporal["temporal_status"], STATUS_UNKNOWN)

    def test_timeout_non_blocking(self):
        def fetch(url):
            return (None, url, "")

        validated, _ = validate_cited_sources(
            [{"index": 1, "url": "https://slow.fr/a"}],
            fetch_fn=fetch,
            run_date="2026-08-27",
        )
        temporal = validated[0]["temporal"]
        self.assertEqual(temporal["access_status"], ACCESS_TIMEOUT)
        self.assertEqual(temporal["temporal_status"], STATUS_UNKNOWN)

    def test_validate_counts(self):
        pages = {
            "https://a.fr/new": (
                200,
                "https://a.fr/new",
                '<meta property="article:published_time" content="2026-08-25">',
            ),
            "https://b.fr/old": (
                200,
                "https://b.fr/old",
                '<meta property="article:published_time" content="2017-11-15">',
            ),
            "https://c.fr/ctx": (
                200,
                "https://c.fr/ctx",
                '<meta property="article:published_time" content="2025-07-18">',
            ),
            "https://d.fr/none": (200, "https://d.fr/none", "<html><body>rien</body></html>"),
        }

        def fetch(url):
            return pages[url]

        sources = [
            {"index": 1, "url": "https://a.fr/new"},
            {"index": 2, "url": "https://b.fr/old"},
            {"index": 3, "url": "https://c.fr/ctx"},
            {"index": 4, "url": "https://d.fr/none"},
        ]
        validated, summary = validate_cited_sources(sources, fetch_fn=fetch, run_date="2026-08-27")
        self.assertEqual(summary["temporal_validation_count"], 4)
        self.assertEqual(summary["direct_date_count"], 3)
        self.assertEqual(summary["unknown_date_count"], 1)
        self.assertEqual(summary["current_count"], 1)
        self.assertEqual(summary["context_count"], 2)
        self.assertEqual(summary["unknown_count"], 1)

    def test_local_to_global_mapping(self):
        def fetch(url):
            return (200, url, json_ld_page(published="2017-11-15"))

        sources = [
            {
                "index": 10,
                "title": "Village justice",
                "url": "https://www.village-justice.com/articles/x,26458.html",
                "source_searches": ["expertise_justice"],
                "original_indices": {"expertise_justice": 72},
            }
        ]
        validated, summary = validate_cited_sources(
            sources,
            local_answers={
                "expertise_justice": "Une mise au point a \u00e9t\u00e9 publi\u00e9e le 20 ao\u00fbt 2026 [72]."
            },
            fetch_fn=fetch,
            run_date="2026-08-27",
        )
        temporal = validated[0]["temporal"]
        self.assertEqual(temporal["claimed_dates"], ["2026-08-20"])
        self.assertEqual(temporal["claimed_from_searches"], ["expertise_justice"])
        self.assertEqual(temporal["temporal_status"], STATUS_MISMATCH)
        self.assertEqual(summary["mismatch_count"], 1)

    def test_fetch_cache_url(self):
        cache = {
            "https://cached.example/a": (200, "https://cached.example/a", json_ld_page(published="2026-08-01"))
        }
        status, final_url, html = fetch_page("https://cached.example/a", cache=cache)
        self.assertEqual(status, 200)
        self.assertIn("datePublished", html)
        self.assertEqual(cache["https://cached.example/a"][0], 200)

    def test_editorial_gemma_injection_compact(self):
        result = {
            "question": "Q",
            "answer_markdown": "Texte [1].",
            "citation_numbers": [1],
            "cited_sources": [
                {
                    "index": 1,
                    "title": "Source test",
                    "url": "https://www.example.com/long/path",
                    "temporal": {
                        "access_status": ACCESS_ACCESSIBLE,
                        "source_date": "2017-11-15",
                        "date_evidence": "json_ld",
                        "date_confidence": "high",
                        "date_verification": "direct",
                        "claimed_dates": ["2026-08-20"],
                        "claimed_from_searches": ["expertise_justice"],
                        "temporal_role": "current",
                        "temporal_status": STATUS_MISMATCH,
                        "note": "Ne pas présenter cette source comme actualité récente.",
                    },
                }
            ],
            "temporal_validation": {
                "status": "completed",
                "temporal_validation_count": 1,
                "current_count": 0,
                "context_count": 0,
                "mismatch_count": 1,
                "unknown_count": 0,
                "direct_date_count": 1,
                "indirect_date_count": 0,
                "unknown_date_count": 0,
            },
        }
        payload = prepare_editorial_input(result)
        entry = payload["cited_sources"][0]
        self.assertEqual(entry["number"], 1)
        self.assertEqual(entry["title"], "Source test")
        self.assertEqual(entry["temporal"]["status"], STATUS_MISMATCH)
        self.assertEqual(entry["temporal"]["source_date"], "2017-11-15")
        self.assertIn("actualit", entry["temporal"]["note"])
        self.assertEqual(payload["temporal_validation"]["mismatch_count"], 1)

    def test_editorial_no_url_in_payload(self):
        result = {
            "question": "Q",
            "answer_markdown": "Texte [1] [3].",
            "citation_numbers": [1, 3],
            "cited_sources": [
                {"index": 1, "title": "A", "url": "https://www.example.com/a"},
                {
                    "index": 3,
                    "title": "B",
                    "url": "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000000000001",
                    "temporal": {
                        "access_status": ACCESS_ACCESSIBLE,
                        "source_date": "2026-08-20",
                        "date_evidence": "legifrance_url",
                        "date_confidence": "high",
                        "date_verification": "direct",
                        "claimed_dates": [],
                        "claimed_from_searches": [],
                        "temporal_role": "current",
                        "temporal_status": STATUS_CURRENT,
                        "note": "",
                    },
                },
            ],
        }
        payload = prepare_editorial_input(result)
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("example.com", raw)
        self.assertNotIn("legifrance.gouv.fr", raw)
        self.assertNotIn("http", raw)
        messages = build_messages("SYSTEM", payload)
        self.assertNotIn("http", messages[1]["content"])

    def test_extract_claimed_dates_with_modes(self):
        detailed = extract_claimed_dates_with_modes(
            "Le Cnam a mis à jour sa fiche le 26 août 2026 [19].\n"
            "Une mise au point a été publiée le 20 août 2026 [72]."
        )
        self.assertEqual(detailed[19]["update_claim"], ["2026-08-26"])
        self.assertEqual(detailed[19].get("publication_claim", []), [])
        self.assertEqual(detailed[72]["publication_claim"], ["2026-08-20"])
        self.assertEqual(detailed[72].get("update_claim", []), [])

    def test_multiple_dates_one_citation_categorized(self):
        detailed = extract_claimed_dates_with_modes(
            "Un article publié le 25 août 2026 présente une nouvelle procédure "
            "issue d'un décret du 5 août 2025 [8]."
        )
        self.assertEqual(detailed[8]["publication_claim"], ["2026-08-25"])
        self.assertEqual(detailed[8]["legal_text_date"], ["2025-08-05"])

    def test_visible_publication_date_extracted(self):
        html = (
            '<html><body><article>'
            "<p>Publication : 27 août 2026</p>"
            "</article></body></html>"
        )
        info = extract_source_date(html, None)
        self.assertEqual(info["visible_publication_date"], date(2026, 8, 27))
        self.assertEqual(info["visible_publication_dates"], [date(2026, 8, 27)])
        self.assertIsNone(info["visible_update_date"])
        self.assertIsNone(info["source_date"])

    def test_visible_update_date_extracted(self):
        html = (
            "<html><body>"
            "<p>Mis à jour le 26 août 2026</p>"
            "<p>Actualisé le 25 août 2026</p>"
            "</body></html>"
        )
        info = extract_source_date(html, None)
        self.assertEqual(info["visible_update_date"], date(2026, 8, 26))
        self.assertEqual(info["visible_update_dates"], [date(2026, 8, 26), date(2026, 8, 25)])
        self.assertIsNone(info["visible_publication_date"])

    def test_visible_update_only_does_not_promote(self):
        status, _ = classify_temporal(
            date(2023, 9, 1),
            "high",
            "direct",
            ACCESS_ACCESSIBLE,
            [],
            "current",
            run_date="2026-08-27",
            modified_date=date(2026, 8, 26),
        )
        self.assertEqual(status, STATUS_CONTEXT)

    def test_service_public_32_no_false_mismatch(self):
        pages = {
            "https://service-public.gouv.fr/actualites/A18459": (
                200,
                "https://service-public.gouv.fr/actualites/A18459",
                '<script type="application/ld+json">{"@type": "NewsArticle", '
                '"datePublished": "2025-09-17", "dateModified": "2026-02-02"}</script>'
                '<div class="publication-date">Publié le 17 septembre 2025 - '
                "Mise à jour le 02 février 2026</div>"
                '<div class="agenda-item">Publié le 27 août 2026</div>'
                "<h1>Une nouvelle procédure pour demander le remboursement</h1>",
            )
        }

        def fetch(url):
            return pages[url]

        sources = [
            {
                "index": 32,
                "title": "Une nouvelle procédure pour demander le remboursement",
                "url": "https://service-public.gouv.fr/actualites/A18459",
                "source_searches": ["mediation"],
                "original_indices": {"mediation": 8},
            }
        ]
        validated, summary = validate_cited_sources(
            sources,
            local_answers={
                "mediation": (
                    "Un article publié le 25 août 2026 présente une nouvelle procédure "
                    "issue d'un décret du 5 août 2025 [8]."
                )
            },
            fetch_fn=fetch,
            run_date="2026-08-27",
        )
        temporal = validated[0]["temporal"]
        self.assertEqual(temporal["temporal_status"], STATUS_CURRENT)
        self.assertEqual(temporal["source_date"], "2025-09-17")
        self.assertEqual(temporal["visible_publication_date"], "2025-09-17")
        self.assertEqual(
            temporal["visible_publication_dates"], ["2025-09-17", "2026-08-27"]
        )
        self.assertEqual(temporal["claimed_dates"], ["2026-08-25"])
        self.assertEqual(temporal["claimed_legal_text_dates"], ["2025-08-05"])
        self.assertEqual(summary["mismatch_count"], 0)

    def test_service_public_34_no_false_mismatch(self):
        pages = {
            "https://service-public.gouv.fr/actualites/A18826": (
                200,
                "https://service-public.gouv.fr/actualites/A18826",
                '<script type="application/ld+json">{"@type": "NewsArticle", '
                '"datePublished": "2026-03-02"}</script>'
                '<div class="publication-date">Publié le 02 mars 2026</div>'
                '<div class="agenda-item">Publié le 27 août 2026</div>',
            )
        }

        def fetch(url):
            return pages[url]

        sources = [
            {
                "index": 34,
                "title": "Instauration d'une contribution pour saisir la justice",
                "url": "https://service-public.gouv.fr/actualites/A18826",
                "source_searches": ["mediation"],
                "original_indices": {"mediation": 20},
            }
        ]
        validated, summary = validate_cited_sources(
            sources,
            local_answers={
                "mediation": (
                    "Le site Service-public a mis en ligne le 25 août 2026 une notice "
                    "rappelant la mise en place d'une contribution [20]."
                )
            },
            fetch_fn=fetch,
            run_date="2026-08-27",
        )
        temporal = validated[0]["temporal"]
        self.assertEqual(temporal["temporal_status"], STATUS_CURRENT)
        self.assertEqual(temporal["source_date"], "2026-03-02")
        self.assertEqual(temporal["visible_publication_date"], "2026-03-02")
        self.assertEqual(
            temporal["visible_publication_dates"], ["2026-03-02", "2026-08-27"]
        )
        self.assertEqual(temporal["claimed_dates"], ["2026-08-25"])
        self.assertEqual(summary["mismatch_count"], 0)

    def test_context_role_visible_recent_stays_context(self):
        status, note = classify_temporal(
            date(2025, 7, 18),
            "high",
            "direct",
            ACCESS_ACCESSIBLE,
            [],
            ROLE_CONTEXT,
            run_date="2026-08-27",
            visible_publication_date=date(2026, 8, 27),
            visible_publication_dates=[date(2025, 7, 18), date(2026, 8, 27)],
        )
        self.assertEqual(status, STATUS_CONTEXT)
        self.assertIn("divergente", note)

    def test_legal_text_date_only_no_mismatch(self):
        pages = {
            "https://service-public.gouv.fr/actualites/A1": (
                200,
                "https://service-public.gouv.fr/actualites/A1",
                "<html><body>notice</body></html>",
            )
        }

        def fetch(url):
            return pages[url]

        sources = [
            {
                "index": 1,
                "url": "https://service-public.gouv.fr/actualites/A1",
                "source_searches": ["mediation"],
                "original_indices": {"mediation": 8},
            }
        ]
        validated, _ = validate_cited_sources(
            sources,
            local_answers={
                "mediation": "La procédure est issue d'un décret du 5 août 2025 [8]."
            },
            fetch_fn=fetch,
            run_date="2026-08-27",
        )
        temporal = validated[0]["temporal"]
        self.assertEqual(temporal.get("claimed_dates"), [])
        self.assertEqual(temporal["claimed_legal_text_dates"], ["2025-08-05"])
        self.assertNotEqual(temporal["temporal_status"], STATUS_MISMATCH)

    def test_village_justice_still_mismatch(self):
        pages = {
            "https://www.village-justice.com/articles/x,26458.html": (
                200,
                "https://www.village-justice.com/articles/x,26458.html",
                '<script type="application/ld+json">{"@type": "Article", '
                '"datePublished": "2017-11-15T12:11:54+01:00", '
                '"dateModified": "2017-11-15T12:11:17+01:00"}</script>'
                '<article><time datetime="2017-11-15">15 novembre 2017</time></article>',
            )
        }

        def fetch(url):
            return pages[url]

        sources = [
            {
                "index": 10,
                "title": "Expertise judiciaire : transmission électronique",
                "url": "https://www.village-justice.com/articles/x,26458.html",
                "source_searches": ["expertise_justice"],
                "original_indices": {"expertise_justice": 72},
            }
        ]
        validated, summary = validate_cited_sources(
            sources,
            local_answers={
                "expertise_justice": (
                    "Une mise au point a été publiée le 20 août 2026 [72]."
                )
            },
            fetch_fn=fetch,
            run_date="2026-08-27",
        )
        temporal = validated[0]["temporal"]
        self.assertEqual(temporal["temporal_status"], STATUS_MISMATCH)
        self.assertEqual(temporal["source_date"], "2017-11-15")
        self.assertEqual(summary["mismatch_count"], 1)

    def test_old_decree_2025_660_still_context(self):
        pages = {
            "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000051919659": (
                200,
                "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000051919659",
                "<html><body>Décret n° 2025-660 du 18 juillet 2025 "
                "reste une référence opérationnelle</body></html>",
            )
        }

        def fetch(url):
            return pages[url]

        sources = [
            {
                "index": 36,
                "title": "Décret n° 2025-660 du 18 juillet 2025",
                "url": "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000051919659",
                "source_searches": ["jurisprudence"],
                "original_indices": {"jurisprudence": 22},
            }
        ]
        validated, _ = validate_cited_sources(
            sources,
            local_answers={
                "jurisprudence": (
                    "Le décret du 18 juillet 2025 reste une référence "
                    "pour le droit applicable [22]."
                )
            },
            fetch_fn=fetch,
            run_date="2026-08-27",
        )
        temporal = validated[0]["temporal"]
        self.assertEqual(temporal["temporal_status"], STATUS_CONTEXT)
        self.assertEqual(temporal["claimed_dates"], [])
        self.assertEqual(temporal["claimed_legal_text_dates"], ["2025-07-18"])

    def test_update_claim_matches_modified_date_not_mismatch(self):
        status, _ = classify_temporal(
            date(2023, 9, 1),
            "high",
            "direct",
            ACCESS_ACCESSIBLE,
            [],
            "current",
            run_date="2026-08-27",
            claimed_updates=["2026-08-26"],
            modified_date=date(2026, 8, 26),
        )
        self.assertEqual(status, STATUS_CURRENT)

    def test_update_claim_without_modified_date_not_mismatch(self):
        status, _ = classify_temporal(
            date(2023, 9, 1),
            "high",
            "direct",
            ACCESS_ACCESSIBLE,
            [],
            "current",
            run_date="2026-08-27",
            claimed_updates=["2026-08-26"],
        )
        self.assertEqual(status, STATUS_CONTEXT)

    def test_publication_claim_still_mismatch_with_update_elsewhere(self):
        status, _ = classify_temporal(
            date(2017, 11, 15),
            "high",
            "direct",
            ACCESS_ACCESSIBLE,
            ["2026-08-20"],
            "current",
            run_date="2026-08-27",
            claimed_updates=["2026-08-21"],
            modified_date=date(2026, 8, 21),
        )
        self.assertEqual(status, STATUS_MISMATCH)

    def test_update_claim_end_to_end_not_mismatch(self):
        pages = {
            "https://cnam.example/fiche": (
                200,
                "https://cnam.example/fiche",
                '<script type="application/ld+json">{"@type": "WebPage", '
                '"datePublished": "2023-09-01", "dateModified": "2026-08-26"}</script>',
            )
        }

        def fetch(url):
            return pages[url]

        sources = [
            {
                "index": 2,
                "url": "https://cnam.example/fiche",
                "source_searches": ["expertise_justice"],
                "original_indices": {"expertise_justice": 19},
            }
        ]
        validated, summary = validate_cited_sources(
            sources,
            local_answers={
                "expertise_justice": "Le Cnam a mis à jour sa fiche le 26 août 2026 [19]."
            },
            fetch_fn=fetch,
            run_date="2026-08-27",
        )
        temporal = validated[0]["temporal"]
        self.assertEqual(temporal["source_date"], "2023-09-01")
        self.assertEqual(temporal["modified_date"], "2026-08-26")
        self.assertEqual(temporal["temporal_status"], STATUS_CURRENT)
        self.assertEqual(summary["mismatch_count"], 0)
