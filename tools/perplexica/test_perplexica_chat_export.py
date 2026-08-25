import unittest
from collections import Counter

from perplexica_chat_export import (
    extract_citation_numbers,
    extract_source_items,
    extract_text_blocks,
    normalize_source,
    transform_message,
)


class PerplexicaChatExportTests(unittest.TestCase):
    def test_concatenates_multiple_text_blocks(self):
        blocks = [
            {"type": "text", "data": "First"},
            {"type": "source", "data": []},
            {"type": "text", "data": "Second"},
        ]
        self.assertEqual(extract_text_blocks(blocks), "First\n\nSecond")

    def test_flattens_multiple_source_blocks(self):
        blocks = [
            {"type": "source", "data": [{"content": "A"}]},
            {"type": "text", "data": "Answer"},
            {"type": "source", "data": [{"content": "B"}, {"content": "C"}]},
        ]
        self.assertEqual(
            [source["content"] for source in extract_source_items(blocks)],
            ["A", "B", "C"],
        )

    def test_extracts_supported_citation_formats(self):
        answer = "Alpha [1] beta [2] gamma [1, 3] delta [1][2]."
        self.assertEqual(extract_citation_numbers(answer), [1, 2, 1, 3, 1, 2])

    def test_does_not_extract_non_numeric_brackets(self):
        answer = "This is [not a citation], nor [1, nope], nor [0]."
        self.assertEqual(extract_citation_numbers(answer), [])

    def test_reports_out_of_range_citation(self):
        message = {
            "responseBlocks": [
                {"type": "text", "data": "Only one source exists [2]."},
                {"type": "source", "data": [{"metadata": {"url": "https://a"}}]},
            ]
        }
        exported = transform_message(message)
        self.assertEqual(exported["unresolved_citations"], [{"number": 2, "citation_count": 1}])
        self.assertEqual(exported["cited_sources"], [])

    def test_source_without_title_url_content_is_tolerated(self):
        normalized = normalize_source({}, 1, Counter({1: 1}))
        self.assertEqual(
            normalized,
            {
                "index": 1,
                "title": None,
                "url": None,
                "content": None,
                "cited": True,
                "citation_count": 1,
            },
        )

    def test_message_without_source_block(self):
        exported = transform_message({"responseBlocks": [{"type": "text", "data": "No sources [1]."}]})
        self.assertEqual(exported["all_sources"], [])
        self.assertEqual(exported["cited_sources"], [])
        self.assertEqual(exported["unresolved_citations"], [{"number": 1, "citation_count": 1}])

    def test_message_without_text_block(self):
        exported = transform_message(
            {"responseBlocks": [{"type": "source", "data": [{"content": "A"}]}]}
        )
        self.assertEqual(exported["answer_markdown"], "")
        self.assertEqual(exported["citation_numbers"], [])
        self.assertFalse(exported["all_sources"][0]["cited"])

    def test_preserves_uncited_sources(self):
        exported = transform_message(
            {
                "responseBlocks": [
                    {"type": "text", "data": "Uses first source [1]."},
                    {
                        "type": "source",
                        "data": [
                            {"content": "A", "metadata": {"title": "A", "url": "https://a"}},
                            {"content": "B", "metadata": {"title": "B", "url": "https://b"}},
                        ],
                    },
                ]
            }
        )
        self.assertTrue(exported["all_sources"][0]["cited"])
        self.assertFalse(exported["all_sources"][1]["cited"])
        self.assertEqual(len(exported["all_sources"]), 2)

    def test_citation_count_is_correct(self):
        exported = transform_message(
            {
                "responseBlocks": [
                    {"type": "text", "data": "Repeated [1], again [1], with second [2]."},
                    {
                        "type": "source",
                        "data": [
                            {"metadata": {"url": "https://a"}},
                            {"metadata": {"url": "https://b"}},
                        ],
                    },
                ]
            }
        )
        self.assertEqual(exported["all_sources"][0]["citation_count"], 2)
        self.assertEqual(exported["all_sources"][1]["citation_count"], 1)
        self.assertEqual(exported["cited_sources"][0]["citation_count"], 2)


if __name__ == "__main__":
    unittest.main()
