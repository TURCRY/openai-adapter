import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Pass1SchemaTests(unittest.TestCase):
    def test_pass1_response_format_is_not_defined_or_used(self):
        source = (ROOT / "adapter.py").read_text(encoding="utf-8")
        self.assertNotIn("def _pass1_response_format", source)
        self.assertNotIn("@app.get(\"/debug/response_format/pass1\")", source)
        self.assertIn("return False", source[source.index("def _canonical_uses_structured_outputs"):])


if __name__ == "__main__":
    unittest.main()
