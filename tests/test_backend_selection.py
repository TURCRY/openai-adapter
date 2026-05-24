import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend_selection  # noqa: E402


class BackendSelectionTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_backend_candidates_deduplicates_and_prioritizes_configured(self):
        candidates = backend_selection.parse_backend_candidates(
            "http://10.0.1.2:5050/",
            "http://10.0.1.10:5050,http://10.0.1.2:5050,http://192.168.0.155:5050",
        )

        self.assertEqual(
            candidates,
            [
                "http://10.0.1.2:5050",
                "http://10.0.1.10:5050",
                "http://192.168.0.155:5050",
            ],
        )

    async def test_select_backend_url_keeps_first_successful_candidate(self):
        calls = []

        async def fake_probe(url, ping_path, headers):
            calls.append((url, ping_path, headers))
            return (url == "http://10.0.1.2:5050", "fake")

        selected, attempts = await backend_selection.select_backend_url(
            [
                "http://10.0.1.10:5050",
                "http://10.0.1.2:5050",
                "http://192.168.0.155:5050",
            ],
            ping_path="/ping",
            headers={"x-api-key": "test"},
            probe=fake_probe,
        )

        self.assertEqual(selected, "http://10.0.1.2:5050")
        self.assertEqual([attempt["url"] for attempt in attempts], ["http://10.0.1.10:5050", "http://10.0.1.2:5050"])
        self.assertEqual(calls[-1], ("http://10.0.1.2:5050", "/ping", {"x-api-key": "test"}))


if __name__ == "__main__":
    unittest.main()
