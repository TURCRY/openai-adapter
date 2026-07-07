import asyncio
import sys
import types
import unittest
from pathlib import Path

import httpx


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _DummyFastAPI:
    def __init__(self, *args, **kwargs):
        self.state = types.SimpleNamespace()

    def add_middleware(self, *args, **kwargs):
        return None

    def on_event(self, *args, **kwargs):
        return lambda fn: fn

    def get(self, *args, **kwargs):
        return lambda fn: fn

    def post(self, *args, **kwargs):
        return lambda fn: fn


def _identity_default(default=None, *args, **kwargs):
    return default


fastapi_stub = types.ModuleType("fastapi")
fastapi_stub.FastAPI = _DummyFastAPI
fastapi_stub.Header = _identity_default
fastapi_stub.File = _identity_default
fastapi_stub.Form = _identity_default
fastapi_stub.Body = _identity_default
fastapi_stub.UploadFile = object
fastapi_stub.Request = object
fastapi_stub.HTTPException = HTTPException

fastapi_middleware_stub = types.ModuleType("fastapi.middleware")
fastapi_cors_stub = types.ModuleType("fastapi.middleware.cors")
fastapi_cors_stub.CORSMiddleware = object
fastapi_responses_stub = types.ModuleType("fastapi.responses")
fastapi_responses_stub.Response = object
fastapi_responses_stub.JSONResponse = object
fastapi_responses_stub.StreamingResponse = object

sys.modules.setdefault("fastapi", fastapi_stub)
sys.modules.setdefault("fastapi.middleware", fastapi_middleware_stub)
sys.modules.setdefault("fastapi.middleware.cors", fastapi_cors_stub)
sys.modules.setdefault("fastapi.responses", fastapi_responses_stub)

if "pydantic" not in sys.modules:
    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.BaseModel = type("BaseModel", (), {})
    sys.modules["pydantic"] = pydantic_stub

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import adapter  # noqa: E402


class FakeHTTP:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_response(status_code: int, url: str, payload: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload or {},
        request=httpx.Request("POST", url),
    )


class RuntimeBackendFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.old_base = adapter.LOCAL_BASE
        self.old_candidates = list(adapter.LOCAL_BACKEND_CANDIDATES)
        self.old_http = adapter._http
        self.old_select = adapter.select_backend_url
        self.old_ping_ts = adapter._last_ping_ok_ts
        adapter.LOCAL_BASE = "http://old:5050"
        adapter.LOCAL_BACKEND_CANDIDATES = ["http://old:5050", "http://new:5050"]
        adapter._last_ping_ok_ts = 0.0
        adapter._backend_reselect_lock = asyncio.Lock()

    def tearDown(self):
        adapter.LOCAL_BASE = self.old_base
        adapter.LOCAL_BACKEND_CANDIDATES = self.old_candidates
        adapter._http = self.old_http
        adapter.select_backend_url = self.old_select
        adapter._last_ping_ok_ts = self.old_ping_ts
        adapter._backend_reselect_lock = asyncio.Lock()

    async def test_timeout_reselects_backend_and_retries_once(self):
        request = httpx.Request("POST", "http://old:5050/chat_orchestre")
        adapter._http = FakeHTTP([
            httpx.ReadTimeout("timed out", request=request),
            make_response(200, "http://new:5050/chat_orchestre", {"text": "ok"}),
        ])

        async def fake_select(candidates, ping_path, headers, probe):
            self.assertEqual(candidates, ["http://new:5050", "http://old:5050"])
            return "http://new:5050", [{"url": "http://new:5050", "ok": "true", "detail": "fake"}]

        adapter.select_backend_url = fake_select

        response = await adapter._local_request_once_with_runtime_fallback(
            "POST",
            "/chat_orchestre",
            timeout=1,
            json={"prompt": "hello"},
            headers={},
        )

        self.assertEqual(response.json()["text"], "ok")
        self.assertEqual(adapter.LOCAL_BASE, "http://new:5050")
        self.assertEqual(
            [call[1] for call in adapter._http.calls],
            ["http://old:5050/chat_orchestre", "http://new:5050/chat_orchestre"],
        )

    async def test_http_503_reselects_backend_and_retries_once(self):
        adapter._http = FakeHTTP([
            make_response(503, "http://old:5050/chat_orchestre", {"error": "busy"}),
            make_response(200, "http://new:5050/chat_orchestre", {"text": "ok"}),
        ])

        async def fake_select(candidates, ping_path, headers, probe):
            return "http://new:5050", [{"url": "http://new:5050", "ok": "true", "detail": "fake"}]

        adapter.select_backend_url = fake_select

        response = await adapter._local_request_once_with_runtime_fallback(
            "POST",
            "/chat_orchestre",
            timeout=1,
            json={"prompt": "hello"},
            headers={},
        )

        self.assertEqual(response.json()["text"], "ok")
        self.assertEqual(adapter.LOCAL_BASE, "http://new:5050")
        self.assertEqual(len(adapter._http.calls), 2)

    async def test_http_400_does_not_reselect_backend(self):
        adapter._http = FakeHTTP([
            make_response(400, "http://old:5050/chat_orchestre", {"error": "bad request"}),
        ])
        select_calls = []

        async def fake_select(candidates, ping_path, headers, probe):
            select_calls.append(candidates)
            return "http://new:5050", []

        adapter.select_backend_url = fake_select

        with self.assertRaises(httpx.HTTPStatusError):
            await adapter._local_request_once_with_runtime_fallback(
                "POST",
                "/chat_orchestre",
                timeout=1,
                json={"prompt": "hello"},
                headers={},
            )

        self.assertEqual(select_calls, [])
        self.assertEqual(adapter.LOCAL_BASE, "http://old:5050")
        self.assertEqual(len(adapter._http.calls), 1)

    async def test_no_candidate_ok_raises_clear_502(self):
        request = httpx.Request("POST", "http://old:5050/chat_orchestre")
        adapter._http = FakeHTTP([
            httpx.ConnectError("connection refused", request=request),
        ])

        async def fake_select(candidates, ping_path, headers, probe):
            return None, [
                {"url": "http://new:5050", "ok": "false", "detail": "down"},
                {"url": "http://old:5050", "ok": "false", "detail": "down"},
            ]

        adapter.select_backend_url = fake_select

        with self.assertRaises(HTTPException) as ctx:
            await adapter._local_request_once_with_runtime_fallback(
                "POST",
                "/chat_orchestre",
                timeout=1,
                json={"prompt": "hello"},
                headers={},
            )

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("Local Flask backend unavailable", str(ctx.exception.detail))
        self.assertEqual(adapter.LOCAL_BASE, "http://old:5050")
        self.assertEqual(len(adapter._http.calls), 1)


if __name__ == "__main__":
    unittest.main()