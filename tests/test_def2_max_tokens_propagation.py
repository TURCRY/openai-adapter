import sys
import types
import unittest
from pathlib import Path

import httpx


class HTTPException(Exception):
    def __init__(self, status_code=None, detail=None):
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

    def middleware(self, *args, **kwargs):
        return lambda fn: fn


def _identity_default(default=None, *args, **kwargs):
    return default


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self):
        return dict(self.__dict__)


def _install_stubs():
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.FastAPI = _DummyFastAPI
    fastapi_stub.Header = _identity_default
    fastapi_stub.File = _identity_default
    fastapi_stub.Form = _identity_default
    fastapi_stub.Body = _identity_default
    fastapi_stub.UploadFile = object
    fastapi_stub.Request = object
    fastapi_stub.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi_stub

    sys.modules["fastapi.middleware"] = types.ModuleType("fastapi.middleware")
    cors_stub = types.ModuleType("fastapi.middleware.cors")
    cors_stub.CORSMiddleware = object
    sys.modules["fastapi.middleware.cors"] = cors_stub
    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.Response = object
    responses_stub.JSONResponse = object
    responses_stub.StreamingResponse = object
    sys.modules["fastapi.responses"] = responses_stub

    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.BaseModel = _BaseModel
    sys.modules["pydantic"] = pydantic_stub


def _remove_stubs():
    for name in (
        "adapter",
        "fastapi",
        "fastapi.middleware",
        "fastapi.middleware.cors",
        "fastapi.responses",
        "pydantic",
    ):
        sys.modules.pop(name, None)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def model_dump(self, **kwargs):
        return {"role": self.role, "content": self.content}


class _Request:
    headers = {}


def _chat_req(**overrides):
    data = {
        "model": "annoter",
        "messages": [_Message("user", "OK")],
        "temperature": None,
        "stream": False,
        "metadata": {},
        "response_format": None,
        "max_tokens": None,
        "max_completion_tokens": None,
        "stop": None,
        "tools": None,
        "tool_choice": None,
        "parallel_tool_calls": None,
    }
    data.update(overrides)
    return types.SimpleNamespace(**data)


class Def2MaxTokensPropagationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _install_stubs()
        import adapter

        self.adapter = adapter
        self.calls = []
        self.old_ensure_local_ready = adapter._ensure_local_ready
        self.old_local_request_once = adapter._local_request_once_with_runtime_fallback
        self.old_model_info = adapter._ensure_local_model_info
        self.old_adapter_api_key = adapter.ADAPTER_API_KEY
        self.old_model_registry = dict(adapter.MODEL_REGISTRY)

        async def fake_ready():
            return True

        async def fake_model_info():
            return {"n_ctx": 4096, "max_tokens": 1024}

        async def fake_local_request(method, path, **kwargs):
            self.calls.append({"method": method, "path": path, **kwargs})
            return httpx.Response(
                200,
                json={"reponse": "OK"},
                request=httpx.Request(method, "http://local" + path),
            )

        adapter._ensure_local_ready = fake_ready
        adapter._ensure_local_model_info = fake_model_info
        adapter._local_request_once_with_runtime_fallback = fake_local_request
        adapter.ADAPTER_API_KEY = ""

    def tearDown(self):
        self.adapter._ensure_local_ready = self.old_ensure_local_ready
        self.adapter._local_request_once_with_runtime_fallback = self.old_local_request_once
        self.adapter._ensure_local_model_info = self.old_model_info
        self.adapter.ADAPTER_API_KEY = self.old_adapter_api_key
        self.adapter.MODEL_REGISTRY = self.old_model_registry
        _remove_stubs()

    async def test_route_text_completion_local_forwards_max_output_tokens(self):
        await self.adapter._route_text_completion(
            requested_model="annoter",
            messages=[{"role": "user", "content": "OK"}],
            max_output_tokens=64,
        )

        self.assertEqual(self.calls[-1]["json"]["max_tokens"], 64)

    async def test_route_text_completion_local_registry_forwards_max_output_tokens(self):
        self.adapter.MODEL_REGISTRY["unit_registry_local"] = {
            "backend": "gpt4all",
            "model": "UnitPhysicalModel",
        }

        await self.adapter._route_text_completion(
            requested_model="unit_registry_local",
            messages=[{"role": "user", "content": "OK"}],
            max_output_tokens=64,
        )

        payload = self.calls[-1]["json"]
        self.assertEqual(payload["model"], "UnitPhysicalModel")
        self.assertEqual(payload["max_tokens"], 64)

    async def test_stream_generator_gpt4all_forwards_max_tokens_override(self):
        self.adapter.MODEL_REGISTRY["unit_stream_local"] = {
            "backend": "gpt4all",
            "model": "UnitStreamModel",
        }

        events = [
            event
            async for event in self.adapter._chat_completions_stream_generator(
                requested_model="unit_stream_local",
                canonical_model="unit_stream_local",
                messages=[{"role": "user", "content": "OK"}],
                max_tokens_override=64,
            )
        ]

        self.assertTrue(events)
        self.assertEqual(self.calls[-1]["json"]["max_tokens"], 64)

    async def test_chat_completions_gpt4all_registry_forwards_max_tokens_override(self):
        self.adapter.MODEL_REGISTRY["unit_chat_local"] = {
            "backend": "gpt4all",
            "model": "UnitChatModel",
        }

        await self.adapter.chat_completions(
            _chat_req(model="unit_chat_local", max_tokens=64),
            _Request(),
            authorization=None,
        )

        payload = self.calls[-1]["json"]
        self.assertEqual(payload["model"], "UnitChatModel")
        self.assertEqual(payload["max_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
