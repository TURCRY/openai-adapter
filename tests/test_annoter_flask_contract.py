"""Contrat adapter -> backend Flask pour le chemin legacy local (ex: model="annoter").

Couvre le payload envoye a /chat_orchestre :
  - mapping annoter -> Qwen_2_5_14B
  - retrieval_top_k canonique (et retrait des alias top_k / memory_top_k)
  - stop : omission quand absent ou None, transmission si explicite
  - temperature : 0.0 reste 0.0
  - max_tokens : transmis si demande, non injecte sinon
  - memory_append : contenu exact, sans fuite de contenu systeme
"""

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
    def __init__(self, role, content, **extra):
        self.role = role
        self.content = content
        self.extra = extra

    def model_dump(self, **kwargs):
        data = {"role": self.role, "content": self.content, **self.extra}
        if kwargs.get("exclude_none"):
            data = {k: v for k, v in data.items() if v is not None or k == "content"}
        return data


class _Request:
    headers = {}


def _chat_req(**overrides):
    """Objet requete minimaliste reproduisant ChatReq cote adapter."""
    data = {
        "model": "annoter",
        "messages": [_Message("user", "Reponds uniquement : OK")],
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


class AnnoterFlaskContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _install_stubs()
        import adapter

        self.adapter = adapter
        self.calls = []

        self.old_ensure_local_ready = adapter._ensure_local_ready
        self.old_local_request_once = adapter._local_request_once_with_runtime_fallback
        self.old_model_info = adapter._ensure_local_model_info
        self.old_adapter_api_key = adapter.ADAPTER_API_KEY

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
        _remove_stubs()

    async def _post(self, **overrides):
        req = _chat_req(**overrides)
        await self.adapter.chat_completions(req, _Request(), authorization=None)
        return self.calls[-1]

    # ---------------------------------------------------------------- 1
    async def test_annoter_maps_to_qwen_and_chat_orchestre(self):
        call = await self._post()
        payload = call["json"]

        self.assertEqual(call["path"], "/chat_orchestre")
        self.assertEqual(payload["model"], "Qwen_2_5_14B")
        self.assertEqual(payload["model_name"], "Qwen_2_5_14B")

    # ---------------------------------------------------------------- 2
    async def test_retrieval_top_k_default_is_4(self):
        payload = (await self._post())["json"]
        self.assertEqual(payload["retrieval_top_k"], 4)

    # ---------------------------------------------------------------- 3
    async def test_metadata_retrieval_top_k_wins(self):
        payload = (await self._post(
            metadata={"retrieval_top_k": 12, "memory_top_k": 7, "top_k": 3}
        ))["json"]
        self.assertEqual(payload["retrieval_top_k"], 12)

    # ---------------------------------------------------------------- 4
    async def test_metadata_memory_top_k_still_accepted_as_input_alias(self):
        payload = (await self._post(metadata={"memory_top_k": 9, "top_k": 3}))["json"]
        self.assertEqual(payload["retrieval_top_k"], 9)

    # ---------------------------------------------------------------- 5
    async def test_legacy_metadata_top_k_still_accepted_as_input_alias(self):
        payload = (await self._post(metadata={"top_k": 6}))["json"]
        self.assertEqual(payload["retrieval_top_k"], 6)

    # ---------------------------------------------------------------- 6
    async def test_payload_has_retrieval_top_k_and_no_ambiguous_aliases(self):
        payload = (await self._post(metadata={"top_k": 5}))["json"]

        self.assertIn("retrieval_top_k", payload)
        self.assertNotIn("memory_top_k", payload)
        self.assertNotIn("top_k", payload)

    # ---------------------------------------------------------------- 7
    async def test_stop_absent_is_omitted_from_payload(self):
        payload = (await self._post())["json"]
        self.assertNotIn("stop", payload)

    # ---------------------------------------------------------------- 8
    async def test_stop_none_in_metadata_is_removed_from_payload(self):
        payload = (await self._post(metadata={"stop": None}))["json"]
        self.assertNotIn("stop", payload)

    # ---------------------------------------------------------------- 9
    async def test_explicit_stop_is_forwarded_unchanged(self):
        payload = (await self._post(stop=["\n\n", "###"]))["json"]
        self.assertEqual(payload["stop"], ["\n\n", "###"])

    async def test_explicit_stop_string_is_forwarded_unchanged(self):
        payload = (await self._post(stop="</s>"))["json"]
        self.assertEqual(payload["stop"], "</s>")

    # --------------------------------------------------------------- 10
    async def test_temperature_zero_is_preserved(self):
        payload = (await self._post(temperature=0.0))["json"]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertIsInstance(payload["temperature"], float)

    # --------------------------------------------------------------- 11
    async def test_temperature_absent_falls_back_to_default(self):
        payload = (await self._post(temperature=None))["json"]
        self.assertEqual(payload["temperature"], 0.4)

    async def test_temperature_nonzero_is_preserved(self):
        payload = (await self._post(temperature=0.85))["json"]
        self.assertEqual(payload["temperature"], 0.85)

    # --------------------------------------------------------------- 12
    async def test_max_tokens_explicit_is_forwarded(self):
        payload = (await self._post(max_tokens=256))["json"]
        self.assertEqual(payload["max_tokens"], 256)

    async def test_max_completion_tokens_takes_priority_over_max_tokens(self):
        payload = (await self._post(max_tokens=256, max_completion_tokens=512))["json"]
        self.assertEqual(payload["max_tokens"], 512)

    # --------------------------------------------------------------- 13
    async def test_max_tokens_absent_is_not_injected(self):
        payload = (await self._post())["json"]
        self.assertNotIn("max_tokens", payload)

    # --------------------------------------------------------------- 14
    async def test_memory_append_exact_content_for_ok_test(self):
        payload = (await self._post(
            messages=[_Message("user", "Reponds uniquement : OK")]
        ))["json"]
        self.assertEqual(payload["memory_append"], "USER: Reponds uniquement : OK")

    async def test_memory_append_excludes_system_messages(self):
        payload = (await self._post(messages=[
            _Message("system", "SECRET_SYSTEM_PROMPT"),
            _Message("user", "Reponds uniquement : OK"),
        ]))["json"]

        self.assertEqual(payload["memory_append"], "USER: Reponds uniquement : OK")
        self.assertNotIn("SECRET_SYSTEM_PROMPT", payload["memory_append"])

    # ------------------------------------------------- invariants memoire
    async def test_conversation_fields_are_present(self):
        payload = (await self._post())["json"]

        self.assertIn("conversation_id", payload)
        self.assertEqual(payload["memory_id"], payload["conversation_id"])
        self.assertEqual(payload["memory_turns"], 6)
        self.assertIn("app_id", payload)

    async def test_no_sampling_params_leak_into_payload(self):
        payload = (await self._post())["json"]

        for key in ("top_p", "repeat_penalty", "presence_penalty", "frequency_penalty"):
            self.assertNotIn(key, payload)


if __name__ == "__main__":
    unittest.main()
