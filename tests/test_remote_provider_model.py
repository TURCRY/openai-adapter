import asyncio
import json
import os
import sys
import types
import unittest
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class HTTPException(Exception):
    def __init__(self, status_code=None, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail



class _DummyStreamingResponse:
    def __init__(self, content, media_type=None, headers=None, **kwargs):
        self.body_iterator = content
        self.media_type = media_type
        self.headers = headers or {}
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

    middleware_stub = types.ModuleType("fastapi.middleware")
    cors_stub = types.ModuleType("fastapi.middleware.cors")
    cors_stub.CORSMiddleware = object
    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.Response = object
    responses_stub.JSONResponse = object
    responses_stub.StreamingResponse = _DummyStreamingResponse
    sys.modules["fastapi.middleware"] = middleware_stub
    sys.modules["fastapi.middleware.cors"] = cors_stub
    sys.modules["fastapi.responses"] = responses_stub

    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.BaseModel = _BaseModel
    sys.modules["pydantic"] = pydantic_stub


def _remove_stubs():
    for name in [
        "adapter",
        "fastapi",
        "fastapi.middleware",
        "fastapi.middleware.cors",
        "fastapi.responses",
        "pydantic",
    ]:
        sys.modules.pop(name, None)



class FakeStreamResponse:
    def __init__(self, status_code=200, lines=None, body=""):
        self.status_code = status_code
        self.lines = list(lines or [])
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for line in self.lines:
            if isinstance(line, BaseException):
                raise line
            yield line

    async def aread(self):
        return self.body.encode("utf-8")
class FakeAsyncClient:
    instances = []
    post_outcomes = []
    stream_outcomes = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        self.stream_calls = []
        FakeAsyncClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json=None, headers=None):
        self.stream_calls.append({"method": method, "url": url, "json": json, "headers": headers})
        if FakeAsyncClient.stream_outcomes:
            return FakeAsyncClient.stream_outcomes.pop(0)
        return FakeStreamResponse(lines=[
            'data: {"choices":[{"delta":{"content":"hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens_details":{"reasoning_tokens":4}}}',
            "data: [DONE]",
        ])

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if FakeAsyncClient.post_outcomes:
            outcome = FakeAsyncClient.post_outcomes.pop(0)
            if isinstance(outcome, httpx.Response):
                return outcome
            return httpx.Response(200, json=outcome, request=httpx.Request("POST", url))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            request=httpx.Request("POST", url),
        )


class _Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def model_dump(self):
        return {"role": self.role, "content": self.content}


class _Request:
    headers = {}


async def _collect_stream(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return "".join(chunks)


def _parse_sse_events(raw):
    events = []
    for block in raw.strip().split("\n\n"):
        event_name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event_name:
            events.append((event_name, data))
    return events
def _responses_req(**overrides):
    data = {
        "model": "deepseek/deepseek-v4-flash",
        "input": "hello",
        "instructions": None,
        "stream": False,
        "max_output_tokens": None,
        "tools": None,
        "tool_choice": None,
        "previous_response_id": None,
        "metadata": {},
        "reasoning": None,
        "parallel_tool_calls": None,
    }
    data.update(overrides)
    return types.SimpleNamespace(**data)



def _tool(name="get_test_value", strict=True):
    return {
        "type": "function",
        "name": name,
        "description": "Retourne une valeur de test",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        "strict": strict,
    }


def _large_codex_tool(name="codex_tool_0", property_count=20, filler_size=512):
    filler = "x" * filler_size
    return {
        "type": "function",
        "name": name,
        "description": "Codex tool summary",
        "parameters": {
            "type": "object",
            "properties": {
                f"arg_{i}": {
                    "type": "string",
                    "description": filler,
                }
                for i in range(property_count)
            },
            "required": ["arg_0"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _namespace_tool(name="mcp__demo", sub_names=("read_file", "list_files")):
    return {
        "type": "namespace",
        "name": name,
        "description": "Grouped Codex tools",
        "tools": [_tool(sub_name) for sub_name in sub_names],
    }


def _web_search_tool():
    return {
        "type": "web_search",
        "name": "web_search",
        "description": "Search the web",
    }


def _tool_call_response(*calls, content=None, finish_reason="tool_calls", usage=None):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content, "tool_calls": list(calls)},
            "finish_reason": finish_reason,
        }],
        "usage": usage or {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _chat_tool_call(call_id="call_123", name="get_test_value", arguments="{\"key\":\"demo\"}"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}

class RemoteProviderModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _install_stubs()
        import adapter

        self.adapter = adapter
        self.old_remote_conf = adapter._REMOTE_CONF
        self.old_async_client = adapter.httpx.AsyncClient
        self.old_local_chat = adapter._local_chat
        self.old_adapter_api_key = adapter.ADAPTER_API_KEY
        self.old_local_models = list(adapter.LOCAL_MODELS)
        self.old_env = {
            "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY"),
            "PLAIN_API_KEY": os.environ.get("PLAIN_API_KEY"),
        }
        os.environ["DEEPSEEK_API_KEY"] = "deepseek-test-key"
        os.environ["PLAIN_API_KEY"] = "plain-test-key"
        FakeAsyncClient.instances = []
        FakeAsyncClient.post_outcomes = []
        FakeAsyncClient.stream_outcomes = []
        adapter.httpx.AsyncClient = FakeAsyncClient
        adapter.ADAPTER_API_KEY = ""
        adapter.LOCAL_MODELS = ["local-test-model"]
        adapter._REMOTE_CONF = {
            "defaults": {
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "PLAIN_API_KEY",
                "use_responses_api": False,
                "force_chat": True,
            },
            "models": {
                "deepseek/deepseek-v4-flash": {
                    "base_url": "https://api.deepseek.com/",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "provider_model": "deepseek-v4-flash",
                    "use_responses_api": False,
                    "force_chat": True,
                    "supports_tools": True,
                    "supports_parallel_tool_calls": True,
                    "supports_strict_tools": True,
                },
                "plain-remote": {
                    "base_url": "https://plain.example/v1",
                    "api_key_env": "PLAIN_API_KEY",
                    "use_responses_api": False,
                    "force_chat": True,
                },
                "loose-tools": {
                    "base_url": "https://plain.example/v1",
                    "api_key_env": "PLAIN_API_KEY",
                    "use_responses_api": False,
                    "force_chat": True,
                    "supports_tools": True,
                    "supports_strict_tools": False,
                },
                "minimaxai/minimax-m3": {
                    "base_url": "https://integrate.api.nvidia.com/v1",
                    "api_key_env": "PLAIN_API_KEY",
                    "use_responses_api": False,
                    "force_chat": True,
                    "supports_tools": True,
                    "supports_parallel_tool_calls": True,
                    "supports_strict_tools": False,
                    "max_tokens": 8192,
                },
            },
        }

    def tearDown(self):
        adapter = self.adapter
        adapter._REMOTE_CONF = self.old_remote_conf
        adapter.httpx.AsyncClient = self.old_async_client
        adapter._local_chat = self.old_local_chat
        adapter.ADAPTER_API_KEY = self.old_adapter_api_key
        adapter.LOCAL_MODELS = self.old_local_models
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _remove_stubs()

    async def test_deepseek_provider_model_is_sent_to_remote_payload(self):
        await self.adapter._remote_chat(
            messages=[{"role": "user", "content": "hello"}],
            model="deepseek/deepseek-v4-flash",
            temperature=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["model"], "deepseek-v4-flash")

    async def test_model_without_provider_model_keeps_requested_model_in_payload(self):
        await self.adapter._remote_chat(
            messages=[{"role": "user", "content": "hello"}],
            model="plain-remote",
            temperature=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["model"], "plain-remote")

    async def test_final_openai_compatible_response_keeps_public_model_id(self):
        req = types.SimpleNamespace(
            model="deepseek/deepseek-v4-flash",
            messages=[_Message("user", "hello")],
            temperature=None,
            stream=False,
            metadata={},
            response_format=None,
        )

        resp = await self.adapter.chat_completions(req, _Request(), authorization=None)

        self.assertEqual(resp.model, "deepseek/deepseek-v4-flash")
        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["model"], "deepseek-v4-flash")

    async def test_streaming_request_uses_provider_model_in_remote_payload(self):
        req = types.SimpleNamespace(
            model="deepseek/deepseek-v4-flash",
            messages=[_Message("user", "hello")],
            temperature=None,
            stream=True,
            metadata={},
            response_format=None,
        )

        resp = await self.adapter.chat_completions(req, _Request(), authorization=None)

        self.assertEqual(resp.model, "deepseek/deepseek-v4-flash")
        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["model"], "deepseek-v4-flash")

    async def test_deepseek_native_responses_uses_responses_endpoint_and_provider_model(self):
        self.adapter._REMOTE_CONF["models"]["deepseek/deepseek-v4-flash"].update({
            "use_responses_api": True,
            "force_chat": False,
            "native_responses_provider": True,
            "supports_tool_choice_in_thinking": True,
        })
        FakeAsyncClient.post_outcomes = [{
            "id": "resp_native",
            "object": "response",
            "status": "completed",
            "model": "deepseek-v4-flash",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }]

        resp = await self.adapter.responses_create(
            _responses_req(
                input=[
                    {"role": "user", "content": "hello"},
                    {"type": "reasoning", "summary": []},
                    {"type": "function_call", "call_id": "call_1", "name": "get_test_value", "arguments": "{\"key\":\"demo\"}"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "VALUE_42"},
                ],
                instructions="be concise",
                tools=[_tool(strict=False)],
                tool_choice={"type": "function", "name": "get_test_value"},
                reasoning={"effort": "low"},
                parallel_tool_calls=True,
                max_output_tokens=123,
            ),
            authorization=None,
        )

        call = FakeAsyncClient.instances[-1].calls[-1]
        payload = call["json"]
        self.assertEqual(call["url"], "https://api.deepseek.com/responses")
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["input"][1]["type"], "reasoning")
        self.assertEqual(payload["input"][2]["type"], "function_call")
        self.assertEqual(payload["input"][3]["type"], "function_call_output")
        self.assertNotIn("messages", payload)
        self.assertEqual(payload["tools"][0]["name"], "get_test_value")
        self.assertNotIn("function", payload["tools"][0])
        self.assertEqual(payload["tool_choice"], {"type": "function", "name": "get_test_value"})
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertTrue(payload["parallel_tool_calls"])
        self.assertEqual(payload["max_output_tokens"], 123)
        self.assertEqual(resp["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(resp["output_text"], "ok")

    async def test_deepseek_native_responses_omits_tool_choice_when_thinking_disallows_it(self):
        self.adapter._REMOTE_CONF["models"]["deepseek/deepseek-v4-flash"].update({
            "use_responses_api": True,
            "force_chat": False,
            "native_responses_provider": True,
            "thinking_default": "enabled",
            "supports_tool_choice_in_thinking": False,
        })
        FakeAsyncClient.post_outcomes = [{
            "id": "resp_native",
            "object": "response",
            "status": "completed",
            "model": "deepseek-v4-flash",
            "output": [{"type": "function_call", "call_id": "call_1", "name": "get_test_value", "arguments": "{\"key\":\"demo\"}"}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }]

        await self.adapter.responses_create(
            _responses_req(
                input='Appelle obligatoirement get_test_value avec key="demo".',
                tools=[_tool(strict=False)],
                tool_choice="required",
                reasoning={"effort": "low"},
                max_output_tokens=128,
            ),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
    async def test_deepseek_native_responses_stream_replays_native_sse_events(self):
        self.adapter._REMOTE_CONF["models"]["deepseek/deepseek-v4-flash"].update({
            "use_responses_api": True,
            "force_chat": False,
            "native_responses_provider": True,
        })
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=[
            "event: response.created",
            "data: " + json.dumps({"type": "response.created", "response": {"id": "resp_1", "object": "response", "model": "deepseek-v4-flash", "status": "in_progress"}}),
            "",
            "event: response.reasoning_text.delta",
            "data: " + json.dumps({"type": "response.reasoning_text.delta", "delta": "hidden"}),
            "",
            "event: response.output_item.added",
            "data: " + json.dumps({"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "call_1", "name": "get_test_value"}}),
            "",
            "event: response.function_call_arguments.delta",
            "data: " + json.dumps({"type": "response.function_call_arguments.delta", "delta": "{\"key\":\"demo\"}"}),
            "",
            "event: response.output_text.delta",
            "data: " + json.dumps({"type": "response.output_text.delta", "delta": "VALUE_42"}),
            "",
            "event: response.completed",
            "data: " + json.dumps({"type": "response.completed", "response": {"id": "resp_1", "object": "response", "model": "deepseek-v4-flash", "status": "completed"}}),
            "",
        ])]

        resp = await self.adapter.responses_create(
            _responses_req(stream=True, tools=[_tool()], reasoning={"effort": "low"}),
            authorization=None,
        )
        raw = await _collect_stream(resp)
        events = _parse_sse_events(raw)
        names = [name for name, _ in events]
        self.assertIn("response.created", names)
        self.assertIn("response.reasoning_text.delta", names)
        self.assertIn("response.function_call_arguments.delta", names)
        self.assertIn("response.output_text.delta", names)
        self.assertIn("response.completed", names)
        payload = FakeAsyncClient.instances[-1].stream_calls[-1]["json"]
        self.assertEqual(FakeAsyncClient.instances[-1].stream_calls[-1]["url"], "https://api.deepseek.com/responses")
        self.assertTrue(payload["stream"])
        self.assertIn("tools", payload)
        completed = [data for name, data in events if name == "response.completed"][0]
        self.assertEqual(completed["response"]["model"], "deepseek/deepseek-v4-flash")

    async def test_deepseek_native_stream_does_not_parse_chat_choices_delta(self):
        self.adapter._REMOTE_CONF["models"]["deepseek/deepseek-v4-flash"].update({
            "use_responses_api": True,
            "force_chat": False,
            "native_responses_provider": True,
        })
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=[
            "event: response.output_text.delta",
            "data: " + json.dumps({"choices": [{"delta": {"content": "chat-style"}}], "type": "response.output_text.delta", "delta": "native"}),
            "",
        ])]

        resp = await self.adapter.responses_create(_responses_req(stream=True), authorization=None)
        events = _parse_sse_events(await _collect_stream(resp))
        self.assertEqual(events[0][0], "response.output_text.delta")
        self.assertEqual(events[0][1]["delta"], "native")
        self.assertIn("choices", events[0][1])
    async def test_responses_string_input_returns_minimal_response_shape(self):
        resp = await self.adapter.responses_create(
            _responses_req(input="hello"),
            authorization=None,
        )

        self.assertEqual(resp["status"], "completed")
        self.assertEqual(resp["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(resp["output"][0]["content"][0]["type"], "output_text")
        self.assertEqual(resp["output_text"], "ok")
        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["model"], "deepseek-v4-flash")

    async def test_responses_list_input_is_converted_to_messages(self):
        await self.adapter.responses_create(
            _responses_req(input=[{"role": "user", "content": "from list"}]),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["messages"], [{"role": "user", "content": "from list"}])

    async def test_responses_instructions_are_converted_to_system_message(self):
        await self.adapter.responses_create(
            _responses_req(input="hello", instructions="be concise"),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "be concise"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "hello"})

    async def test_responses_max_output_tokens_is_forwarded_to_remote_payload(self):
        await self.adapter.responses_create(
            _responses_req(max_output_tokens=123),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["max_tokens"], 123)

    async def test_responses_stream_true_is_accepted_as_sse(self):
        resp = await self.adapter.responses_create(
            _responses_req(stream=True),
            authorization=None,
        )

        self.assertEqual(resp.media_type, "text/event-stream")

    async def test_responses_tool_is_forwarded_to_chat_completions(self):
        await self.adapter.responses_create(
            _responses_req(tools=[_tool()]),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["function"]["name"], "get_test_value")
        self.assertTrue(payload["tools"][0]["function"]["strict"])

    async def test_responses_function_call_output_is_converted_to_tool_message(self):
        await self.adapter.responses_create(
            _responses_req(input=[{"type": "function_call_output", "call_id": "call_123", "output": "done"}]),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["messages"], [{"role": "tool", "tool_call_id": "call_123", "content": "done"}])

    async def test_responses_auth_requires_bearer_when_adapter_key_is_set(self):
        self.adapter.ADAPTER_API_KEY = "secret"

        with self.assertRaises(HTTPException) as ctx:
            await self.adapter.responses_create(
                _responses_req(),
                authorization=None,
            )

        self.assertEqual(ctx.exception.status_code, 401)

        resp = await self.adapter.responses_create(
            _responses_req(),
            authorization="Bearer secret",
        )
        self.assertEqual(resp["status"], "completed")

    async def test_responses_routes_to_mocked_local_model(self):
        calls = []

        async def fake_local_chat(prompt, route_hint=None, temperature=None, meta=None, **kwargs):
            calls.append({
                "prompt": prompt,
                "route_hint": route_hint,
                "messages": kwargs.get("messages"),
            })
            return "local ok"

        self.adapter._local_chat = fake_local_chat
        resp = await self.adapter.responses_create(
            _responses_req(model="local-test-model", input="local hello"),
            authorization=None,
        )

        self.assertEqual(resp["output_text"], "local ok")
        self.assertEqual(calls[-1]["route_hint"], "local-test-model")
        self.assertEqual(calls[-1]["messages"], [{"role": "user", "content": "local hello"}])
    async def test_responses_stream_true_returns_sse_response(self):
        resp = await self.adapter.responses_create(
            _responses_req(stream=True),
            authorization=None,
        )

        self.assertEqual(resp.media_type, "text/event-stream")
        self.assertEqual(resp.headers["Cache-Control"], "no-cache")
        self.assertEqual(resp.headers["Connection"], "keep-alive")

    async def test_responses_stream_event_order_and_completed_payload(self):
        resp = await self.adapter.responses_create(
            _responses_req(stream=True),
            authorization=None,
        )
        events = _parse_sse_events(await _collect_stream(resp))
        names = [name for name, _ in events]

        self.assertEqual(names[0:4], [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
        ])
        self.assertIn("response.output_text.delta", names)
        self.assertEqual(names[-4:], [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ])
        completed = events[-1][1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(completed["output_text"], "hello")

    async def test_responses_stream_uses_provider_model_upstream(self):
        resp = await self.adapter.responses_create(
            _responses_req(stream=True),
            authorization=None,
        )
        await _collect_stream(resp)

        payload = FakeAsyncClient.instances[-1].stream_calls[-1]["json"]
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["model"], "deepseek-v4-flash")

    async def test_responses_stream_converts_chat_completion_chunks_and_usage(self):
        resp = await self.adapter.responses_create(
            _responses_req(stream=True),
            authorization=None,
        )
        events = _parse_sse_events(await _collect_stream(resp))
        deltas = [data["delta"] for name, data in events if name == "response.output_text.delta"]
        completed = events[-1][1]["response"]

        self.assertEqual(deltas, ["hel", "lo"])
        self.assertEqual(completed["usage"]["input_tokens"], 3)
        self.assertEqual(completed["usage"]["output_tokens"], 2)
        self.assertEqual(completed["usage"]["total_tokens"], 5)
        self.assertEqual(completed["usage"]["cached_tokens"], 1)
        self.assertEqual(completed["usage"]["reasoning_tokens"], 4)

    async def test_responses_stream_non_stream_backend_is_adapted_as_single_delta(self):
        self.adapter._REMOTE_CONF["models"]["plain-remote"]["supports_stream"] = False
        resp = await self.adapter.responses_create(
            _responses_req(model="plain-remote", stream=True),
            authorization=None,
        )
        events = _parse_sse_events(await _collect_stream(resp))
        deltas = [data["delta"] for name, data in events if name == "response.output_text.delta"]

        self.assertEqual(deltas, ["ok"])
        self.assertEqual(FakeAsyncClient.instances[-1].calls[-1]["json"]["model"], "plain-remote")

    async def test_responses_stream_upstream_error_becomes_error_event(self):
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(status_code=400, body="bad upstream")]
        resp = await self.adapter.responses_create(
            _responses_req(stream=True),
            authorization=None,
        )
        events = _parse_sse_events(await _collect_stream(resp))

        self.assertEqual(events[-1][0], "error")
        self.assertEqual(events[-1][1]["error"]["status_code"], 400)
        self.assertIn("bad upstream", str(events[-1][1]["error"]["detail"]))

    async def test_responses_stream_error_after_start_becomes_error_event(self):
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=[
            'data: {"choices":[{"delta":{"content":"partial"}}]}',
            RuntimeError("stream broke"),
        ])]
        resp = await self.adapter.responses_create(
            _responses_req(stream=True),
            authorization=None,
        )
        events = _parse_sse_events(await _collect_stream(resp))
        names = [name for name, _ in events]

        self.assertIn("response.output_text.delta", names)
        self.assertEqual(events[-1][0], "error")
        self.assertIn("stream broke", str(events[-1][1]["error"]))

    async def test_responses_stream_client_cancellation_is_propagated(self):
        async def cancelled_parts(**kwargs):
            yield {"delta": "partial"}
            raise asyncio.CancelledError()

        self.adapter._remote_chat_stream_parts = cancelled_parts
        gen = self.adapter._responses_stream_generator(
            requested_model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
        )

        chunks = []
        with self.assertRaises(asyncio.CancelledError):
            async for chunk in gen:
                chunks.append(chunk)

        self.assertTrue(any("response.output_text.delta" in chunk for chunk in chunks))


    async def test_tool_choice_auto_is_forwarded(self):
        await self.adapter.responses_create(_responses_req(tools=[_tool()], tool_choice="auto"), authorization=None)
        self.assertEqual(FakeAsyncClient.instances[-1].calls[-1]["json"]["tool_choice"], "auto")

    async def test_tool_choice_required_is_forwarded(self):
        await self.adapter.responses_create(_responses_req(tools=[_tool()], tool_choice="required"), authorization=None)
        self.assertEqual(FakeAsyncClient.instances[-1].calls[-1]["json"]["tool_choice"], "required")

    async def test_tool_choice_targeted_function_is_converted(self):
        await self.adapter.responses_create(
            _responses_req(tools=[_tool()], tool_choice={"type": "function", "name": "get_test_value"}),
            authorization=None,
        )
        self.assertEqual(
            FakeAsyncClient.instances[-1].calls[-1]["json"]["tool_choice"],
            {"type": "function", "function": {"name": "get_test_value"}},
        )

    async def test_responses_input_user_only_is_converted_to_chat_message(self):
        await self.adapter.responses_create(
            _responses_req(model="minimaxai/minimax-m3", input=[{"role": "user", "content": "hello"}]),
            authorization=None,
        )

        messages = FakeAsyncClient.instances[-1].calls[-1]["json"]["messages"]
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])

    async def test_responses_input_developer_and_user_maps_developer_to_system_for_chat(self):
        await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                input=[
                    {"role": "developer", "content": "dev rules"},
                    {"role": "user", "content": "hello"},
                ],
            ),
            authorization=None,
        )

        messages = FakeAsyncClient.instances[-1].calls[-1]["json"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual([m["content"] for m in messages], ["dev rules", "hello"])

    async def test_responses_input_system_developer_user_preserves_order_for_chat(self):
        await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                input=[
                    {"role": "system", "content": "sys"},
                    {"role": "developer", "content": "dev"},
                    {"role": "user", "content": "usr"},
                ],
            ),
            authorization=None,
        )

        messages = FakeAsyncClient.instances[-1].calls[-1]["json"]["messages"]
        self.assertEqual(
            [(m["role"], m["content"]) for m in messages],
            [("system", "sys"), ("system", "dev"), ("user", "usr")],
        )

    async def test_responses_input_developer_is_preserved_when_chat_provider_supports_it(self):
        self.adapter._REMOTE_CONF["models"]["minimaxai/minimax-m3"]["supports_developer_role"] = True
        await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                input=[
                    {"role": "developer", "content": "dev"},
                    {"role": "user", "content": "usr"},
                ],
            ),
            authorization=None,
        )

        messages = FakeAsyncClient.instances[-1].calls[-1]["json"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["developer", "user"])

    async def test_responses_input_developer_with_tools(self):
        await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                input=[
                    {"role": "developer", "content": "dev"},
                    {"role": "user", "content": "call tool"},
                ],
                tools=[_tool()],
                tool_choice="required",
            ),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "required")

    async def test_responses_input_developer_with_function_call_output_second_turn(self):
        await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                input=[
                    {"role": "developer", "content": "dev"},
                    {"role": "user", "content": "call tool"},
                    {"type": "function_call", "call_id": "call_dev", "name": "get_test_value", "arguments": "{\"key\":\"demo\"}"},
                    {"type": "function_call_output", "call_id": "call_dev", "output": "VALUE_42"},
                    {"role": "user", "content": "final"},
                ],
                tools=[_tool()],
                tool_choice="none",
            ),
            authorization=None,
        )

        messages = FakeAsyncClient.instances[-1].calls[-1]["json"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call_dev")
        self.assertEqual(messages[3]["tool_call_id"], "call_dev")

    async def test_responses_input_developer_streaming(self):
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=["data: [DONE]"])]
        resp = await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                stream=True,
                input=[
                    {"role": "developer", "content": "dev"},
                    {"role": "user", "content": "hello"},
                ],
            ),
            authorization=None,
        )
        await _collect_stream(resp)

        messages = FakeAsyncClient.instances[-1].stream_calls[-1]["json"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])

    async def test_deepseek_native_responses_preserves_developer_input_item(self):
        self.adapter._REMOTE_CONF["models"]["deepseek/deepseek-v4-flash"].update({
            "use_responses_api": True,
            "force_chat": False,
            "native_responses_provider": True,
        })
        await self.adapter.responses_create(
            _responses_req(
                model="deepseek/deepseek-v4-flash",
                input=[
                    {"role": "developer", "content": "dev"},
                    {"role": "user", "content": "hello"},
                ],
            ),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["input"][0]["role"], "developer")
        self.assertEqual(payload["input"][1]["role"], "user")

    async def test_non_stream_function_call_is_converted_to_response_item(self):
        FakeAsyncClient.post_outcomes = [_tool_call_response(_chat_tool_call())]
        resp = await self.adapter.responses_create(_responses_req(tools=[_tool()]), authorization=None)

        self.assertEqual(resp["output_text"], "")
        self.assertEqual(len(resp["output"]), 1)
        item = resp["output"][0]
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["call_id"], "call_123")
        self.assertEqual(item["name"], "get_test_value")
        self.assertEqual(item["arguments"], "{\"key\":\"demo\"}")

    async def test_non_stream_multiple_tool_calls_preserve_order(self):
        FakeAsyncClient.post_outcomes = [_tool_call_response(
            _chat_tool_call("call_1", arguments="{\"key\":\"a\"}"),
            _chat_tool_call("call_2", arguments="{\"key\":\"b\"}"),
        )]
        resp = await self.adapter.responses_create(_responses_req(tools=[_tool()]), authorization=None)

        self.assertEqual([item["call_id"] for item in resp["output"]], ["call_1", "call_2"])
        self.assertEqual(resp["output_text"], "")

    async def test_non_stream_text_and_tool_call_are_both_returned(self):
        FakeAsyncClient.post_outcomes = [_tool_call_response(_chat_tool_call(), content="I will call it")]
        resp = await self.adapter.responses_create(_responses_req(tools=[_tool()]), authorization=None)

        self.assertEqual(resp["output"][0]["type"], "message")
        self.assertEqual(resp["output"][1]["type"], "function_call")
        self.assertEqual(resp["output_text"], "I will call it")

    async def test_function_call_output_object_is_serialized(self):
        await self.adapter.responses_create(
            _responses_req(input=[{"type": "function_call_output", "call_id": "call_123", "output": {"value": 42}}]),
            authorization=None,
        )
        self.assertEqual(FakeAsyncClient.instances[-1].calls[-1]["json"]["messages"][0]["content"], "{\"value\":42}")

    async def test_assistant_function_call_history_is_reconstructed(self):
        await self.adapter.responses_create(
            _responses_req(input=[
                {"role": "user", "content": "demo"},
                {"type": "function_call", "call_id": "call_123", "name": "get_test_value", "arguments": "{\"key\":\"demo\"}"},
                {"type": "function_call_output", "call_id": "call_123", "output": "VALUE_42"},
            ]),
            authorization=None,
        )
        messages = FakeAsyncClient.instances[-1].calls[-1]["json"]["messages"]
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(messages[2]["role"], "tool")

    async def test_model_without_tools_support_fails_without_fallback(self):
        with self.assertRaises(HTTPException) as ctx:
            await self.adapter.responses_create(_responses_req(model="plain-remote", tools=[_tool()]), authorization=None)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("supports_tools", str(ctx.exception.detail))
        self.assertEqual(FakeAsyncClient.instances, [])

    async def test_provider_model_is_applied_when_tools_are_sent(self):
        await self.adapter.responses_create(_responses_req(tools=[_tool()]), authorization=None)
        self.assertEqual(FakeAsyncClient.instances[-1].calls[-1]["json"]["model"], "deepseek-v4-flash")

    async def test_strict_is_removed_only_when_capability_is_false(self):
        await self.adapter.responses_create(_responses_req(model="loose-tools", tools=[_tool()]), authorization=None)
        tool_payload = FakeAsyncClient.instances[-1].calls[-1]["json"]["tools"][0]["function"]
        self.assertNotIn("strict", tool_payload)
        self.assertIn("parameters", tool_payload)

    async def test_function_tool_is_converted_for_chat_completions(self):
        await self.adapter.responses_create(_responses_req(model="minimaxai/minimax-m3", tools=[_tool("get_test_value")]), authorization=None)

        tool_payload = FakeAsyncClient.instances[-1].calls[-1]["json"]["tools"][0]
        self.assertEqual(tool_payload["type"], "function")
        self.assertEqual(tool_payload["function"]["name"], "get_test_value")

    async def test_namespace_tool_is_flattened_for_chat_completions(self):
        await self.adapter.responses_create(_responses_req(model="minimaxai/minimax-m3", tools=[_namespace_tool()]), authorization=None)

        tool_payloads = FakeAsyncClient.instances[-1].calls[-1]["json"]["tools"]
        names = [tool["function"]["name"] for tool in tool_payloads]
        self.assertEqual(names, ["mcp__demo__read_file", "mcp__demo__list_files"])
        self.assertIn("parameters", tool_payloads[0]["function"])

    async def test_namespace_flattening_rejects_name_collisions(self):
        tools = [_tool("mcp__demo__read_file"), _namespace_tool("mcp__demo", ("read_file",))]
        with self.assertRaises(HTTPException) as ctx:
            await self.adapter.responses_create(_responses_req(model="minimaxai/minimax-m3", tools=tools), authorization=None)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("collision", str(ctx.exception.detail))
        self.assertEqual(FakeAsyncClient.instances, [])

    async def test_namespace_provider_tool_call_is_restored_for_codex_responses(self):
        FakeAsyncClient.post_outcomes = [
            _tool_call_response(
                _chat_tool_call(
                    call_id="call_ns_123",
                    name="mcp__demo__read_file",
                    arguments="{\"path\":\"test.txt\"}",
                )
            )
        ]

        resp = await self.adapter.responses_create(
            _responses_req(model="minimaxai/minimax-m3", tools=[_namespace_tool("mcp__demo", ("read_file",))]),
            authorization=None,
        )

        item = resp["output"][0]
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["namespace"], "mcp__demo")
        self.assertEqual(item["name"], "read_file")
        self.assertEqual(item["call_id"], "call_ns_123")
        self.assertEqual(item["arguments"], "{\"path\":\"test.txt\"}")

    async def test_streamed_namespace_provider_tool_call_is_restored_for_codex_responses(self):
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=[
            "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_ns_stream", "function": {"name": "mcp__demo__read_file", "arguments": "{\"path\":"}}]}}]}),
            "",
            "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"test.txt\"}"}}]}}]}),
            "",
            "data: [DONE]",
            "",
        ])]

        resp = await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                stream=True,
                tools=[_namespace_tool("mcp__demo", ("read_file",))],
            ),
            authorization=None,
        )
        events = _parse_sse_events(await _collect_stream(resp))
        completed = [data for name, data in events if name == "response.completed"][0]
        item = completed["response"]["output"][0]
        self.assertEqual(item["namespace"], "mcp__demo")
        self.assertEqual(item["name"], "read_file")
        self.assertEqual(item["call_id"], "call_ns_stream")
        self.assertEqual(item["arguments"], "{\"path\":\"test.txt\"}")

    async def test_function_and_hosted_web_search_filters_only_hosted_tool(self):
        await self.adapter.responses_create(
            _responses_req(model="minimaxai/minimax-m3", tools=[_tool("direct_fn"), _web_search_tool()]),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual([tool["function"]["name"] for tool in payload["tools"]], ["direct_fn"])

    async def test_namespace_and_hosted_web_search_flattens_namespace_and_filters_web_search(self):
        await self.adapter.responses_create(
            _responses_req(model="minimaxai/minimax-m3", tools=[_namespace_tool("mcp__demo", ("read_file",)), _web_search_tool()]),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual([tool["function"]["name"] for tool in payload["tools"]], ["mcp__demo__read_file"])

    async def test_mcp_search_web_function_is_kept_when_hosted_web_search_is_filtered(self):
        await self.adapter.responses_create(
            _responses_req(model="minimaxai/minimax-m3", tools=[_tool("search_web"), _web_search_tool()]),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual([tool["function"]["name"] for tool in payload["tools"]], ["search_web"])

    async def test_explicit_hosted_web_search_tool_choice_is_rejected_before_provider_call(self):
        with self.assertRaises(HTTPException) as ctx:
            await self.adapter.responses_create(
                _responses_req(
                    model="minimaxai/minimax-m3",
                    tools=[_tool("search_web"), _web_search_tool()],
                    tool_choice={"type": "web_search"},
                ),
                authorization=None,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("explicitly requires hosted web_search", str(ctx.exception.detail))
        self.assertEqual(FakeAsyncClient.instances, [])

    async def test_deepseek_native_responses_preserves_namespace_and_web_search_tools(self):
        self.adapter._REMOTE_CONF["models"]["deepseek/deepseek-v4-flash"].update({
            "use_responses_api": True,
            "force_chat": False,
            "native_responses_provider": True,
        })
        await self.adapter.responses_create(
            _responses_req(tools=[_namespace_tool(), _web_search_tool()]),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["tools"][0]["type"], "namespace")
        self.assertEqual(payload["tools"][0]["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][1]["type"], "web_search")

    async def test_too_many_tools_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            await self.adapter.responses_create(_responses_req(tools=[_tool(f"tool_{i}") for i in range(self.adapter.MAX_TOOLS + 1)]), authorization=None)
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_default_tool_count_limit_allows_codex_tool_sets(self):
        self.assertGreaterEqual(self.adapter.MAX_TOOLS, 64)

    async def test_tool_schema_just_under_limit_is_accepted(self):
        old = self.adapter.MAX_TOOL_SCHEMA_BYTES
        tools = [_large_codex_tool("codex_under_limit", property_count=8, filler_size=256)]
        converted = self.adapter._responses_tools_to_chat_result(
            tools,
            self.adapter._model_cfg("deepseek/deepseek-v4-flash"),
            "deepseek/deepseek-v4-flash",
        ).tools
        self.adapter.MAX_TOOL_SCHEMA_BYTES = self.adapter._json_bytes(converted)
        try:
            await self.adapter.responses_create(_responses_req(tools=tools), authorization=None)
            payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
            self.assertEqual(len(payload["tools"]), 1)
        finally:
            self.adapter.MAX_TOOL_SCHEMA_BYTES = old

    async def test_tool_schema_just_above_limit_is_rejected(self):
        old = self.adapter.MAX_TOOL_SCHEMA_BYTES
        tools = [_large_codex_tool("codex_over_limit", property_count=8, filler_size=256)]
        self.adapter.MAX_TOOL_SCHEMA_BYTES = self.adapter._json_bytes(tools) - 1
        try:
            with self.assertRaises(HTTPException) as ctx:
                await self.adapter.responses_create(_responses_req(tools=tools), authorization=None)
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            self.adapter.MAX_TOOL_SCHEMA_BYTES = old

    async def test_tool_schema_size_limit_is_rejected(self):
        old = self.adapter.MAX_TOOL_SCHEMA_BYTES
        self.adapter.MAX_TOOL_SCHEMA_BYTES = 10
        try:
            with self.assertRaises(HTTPException) as ctx:
                await self.adapter.responses_create(_responses_req(tools=[_tool()]), authorization=None)
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            self.adapter.MAX_TOOL_SCHEMA_BYTES = old

    async def test_codex_sized_tool_set_exceeding_old_limit_is_accepted(self):
        tools = [_large_codex_tool(f"codex_tool_{i}", property_count=12, filler_size=256) for i in range(24)]
        total_size = self.adapter._json_bytes(tools)
        self.assertGreater(total_size, 65536)
        self.assertLess(total_size, self.adapter.MAX_TOOL_SCHEMA_BYTES)

        await self.adapter.responses_create(_responses_req(model="minimaxai/minimax-m3", tools=tools), authorization=None)

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertEqual(payload["model"], "minimaxai/minimax-m3")
        self.assertEqual(len(payload["tools"]), len(tools))
        self.assertTrue(FakeAsyncClient.instances[-1].calls[-1]["url"].startswith("https://integrate.api.nvidia.com/v1"))

    async def test_function_arguments_limit_is_rejected(self):
        old = self.adapter.MAX_FUNCTION_ARGUMENTS_BYTES
        self.adapter.MAX_FUNCTION_ARGUMENTS_BYTES = 8
        FakeAsyncClient.post_outcomes = [_tool_call_response(_chat_tool_call(arguments="{\"key\":\"too-long\"}"))]
        try:
            with self.assertRaises(HTTPException) as ctx:
                await self.adapter.responses_create(_responses_req(tools=[_tool()]), authorization=None)
            self.assertEqual(ctx.exception.status_code, 502)
        finally:
            self.adapter.MAX_FUNCTION_ARGUMENTS_BYTES = old

    async def test_function_output_limit_is_rejected(self):
        old = self.adapter.MAX_FUNCTION_OUTPUT_BYTES
        self.adapter.MAX_FUNCTION_OUTPUT_BYTES = 4
        try:
            with self.assertRaises(HTTPException) as ctx:
                await self.adapter.responses_create(_responses_req(input=[{"type": "function_call_output", "call_id": "call_123", "output": "VALUE_42"}]), authorization=None)
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            self.adapter.MAX_FUNCTION_OUTPUT_BYTES = old

    async def test_finish_reason_length_returns_incomplete(self):
        FakeAsyncClient.post_outcomes = [_tool_call_response(content=None, finish_reason="length", usage={"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4})]
        resp = await self.adapter.responses_create(_responses_req(), authorization=None)

        self.assertEqual(resp["status"], "incomplete")
        self.assertEqual(resp["incomplete_details"]["reason"], "max_output_tokens")
        self.assertEqual(resp["usage"]["input_tokens"], 4)

    async def test_previous_response_id_is_rejected_stateless(self):
        with self.assertRaises(HTTPException) as ctx:
            await self.adapter.responses_create(_responses_req(previous_response_id="resp_old"), authorization=None)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("complete input history", str(ctx.exception.detail))


    async def test_second_turn_final_omits_tools_when_tool_choice_none(self):
        await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                input=[
                    {"role": "user", "content": "Appelle obligatoirement get_test_value avec key=\"demo\"."},
                    {"type": "function_call", "call_id": "call_123", "name": "get_test_value", "arguments": "{\"key\":\"demo\"}"},
                    {"type": "function_call_output", "call_id": "call_123", "output": "VALUE_42"},
                    {"role": "user", "content": "Reponds uniquement avec la valeur retournee par l'outil."},
                ],
                tools=[_tool()],
                tool_choice="none",
                max_output_tokens=999,
            ),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["model"], "minimaxai/minimax-m3")
        self.assertTrue(FakeAsyncClient.instances[-1].calls[-1]["url"].startswith("https://integrate.api.nvidia.com/v1"))

    async def test_second_turn_can_keep_tools_when_new_call_is_explicitly_allowed(self):
        await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                input=[
                    {"role": "user", "content": "demo"},
                    {"type": "function_call", "call_id": "call_123", "name": "get_test_value", "arguments": "{\"key\":\"demo\"}"},
                    {"type": "function_call_output", "call_id": "call_123", "output": "VALUE_42"},
                    {"role": "user", "content": "call again if needed"},
                ],
                tools=[_tool()],
                tool_choice="required",
                max_output_tokens=128,
            ),
            authorization=None,
        )

        payload = FakeAsyncClient.instances[-1].calls[-1]["json"]
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(payload["max_tokens"], 128)

    async def test_second_turn_reconstructed_history_has_call_id_and_no_duplication(self):
        await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                input=[
                    {"role": "user", "content": "first"},
                    {"type": "function_call", "call_id": "call_abc", "name": "get_test_value", "arguments": "{\"key\":\"demo\"}"},
                    {"type": "function_call_output", "call_id": "call_abc", "output": "VALUE_42"},
                    {"role": "user", "content": "final"},
                ],
                tools=[_tool()],
                tool_choice="none",
            ),
            authorization=None,
        )

        messages = FakeAsyncClient.instances[-1].calls[-1]["json"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "user"])
        self.assertEqual(len(messages), 4)
        self.assertIsNone(messages[1]["content"])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_abc")
        self.assertEqual(messages[2]["tool_call_id"], "call_abc")
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], "{\"key\":\"demo\"}")
    async def test_streamed_second_turn_final_omits_tools_when_tool_choice_none(self):
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=["data: [DONE]"])]
        resp = await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                stream=True,
                input=[
                    {"role": "user", "content": "first"},
                    {"type": "function_call", "call_id": "call_stream", "name": "get_test_value", "arguments": "{\"key\":\"demo\"}"},
                    {"type": "function_call_output", "call_id": "call_stream", "output": "VALUE_42"},
                    {"role": "user", "content": "final"},
                ],
                tools=[_tool()],
                tool_choice="none",
                max_output_tokens=999,
            ),
            authorization=None,
        )
        await _collect_stream(resp)

        payload = FakeAsyncClient.instances[-1].stream_calls[-1]["json"]
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["messages"][1]["tool_calls"][0]["id"], "call_stream")
        self.assertEqual(payload["messages"][2]["tool_call_id"], "call_stream")

    async def test_streamed_second_turn_can_keep_tools_when_new_call_is_explicitly_allowed(self):
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=["data: [DONE]"])]
        resp = await self.adapter.responses_create(
            _responses_req(
                model="minimaxai/minimax-m3",
                stream=True,
                input=[
                    {"role": "user", "content": "first"},
                    {"type": "function_call", "call_id": "call_stream", "name": "get_test_value", "arguments": "{\"key\":\"demo\"}"},
                    {"type": "function_call_output", "call_id": "call_stream", "output": "VALUE_42"},
                    {"role": "user", "content": "maybe call again"},
                ],
                tools=[_tool()],
                tool_choice="required",
                max_output_tokens=128,
            ),
            authorization=None,
        )
        await _collect_stream(resp)

        payload = FakeAsyncClient.instances[-1].stream_calls[-1]["json"]
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(payload["max_tokens"], 128)
    async def test_streamed_tool_call_arguments_are_aggregated(self):
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=[
            "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_123", "type": "function", "function": {"name": "get_test_value", "arguments": "{\"key\""}}]}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":\"demo\"}"}}]}}]}),
            "data: " + json.dumps({"choices": [{"finish_reason": "tool_calls", "delta": {}}]}),
            "data: [DONE]",
        ])]
        resp = await self.adapter.responses_create(_responses_req(stream=True, tools=[_tool()]), authorization=None)
        events = _parse_sse_events(await _collect_stream(resp))
        names = [name for name, _ in events]
        self.assertIn("response.function_call_arguments.delta", names)
        self.assertIn("response.function_call_arguments.done", names)
        done = [data for name, data in events if name == "response.function_call_arguments.done"][0]
        self.assertEqual(done["arguments"], "{\"key\":\"demo\"}")
        completed = events[-1][1]["response"]
        self.assertEqual(completed["output"][0]["type"], "function_call")
        self.assertEqual(completed["output_text"], "")

    async def test_streamed_interleaved_tool_calls_do_not_mix_arguments(self):
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=[
            "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_test_value", "arguments": "A"}}, {"index": 1, "id": "call_2", "function": {"name": "get_test_value", "arguments": "X"}}]}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "Y"}}, {"index": 0, "function": {"arguments": "B"}}]}}]}),
            "data: " + json.dumps({"choices": [{"finish_reason": "tool_calls", "delta": {}}]}),
            "data: [DONE]",
        ])]
        resp = await self.adapter.responses_create(_responses_req(stream=True, tools=[_tool()]), authorization=None)
        events = _parse_sse_events(await _collect_stream(resp))
        completed = events[-1][1]["response"]
        by_call = {item["call_id"]: item["arguments"] for item in completed["output"]}
        self.assertEqual(by_call, {"call_1": "AB", "call_2": "XY"})

    async def test_streamed_tool_call_uses_provider_model_upstream(self):
        FakeAsyncClient.stream_outcomes = [FakeStreamResponse(lines=["data: [DONE]"])]
        resp = await self.adapter.responses_create(_responses_req(stream=True, tools=[_tool()]), authorization=None)
        await _collect_stream(resp)
        payload = FakeAsyncClient.instances[-1].stream_calls[-1]["json"]
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertIn("tools", payload)

if __name__ == "__main__":
    unittest.main()




