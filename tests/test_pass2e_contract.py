import importlib
import sys
import types
import unittest


class _DummyHTTPException(Exception):
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


def _install_stubs():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _DummyFastAPI
    fastapi.File = _identity_default
    fastapi.Form = _identity_default
    fastapi.Body = _identity_default
    fastapi.Header = _identity_default
    fastapi.HTTPException = _DummyHTTPException
    fastapi.Request = object
    fastapi.UploadFile = object
    sys.modules.setdefault("fastapi", fastapi)

    middleware = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = object
    sys.modules.setdefault("fastapi.middleware", middleware)
    sys.modules.setdefault("fastapi.middleware.cors", cors)

    responses = types.ModuleType("fastapi.responses")
    responses.Response = object
    responses.JSONResponse = object
    responses.StreamingResponse = object
    sys.modules.setdefault("fastapi.responses", responses)


_install_stubs()
adapter = importlib.import_module("adapter")
for _name in [
    "adapter",
    "fastapi",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "fastapi.responses",
]:
    sys.modules.pop(_name, None)


class Pass2EContractTests(unittest.TestCase):
    def test_segment_schema_is_mapped_without_loss(self):
        out = adapter.normalize_pass2e_compact({
            "resume_segment": "Resume court",
            "themes": ["Point factuel"],
            "actions": ["Action a conserver"],
            "problems": ["Desaccord a conserver"],
        })

        self.assertEqual(out["resume_factuel"], "Resume court")
        self.assertEqual(out["points_cles"], ["Point factuel"])
        self.assertEqual(out["actions"], ["Action a conserver"])
        self.assertEqual(out["desaccords"], ["Desaccord a conserver"])

    def test_pass2e_fallback_is_dedicated(self):
        self.assertEqual(adapter._fallback_chain_for("pass2e_remote"), ["pass2e_remote"])
        fb = adapter._fallback_json_for_model("pass2e_remote")
        self.assertEqual(sorted(fb.keys()), sorted(adapter.PASS2E_KEYS))
        self.assertNotIn("resume_segment", fb)
        self.assertNotIn("themes", fb)
        self.assertNotIn("problems", fb)

    def test_empty_detection(self):
        self.assertTrue(adapter._is_effectively_empty_pass2e(adapter._fallback_json_for_model("pass2e_remote")))
        self.assertFalse(adapter._is_effectively_empty_pass2e({"resume_factuel": "fait", "points_cles": []}))


if __name__ == "__main__":
    unittest.main()


def tearDownModule():
    for name in [
        "adapter",
        "fastapi",
        "fastapi.middleware",
        "fastapi.middleware.cors",
        "fastapi.responses",
    ]:
        sys.modules.pop(name, None)
