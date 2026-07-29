import sys
import types
import unittest
from pathlib import Path


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


class DebriefContractTests(unittest.TestCase):
    def test_valid_debrief_preserves_roots_and_data(self):
        payload = {
            "mode_debrief": "complement",
            "sujets": [
                {
                    "numero": 1,
                    "titre": "Fondations",
                    "orientation_expert": "Verifier les reprises en sous-oeuvre.",
                    "demandes_documents": [{"objet": "Diagnostic structurel"}],
                }
            ],
            "demandes_documents_hors_sujet": [{"objet": "Devis de reprise"}],
            "global_debrief": {
                "resume": "Le debrief insiste sur les suites techniques.",
                "themes_abordes": [{"titre": "Travaux", "synthese": ["Reprise a chiffrer"]}],
                "actions": [{"action": "Demander les devis"}],
                "perspectives": [{"probleme": "Incertitude structurelle", "solution": "Diagnostic"}],
            },
        }

        out = adapter.normalize_debrief_annotation(payload)

        self.assertEqual(set(out.keys()), adapter.DEBRIEF_ROOT_KEYS)
        self.assertEqual(out["mode_debrief"], "complement")
        self.assertEqual(out["sujets"][0]["orientation_expert"], "Verifier les reprises en sous-oeuvre.")
        self.assertEqual(out["demandes_documents_hors_sujet"][0]["objet"], "Devis de reprise")
        self.assertEqual(out["global_debrief"]["resume"], "Le debrief insiste sur les suites techniques.")
        self.assertEqual(out["global_debrief"]["themes_abordes"][0]["titre"], "Travaux")

    def test_missing_optional_fields_are_completed(self):
        out = adapter.normalize_debrief_annotation({
            "mode_debrief": True,
            "sujets": [{"numero": 2, "orientation_expert": "Conserver."}],
            "global_debrief": {"resume": "Observation transversale."},
        })

        self.assertEqual(out["mode_debrief"], "complement")
        self.assertEqual(out["sujets"][0]["demandes_documents"], [])
        self.assertEqual(out["demandes_documents_hors_sujet"], [])
        self.assertEqual(out["global_debrief"]["themes_abordes"], [])
        self.assertEqual(out["global_debrief"]["actions"], [])
        self.assertEqual(out["global_debrief"]["perspectives"], [])

    def test_report_schema_is_not_accepted_as_debrief_success(self):
        with self.assertRaises(ValueError):
            adapter.normalize_debrief_annotation(adapter._fallback_json_for_model("report_remote"))

    def test_debrief_fallback_is_dedicated_and_report_fallback_unchanged(self):
        debrief = adapter._fallback_json_for_model("report_debrief_remote")
        report = adapter._fallback_json_for_model("report_remote")

        self.assertEqual(set(debrief.keys()), adapter.DEBRIEF_ROOT_KEYS)
        self.assertIn("global_debrief", debrief)
        self.assertIn("resume_global", report)
        self.assertNotIn("resume_global", debrief)
        self.assertNotIn("global_debrief", report)

    def test_debrief_normalizer_does_not_call_report_normalizer(self):
        original = adapter.normalize_report_annotation
        try:
            def fail(_payload):
                raise AssertionError("normalize_report_annotation must not be called")
            adapter.normalize_report_annotation = fail
            out = adapter.normalize_debrief_annotation({
                "mode_debrief": "complement",
                "global_debrief": {"resume": "Resume utile."},
            })
            self.assertEqual(out["global_debrief"]["resume"], "Resume utile.")
        finally:
            adapter.normalize_report_annotation = original




def tearDownModule():
    for name in [
        "adapter",
        "fastapi",
        "fastapi.middleware",
        "fastapi.middleware.cors",
        "fastapi.responses",
    ]:
        sys.modules.pop(name, None)

if __name__ == "__main__":
    unittest.main()