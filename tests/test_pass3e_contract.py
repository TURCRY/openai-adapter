import asyncio
import importlib
import json
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


class Pass3EContractTests(unittest.TestCase):
    def test_complete_payload_is_preserved(self):
        out = adapter.normalize_pass3e_synthesis({
            "numero": 2,
            "titre": "Desordres allegues",
            "localisation": "Facade nord",
            "description": "Fissures et decalages declares",
            "avis_participants": [{"nom": "P2", "role": "Proprietaire", "resume": "Fenetre bloquee."}],
            "synthese_echanges": "Des fissures et decalages sont decrits.",
            "conclusion_expert": "Constat a verifier techniquement.",
        })

        self.assertEqual(out["numero"], 2)
        self.assertEqual(out["titre"], "Desordres allegues")
        self.assertEqual(out["localisation"], "Facade nord")
        self.assertEqual(out["description"], "Fissures et decalages declares")
        self.assertEqual(out["avis_participants"][0]["resume"], "Fenetre bloquee.")
        self.assertEqual(out["synthese_echanges"], "Des fissures et decalages sont decrits.")
        self.assertFalse(adapter._is_effectively_empty_pass3e(out))

    def test_numero_string_is_normalized(self):
        out = adapter.normalize_pass3e_synthesis({"numero": "007", "synthese_echanges": "Travaux evoques."})
        self.assertEqual(out["numero"], 7)

    def test_participants_string_becomes_list(self):
        out = adapter.normalize_pass3e_synthesis({
            "numero": 4,
            "avis_participants": "Le proprietaire indique disposer de photos.",
        })
        self.assertEqual(out["avis_participants"], [{
            "nom": "",
            "role": "",
            "resume": "Le proprietaire indique disposer de photos.",
        }])
        self.assertFalse(adapter._is_effectively_empty_pass3e(out))

    def test_partial_useful_payload_is_not_discarded(self):
        out = adapter.normalize_pass3e_synthesis({
            "numero": 8,
            "conclusion": "La necessite de travaux urgents ne peut pas etre tranchee sans constat.",
        })
        self.assertEqual(out["conclusion_expert"], "La necessite de travaux urgents ne peut pas etre tranchee sans constat.")
        self.assertFalse(adapter._is_effectively_empty_pass3e(out))

    def test_localisation_and_description_aliases_are_preserved(self):
        out = adapter.normalize_pass3e_synthesis({
            "numero": 5,
            "location": "Cuisine / mur est",
            "objet": "Demande de verification des infiltrations",
            "synthese_echanges": "Une reprise est demandee.",
        })
        self.assertEqual(out["localisation"], "Cuisine / mur est")
        self.assertEqual(out["description"], "Demande de verification des infiltrations")

        out = adapter.normalize_pass3e_synthesis({
            "numero": 6,
            "lieu": "Garage",
            "contexte": "Reserve sur le seuil",
            "synthese_echanges": "Le point reste ouvert.",
        })
        self.assertEqual(out["localisation"], "Garage")
        self.assertEqual(out["description"], "Reserve sur le seuil")

    def test_empty_payload_is_detected(self):
        out = adapter.normalize_pass3e_synthesis({})
        self.assertTrue(adapter._is_effectively_empty_pass3e(out))
        self.assertTrue(adapter._is_effectively_empty_pass3e(adapter._fallback_json_for_model("pass3e_remote")) is False)

    def test_non_json_payload_is_rejected_by_normalizer(self):
        with self.assertRaises(ValueError):
            adapter.normalize_pass3e_synthesis("texte hors JSON")

    def test_unknown_fields_with_useful_text_are_preserved_when_reasonable(self):
        out = adapter.normalize_pass3e_synthesis({
            "num": "9",
            "title": "Responsabilites",
            "synthese": ["Declaration de sinistre evoquee.", "Investigations demandees."],
            "avis_expert": "Responsabilites non tranchees a ce stade.",
        })
        self.assertEqual(out["numero"], 9)
        self.assertIn("Declaration de sinistre", out["synthese_echanges"])
        self.assertEqual(out["conclusion_expert"], "Responsabilites non tranchees a ce stade.")

    def test_pass3e_fallback_is_dedicated(self):
        self.assertEqual(adapter._fallback_chain_for("pass3e_remote"), ["pass3e_remote"])
        fb = adapter._fallback_json_for_model("pass3e_remote")
        self.assertEqual(sorted(fb.keys()), sorted(adapter.PASS3E_KEYS))
        self.assertIn("localisation", fb)
        self.assertIn("description", fb)
        self.assertEqual(fb["localisation"], "")
        self.assertEqual(fb["description"], "")
        self.assertIn("Pass3E", fb["conclusion_expert"])
        self.assertNotIn("resume_global", fb)
        self.assertNotIn("themes", fb)

    def test_chat_completions_routes_pass3e_remote_to_pass3e_normalizer(self):
        calls = []

        async def fake_remote_chat(messages, model_name, temperature, retries=2, response_format=None):
            calls.append(model_name)
            return json.dumps({
                "numero": "12",
                "titre": "Menuiseries",
                "lieu": "Salon",
                "contexte": "Reserve sur les reglages",
                "synthese_echanges": "Le reglage est evoque sans decision acquise.",
                "conclusion_expert": "Verification contradictoire necessaire.",
            })

        def fail_pass2e(_parsed):
            raise AssertionError("pass3e_remote must not use Pass2E normalization")

        def fail_report(_parsed):
            raise AssertionError("pass3e_remote must not use report normalization")

        old_remote = adapter._remote_chat_with_retry
        old_pass2e = adapter.normalize_pass2e_compact
        old_report = adapter.normalize_report_annotation
        try:
            adapter._remote_chat_with_retry = fake_remote_chat
            adapter.normalize_pass2e_compact = fail_pass2e
            adapter.normalize_report_annotation = fail_report
            req = adapter.ChatReq(
                model="pass3e_remote",
                messages=[adapter.ChatMessage(role="user", content="{}")],
                temperature=0.0,
            )
            request = types.SimpleNamespace(headers={})
            resp = asyncio.run(adapter.chat_completions(req=req, request=request))
        finally:
            adapter._remote_chat_with_retry = old_remote
            adapter.normalize_pass2e_compact = old_pass2e
            adapter.normalize_report_annotation = old_report

        self.assertEqual(calls, ["pass3e_remote"])
        payload = json.loads(resp.choices[0].message["content"])
        self.assertEqual(payload["numero"], 12)
        self.assertEqual(payload["localisation"], "Salon")
        self.assertEqual(payload["description"], "Reserve sur les reglages")
        self.assertEqual(payload["synthese_echanges"], "Le reglage est evoque sans decision acquise.")



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
