import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from editorial_rewrite import (
    DEFAULT_TIMEOUT_SECONDS,
    ENV_ADAPTER_API_KEY,
    ENV_EDITORIAL_API_KEY,
    ENV_EDITORIAL_BASE_URL,
    MAX_TIMEOUT_SECONDS,
    EditorialConfig,
    EditorialOutputValidationError,
    EditorialRewriteError,
    EditorialTemporalViolationError,
    build_messages,
    call_editorial_llm,
    editorial_config_from_env,
    editorial_temporal_violations,
    extract_citation_numbers,
    invalid_claimed_dates_for_source,
    invalid_date_forms,
    neutralize_mismatch_claims,
    neutralize_mismatch_freshness,
    parse_editorial_output,
    temporal_safe_raw_markdown,
    prepare_editorial_input,
    RETRY_STRUCTURE_INSTRUCTION,
    reinforce_editorial_prompt,
    rewrite_editorial,
    validate_editorial_citations,
    validate_editorial_output,
)


def canonical_result():
    return {
        "chat_id": "chat123",
        "message_id": "msg123",
        "question": "Sujet ?",
        "answer_markdown": "Texte Perplexica [3][1]. Invitation \u00e0 poursuivre.",
        "all_sources": [
            {"index": 1, "title": "Cit\u00e9e 1", "url": "https://example.com/1", "content": "secret long content"},
            {"index": 2, "title": "Non cit\u00e9e", "url": "https://example.com/2", "content": "must not be sent"},
            {"index": 3, "title": "Cit\u00e9e 3", "url": "https://example.com/3"},
        ],
        "cited_sources": [
            {"index": 3, "title": "Cit\u00e9e 3", "url": "https://example.com/3"},
            {"index": 1, "title": "Cit\u00e9e 1", "url": "https://example.com/1"},
        ],
        "citation_numbers": [3, 1],
        "status": "completed",
    }


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self.body


class EditorialRewriteTests(unittest.TestCase):
    def clear_editorial_env(self):
        old = {
            ENV_EDITORIAL_BASE_URL: os.environ.pop(ENV_EDITORIAL_BASE_URL, None),
            ENV_EDITORIAL_API_KEY: os.environ.pop(ENV_EDITORIAL_API_KEY, None),
            ENV_ADAPTER_API_KEY: os.environ.pop(ENV_ADAPTER_API_KEY, None),
        }
        return old

    def restore_env(self, old):
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_extract_citation_numbers_preserves_first_seen_order(self):
        self.assertEqual(extract_citation_numbers("A [3][1][3] [42]"), [3, 1, 42])

    def test_default_timeout_is_300(self):
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, 300)
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(f"{ENV_EDITORIAL_BASE_URL}=http://adapter.local\n", encoding="utf-8")
            old = self.clear_editorial_env()
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.timeout, 300)

    def test_prepare_input_sends_only_cited_sources(self):
        result = canonical_result()
        payload = prepare_editorial_input(result)
        self.assertEqual(payload["citation_numbers"], [1, 3])
        self.assertEqual(
            payload["cited_sources"],
            [
                {"number": 3, "title": "Cit\u00e9e 3"},
                {"number": 1, "title": "Cit\u00e9e 1"},
            ],
        )
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("all_sources", raw)
        self.assertNotIn("Non cit\u00e9e", raw)
        self.assertNotIn("must not be sent", raw)
        self.assertNotIn("https://example.com/1", raw)
        self.assertNotIn("https://example.com/3", raw)
        self.assertEqual(result["cited_sources"][0]["url"], "https://example.com/3")

    def test_prepare_aggregated_input_uses_editorial_answer_and_completed_searches_only(self):
        result = canonical_result()
        result["answer_markdown"] = "## A\n\nTexte brut avec empty."
        result["editorial_answer_markdown"] = "## A\n\nTexte utile [1]."
        result["searches"] = {
            "a": {"status": "completed", "source_count": 2, "cited_source_count": 1},
            "b": {"status": "empty", "source_count": 0, "cited_source_count": 0},
            "c": {"status": "failed", "source_count": 0, "cited_source_count": 0},
        }
        payload = prepare_editorial_input(result)
        self.assertEqual(payload["response"], "## A\n\nTexte utile [1].")
        self.assertEqual(list(payload["searches"].keys()), ["a"])
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("all_sources", raw)
        self.assertNotIn("empty", raw)

    def test_build_messages_contains_prompt_and_limited_payload(self):
        payload = prepare_editorial_input(canonical_result())
        messages = build_messages("SYSTEM", payload)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("SYSTEM", messages[0]["content"])
        self.assertIn("cited_sources", messages[1]["content"])
        self.assertNotIn("all_sources", messages[1]["content"])

    def test_parse_strict_json_output(self):
        parsed = parse_editorial_output('{"title":"Titre","body_markdown":"Corps [1]","citation_numbers":[1]}')
        self.assertEqual(parsed["title"], "Titre")
        self.assertEqual(parsed["citation_numbers"], [1])

    def test_parse_json_inside_text_or_fence(self):
        raw = 'Voici:\n```json\n{"title":"Titre","body_markdown":"Corps [3]","citation_numbers":[3]}\n```'
        parsed = parse_editorial_output(raw)
        self.assertEqual(parsed["body_markdown"], "Corps [3]")

    def test_invalid_json_output_raises(self):
        with self.assertRaises(EditorialRewriteError):
            parse_editorial_output("pas du json")

    def test_citations_conserved(self):
        editorial = {"title": "Titre", "body_markdown": "Texte [3][1]", "citation_numbers": [3, 1]}
        validate_editorial_citations(editorial, [1, 3])
        self.assertEqual(editorial["citation_numbers"], [3, 1])

    def test_unknown_citation_raises(self):
        editorial = {"title": "Titre", "body_markdown": "Texte [99]", "citation_numbers": [99]}
        with self.assertRaises(EditorialRewriteError):
            validate_editorial_citations(editorial, [1, 3])

    def test_rewrite_editorial_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text("Prompt \u00e9ditorial", encoding="utf-8")
            config = EditorialConfig(base_url="http://local", model="local-gemma-4", prompt_file=prompt)
            raw = '{"title":"Note","body_markdown":"Synth\u00e8se [3]","citation_numbers":[3]}'
            editorial, raw_text = rewrite_editorial(canonical_result(), config, llm_func=lambda cfg, msgs: raw)
            self.assertEqual(editorial["status"], "completed")
            self.assertEqual(editorial["model"], "local-gemma-4")
            self.assertEqual(editorial["citation_numbers"], [3])
            self.assertEqual(raw_text, raw)

    def test_call_editorial_llm_uses_openai_compatible_endpoint(self):
        captured = {}
        response_body = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(response_body)

        config = EditorialConfig(
            base_url="http://adapter.local",
            model="local-gemma-4",
            prompt_file=Path("prompt"),
            api_key="secret",
            timeout=12,
        )
        with patch("editorial_rewrite.urlopen", fake_urlopen):
            content = call_editorial_llm(config, [{"role": "user", "content": "Bonjour"}])
        self.assertEqual(content, "{}")
        self.assertEqual(captured["url"], "http://adapter.local/v1/chat/completions")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["body"]["model"], "local-gemma-4")
        self.assertFalse(captured["body"]["stream"])
        self.assertNotIn("max_tokens", captured["body"])
        self.assertIn("Authorization", captured["headers"])

    def test_call_editorial_llm_http_error(self):
        def fake_urlopen(request, timeout):
            raise HTTPError(request.full_url, 500, "boom", {}, None)

        config = EditorialConfig(base_url="http://adapter.local", prompt_file=Path("prompt"))
        with patch("editorial_rewrite.urlopen", fake_urlopen):
            with self.assertRaises(EditorialRewriteError):
                call_editorial_llm(config, [])

    def test_call_editorial_llm_network_error(self):
        def fake_urlopen(request, timeout):
            raise URLError("offline")

        config = EditorialConfig(base_url="http://adapter.local", prompt_file=Path("prompt"))
        with patch("editorial_rewrite.urlopen", fake_urlopen):
            with self.assertRaises(EditorialRewriteError):
                call_editorial_llm(config, [])

    def test_invalid_timeout_is_rejected_before_http_call(self):
        for timeout in (0, -1, MAX_TIMEOUT_SECONDS + 1):
            config = EditorialConfig(base_url="http://adapter.local", prompt_file=Path("prompt"), timeout=timeout)
            with self.subTest(timeout=timeout):
                with patch("editorial_rewrite.urlopen") as fake_urlopen:
                    with self.assertRaises(EditorialRewriteError):
                        call_editorial_llm(config, [])
                fake_urlopen.assert_not_called()

    def test_editorial_config_reads_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                f"{ENV_EDITORIAL_BASE_URL}=http://adapter.local\n"
                f"{ENV_EDITORIAL_API_KEY}='secret-from-file'\n",
                encoding="utf-8",
            )
            old = self.clear_editorial_env()
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.base_url, "http://adapter.local")
        self.assertEqual(config.api_key, "secret-from-file")

    def test_system_env_has_priority_over_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(f"{ENV_EDITORIAL_BASE_URL}=http://from-file\n", encoding="utf-8")
            old = self.clear_editorial_env()
            os.environ[ENV_EDITORIAL_BASE_URL] = "http://from-env"
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.base_url, "http://from-env")

    def test_adapter_api_key_is_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                f"{ENV_EDITORIAL_BASE_URL}=http://adapter.local\n"
                f"{ENV_ADAPTER_API_KEY}='adapter-secret'\n",
                encoding="utf-8",
            )
            old = self.clear_editorial_env()
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.api_key, "adapter-secret")

    def test_editorial_api_key_has_priority_over_adapter_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                f"{ENV_EDITORIAL_BASE_URL}=http://adapter.local\n"
                f"{ENV_EDITORIAL_API_KEY}=editorial-secret\n"
                f"{ENV_ADAPTER_API_KEY}=adapter-secret\n",
                encoding="utf-8",
            )
            old = self.clear_editorial_env()
            try:
                config = editorial_config_from_env(env_file=env_path)
            finally:
                self.restore_env(old)
        self.assertEqual(config.api_key, "editorial-secret")


def mismatch_source():
    """Village-Justice-style mismatch source: claimed 2026-08-20, real 2017-11-15."""
    return {
        "index": 10,
        "title": "Village Justice - Dematerialisation des echanges",
        "url": "https://www.village-justice.com/articles/dematerialisation",
        "temporal": {
            "access_status": "accessible",
            "source_date": "2017-11-15",
            "date_evidence": "json_ld",
            "date_confidence": "high",
            "date_verification": "direct",
            "claimed_dates": ["2026-08-20"],
            "claimed_from_searches": ["expertise_justice"],
            "temporal_role": "current",
            "temporal_status": "mismatch",
            "note": "Ne pas presenter cette source comme actualite recente.",
        },
    }


def source_with_status(status, claimed_dates):
    return {
        "index": 5,
        "title": "Source " + status,
        "url": "https://example.com/source-5",
        "temporal": {
            "source_date": "2026-08-01",
            "date_evidence": "meta",
            "date_confidence": "medium",
            "claimed_dates": claimed_dates,
            "temporal_role": status,
            "temporal_status": status,
        },
    }


class EditorialTemporalTests(unittest.TestCase):
    def test_neutralize_mismatch_claim_removes_invalid_date(self):
        response = (
            "Une publication du 20 aout 2026 traite de la dematerialisation "
            "des echanges d'expertise [10]."
        )
        result = neutralize_mismatch_claims(response, [mismatch_source()])
        self.assertNotIn("20 aout 2026", result)
        self.assertNotIn("20 août 2026", result)
        self.assertIn("[10]", result)
        self.assertIn("dematerialisation", result)

    def test_neutralize_keeps_non_temporal_content_and_citation(self):
        response = (
            "Cette source traite des enjeux pratiques de la dematerialisation "
            "des echanges [10]."
        )
        result = neutralize_mismatch_claims(response, [mismatch_source()])
        self.assertEqual(result, response)

    def test_neutralize_does_not_touch_other_dates(self):
        response = (
            "Une publication du 15 novembre 2017 analyse la dematerialisation [10]."
        )
        result = neutralize_mismatch_claims(response, [mismatch_source()])
        self.assertEqual(result, response)

    def test_neutralize_does_not_touch_date_ranges(self):
        response = "Du 20 au 27 aout 2026, la dematerialisation a progresse [10]."
        result = neutralize_mismatch_claims(response, [mismatch_source()])
        self.assertEqual(result, response)

    def test_neutralize_ignores_context_unknown_current(self):
        for status in ("context", "unknown", "current"):
            with self.subTest(status=status):
                response = "Une publication du 20 aout 2026 presente le sujet [5]."
                sources = [source_with_status(status, ["2026-08-20"])]
                result = neutralize_mismatch_claims(response, sources)
                self.assertEqual(result, response)

    def test_prepare_editorial_input_mismatch_payload(self):
        result = canonical_result()
        result["citation_numbers"] = [3, 1, 10]
        result["answer_markdown"] = (
            "Texte Perplexica [3][1]. Une publication du 20 aout 2026 "
            "traite du sujet [10]."
        )
        result["cited_sources"].append(mismatch_source())
        payload = prepare_editorial_input(result)
        mismatch_entry = next(
            entry for entry in payload["cited_sources"] if entry["number"] == 10
        )
        self.assertEqual(mismatch_entry["temporal"]["status"], "mismatch")
        self.assertEqual(
            mismatch_entry["temporal"]["invalid_claimed_dates"], ["2026-08-20"]
        )
        self.assertIn("présenter", mismatch_entry["temporal"]["note"])
        self.assertIn("date invalidée", mismatch_entry["temporal"]["note"])
        self.assertNotIn("20 aout 2026", payload["response"])
        self.assertNotIn("20 août 2026", payload["response"])
        self.assertIn("[10]", payload["response"])
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("https://www.village-justice.com", raw)

    def test_prepare_editorial_input_context_note(self):
        result = canonical_result()
        result["citation_numbers"] = [3, 1, 5]
        result["answer_markdown"] = "Texte Perplexica [3][1]. Rappel du decret [5]."
        result["cited_sources"].append(source_with_status("context", []))
        payload = prepare_editorial_input(result)
        entry = next(e for e in payload["cited_sources"] if e["number"] == 5)
        self.assertEqual(entry["temporal"]["status"], "context")
        self.assertIn("contexte", entry["temporal"]["note"].lower())
        self.assertIn("Rappel du decret [5]", payload["response"])

    def test_editorial_temporal_violations_detects_invalid_date(self):
        body = "La dematerialisation a ete publiee le 20 aout 2026 [10]."
        violations = editorial_temporal_violations(body, [mismatch_source()])
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["source_number"], 10)
        self.assertEqual(violations[0]["invalid_claimed_date"], "2026-08-20")

    def test_editorial_temporal_violations_clean_body(self):
        body = "La dematerialisation des echanges est un enjeu majeur [10]."
        violations = editorial_temporal_violations(body, [mismatch_source()])
        self.assertEqual(violations, [])

    def test_rewrite_editorial_raises_on_temporal_violation(self):
        result = canonical_result()
        result["citation_numbers"] = [3, 1, 10]
        result["answer_markdown"] = (
            "Texte Perplexica [3][1]. Une publication du 20 aout 2026 "
            "traite du sujet [10]."
        )
        result["cited_sources"].append(mismatch_source())
        raw = (
            '{"title":"Note","body_markdown":"Une publication du 20 aout 2026 '
            'traite du sujet [10].","citation_numbers":[10]}'
        )
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text("Prompt editorial", encoding="utf-8")
            config = EditorialConfig(
                base_url="http://local", model="local-gemma-4", prompt_file=prompt
            )
            with self.assertRaises(EditorialTemporalViolationError) as ctx:
                rewrite_editorial(result, config, llm_func=lambda cfg, msgs: raw)
        self.assertEqual(ctx.exception.violation_count, 1)

    def test_rewrite_editorial_clean_body_no_violation(self):
        result = canonical_result()
        result["citation_numbers"] = [3, 1, 10]
        result["answer_markdown"] = (
            "Texte Perplexica [3][1]. Une publication du 20 aout 2026 "
            "traite du sujet [10]."
        )
        result["cited_sources"].append(mismatch_source())
        raw = (
            '{"title":"Note","body_markdown":"Synthèse de la veille [10].",'
            '"citation_numbers":[10]}'
        )
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text("Prompt editorial", encoding="utf-8")
            config = EditorialConfig(
                base_url="http://local", model="local-gemma-4", prompt_file=prompt
            )
            editorial, _ = rewrite_editorial(
                result, config, llm_func=lambda cfg, msgs: raw
            )
        self.assertEqual(editorial["status"], "completed")
        self.assertEqual(editorial["editorial_temporal_violation_count"], 0)
        self.assertEqual(editorial["temporal_violation_count"], 0)

    def test_invalid_date_forms_french_and_iso(self):
        forms = invalid_date_forms("2026-08-20")
        self.assertIn("20 aout 2026", forms)
        self.assertIn("20 août 2026", forms)
        self.assertIn("2026-08-20", forms)

    def test_invalid_claimed_dates_only_divergent_subset(self):
        source = dict(mismatch_source())
        source["temporal"] = dict(source["temporal"])
        source["temporal"]["claimed_dates"] = ["2026-08-20", "2017-11-15"]
        invalid = invalid_claimed_dates_for_source(source["temporal"])
        self.assertEqual(invalid, ["2026-08-20"])
        response = (
            "Une publication du 20 aout 2026 et une analyse du 15 novembre 2017 "
            "portent sur la dematerialisation [10]."
        )
        result = neutralize_mismatch_claims(response, [source])
        self.assertNotIn("20 aout 2026", result)
        self.assertNotIn("20 août 2026", result)
        self.assertIn("15 novembre 2017", result)
        self.assertIn("[10]", result)

    def test_neutralize_produces_cette_source_representation(self):
        response = (
            "Une publication du 20 aout 2026 traite de la dematerialisation "
            "des echanges d'expertise [10]."
        )
        result = neutralize_mismatch_claims(response, [mismatch_source()])
        self.assertIn("Cette source", result)
        self.assertNotIn("20 aout 2026", result)
        self.assertNotIn("20 août 2026", result)
        self.assertIn("[10]", result)
        self.assertIn("dematerialisation", result)

    def test_neutralize_mid_sentence_uses_neutral_source(self):
        response = (
            "Selon une publication du 20 aout 2026, la dematerialisation "
            "progresse [10]."
        )
        result = neutralize_mismatch_claims(response, [mismatch_source()])
        self.assertIn("une source", result)
        self.assertNotIn("20 aout 2026", result)
        self.assertIn("[10]", result)

    def test_violation_ignores_valid_claimed_date(self):
        source = dict(mismatch_source())
        source["temporal"] = dict(source["temporal"])
        source["temporal"]["claimed_dates"] = ["2026-08-20", "2017-11-15"]
        body = "Une publication du 15 novembre 2017 reste une reference [10]."
        self.assertEqual(editorial_temporal_violations(body, [source]), [])

    def test_payload_current_and_unknown_notes(self):
        result = canonical_result()
        result["citation_numbers"] = [3, 1, 5, 6]
        result["answer_markdown"] = "Texte Perplexica [3][1]. Sujet [5] et [6]."
        unknown = source_with_status("unknown", [])
        unknown["index"] = 5
        current = source_with_status("current", [])
        current["index"] = 6
        result["cited_sources"].extend([unknown, current])
        payload = prepare_editorial_input(result)
        by_number = {entry["number"]: entry for entry in payload["cited_sources"]}
        unknown_entry = by_number[5]
        current_entry = by_number[6]
        self.assertEqual(unknown_entry["temporal"]["status"], "unknown")
        self.assertIn("non vérifiée", unknown_entry["temporal"]["note"])
        self.assertIn("ne pas affirmer une date certaine", unknown_entry["temporal"]["note"].lower())
        self.assertEqual(current_entry["temporal"]["status"], "current")
        self.assertNotIn("note", current_entry["temporal"])


    def test_freshness_marker_near_mismatch_citation_is_violation(self):
        body = "Des travaux récents portent sur la transmission électronique [10]."
        violations = editorial_temporal_violations(body, [mismatch_source()])
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["source_number"], 10)
        self.assertEqual(violations[0]["type"], "freshness_marker")
        self.assertEqual(violations[0]["matched_marker"], "récents")

    def test_freshness_neutral_formulations_are_ok(self):
        neutral = [
            "Cette source traite de la dématérialisation des échanges [10].",
            "Cette référence porte sur la transmission électronique [10].",
            "Des travaux portent sur la normalisation des flux dématérialisés [10].",
        ]
        for body in neutral:
            with self.subTest(body=body):
                self.assertEqual(
                    editorial_temporal_violations(body, [mismatch_source()]), []
                )

    def test_freshness_marker_not_misattributed_from_other_citation(self):
        body = (
            "Une publication récente [11] traite du sujet. "
            "L’ancienne référence [10] reste utile."
        )
        violations = editorial_temporal_violations(body, [mismatch_source()])
        self.assertEqual(violations, [])

    def test_freshness_marker_after_citation_detected(self):
        body = "Le rapport [10] a été publié récemment."
        violations = editorial_temporal_violations(body, [mismatch_source()])
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["type"], "freshness_marker")

    def test_invalid_date_and_freshness_both_detected(self):
        body = "Une publication du 20 aout 2026, récente, traite du sujet [10]."
        violations = editorial_temporal_violations(body, [mismatch_source()])
        self.assertEqual(len(violations), 2)
        types = {violation["type"] for violation in violations}
        self.assertEqual(types, {"invalid_date", "freshness_marker"})

    def test_freshness_marker_ignored_for_context_and_current(self):
        body = "Un arrêt récent de la Cour [5] précise la portée du texte."
        for status in ("context", "current"):
            source = source_with_status(status, [])
            with self.subTest(status=status):
                self.assertEqual(
                    editorial_temporal_violations(body, [source]), []
                )

    def test_rewrite_editorial_raises_on_freshness_violation(self):
        result = canonical_result()
        result["citation_numbers"] = [3, 1, 10]
        result["answer_markdown"] = "Texte Perplexica [3][1]. Sujet [10]."
        result["cited_sources"].append(mismatch_source())
        raw = (
            '{"title":"Note","body_markdown":"Des travaux récents portent sur '
            'le sujet [10].","citation_numbers":[10]}'
        )
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.md"
            prompt.write_text("Prompt éditorial", encoding="utf-8")
            config = EditorialConfig(
                base_url="http://local", model="local-gemma-4", prompt_file=prompt
            )
            with self.assertRaises(EditorialTemporalViolationError) as ctx:
                rewrite_editorial(result, config, llm_func=lambda cfg, msgs: raw)
        self.assertEqual(ctx.exception.violation_count, 1)
        self.assertEqual(ctx.exception.reason, "temporal_violation")

    def test_neutralize_freshness_marker_near_mismatch_citation(self):
        response = (
            "Des travaux récents portent sur la transmission électronique "
            "des échanges d'expertise [10]."
        )
        result = neutralize_mismatch_freshness(response, [mismatch_source()])
        self.assertNotIn("récents", result)
        self.assertIn("Des travaux portent sur", result)
        self.assertIn("[10]", result)

    def test_neutralize_freshness_keeps_marker_attributed_to_other_citation(self):
        response = (
            "Une publication récente [11] compare une ancienne référence [10]."
        )
        sources = [
            mismatch_source(),
            {
                "index": 11,
                "title": "Autre source",
                "url": "https://example.com/11",
                "temporal": {
                    "source_date": "2026-08-01",
                    "claimed_dates": [],
                    "temporal_status": "current",
                },
            },
        ]
        result = neutralize_mismatch_freshness(response, sources)
        self.assertIn("récente", result)
        self.assertEqual(result, response)

    def test_neutralize_freshness_after_citation_keeps_fact(self):
        response = "Le rapport [10] a été publié récemment."
        result = neutralize_mismatch_freshness(response, [mismatch_source()])
        self.assertNotIn("récemment", result)
        self.assertIn("a été publié", result)
        self.assertIn("[10]", result)

    def test_temporal_safe_raw_markdown_neutralizes_dates_and_freshness(self):
        raw = (
            "Une publication du 20 août 2026 analyse la dématérialisation "
            "des échanges [10]. Des travaux récents portent sur le même sujet [10]."
        )
        safe = temporal_safe_raw_markdown(raw, [mismatch_source()])
        self.assertNotIn("20 août 2026", safe)
        self.assertNotIn("récents", safe)
        self.assertIn("dématérialisation", safe)
        self.assertIn("[10]", safe)
        self.assertEqual(safe.count("[10]"), 2)


def counting_llm(responses):
    """Return an llm_func that records calls and replays the given responses."""
    calls: list[list[dict[str, str]]] = []

    def _llm(config, messages):
        calls.append(list(messages))
        index = min(len(calls) - 1, len(responses) - 1)
        return responses[index]

    return _llm, calls


def long_editorial_body(citations, min_length=2000):
    filler = "La veille de la période couvre les évolutions en cours et leurs implications. "
    body = ""
    while len(body) < min_length:
        body += filler
    return body + "[" + "][".join(str(number) for number in citations) + "]"


def sectioned_editorial():
    return {
        "title": "Veille Expertise Judiciaire, Construction et Médiation",
        "body_markdown": "## À retenir\nLa période récente est marquée par plusieurs évolutions. [1]",
        "expertise_justice": "Plusieurs mises à jour pratiques concernent les experts judiciaires. " * 10,
        "expertise_construction": "Le secteur de la construction connaît des évolutions réglementaires. " * 10,
        "médiation": "La médiation se développe dans les litiges civils. " * 10,
        "citation_numbers": [1],
    }


class EditorialOutputValidationTests(unittest.TestCase):
    def make_config(self, tmp, min_body_chars=2000):
        prompt = Path(tmp) / "prompt.md"
        prompt.write_text("Prompt éditorial", encoding="utf-8")
        return EditorialConfig(
            base_url="http://local",
            model="local-gemma-4",
            prompt_file=prompt,
            min_body_chars=min_body_chars,
        )

    def test_valid_long_body_accepted_without_retry(self):
        raw = json.dumps(
            {
                "title": "Note",
                "body_markdown": long_editorial_body([3, 1], 2000),
                "citation_numbers": [3, 1],
            },
            ensure_ascii=False,
        )
        llm, calls = counting_llm([raw])
        with tempfile.TemporaryDirectory() as tmp:
            editorial, _ = rewrite_editorial(canonical_result(), self.make_config(tmp), llm_func=llm)
        self.assertEqual(editorial["status"], "completed")
        self.assertEqual(editorial["editorial_retry_count"], 0)
        self.assertEqual(editorial["editorial_output_validation_status"], "ok")
        self.assertIsNone(editorial["editorial_output_invalid_reason"])
        self.assertEqual(len(calls), 1)

    def test_sectioned_short_body_rejected(self):
        editorial = sectioned_editorial()
        with self.assertRaises(EditorialOutputValidationError) as ctx:
            validate_editorial_output(editorial, allowed_numbers=[1, 3], min_body_chars=2000)
        self.assertEqual(ctx.exception.reason, "unexpected_sectioned_output")

    def test_short_body_without_extra_keys_rejected_as_body_too_short(self):
        editorial = {
            "title": "Note",
            "body_markdown": "## Résumé\nContenu bref. [1]",
            "citation_numbers": [1],
        }
        with self.assertRaises(EditorialOutputValidationError) as ctx:
            validate_editorial_output(editorial, allowed_numbers=[1], min_body_chars=2000)
        self.assertEqual(ctx.exception.reason, "body_too_short")

    def test_truncated_body_rejected(self):
        editorial = {
            "title": "Note",
            "body_markdown": "## Résumé\n" + "Texte long. " * 500 + "\n## Conclusion inachevée",
            "citation_numbers": [],
        }
        with self.assertRaises(EditorialOutputValidationError) as ctx:
            validate_editorial_output(editorial, allowed_numbers=[1, 3], min_body_chars=2000)
        self.assertEqual(ctx.exception.reason, "body_truncated")

    def test_body_citations_absent_from_declaration_are_normalized(self):
        editorial = {
            "title": "Note",
            "body_markdown": "Corps de la note [1] [3].",
            "citation_numbers": [1],
        }
        stats = validate_editorial_output(
            editorial, allowed_numbers=[1, 3], min_body_chars=None
        )
        self.assertEqual(editorial["citation_numbers"], [1, 3])
        self.assertTrue(stats["normalized"])
        self.assertEqual(stats["declared_count"], 1)
        self.assertEqual(stats["actual_count"], 2)

    def test_declared_superset_of_body_is_accepted_and_normalized(self):
        editorial = {
            "title": "Note",
            "body_markdown": "Corps de la note [1].",
            "citation_numbers": [1, 3],
        }
        stats = validate_editorial_output(
            editorial, allowed_numbers=[1, 3], min_body_chars=None
        )
        self.assertEqual(editorial["citation_numbers"], [1])
        self.assertTrue(stats["normalized"])
        self.assertEqual(stats["declared_count"], 2)
        self.assertEqual(stats["actual_count"], 1)
        self.assertEqual(stats["declared_unused_count"], 1)

    def test_order_mismatch_normalized_to_body_order(self):
        editorial = {
            "title": "Note",
            "body_markdown": "Corps de la note [1] [3].",
            "citation_numbers": [3, 1],
        }
        stats = validate_editorial_output(
            editorial, allowed_numbers=[1, 3], min_body_chars=None
        )
        self.assertEqual(editorial["citation_numbers"], [1, 3])
        self.assertTrue(stats["normalized"])

    def test_declared_out_of_range_absent_from_body_is_ignored(self):
        editorial = {
            "title": "Note",
            "body_markdown": "Corps de la note [1].",
            "citation_numbers": [1, 99],
        }
        stats = validate_editorial_output(
            editorial, allowed_numbers=[1, 3], min_body_chars=None
        )
        self.assertEqual(editorial["citation_numbers"], [1])
        self.assertTrue(stats["normalized"])
        self.assertEqual(stats["declared_unused_count"], 1)

    def test_invented_citation_rejected(self):
        editorial = {
            "title": "Note",
            "body_markdown": "Corps de la note [99].",
            "citation_numbers": [99],
        }
        with self.assertRaises(EditorialOutputValidationError) as ctx:
            validate_editorial_output(editorial, allowed_numbers=[1, 3], min_body_chars=None)
        self.assertEqual(ctx.exception.reason, "invented_citation")

    def test_retry_valid_after_sectioned_output(self):
        first = json.dumps(sectioned_editorial(), ensure_ascii=False)
        second = json.dumps(
            {
                "title": "Note",
                "body_markdown": long_editorial_body([1, 3], 2000),
                "citation_numbers": [1, 3],
            },
            ensure_ascii=False,
        )
        llm, calls = counting_llm([first, second])
        with tempfile.TemporaryDirectory() as tmp:
            editorial, _ = rewrite_editorial(canonical_result(), self.make_config(tmp), llm_func=llm)
        self.assertEqual(editorial["editorial_retry_count"], 1)
        self.assertEqual(editorial["editorial_output_validation_status"], "ok")
        self.assertEqual(len(calls), 2)
        self.assertIn("Retourne exactement et uniquement les clés", calls[1][0]["content"])
        self.assertNotIn("Retourne exactement et uniquement les clés", calls[0][0]["content"])

    def test_retry_invalid_propagates_exception(self):
        first = json.dumps(sectioned_editorial(), ensure_ascii=False)
        llm, calls = counting_llm([first, first])
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            with self.assertRaises(EditorialOutputValidationError) as ctx:
                rewrite_editorial(canonical_result(), config, llm_func=llm)
        self.assertEqual(ctx.exception.reason, "unexpected_sectioned_output")
        self.assertEqual(ctx.exception.retry_count, 1)
        self.assertEqual(len(calls), 2)

    def test_temporal_violation_retried_once_then_raises(self):
        result = canonical_result()
        result["citation_numbers"] = [3, 1, 10]
        result["answer_markdown"] = (
            "Texte Perplexica [3][1]. Une publication du 20 aout 2026 "
            "traite du sujet [10]."
        )
        result["cited_sources"].append(mismatch_source())
        violating = (
            '{"title":"Note","body_markdown":"Une publication du 20 aout 2026 '
            'traite du sujet [10].","citation_numbers":[10]}'
        )
        llm, calls = counting_llm([violating, violating])
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp, min_body_chars=None)
            with self.assertRaises(EditorialTemporalViolationError) as ctx:
                rewrite_editorial(result, config, llm_func=llm)
        self.assertEqual(ctx.exception.reason, "temporal_violation")
        self.assertEqual(ctx.exception.retry_count, 1)
        self.assertEqual(ctx.exception.violation_count, 1)
        self.assertEqual(len(calls), 2)

    def test_transport_error_is_not_retried(self):
        def boom(config, messages):
            raise EditorialRewriteError("offline")

        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp)
            with self.assertRaises(EditorialRewriteError):
                rewrite_editorial(canonical_result(), config, llm_func=boom)

    def test_rewrite_editorial_auto_normalizes_citation_numbers(self):
        raw = json.dumps(
            {
                "title": "Note",
                "body_markdown": long_editorial_body([1, 3], 2000),
                "citation_numbers": [1],
            },
            ensure_ascii=False,
        )
        llm, calls = counting_llm([raw])
        with tempfile.TemporaryDirectory() as tmp:
            editorial, _ = rewrite_editorial(
                canonical_result(), self.make_config(tmp), llm_func=llm
            )
        self.assertEqual(editorial["status"], "completed")
        self.assertEqual(editorial["citation_numbers"], [1, 3])
        self.assertEqual(editorial["editorial_citation_numbers_normalized"], True)
        self.assertEqual(editorial["editorial_declared_citation_count"], 1)
        self.assertEqual(editorial["editorial_actual_citation_count"], 2)
        self.assertEqual(editorial["editorial_retry_count"], 0)
        self.assertEqual(len(calls), 1)

    def test_rewrite_editorial_citation_numbers_not_normalized_when_matching(self):
        raw = json.dumps(
            {
                "title": "Note",
                "body_markdown": long_editorial_body([3, 1], 2000),
                "citation_numbers": [3, 1],
            },
            ensure_ascii=False,
        )
        llm, calls = counting_llm([raw])
        with tempfile.TemporaryDirectory() as tmp:
            editorial, _ = rewrite_editorial(
                canonical_result(), self.make_config(tmp), llm_func=llm
            )
        self.assertEqual(editorial["status"], "completed")
        self.assertEqual(editorial["editorial_citation_numbers_normalized"], False)
        self.assertEqual(editorial["editorial_declared_citation_count"], 2)
        self.assertEqual(editorial["editorial_actual_citation_count"], 2)
        self.assertEqual(len(calls), 1)

    def test_run_20260831_style_declared_all_valid_body_subset_is_accepted(self):
        actual = [5, 6, 10, 14, 23, 26, 28, 29, 35, 42, 59, 60]
        editorial = {
            "title": "Note",
            "body_markdown": long_editorial_body(actual, 2000),
            "citation_numbers": list(range(1, 62)),
        }
        stats = validate_editorial_output(
            editorial, allowed_numbers=list(range(1, 62)), min_body_chars=2000
        )
        self.assertEqual(editorial["citation_numbers"], actual)
        self.assertTrue(stats["normalized"])
        self.assertEqual(stats["declared_count"], 61)
        self.assertEqual(stats["actual_count"], len(actual))
        self.assertEqual(stats["declared_unused_count"], 49)

    def test_declared_subset_of_valid_body_is_accepted_and_normalized(self):
        editorial = {
            "title": "Note",
            "body_markdown": "Corps de la note [1] [3].",
            "citation_numbers": [1],
        }
        stats = validate_editorial_output(
            editorial, allowed_numbers=[1, 3], min_body_chars=None
        )
        self.assertEqual(editorial["citation_numbers"], [1, 3])
        self.assertTrue(stats["normalized"])
        self.assertEqual(stats["declared_count"], 1)
        self.assertEqual(stats["actual_count"], 2)

    def test_body_without_citation_is_accepted_as_empty_canonical_citations(self):
        editorial = {
            "title": "Note",
            "body_markdown": "Corps de la note sans citation.",
            "citation_numbers": [1, 3],
        }
        stats = validate_editorial_output(
            editorial, allowed_numbers=[1, 3], min_body_chars=None
        )
        self.assertEqual(editorial["citation_numbers"], [])
        self.assertTrue(stats["normalized"])
        self.assertEqual(stats["actual_count"], 0)

    def test_diagnostics_written_for_rejected_attempt_only(self):
        first = json.dumps(sectioned_editorial(), ensure_ascii=False)
        second = json.dumps(
            {
                "title": "Note",
                "body_markdown": long_editorial_body([1, 3], 2000),
                "citation_numbers": [1, 3],
            },
            ensure_ascii=False,
        )
        llm, _ = counting_llm([first, second])
        with tempfile.TemporaryDirectory() as tmp:
            diagnostics = Path(tmp) / "diagnostics"
            editorial, _ = rewrite_editorial(
                canonical_result(),
                self.make_config(tmp),
                llm_func=llm,
                diagnostics_dir=diagnostics,
            )
            self.assertEqual(editorial["editorial_retry_count"], 1)
            self.assertTrue((diagnostics / "editorial_attempt_1_raw.txt").exists())
            self.assertTrue((diagnostics / "editorial_attempt_1_error.json").exists())
            self.assertFalse((diagnostics / "editorial_attempt_2_raw.txt").exists())
            error = json.loads((diagnostics / "editorial_attempt_1_error.json").read_text(encoding="utf-8"))
        self.assertEqual(error["reason"], "unexpected_sectioned_output")
        self.assertEqual(error["declared_citations"], [1])
        self.assertEqual(error["actual_citations"], [1])
        self.assertIsNone(error["temporal_violation_count"])

    def test_diagnostics_not_written_for_valid_output(self):
        raw = json.dumps(
            {
                "title": "Note",
                "body_markdown": long_editorial_body([1, 3], 2000),
                "citation_numbers": [1, 3],
            },
            ensure_ascii=False,
        )
        llm, _ = counting_llm([raw])
        with tempfile.TemporaryDirectory() as tmp:
            diagnostics = Path(tmp) / "diagnostics"
            rewrite_editorial(
                canonical_result(),
                self.make_config(tmp),
                llm_func=llm,
                diagnostics_dir=diagnostics,
            )
            self.assertFalse(diagnostics.exists())
    def test_reinforce_editorial_prompt_appends_structure_instruction(self):
        prompt = "Consignes éditoriales."
        reinforced = reinforce_editorial_prompt(prompt)
        self.assertTrue(reinforced.startswith("Consignes éditoriales."))
        self.assertIn(RETRY_STRUCTURE_INSTRUCTION, reinforced)
        self.assertIn("Retourne exactement et uniquement les clés", reinforced)
        self.assertIn("Tout le contenu éditorial doit être dans body_markdown", reinforced)


if __name__ == "__main__":
    unittest.main()
