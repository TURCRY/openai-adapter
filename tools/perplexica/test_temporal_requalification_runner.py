#!/usr/bin/env python3
"""Tests d'intégration du runner de requalification temporelle V3.

Aucun appel réseau réel : call_gemma_batch est systématiquement mocké.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from editorial_rewrite import prepare_editorial_input
from temporal_requalification_runner import (
    CONTEXT_NOTE,
    UNKNOWN_NOTE,
    build_requalification_plans,
    requalification_config_from_job,
    requalification_enabled,
    run_temporal_requalification,
)
from temporal_validation import STATUS_CONTEXT, STATUS_CURRENT, STATUS_MISMATCH, STATUS_UNKNOWN

LOCAL_ANSWER = (
    "Un article publié le 25 août 2026 présente une procédure récente[18].\n\n"
    "Ce second sujet est documenté dans la fiche de référence[19]."
)
LOCAL_ANSWERS = {"expertise_construction": LOCAL_ANSWER}


def make_source(index, status=STATUS_UNKNOWN, local=18):
    return {
        "index": index,
        "title": "Source {0}".format(index),
        "url": "https://exemple.fr/article/{0}".format(index),
        "source_searches": ["expertise_construction"],
        "original_indices": {"expertise_construction": local},
        "temporal": {
            "temporal_status": status,
            "temporal_role": "unknown" if status == STATUS_UNKNOWN else "context",
            "access_status": "accessible",
            "source_date": None,
            "claimed_dates": [],
            "date_confidence": None,
        },
    }


def make_result(*sources):
    return {
        "question": "Question de test",
        "answer_markdown": "Réponse avec citation [33].",
        "citation_numbers": [source["index"] for source in sources],
        "cited_sources": list(sources),
        "searches": {},
        "temporal_validation": {
            "status": "completed",
            "temporal_validation_count": len(sources),
        },
    }


def make_job(req_enabled=True, **overrides):
    job = {"name": "test", "temporal": {"enabled": True, "run_date": "2026-08-27"}}
    if req_enabled is None:
        return job
    requalification = {
        "enabled": req_enabled,
        "model": "local-gemma-4",
        "batch_size": 4,
        "timeout": 600,
    }
    requalification.update(overrides)
    job["temporal"]["requalification"] = requalification
    return job


def batch_payload(*entries):
    return {"content": json.dumps({"sources": list(entries)}, ensure_ascii=False), "usage": {}}


def context_entry(source_number):
    return {
        "source_number": source_number,
        "recommended_status": "context",
        "confidence": "high",
        "reason_code": "role_context_explicit",
        "reason": "Document de référence sans nouveauté.",
    }


class RequalificationRunnerTests(unittest.TestCase):
    def test_disabled_keeps_python_only(self):
        job = {"name": "test", "temporal": {"enabled": True}}
        result = make_result(make_source(33))
        with patch(
            "temporal_requalification_runner.call_gemma_batch",
            side_effect=AssertionError("ne doit pas appeler Gemma"),
        ):
            summary = run_temporal_requalification("ignored", result, job)
        self.assertEqual(summary["status"], "disabled")
        self.assertEqual(result["temporal_requalification"]["status"], "disabled")
        temporal = result["cited_sources"][0]["temporal"]
        self.assertEqual(temporal["temporal_status"], STATUS_UNKNOWN)
        self.assertNotIn("gemma_recommended_status", temporal)

    def test_legacy_job_without_temporal_config_disabled(self):
        job = {"name": "test"}
        self.assertFalse(requalification_enabled(job))
        result = make_result(make_source(33))
        with patch(
            "temporal_requalification_runner.call_gemma_batch",
            side_effect=AssertionError("ne doit pas appeler Gemma"),
        ):
            summary = run_temporal_requalification("ignored", result, job)
        self.assertEqual(summary["status"], "disabled")

    def test_enabled_merges_final_status_and_editorial_uses_it(self):
        result = make_result(make_source(33))
        job = make_job()

        def fake_call(config, messages):
            return batch_payload(context_entry(33)), None

        with patch("temporal_requalification_runner.call_gemma_batch", side_effect=fake_call), patch(
            "temporal_requalification_runner.load_local_answers_from_run",
            return_value=LOCAL_ANSWERS,
        ):
            summary = run_temporal_requalification("ignored", result, job)
        self.assertEqual(summary["temporal_requalification_accepted_count"], 1)
        temporal = result["cited_sources"][0]["temporal"]
        self.assertEqual(temporal["temporal_status"], STATUS_CONTEXT)
        self.assertEqual(temporal["final_status"], STATUS_CONTEXT)
        self.assertEqual(temporal["python_status"], STATUS_UNKNOWN)
        self.assertEqual(temporal["gemma_recommended_status"], "context")
        self.assertEqual(temporal["gemma_validation"], "accepted")
        self.assertEqual(temporal["note"], CONTEXT_NOTE)
        entry = prepare_editorial_input(result)["cited_sources"][0]
        self.assertEqual(entry["temporal"]["status"], STATUS_CONTEXT)
        self.assertEqual(entry["temporal"]["note"], CONTEXT_NOTE)

    def test_timeout_falls_back_to_python(self):
        result = make_result(make_source(33))
        job = make_job()
        with patch(
            "temporal_requalification_runner.call_gemma_batch",
            return_value=(None, "TimeoutError: timeout"),
        ), patch(
            "temporal_requalification_runner.load_local_answers_from_run",
            return_value=LOCAL_ANSWERS,
        ):
            summary = run_temporal_requalification("ignored", result, job)
        self.assertEqual(summary["temporal_requalification_error_count"], 1)
        temporal = result["cited_sources"][0]["temporal"]
        self.assertEqual(temporal["temporal_status"], STATUS_UNKNOWN)
        self.assertEqual(temporal["final_status"], STATUS_UNKNOWN)
        self.assertEqual(temporal["gemma_validation"], "error")
        self.assertIn("TimeoutError", temporal["gemma_error"])
        self.assertEqual(temporal["note"], UNKNOWN_NOTE)

    def test_invalid_json_falls_back_to_python(self):
        result = make_result(make_source(33))
        job = make_job()
        with patch(
            "temporal_requalification_runner.call_gemma_batch",
            return_value=({"content": "ceci n'est pas du JSON", "usage": {}}, None),
        ), patch(
            "temporal_requalification_runner.load_local_answers_from_run",
            return_value=LOCAL_ANSWERS,
        ):
            summary = run_temporal_requalification("ignored", result, job)
        self.assertEqual(summary["temporal_requalification_error_count"], 1)
        temporal = result["cited_sources"][0]["temporal"]
        self.assertEqual(temporal["temporal_status"], STATUS_UNKNOWN)
        self.assertEqual(temporal["gemma_validation"], "error")

    def test_partial_batch_rejection_preserves_others(self):
        result = make_result(make_source(33, local=18), make_source(34, local=19))
        job = make_job()
        bad_entry = {
            "source_number": 34,
            "recommended_status": "current",
            "confidence": "high",
            "reason_code": "role_current_recent_context",
            "reason": "Contexte présenté comme récent.",
        }

        def fake_call(config, messages):
            return batch_payload(context_entry(33), bad_entry), None

        with patch("temporal_requalification_runner.call_gemma_batch", side_effect=fake_call), patch(
            "temporal_requalification_runner.load_local_answers_from_run",
            return_value=LOCAL_ANSWERS,
        ):
            summary = run_temporal_requalification("ignored", result, job)
        self.assertEqual(summary["temporal_requalification_accepted_count"], 1)
        self.assertEqual(summary["temporal_requalification_rejected_count"], 1)
        first = result["cited_sources"][0]["temporal"]
        second = result["cited_sources"][1]["temporal"]
        self.assertEqual(first["temporal_status"], STATUS_CONTEXT)
        self.assertEqual(first["gemma_validation"], "accepted")
        self.assertEqual(second["temporal_status"], STATUS_UNKNOWN)
        self.assertEqual(second["gemma_validation"], "rejected")
        self.assertIn("role_current_recent_context", second["gemma_error"])

    def test_mismatch_safe_not_sent(self):
        result = make_result(make_source(10, status=STATUS_MISMATCH))
        job = make_job()
        with patch(
            "temporal_requalification_runner.call_gemma_batch",
            side_effect=AssertionError("ne doit pas appeler Gemma"),
        ), patch(
            "temporal_requalification_runner.load_local_answers_from_run",
            return_value=LOCAL_ANSWERS,
        ):
            summary = run_temporal_requalification("ignored", result, job)
        self.assertEqual(summary["temporal_requalification_eligible_count"], 0)
        self.assertEqual(summary["temporal_requalification_processed_count"], 0)

    def test_current_safe_not_sent(self):
        result = make_result(make_source(1, status=STATUS_CURRENT))
        job = make_job()
        with patch(
            "temporal_requalification_runner.call_gemma_batch",
            side_effect=AssertionError("ne doit pas appeler Gemma"),
        ):
            summary = run_temporal_requalification("ignored", result, job)
        self.assertEqual(summary["temporal_requalification_eligible_count"], 0)

    def test_no_url_in_requalification_or_editorial_payloads(self):
        result = make_result(make_source(33))
        config = {
            "enabled": True,
            "model": "local-gemma-4",
            "batch_size": 4,
            "timeout": 600,
            "window_mode": "strict_7d",
            "run_date": "2026-08-27",
            "base_url": "",
            "api_key": None,
        }
        plans = build_requalification_plans(result, LOCAL_ANSWERS, config)
        self.assertEqual(len(plans), 1)
        self.assertNotIn("url", plans[0]["payload"])
        dumped = json.dumps(plans[0]["payload"], ensure_ascii=False)
        self.assertNotIn("https://exemple.fr", dumped)
        editorial = prepare_editorial_input(result)
        self.assertNotIn("https://exemple.fr", json.dumps(editorial, ensure_ascii=False))

    def test_batch_size_clamped_to_four(self):
        config = requalification_config_from_job(make_job(batch_size=10))
        self.assertEqual(config["batch_size"], 4)
        config = requalification_config_from_job(make_job(batch_size=0))
        self.assertEqual(config["batch_size"], 1)

    def test_unknown_eligible_and_old_context_not_eligible(self):
        s_unknown = make_source(33, local=18)
        s_ctx = make_source(37, status=STATUS_CONTEXT, local=19)
        s_ctx["temporal"]["modified_date"] = "2026-01-01"
        result = make_result(s_unknown, s_ctx)
        config = {
            "enabled": True,
            "model": "local-gemma-4",
            "batch_size": 4,
            "timeout": 600,
            "window_mode": "strict_7d",
            "run_date": "2026-08-27",
            "base_url": "",
            "api_key": None,
        }
        plans = build_requalification_plans(result, LOCAL_ANSWERS, config)
        self.assertEqual({plan["index"] for plan in plans}, {33})

    def test_final_status_current_has_no_negative_note(self):
        result = make_result(make_source(33, local=18))
        job = make_job()
        entry = {
            "source_number": 33,
            "recommended_status": "current",
            "confidence": "high",
            "reason_code": "role_current_recent_context",
            "reason": "Publication récente dans le contexte.",
        }

        def fake_call(config, messages):
            return batch_payload(entry), None

        with patch("temporal_requalification_runner.call_gemma_batch", side_effect=fake_call), patch(
            "temporal_requalification_runner.load_local_answers_from_run",
            return_value=LOCAL_ANSWERS,
        ):
            summary = run_temporal_requalification("ignored", result, job)
        self.assertEqual(summary["temporal_requalification_current_count"], 1)
        temporal = result["cited_sources"][0]["temporal"]
        self.assertEqual(temporal["temporal_status"], STATUS_CURRENT)
        self.assertNotIn("note", temporal)


if __name__ == "__main__":
    unittest.main()