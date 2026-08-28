#!/usr/bin/env python3
"""Unit tests for temporal_requalification (no LLM, no network)."""

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_validation import (
    STATUS_CONTEXT,
    STATUS_CURRENT,
    STATUS_MISMATCH,
    STATUS_UNKNOWN,
)
from temporal_requalification import (
    WINDOW_EXTENDED_30D,
    WINDOW_STRICT_7D,
    allowed_transitions,
    apply_requalification,
    build_claim_contexts_for_source,
    build_requalification_payload,
    context_to_current_allowed,
    deterministic_pre_positioning,
    eligibility_reason,
    extract_claim_and_neighbor_contexts,
    extract_claim_context,
    extract_recent_context_signals,
    extract_title_date,
    final_status_with_pre_positioning,
    reason_code_coherent,
    requalification_plan,
    resolve_window_params,
    validate_gemma_response,
    validate_gemma_response_v1,
)


class ClaimContextTests(unittest.TestCase):
    def test_full_sentence_with_citation(self):
        answer = (
            "Première phrase sans intérêt temporel. "
            "Un article publié le 25 août 2026 présente une procédure issue "
            "d'un décret du 5 août 2025[8]. "
            "Troisième phrase à écarter."
        )
        context = extract_claim_context(answer, 8)
        self.assertIn("publié le 25 août 2026", context)
        self.assertIn("décret du 5 août 2025", context)
        self.assertNotIn("Première phrase", context)
        self.assertNotIn("Troisième phrase", context)

    def test_dash_bullets_not_mixed(self):
        answer = (
            "L'annuaire a été mis à jour au 26/08/2026 pour cette juridiction[82]. "
            "- Une autre actualité publiée le 20 août 2026 concerne un autre sujet[72]. "
            "- Troisième item sans rapport[19]."
        )
        context = extract_claim_context(answer, 82)
        self.assertIn("26/08/2026", context)
        self.assertNotIn("Une autre actualité", context)
        self.assertNotIn("Troisième item", context)

    def test_dated_sentence_preferred(self):
        answer = (
            "Hmm, sorry I could not find any relevant information on this topic[4]. "
            "Le document le plus récent est le décret publié le 29 juillet 2026[4]."
        )
        context = extract_claim_context(answer, 4)
        self.assertIn("29 juillet 2026", context)
        self.assertNotIn("sorry I could not", context)

    def test_dated_isolated_sentence_preferred_over_intro(self):
        answer = (
            "Introduction — périmètre et méthode de veille. Je fournis ci-dessous "
            "les actualités sur la médiation publiées dans les sept derniers jours "
            "(période autour du 20-27 août 2026) en écartant les contenus anciens "
            "non actualisés[6][8][9][20].\n\n"
            "## 1) Circulaire relative à l'exécution des contrats "
            "La Direction des affaires juridiques (DAJ) a publié une information "
            "indiquant la diffusion d'une circulaire (n° 6553/SG du 12 août 2026) "
            "qui invite à mobiliser les outils de médiation[9]."
        )
        context = extract_claim_context(answer, 9)
        self.assertIn("12 août 2026", context)
        self.assertIn("DAJ", context)
        self.assertNotIn("sept derniers jours", context)

    def test_short_sentence_still_returned(self):
        context = extract_claim_context("[8] note", 8)
        self.assertIsNotNone(context)
        self.assertIn("[8]", context)

    def test_missing_citation_returns_none(self):
        self.assertIsNone(extract_claim_context("Texte sans citation.", 12))

    def test_provenance_preserved(self):
        source = {
            "index": 38,
            "title": "Décret n° 2026-683",
            "url": "https://lacour-avocat.fr/x",
            "source_searches": ["mard_textes"],
            "original_indices": {"mard_textes": 4},
        }
        local_answers = {
            "mard_textes": "Le décret publié le 29 juillet 2026 figure dans les résultats[4]."
        }
        contexts = build_claim_contexts_for_source(source, local_answers)
        self.assertEqual(len(contexts), 1)
        entry = contexts[0]
        self.assertEqual(entry["search_name"], "mard_textes")
        self.assertEqual(entry["local_citation"], 4)
        self.assertEqual(entry["global_index"], 38)
        self.assertIn("29 juillet 2026", entry["claim_context"])


class TitleDateTests(unittest.TestCase):
    def test_title_date_jorf(self):
        value, confidence = extract_title_date(
            "JORF n° 0199 du 27 août 2026 - Légifrance"
        )
        self.assertEqual(value, "2026-08-27")
        self.assertEqual(confidence, "high")

    def test_title_date_cassation_bare(self):
        value, confidence = extract_title_date(
            "Cour de cassation, civile, Chambre civile 3, 8 janvier 2026, "
            "23-22.803, Publié au bulletin - Légifrance"
        )
        self.assertEqual(value, "2026-01-08")
        self.assertEqual(confidence, "medium")

    def test_title_date_none(self):
        self.assertEqual(
            extract_title_date("Décret n° 2026-683 : Réforme Justice Négociée et MARD"),
            (None, None),
        )

    def test_title_date_empty(self):
        self.assertEqual(extract_title_date(""), (None, None))

    def test_payload_contains_title_date(self):
        source = {
            "index": 6,
            "title": "JORF n° 0199 du 27 août 2026 - Légifrance",
            "url": "https://www.legifrance.gouv.fr/jorf/id/0",
        }
        payload = build_requalification_payload(source, {}, [])
        self.assertEqual(payload["title_date"], "2026-08-27")
        self.assertEqual(payload["title_date_confidence"], "high")


class EligibilityTests(unittest.TestCase):
    def test_mismatch_not_eligible(self):
        temporal = {"temporal_status": STATUS_MISMATCH, "temporal_role": "current"}
        eligible, reason = eligibility_reason({"index": 10}, temporal, [], run_date="2026-08-27")
        self.assertFalse(eligible)
        self.assertIn("mismatch", reason)

    def test_current_not_eligible(self):
        temporal = {"temporal_status": STATUS_CURRENT}
        eligible, _ = eligibility_reason({"index": 1}, temporal, [])
        self.assertFalse(eligible)

    def test_unknown_with_context_eligible(self):
        temporal = {"temporal_status": STATUS_UNKNOWN}
        contexts = [{"claim_context": "Un article publié le 25 août 2026 présente une procédure[8]."}]
        eligible, reason = eligibility_reason({"index": 33}, temporal, contexts)
        self.assertTrue(eligible)
        self.assertIn("unknown", reason)

    def test_unknown_without_matter_not_eligible(self):
        temporal = {"temporal_status": STATUS_UNKNOWN}
        eligible, _ = eligibility_reason({"index": 60}, temporal, [])
        self.assertFalse(eligible)

    def test_context_without_recent_signal_not_eligible(self):
        temporal = {"temporal_status": STATUS_CONTEXT, "modified_date": "2026-01-01"}
        eligible, _ = eligibility_reason({"index": 37}, temporal, [], run_date="2026-08-27")
        self.assertFalse(eligible)

    def test_context_with_recent_modified_eligible_but_no_current(self):
        temporal = {
            "temporal_status": STATUS_CONTEXT,
            "source_date": "2026-07-29",
            "modified_date": "2026-08-24",
            "date_confidence": "high",
        }
        eligible, reason = eligibility_reason(
            {"index": 38, "title": "Décret n° 2026-683"}, temporal, [], run_date="2026-08-27"
        )
        self.assertTrue(eligible)
        self.assertIn("modified_date", reason)
        ok, guard = context_to_current_allowed(
            temporal, [], "Décret n° 2026-683", run_date="2026-08-27"
        )
        self.assertFalse(ok)
        self.assertIn("insuffisante", guard)
        transitions = allowed_transitions(STATUS_CONTEXT, context_to_current_ok=ok)
        self.assertEqual(transitions["allowed"], [STATUS_CONTEXT])


class TransitionTests(unittest.TestCase):
    def test_unknown_allows_current_and_context(self):
        transitions = allowed_transitions(STATUS_UNKNOWN)
        self.assertEqual(transitions["allowed"], [STATUS_CURRENT, STATUS_CONTEXT])
        self.assertIn(STATUS_MISMATCH, transitions["forbidden"])

    def test_context_update_claim_allows_current(self):
        temporal = {
            "temporal_status": STATUS_CONTEXT,
            "modified_date": "2026-08-24",
            "claimed_updates": ["2026-08-24"],
        }
        ok, reason = context_to_current_allowed(temporal, [], "Titre", run_date="2026-08-27")
        self.assertTrue(ok)
        self.assertIn("update_claim", reason)
        transitions = allowed_transitions(STATUS_CONTEXT, context_to_current_ok=ok)
        self.assertEqual(transitions["allowed"], [STATUS_CONTEXT, STATUS_CURRENT])

    def test_context_visible_update_allows_current(self):
        temporal = {"temporal_status": STATUS_CONTEXT, "visible_update_date": "2026-08-26"}
        ok, _ = context_to_current_allowed(temporal, [], "Titre", run_date="2026-08-27")
        self.assertTrue(ok)

    def test_context_visible_publication_allows_current(self):
        temporal = {"temporal_status": STATUS_CONTEXT, "visible_publication_date": "2026-08-27"}
        ok, _ = context_to_current_allowed(temporal, [], "Titre", run_date="2026-08-27")
        self.assertTrue(ok)

    def test_context_title_date_allows_current(self):
        temporal = {"temporal_status": STATUS_CONTEXT, "source_date": "2025-07-18"}
        ok, _ = context_to_current_allowed(
            temporal, [], "Décret n° 2026-778 du 13 août 2026", run_date="2026-08-27"
        )
        self.assertTrue(ok)

    def test_context_modified_plus_update_formulation_allows_current(self):
        temporal = {"temporal_status": STATUS_CONTEXT, "modified_date": "2026-08-24"}
        contexts = [{"claim_context": "La fiche a été mise à jour le 24 août 2026[8]."}]
        ok, _ = context_to_current_allowed(temporal, contexts, "Titre", run_date="2026-08-27")
        self.assertTrue(ok)

    def test_context_modified_alone_not_enough(self):
        temporal = {"temporal_status": STATUS_CONTEXT, "modified_date": "2026-08-24"}
        ok, guard = context_to_current_allowed(temporal, [], "Titre", run_date="2026-08-27")
        self.assertFalse(ok)
        self.assertIn("insuffisante", guard)

    def test_current_no_transition(self):
        transitions = allowed_transitions(STATUS_CURRENT)
        self.assertEqual(transitions["allowed"], [STATUS_CURRENT])

    def test_mismatch_no_transition(self):
        transitions = allowed_transitions(STATUS_MISMATCH)
        self.assertEqual(transitions["allowed"], [STATUS_MISMATCH])


def make_payload(**overrides):
    payload = {
        "source_number": 38,
        "title": "Décret n° 2026-683 : Réforme Justice Négociée et MARD",
        "domain": "lacour-avocat.fr",
        "python_status": STATUS_CONTEXT,
        "source_date": "2026-07-29",
        "modified_date": "2026-08-24",
        "visible_publication_date": None,
        "visible_update_date": None,
        "date_confidence": "high",
        "claimed_dates": ["2026-07-29"],
        "claimed_updates": [],
        "access_status": "accessible",
        "claim_types": ["publication_claim"],
        "claim_contexts": [
            {
                "search_name": "mard_textes",
                "local_citation": 4,
                "global_index": 38,
                "claim_context": "Le document le plus récent est le décret publié le 29 juillet 2026[4].",
                "claim_types": ["publication_claim"],
            }
        ],
    }
    payload.update(overrides)
    return payload


class ResponseValidationTests(unittest.TestCase):
    def test_valid_response_accepted(self):
        ok, error, normalized = validate_gemma_response(
            make_payload(),
            {
                "source_number": 38,
                "recommended_status": STATUS_CONTEXT,
                "confidence": "high",
                "reason_code": "role_context_explicit",
                "reason": "Le décret est cité comme cadre réglementaire, pas comme nouveauté de la période.",
            },
        )
        self.assertTrue(ok, error)
        self.assertTrue(normalized["applied"])

    def test_wrong_source_number_rejected(self):
        ok, error, _ = validate_gemma_response(
            make_payload(),
            {
                "source_number": 39,
                "recommended_status": STATUS_CONTEXT,
                "confidence": "high",
                "reason_code": "role_context_explicit",
                "reason": "Cadre réglementaire.",
            },
        )
        self.assertFalse(ok)
        self.assertIn("source_number", error)

    def test_bad_status_rejected(self):
        ok, error, _ = validate_gemma_response(
            make_payload(),
            {
                "source_number": 38,
                "recommended_status": "autre",
                "confidence": "high",
                "reason_code": "no_signal",
                "reason": "X.",
            },
        )
        self.assertFalse(ok)
        self.assertIn("recommended_status", error)

    def test_forbidden_transition_unknown_to_mismatch(self):
        payload = make_payload(python_status=STATUS_UNKNOWN)
        ok, error, _ = validate_gemma_response(
            payload,
            {
                "source_number": 38,
                "recommended_status": STATUS_MISMATCH,
                "confidence": "high",
                "reason_code": "mismatch_confirmed",
                "reason": "X.",
            },
        )
        self.assertFalse(ok)
        self.assertIn("transition", error)

    def test_confidence_low_keeps_python_status(self):
        payload = make_payload(python_status=STATUS_UNKNOWN)
        ok, error, normalized = validate_gemma_response(
            payload,
            {
                "source_number": 38,
                "recommended_status": STATUS_CURRENT,
                "confidence": "low",
                "reason_code": "role_current_publication_context",
                "reason": "Publication récente signalée.",
            },
        )
        self.assertTrue(ok, error)
        self.assertFalse(normalized["applied"])

    def test_new_date_in_reason_rejected(self):
        ok, error, _ = validate_gemma_response(
            make_payload(),
            {
                "source_number": 38,
                "recommended_status": STATUS_CONTEXT,
                "confidence": "high",
                "reason_code": "role_context_explicit",
                "reason": "Document daté du 12 janvier 2015.",
            },
        )
        self.assertFalse(ok)
        self.assertIn("date nouvelle", error)

    def test_existing_date_in_reason_accepted(self):
        ok, error, _ = validate_gemma_response(
            make_payload(),
            {
                "source_number": 38,
                "recommended_status": STATUS_CONTEXT,
                "confidence": "high",
                "reason_code": "role_context_explicit",
                "reason": "Publié le 29 juillet 2026, hors fenêtre.",
            },
        )
        self.assertTrue(ok, error)

    def test_citation_in_reason_rejected(self):
        ok, error, _ = validate_gemma_response(
            make_payload(),
            {
                "source_number": 38,
                "recommended_status": STATUS_CONTEXT,
                "confidence": "high",
                "reason_code": "role_context_explicit",
                "reason": "Voir la source [5] pour le cadre.",
            },
        )
        self.assertFalse(ok)
        self.assertIn("citation", error)

    def test_url_in_reason_rejected(self):
        ok, error, _ = validate_gemma_response(
            make_payload(),
            {
                "source_number": 38,
                "recommended_status": STATUS_CONTEXT,
                "confidence": "high",
                "reason_code": "role_context_explicit",
                "reason": "Voir https://example.com pour le cadre.",
            },
        )
        self.assertFalse(ok)
        self.assertIn("URL", error)

    def test_reason_code_enum_rejected(self):
        ok, error, _ = validate_gemma_response(
            make_payload(),
            {
                "source_number": 38,
                "recommended_status": STATUS_CONTEXT,
                "confidence": "high",
                "reason_code": "nimporte_quoi",
                "reason": "Cadre réglementaire.",
            },
        )
        self.assertFalse(ok)
        self.assertIn("reason_code", error)


class ApplyTests(unittest.TestCase):
    def test_apply_updates_status_and_keeps_facts(self):
        temporal = {
            "temporal_status": STATUS_CONTEXT,
            "source_date": "2026-07-29",
            "note": "Ancienne note Python.",
        }
        recommendation = {
            "source_number": 38,
            "recommended_status": STATUS_CURRENT,
            "confidence": "high",
            "reason_code": "role_current_update_claim",
            "reason": "Mise à jour récente documentée.",
            "applied": True,
        }
        updated = apply_requalification(temporal, recommendation)
        self.assertEqual(updated["temporal_status"], STATUS_CURRENT)
        self.assertEqual(updated["source_date"], "2026-07-29")
        self.assertIn("Requalifié Gemma", updated["note"])
        self.assertTrue(updated["requalification"]["applied"])

    def test_apply_low_confidence_keeps_status(self):
        temporal = {"temporal_status": STATUS_CONTEXT, "source_date": "2026-07-29"}
        recommendation = {
            "recommended_status": STATUS_CURRENT,
            "confidence": "low",
            "reason_code": "role_current_update_claim",
            "reason": "Hypothèse.",
            "applied": False,
        }
        updated = apply_requalification(temporal, recommendation)
        self.assertEqual(updated["temporal_status"], STATUS_CONTEXT)
        self.assertFalse(updated["requalification"]["applied"])

    def test_apply_same_status_not_applied(self):
        temporal = {"temporal_status": STATUS_CONTEXT}
        recommendation = {
            "recommended_status": STATUS_CONTEXT,
            "confidence": "high",
            "reason_code": "role_context_explicit",
            "reason": "Confirmation du cadre.",
            "applied": True,
        }
        updated = apply_requalification(temporal, recommendation)
        self.assertEqual(updated["temporal_status"], STATUS_CONTEXT)
        self.assertFalse(updated["requalification"]["applied"])


class PayloadTests(unittest.TestCase):
    def test_payload_fields_and_no_url(self):
        source = {
            "index": 38,
            "title": "Décret n° 2026-683",
            "url": "https://lacour-avocat.fr/x/y/z",
            "source_searches": ["mard_textes"],
            "original_indices": {"mard_textes": 4},
        }
        temporal = {
            "temporal_status": STATUS_CONTEXT,
            "temporal_role": "current",
            "source_date": "2026-07-29",
            "modified_date": "2026-08-24",
            "date_confidence": "high",
            "claimed_dates": ["2026-07-29"],
            "access_status": "accessible",
        }
        contexts = [
            {
                "search_name": "mard_textes",
                "local_citation": 4,
                "global_index": 38,
                "claim_context": "Le décret publié le 29 juillet 2026[4].",
                "claim_types": ["publication_claim"],
            }
        ]
        payload = build_requalification_payload(source, temporal, contexts)
        for key in (
            "source_number",
            "title",
            "domain",
            "python_status",
            "source_date",
            "modified_date",
            "claimed_dates",
            "access_status",
            "claim_types",
            "claim_contexts",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["domain"], "lacour-avocat.fr")
        self.assertNotIn("url", payload)


class ReasonCodeCoherenceTests(unittest.TestCase):
    def _payload(self, **overrides):
        payload = {
            "source_number": 6,
            "title": "JORF n° 0199 du 27 août 2026 - Légifrance",
            "title_date": "2026-08-27",
            "python_status": "unknown",
            "claimed_updates": [],
            "visible_update_date": None,
            "claim_types": [],
            "claim_contexts": [],
        }
        payload.update(overrides)
        return payload

    def _response(self, status="current", reason_code="role_current_title_date", confidence="high"):
        return {
            "source_number": 6,
            "recommended_status": status,
            "confidence": confidence,
            "reason_code": reason_code,
            "reason": "Justification de test.",
        }

    def test_title_date_in_window_accepted(self):
        ok, _, normalized = validate_gemma_response_v1(
            self._payload(), self._response(), run_date="2026-08-27"
        )
        self.assertTrue(ok)
        self.assertEqual(normalized["reason_code"], "role_current_title_date")
        self.assertEqual(normalized["recommended_status"], "current")

    def test_title_date_without_title_date_rejected(self):
        payload = self._payload(title_date=None, claim_contexts=[])
        ok, why, _ = validate_gemma_response_v1(
            payload, self._response(), run_date="2026-08-27"
        )
        self.assertFalse(ok)
        self.assertIn("role_current_title_date sans title_date", why)

    def test_title_date_out_of_window_rejected(self):
        payload = self._payload(title_date="2026-01-08")
        ok, why, _ = validate_gemma_response_v1(
            payload, self._response(), run_date="2026-08-27"
        )
        self.assertFalse(ok)
        self.assertIn("hors fenêtre", why)

    def test_circulaire_out_of_window_without_signal_not_current(self):
        payload = self._payload(
            title_date=None,
            source_date="2026-08-12",
            claim_contexts=[
                {
                    "claim_context": (
                        "Circulaire (n° 6553/SG du 12 août 2026) diffusée "
                        "comme cadre pour l'exécution des contrats[9]."
                    )
                }
            ],
        )
        ok, why, _ = validate_gemma_response_v1(
            payload,
            self._response(reason_code="role_current_publication_context"),
            run_date="2026-08-27",
        )
        self.assertFalse(ok)
        self.assertIn("sans signal récent explicite", why)

    def test_out_of_window_with_recent_signal_current_possible(self):
        payload = self._payload(
            title_date=None,
            source_date="2026-08-12",
            claimed_updates=["2026-08-26"],
        )
        ok, _, normalized = validate_gemma_response_v1(
            payload,
            self._response(reason_code="role_current_update_claim"),
            run_date="2026-08-27",
        )
        self.assertTrue(ok)
        self.assertEqual(normalized["recommended_status"], "current")

    def test_title_date_missing_with_recent_signal_normalized(self):
        payload = self._payload(
            title_date=None,
            claim_contexts=[
                {"claim_context": "une nouvelle étape à partir du 22 août 2026[9]."}
            ],
        )
        ok, _, normalized = validate_gemma_response_v1(
            payload,
            self._response(reason_code="role_current_title_date"),
            run_date="2026-08-27",
        )
        self.assertTrue(ok)
        self.assertEqual(normalized["reason_code"], "role_current_publication_context")

    def test_update_claim_without_update_signal_rejected(self):
        payload = self._payload(
            title_date=None, claimed_updates=[], visible_update_date=None
        )
        ok, why, _ = validate_gemma_response_v1(
            payload,
            self._response(reason_code="role_current_update_claim"),
            run_date="2026-08-27",
        )
        self.assertFalse(ok)
        self.assertIn("role_current_update_claim sans", why)

    def test_context_legal_text_without_legal_signal_rejected(self):
        payload = self._payload(
            title="Publication quelconque",
            claim_types=[],
            claim_contexts=[
                {"claim_context": "Information générale sans élément juridique[4]."}
            ],
        )
        ok, why, _ = validate_gemma_response_v1(
            payload,
            self._response(status="context", reason_code="role_context_legal_text"),
            run_date="2026-08-27",
        )
        self.assertFalse(ok)
        self.assertIn("role_context_legal_text sans signal juridique", why)


class PlanTests(unittest.TestCase):
    def test_plan_10_mismatch_not_eligible(self):
        source = {
            "index": 10,
            "title": "Expertise judiciaire : transmission électronique",
            "url": "https://www.village-justice.com/articles/x",
        }
        temporal = {
            "temporal_status": STATUS_MISMATCH,
            "temporal_role": "current",
            "source_date": "2017-11-15",
        }
        plan = requalification_plan(source, temporal, {}, run_date="2026-08-27")
        self.assertFalse(plan["eligible"])
        self.assertEqual(plan["python_status"], STATUS_MISMATCH)

    def test_plan_1_current_not_eligible(self):
        source = {"index": 1, "title": "t", "url": "https://aicvf.org/x"}
        temporal = {"temporal_status": STATUS_CURRENT, "temporal_role": "current"}
        plan = requalification_plan(source, temporal, {}, run_date="2026-08-27")
        self.assertFalse(plan["eligible"])

    def test_plan_38_context_eligible_but_current_blocked(self):
        source = {
            "index": 38,
            "title": "Décret n° 2026-683 : Réforme Justice Négociée et MARD",
            "url": "https://lacour-avocat.fr/x",
            "source_searches": ["mard_textes"],
            "original_indices": {"mard_textes": 4},
        }
        temporal = {
            "temporal_status": STATUS_CONTEXT,
            "temporal_role": "current",
            "source_date": "2026-07-29",
            "modified_date": "2026-08-24",
            "date_confidence": "high",
            "claimed_dates": ["2026-07-29"],
            "access_status": "accessible",
        }
        local_answers = {
            "mard_textes": "Le document le plus récent est le décret publié le 29 juillet 2026[4]."
        }
        plan = requalification_plan(source, temporal, local_answers, run_date="2026-08-27")
        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["python_status"], STATUS_CONTEXT)
        self.assertNotIn(STATUS_CURRENT, plan["transitions"]["allowed"])
        self.assertTrue(plan["guardrails"])
        self.assertTrue(any("insuffisante" in guard for guard in plan["guardrails"]))
        self.assertEqual(plan["payload"]["source_number"], 38)





class V2ContextAndSignalsTests(unittest.TestCase):
    """V2 : contexte de citation + voisins + signaux temporels récents."""

    def test_signal_in_previous_sentence(self):
        answer = (
            "Le site a été mis à jour le 26 août 2026[5]. "
            "La circulaire diffusée pour l'exécution des contrats est détaillée[9]."
        )
        enriched = extract_claim_and_neighbor_contexts(
            answer, 9, run_date="2026-08-27", window_days=7, recent_tolerance_days=7
        )
        self.assertIn("circulaire", enriched["claim_context"])
        self.assertIn("mis à jour", enriched["neighbor_context"])
        self.assertTrue(enriched["recent_context_signals"])
        signal = enriched["recent_context_signals"][0]
        self.assertEqual(signal["type"], "recent_update")
        self.assertEqual(signal["date"], "2026-08-26")
        self.assertEqual(signal["proximity"], "previous_sentence")

    def test_signal_in_next_sentence(self):
        answer = (
            "La DAJ a publié une information relative à une circulaire[9]. "
            "Cette information marque une nouvelle étape à partir du 22 août 2026[9]."
        )
        enriched = extract_claim_and_neighbor_contexts(
            answer, 9, run_date="2026-08-27", window_days=7, recent_tolerance_days=7
        )
        self.assertIn("circulaire", enriched["claim_context"])
        self.assertIn("nouvelle étape", enriched["neighbor_context"])
        self.assertTrue(enriched["recent_context_signals"])
        signal = enriched["recent_context_signals"][0]
        self.assertEqual(signal["type"], "recent_event")
        self.assertEqual(signal["date"], "2026-08-22")
        self.assertEqual(signal["proximity"], "next_sentence")

    def test_no_paragraph_boundary_crossing(self):
        answer = (
            "## 1) Titre de section\n"
            "Une information antérieure datée du 1er janvier 2026 ne doit pas "
            "être rattachée à la citation suivante[4].\n"
            "\n"
            "## 2) Autre section\n"
            "La citation porte sur la circulaire publiée le 12 août 2026[9]."
        )
        enriched = extract_claim_and_neighbor_contexts(
            answer, 9, run_date="2026-08-27", window_days=7, recent_tolerance_days=7
        )
        self.assertIn("circulaire", enriched["claim_context"])
        self.assertNotIn("janvier 2026", enriched["claim_context"])
        self.assertNotIn("Titre de section", enriched["neighbor_context"] or "")
        self.assertNotIn("Autre section", enriched["neighbor_context"] or "")
        for signal in enriched["recent_context_signals"]:
            self.assertNotEqual(signal["date"], "2026-01-01")

    def test_legal_text_date_not_recent_signal(self):
        answer = (
            "Un article publié le 25 août 2026 présente une procédure issue "
            "d'un décret du 5 août 2025[8]."
        )
        enriched = extract_claim_and_neighbor_contexts(
            answer, 8, run_date="2026-08-27", window_days=7, recent_tolerance_days=7
        )
        signals = enriched["recent_context_signals"]
        dates = {signal["date"] for signal in signals}
        self.assertNotIn("2025-08-05", dates)

    def test_nouvelle_etape_from_date(self):
        answer = "La DAJ signale une nouvelle étape à partir du 22 août 2026 dans le suivi de ces questions[6]."
        enriched = extract_claim_and_neighbor_contexts(
            answer, 6, run_date="2026-08-27", window_days=7, recent_tolerance_days=7
        )
        self.assertTrue(enriched["recent_context_signals"])
        signal = enriched["recent_context_signals"][0]
        self.assertEqual(signal["type"], "recent_event")
        self.assertEqual(signal["date"], "2026-08-22")

    def test_mise_a_jour_range(self):
        answer = (
            "Mises à jour du Code des assurances avec mentions datées "
            "du 22/08/2026 au 26/08/2026, ce qui indique des actualisations "
            "réglementaires récentes[47][46]."
        )
        enriched = extract_claim_and_neighbor_contexts(
            answer, 47, run_date="2026-08-27", window_days=7, recent_tolerance_days=7
        )
        self.assertTrue(enriched["recent_context_signals"])
        signal = enriched["recent_context_signals"][0]
        self.assertEqual(signal["type"], "recent_update")
        self.assertEqual(signal["date"], "2026-08-26")

    def test_claim_with_signal_preferred_over_introduction(self):
        # Régression [29] : une puce datée avec signal récent doit gagner
        # contre une phrase d'introduction générique qui contient la citation.
        answer = (
            "Introduction J'ai examiné les résultats fournis pour repérer "
            "les actualités récentes sur l'assurance construction[1][47][31].\n\n"
            "- Mises à jour du Code des assurances (pages Légifrance) datées "
            "du 22/08/2026 au 26/08/2026, ce qui indique des actualisations "
            "réglementaires récentes[47][46].\n\n"
            "- Pages Légifrance liées au Code des assurances[47][41]."
        )
        enriched = extract_claim_and_neighbor_contexts(
            answer, 47, run_date="2026-08-27", window_days=7, recent_tolerance_days=7
        )
        self.assertIn("Mises à jour du Code des assurances", enriched["claim_context"])
        self.assertNotIn("Introduction", enriched["claim_context"])
        self.assertTrue(enriched["recent_context_signals"])
        signal = enriched["recent_context_signals"][0]
        self.assertEqual(signal["type"], "recent_update")
        self.assertEqual(signal["date"], "2026-08-26")
        self.assertEqual(signal["proximity"], "same_sentence")

    def test_heading_not_merged_into_claim(self):
        # Régression [33] : un titre markdown ne doit pas être fusionné dans
        # la phrase de citation ; le signal voisin reste exploitable.
        answer = (
            "## 1) Titre de la circulaire\n"
            "La DAJ a publié une information indiquant la diffusion d'une "
            "circulaire (n° 6553/SG du 12 août 2026)[9].\n"
            "Cette information a été mise en ligne récemment et signale une "
            "nouvelle étape à partir du 22 août 2026[9][6]."
        )
        enriched = extract_claim_and_neighbor_contexts(
            answer, 9, run_date="2026-08-27", window_days=7, recent_tolerance_days=7
        )
        self.assertIn("DAJ", enriched["claim_context"])
        self.assertNotIn("Titre de la circulaire", enriched["claim_context"])
        self.assertIn("mise en ligne", enriched["neighbor_context"] or "")
        self.assertTrue(enriched["recent_context_signals"])
        signal = enriched["recent_context_signals"][0]
        self.assertEqual(signal["proximity"], "next_sentence")

    def test_reason_code_coherent_with_recent_context(self):
        payload = {
            "source_number": 33,
            "title": "Publication d'une circulaire",
            "title_date": None,
            "python_status": "unknown",
            "claimed_updates": [],
            "visible_update_date": None,
            "claim_types": [],
            "recent_context_signals": [
                {
                    "type": "recent_event",
                    "date": "2026-08-22",
                    "text": "Nouvelle étape à partir du 22 août 2026.",
                    "proximity": "next_sentence",
                }
            ],
            "claim_contexts": [],
        }
        ok, _, normalized = validate_gemma_response_v1(
            payload,
            {
                "source_number": 33,
                "recommended_status": "current",
                "confidence": "high",
                "reason_code": "role_current_recent_context",
                "reason": "Nouvelle étape récente dans la fenêtre de veille.",
            },
            run_date="2026-08-27",
            window_days=7,
            recent_tolerance_days=7,
        )
        self.assertTrue(ok)
        self.assertEqual(normalized["reason_code"], "role_current_recent_context")

    def test_payload_includes_neighbor_and_signals(self):
        source = {
            "index": 33,
            "title": "Publication d'une circulaire",
            "url": "https://www.economie.gouv.fr/daj/publication",
            "source_searches": ["mediation"],
            "original_indices": {"mediation": 9},
        }
        local_answers = {
            "mediation": (
                "La DAJ a publié une information indiquant la diffusion d'une "
                "circulaire (n° 6553/SG du 12 août 2026)[9]. "
                "Cette information a été mise en ligne récemment et signale une "
                "nouvelle étape à partir du 22 août 2026 dans le suivi[6]."
            )
        }
        contexts = build_claim_contexts_for_source(
            source, local_answers, run_date="2026-08-27", window_days=7, recent_tolerance_days=7
        )
        payload = build_requalification_payload(source, {}, contexts)
        self.assertEqual(payload["source_number"], 33)
        self.assertTrue(payload["recent_context_signals"])
        entry = contexts[0]
        self.assertIn("neighbor_context", entry)
        self.assertIn("recent_context_signals", entry)
        self.assertIn("DAJ", entry["claim_context"])
        self.assertIn("mise en ligne", entry["neighbor_context"])


class WindowModeTests(unittest.TestCase):
    def test_default_is_strict_7d(self):
        self.assertEqual(resolve_window_params(None), (7, 7))
        self.assertEqual(resolve_window_params(WINDOW_STRICT_7D), (7, 7))

    def test_extended_30d(self):
        self.assertEqual(resolve_window_params(WINDOW_EXTENDED_30D), (30, 7))

    def test_explicit_mode_overrides_raw_params(self):
        self.assertEqual(
            resolve_window_params(WINDOW_EXTENDED_30D, window_days=15, recent_tolerance_days=3),
            (30, 7),
        )

    def test_suites_procedurales_recentes_strict_vs_extended(self):
        sentences = [
            "La cour a tenu audience d'incident le 01/07/2026 et a mis l'affaire "
            "en délibéré au 12/08/2026, montrant des suites procédurales récentes[5]."
        ]
        strict = extract_recent_context_signals(
            sentences, 5, run_date="2026-08-27", window_mode=WINDOW_STRICT_7D
        )
        self.assertEqual(strict, [])
        extended = extract_recent_context_signals(
            sentences, 5, run_date="2026-08-27", window_mode=WINDOW_EXTENDED_30D
        )
        self.assertEqual(len(extended), 1)
        self.assertEqual(extended[0]["type"], "recent_event")
        self.assertEqual(extended[0]["date"], "2026-08-12")

    def test_designation_expert_le_date(self):
        sentences = [
            "Un dossier local mentionne la désignation d'un expert le 17/08/2026 "
            "pour un contentieux en cours[21]."
        ]
        signals = extract_recent_context_signals(
            sentences, 21, run_date="2026-08-27", window_mode=WINDOW_EXTENDED_30D
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "recent_event")
        self.assertEqual(signals[0]["date"], "2026-08-17")

    def test_recentes_decisions_with_date(self):
        sentences = [
            "Plusieurs récentes décisions ont été publiées le 22 août 2026[3]."
        ]
        signals = extract_recent_context_signals(
            sentences, 3, run_date="2026-08-27", window_mode=WINDOW_STRICT_7D
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["date"], "2026-08-22")


class DeterministicPrePositioningTests(unittest.TestCase):
    def _payload(self, **overrides):
        payload = {
            "source_number": 48,
            "title": ("Cour de cassation, civile, Chambre civile 3, 8 janvier 2026, "
                      "23-22.803, Publié au bulletin - Légifrance"),
            "title_date": "2026-01-08",
            "python_status": "unknown",
            "domain": "legifrance.gouv.fr",
            "claimed_updates": [],
            "visible_update_date": None,
            "claim_types": [],
            "claim_contexts": [],
            "recent_context_signals": [],
        }
        payload.update(overrides)
        return payload

    def test_old_title_date_blocks_current_and_forces_context(self):
        pre = deterministic_pre_positioning(self._payload(), run_date="2026-08-27")
        self.assertFalse(pre["current_allowed"])
        self.assertEqual(pre["forced_status"], "context")
        self.assertIn("hors fenêtre", pre["reason"])

    def test_old_title_date_with_recent_signal_allows_current(self):
        pre = deterministic_pre_positioning(
            self._payload(
                recent_context_signals=[
                    {"type": "recent_update", "date": "2026-08-26", "proximity": "same_sentence"}
                ]
            ),
            run_date="2026-08-27",
        )
        self.assertTrue(pre["current_allowed"])

    def test_recent_title_date_allows_current(self):
        pre = deterministic_pre_positioning(
            self._payload(title_date="2026-08-27"), run_date="2026-08-27"
        )
        self.assertTrue(pre["current_allowed"])

    def test_old_decision_context_blocks_current(self):
        pre = deterministic_pre_positioning(
            self._payload(
                title_date=None,
                claim_contexts=[
                    {
                        "claim_context": (
                            "Je n'ai pas identifié de décision judiciaire nouvelle rendue "
                            "entre le 20/08/2026 et le 27/08/2026 (les décisions portent "
                            "majoritairement des dates antérieures à cette fenêtre)[1]."
                        )
                    }
                ],
            ),
            run_date="2026-08-27",
        )
        self.assertFalse(pre["current_allowed"])
        self.assertIsNone(pre["forced_status"])

    def test_48_gemma_current_rejected_and_context_forced(self):
        payload = self._payload()
        ok, _, _ = validate_gemma_response_v1(
            payload,
            {
                "source_number": 48,
                "recommended_status": "current",
                "confidence": "high",
                "reason_code": "role_current_recent_context",
                "reason": "La date du titre est comprise dans la fenêtre de veille.",
            },
            run_date="2026-08-27",
        )
        self.assertFalse(ok)
        pre = deterministic_pre_positioning(payload, run_date="2026-08-27")
        final_status, note = final_status_with_pre_positioning(
            "unknown", {"recommended_status": "current", "applied": True}, False, pre
        )
        self.assertEqual(final_status, "context")
        self.assertIn("pré-positionnement", note)

    def test_13_payload_not_forced_by_pre_positioning(self):
        payload = {
            "source_number": 13,
            "title": "Décision Tribunal judiciaire d'Évry : RG n°26/00086",
            "title_date": None,
            "python_status": "unknown",
            "domain": "courdecassation.fr",
            "claim_contexts": [
                {
                    "claim_context": (
                        "Plusieurs décisions de tribunaux judiciaires relatives à "
                        "contentieux techniques ont été publiées en août 2026 et "
                        "figurent dans les résultats, avec des dates explicites[2]."
                    )
                }
            ],
        }
        pre = deterministic_pre_positioning(payload, run_date="2026-08-27")
        self.assertTrue(pre["current_allowed"])
        self.assertIsNone(pre["forced_status"])


class LegalUpdateReasonCodeTests(unittest.TestCase):
    def _payload(self, **overrides):
        payload = {
            "source_number": 29,
            "title": "Code des assurances - Légifrance",
            "title_date": None,
            "python_status": "unknown",
            "domain": "legifrance.gouv.fr",
            "claimed_updates": [],
            "visible_update_date": None,
            "claim_types": [],
            "recent_context_signals": [
                {
                    "type": "recent_update",
                    "date": "2026-08-26",
                    "text": ("Mises à jour du Code des assurances (pages Légifrance) "
                             "avec mentions Version en vigueur datées du 22/08/2026 "
                             "au 26/08/2026."),
                    "proximity": "same_sentence",
                }
            ],
            "claim_contexts": [
                {
                    "claim_context": (
                        "Mises à jour du Code des assurances (pages Légifrance) avec "
                        "mentions « Version en vigueur » datées du 22/08/2026 au "
                        "26/08/2026[47]."
                    )
                }
            ],
        }
        payload.update(overrides)
        return payload

    def test_recent_context_normalized_to_legal_update(self):
        ok, _, normalized = validate_gemma_response_v1(
            self._payload(),
            {
                "source_number": 29,
                "recommended_status": "current",
                "confidence": "high",
                "reason_code": "role_current_recent_context",
                "reason": "Mise à jour du Code des assurances.",
            },
            run_date="2026-08-27",
        )
        self.assertTrue(ok)
        self.assertEqual(normalized["reason_code"], "role_current_legal_update")

    def test_legal_update_code_coherent(self):
        ok, effective, _ = reason_code_coherent(
            self._payload(), "role_current_legal_update", run_date="2026-08-27"
        )
        self.assertTrue(ok)
        self.assertEqual(effective, "role_current_legal_update")

    def test_legal_update_without_evidence_rejected(self):
        ok, _, _ = reason_code_coherent(
            self._payload(recent_context_signals=[], claim_contexts=[]),
            "role_current_legal_update",
            run_date="2026-08-27",
        )
        self.assertFalse(ok)

    def test_recent_event_not_normalized_to_legal_update(self):
        ok, _, normalized = validate_gemma_response_v1(
            self._payload(
                recent_context_signals=[
                    {
                        "type": "recent_event",
                        "date": "2026-08-22",
                        "text": "Nouvelle étape à partir du 22 août 2026.",
                        "proximity": "same_sentence",
                    }
                ],
                claim_contexts=[
                    {"claim_context": "Nouvelle étape à partir du 22 août 2026[9]."}
                ],
            ),
            {
                "source_number": 29,
                "recommended_status": "current",
                "confidence": "high",
                "reason_code": "role_current_recent_context",
                "reason": "Nouvelle étape récente.",
            },
            run_date="2026-08-27",
        )
        self.assertTrue(ok)
        self.assertEqual(normalized["reason_code"], "role_current_recent_context")


class RelaxedLegalContextTests(unittest.TestCase):
    def test_pdf_justice_manifest_legal_context_accepted(self):
        # Cas [3] : PDF « La justice et les experts de justice » sans marqueur
        # juridique explicite précis dans le claim_context.
        payload = {
            "source_number": 3,
            "title": "La justice et les experts de justice à l'heure de la dématérialisation",
            "title_date": None,
            "python_status": "unknown",
            "domain": "cours-appel.justice.fr",
            "claim_types": [],
            "claim_contexts": [
                {
                    "claim_context": (
                        "La tenue des listes d'experts et la formalisation des fiches "
                        "expert (RPJ) sont documentées dans les textes et guides "
                        "administratifs disponibles[20][46]."
                    )
                }
            ],
        }
        ok, _, normalized = validate_gemma_response_v1(
            payload,
            {
                "source_number": 3,
                "recommended_status": "context",
                "confidence": "high",
                "reason_code": "role_context_legal_text",
                "reason": "Document de référence sur l'organisation des experts.",
            },
            run_date="2026-08-27",
        )
        self.assertTrue(ok)
        self.assertEqual(normalized["recommended_status"], "context")

    def test_legal_domain_alone_accepted(self):
        payload = {
            "source_number": 60,
            "title": "Vos droits et démarches",
            "domain": "justice.fr",
            "python_status": "unknown",
            "claim_types": [],
            "claim_contexts": [],
        }
        ok, _, _ = validate_gemma_response_v1(
            payload,
            {
                "source_number": 60,
                "recommended_status": "context",
                "confidence": "high",
                "reason_code": "role_context_legal_text",
                "reason": "Portail institutionnel de référence.",
            },
            run_date="2026-08-27",
        )
        self.assertTrue(ok)

    def test_non_legal_source_still_rejected(self):
        payload = {
            "source_number": 61,
            "title": "Publication quelconque",
            "domain": "exemple.fr",
            "python_status": "unknown",
            "claim_types": [],
            "claim_contexts": [
                {"claim_context": "Information générale sans élément juridique[4]."}
            ],
        }
        ok, why, _ = validate_gemma_response_v1(
            payload,
            {
                "source_number": 61,
                "recommended_status": "context",
                "confidence": "high",
                "reason_code": "role_context_legal_text",
                "reason": "Cadre général.",
            },
            run_date="2026-08-27",
        )
        self.assertFalse(ok)
        self.assertIn("role_context_legal_text sans signal juridique", why)


class StrictSignalGuardrailTests(unittest.TestCase):
    """Correctif V3bis : les marqueurs génériques dérivés d'une phrase méta
    de l'answer (ex. négation « aucune décision nouvelle entre le 20/08 et le
    27/08 ») ne doivent pas neutraliser le pré-positionnement déterministe.
    Cas réel : [48] Cass. 3e civ. 8 janvier 2026."""

    REAL_48_CONTEXT = (
        "Si votre besoin est de repérer des arrêts intervenus strictement dans "
        "les 7 derniers jours (20–27 août 2026), les sources fournies ne "
        "contiennent pas, pour cette fenêtre, de décision nouvelle applicable "
        "au cœur des thèmes demandés, et il faudra étendre la recherche ou "
        "interroger directement les bases[21]."
    )

    def _payload(self, **overrides):
        payload = {
            "source_number": 48,
            "title": ("Cour de cassation, civile, Chambre civile 3, 8 janvier 2026, "
                      "23-22.803, Publié au bulletin - Légifrance"),
            "title_date": "2026-01-08",
            "python_status": "unknown",
            "domain": "legifrance.gouv.fr",
            "claimed_updates": [],
            "visible_update_date": None,
            "claim_types": [],
            "claim_contexts": [],
            "recent_context_signals": [],
        }
        payload.update(overrides)
        return payload

    def test_48_real_negation_context_blocks_current(self):
        payload = self._payload(
            claim_contexts=[{"claim_context": self.REAL_48_CONTEXT}]
        )
        pre = deterministic_pre_positioning(payload, run_date="2026-08-27")
        self.assertFalse(pre["current_allowed"])
        self.assertEqual(pre["forced_status"], "context")
        self.assertNotIn("event_in_window", pre["recent_signals"])

    def test_48_real_context_gemma_publication_context_rejected(self):
        payload = self._payload(
            claim_contexts=[{"claim_context": self.REAL_48_CONTEXT}]
        )
        ok, why, _ = validate_gemma_response_v1(
            payload,
            {
                "source_number": 48,
                "recommended_status": "current",
                "confidence": "high",
                "reason_code": "role_current_publication_context",
                "reason": "Décision publiée au bulletin.",
            },
            run_date="2026-08-27",
        )
        self.assertFalse(ok)
        self.assertIn("sans signal récent explicite", why)

    def test_publication_context_without_evidence_rejected(self):
        ok, _, message = reason_code_coherent(
            self._payload(),
            "role_current_publication_context",
            run_date="2026-08-27",
        )
        self.assertFalse(ok)
        self.assertIn("sans signal récent explicite daté", message)

    def test_publication_context_with_dated_signal_accepted(self):
        payload = self._payload(
            title_date=None,
            recent_context_signals=[
                {"type": "recent_update", "date": "2026-08-26",
                 "proximity": "same_sentence"}
            ],
        )
        ok, _, _ = reason_code_coherent(
            payload, "role_current_publication_context", run_date="2026-08-27"
        )
        self.assertTrue(ok)

    def test_23_designation_expert_signal_not_forced(self):
        payload = self._payload(
            title_date=None,
            claim_contexts=[
                {
                    "claim_context": (
                        "Un dossier local sur la tempête Alex mentionne la "
                        "désignation d'un expert le 17/08/2026 pour un "
                        "contentieux important[21]."
                    )
                }
            ],
            recent_context_signals=[
                {"type": "recent_event", "date": "2026-08-17",
                 "proximity": "same_sentence"}
            ],
        )
        pre = deterministic_pre_positioning(payload, run_date="2026-08-27")
        self.assertTrue(pre["current_allowed"])
        self.assertIsNone(pre["forced_status"])

    def test_13_positive_publication_context_still_not_forced(self):
        payload = {
            "source_number": 13,
            "title": "Décision Tribunal judiciaire d'Évry : RG n°26/00086",
            "title_date": None,
            "python_status": "unknown",
            "domain": "courdecassation.fr",
            "claim_contexts": [
                {
                    "claim_context": (
                        "Plusieurs décisions de tribunaux judiciaires relatives à "
                        "contentieux techniques ont été publiées en août 2026 et "
                        "figurent dans les résultats, avec des dates explicites[2]."
                    )
                }
            ],
        }
        pre = deterministic_pre_positioning(payload, run_date="2026-08-27")
        self.assertTrue(pre["current_allowed"])
        self.assertIsNone(pre["forced_status"])


if __name__ == "__main__":
    unittest.main()
