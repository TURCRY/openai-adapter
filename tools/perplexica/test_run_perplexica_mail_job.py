import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from editorial_rewrite import EditorialRewriteError
from mail_builder import MailBuildError
from mail_sender import MailSendError, SMTPConfig
from perplexica_client import PerplexicaClientError
from run_perplexica_mail_job import (
    JobConfigError,
    aggregate_search_results,
    classify_search_result,
    load_job,
    read_prompt,
    resolve_prompt_path,
    run_job,
    subject_for_variant,
)


def result_for(name="default", *, source_url=None, local_index=1, answer=None, sources=1):
    source_url = source_url or f"https://example.com/{name}"
    if sources == 0:
        return {
            "chat_id": f"chat-{name}",
            "message_id": f"msg-{name}",
            "question": f"Question {name}",
            "answer_markdown": answer or "I could not find any relevant information.",
            "all_sources": [],
            "cited_sources": [],
            "citation_numbers": [],
            "status": "completed",
        }
    return {
        "chat_id": f"chat-{name}",
        "message_id": f"msg-{name}",
        "question": f"Question {name}",
        "answer_markdown": answer or f"Réponse {name} [{local_index}].",
        "all_sources": [
            {"index": local_index, "title": f"Source {name}", "url": source_url},
            {"index": 99, "title": f"Consultée {name}", "url": f"https://consulted.example/{name}"},
        ],
        "cited_sources": [
            {"index": local_index, "title": f"Source {name}", "url": source_url, "citation_count": 1}
        ],
        "citation_numbers": [local_index],
        "unresolved_citations": [],
        "created_at": "2026-08-26T00:00:00Z",
        "status": "completed",
    }


def editorial_payload():
    return {
        "status": "completed",
        "model": "local-gemma-4",
        "title": "Titre éditorial",
        "body_markdown": "Corps éditorial [1].",
        "citation_numbers": [1],
    }


def mail_for_call(result, subject=None, editorial=None, display_title=None):
    label = "editorial" if editorial else "raw"
    return {
        "subject": subject or f"Sujet {label}",
        "text": f"Texte {label} UTF-8 éà",
        "html": f"<html><body>Texte {label} UTF-8 éà</body></html>",
        "metadata": {
            "subject": subject or f"Sujet {label}",
            "chat_id": result.get("chat_id"),
            "message_id": result.get("message_id"),
            "editorial_used": bool(editorial),
            "display_title": display_title,
        },
    }


class FakeClient:
    instances = []
    calls = []
    results = []

    def __init__(self, base_url, timeout=60):
        self.base_url = base_url
        self.timeout = timeout
        FakeClient.instances.append(self)

    def ask(self, prompt, **options):
        FakeClient.calls.append((self, prompt, options))
        item = FakeClient.results.pop(0) if FakeClient.results else result_for(str(len(FakeClient.calls)))
        if isinstance(item, Exception):
            raise item
        return item


class FailingClient(FakeClient):
    def ask(self, prompt, **options):
        raise PerplexicaClientError("stream interrupted")


class RunPerplexicaMailJobTests(unittest.TestCase):
    def setUp(self):
        FakeClient.instances = []
        FakeClient.calls = []
        FakeClient.results = []
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.jobs_dir = self.root / "jobs"
        self.prompts_dir = self.root / "prompts"
        self.output_root = self.root / "output"
        self.jobs_dir.mkdir()
        self.prompts_dir.mkdir()
        self.prompt_path = self.prompts_dir / "prompt.md"
        self.prompt_path.write_text("Question UTF-8 médiation", encoding="utf-8")
        self.editorial_prompt_path = self.prompts_dir / "editorial.md"
        self.editorial_prompt_path.write_text("Prompt éditorial", encoding="utf-8")
        self.job_path = self.jobs_dir / "job.json"
        self.write_job()

    def tearDown(self):
        self.tmp.cleanup()

    def write_job(self, **overrides):
        payload = {
            "name": "veille_test",
            "prompt_file": "../prompts/prompt.md",
            "base_url": "https://perplexica.example",
            "send_mail": True,
            "subject": "Sujet test",
            "mail_mode": "editorial",
            "perplexica_options": {"sources": ["web"]},
            "editorial": {"enabled": True, "prompt_file": "../prompts/editorial.md", "model": "local-gemma-4"},
        }
        payload.update(overrides)
        self.job_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def write_multi_job(self, searches=None, **overrides):
        if searches is None:
            searches = [
                {"name": "expertise_justice", "prompt_file": "../prompts/expertise_justice.md"},
                {"name": "expertise_construction", "prompt_file": "../prompts/expertise_construction.md"},
                {"name": "mediation", "prompt_file": "../prompts/mediation.md"},
                {"name": "mard_textes", "prompt_file": "../prompts/mard_textes.md"},
                {"name": "jurisprudence", "prompt_file": "../prompts/jurisprudence.md"},
                {"name": "institutionnelle", "prompt_file": "../prompts/institutionnelle.md"},
            ]
        for search in searches:
            (self.prompts_dir / Path(search["prompt_file"]).name).write_text(f"Prompt {search['name']}", encoding="utf-8")
        payload = {
            "name": "veille_test",
            "base_url": "https://perplexica.example",
            "send_mail": True,
            "subject": "Veille expertise et médiation",
            "display_title": "Veille hebdomadaire — Expertise de justice et médiation",
            "mail_mode": "both",
            "searches": searches,
            "editorial": {"enabled": True, "prompt_file": "../prompts/editorial.md", "model": "local-gemma-4"},
        }
        payload.update(overrides)
        self.job_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def test_loads_valid_job(self):
        job = load_job(self.job_path)
        self.assertEqual(job["name"], "veille_test")
        self.assertEqual(job["mail_mode"], "editorial")

    def test_subject_for_both_variants_preserves_unicode(self):
        base = "Veille expertise et médiation"
        self.assertEqual(subject_for_variant(base, "raw", "both"), "Veille expertise et médiation — Source Perplexica")
        self.assertEqual(
            subject_for_variant(base, "editorial", "both"),
            "Veille expertise et médiation — Synthèse éditoriale",
        )

    def test_display_title_is_passed_to_raw_and_editorial_mails(self):
        title = "Veille hebdomadaire — Expertise de justice et médiation"
        self.write_job(mail_mode="both", display_title=title)
        run_dir, metadata = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(metadata["status"], "completed_no_mail")
        raw_meta = json.loads((run_dir / "raw_mail.json").read_text(encoding="utf-8"))
        editorial_meta = json.loads((run_dir / "editorial_mail.json").read_text(encoding="utf-8"))
        self.assertEqual(raw_meta["display_title"], title)
        self.assertEqual(editorial_meta["display_title"], title)
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["question"], "Question 1")

    def test_rejects_invalid_mail_mode(self):
        self.write_job(mail_mode="invalid")
        with self.assertRaises(JobConfigError):
            load_job(self.job_path)

    def test_rejects_non_string_display_title(self):
        self.write_job(display_title=123)
        with self.assertRaises(JobConfigError):
            load_job(self.job_path)

    def test_resolves_prompt_relative_to_job_file(self):
        job = load_job(self.job_path)
        resolved = resolve_prompt_path(self.job_path, job)
        self.assertEqual(resolved.resolve(), self.prompt_path.resolve())

    def test_editorial_timeout_override_is_passed_to_rewriter(self):
        self.write_job(editorial={"enabled": True, "prompt_file": "../prompts/editorial.md", "model": "local-gemma-4", "timeout": 600})
        captured = {}

        def rewriter(result, config):
            captured["timeout"] = config.timeout
            return editorial_payload(), "raw llm"

        _, metadata = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            editorial_rewrite_func=rewriter,
        )
        self.assertEqual(metadata["status"], "completed_no_mail")
        self.assertEqual(captured["timeout"], 600)

    def test_rejects_invalid_editorial_timeout(self):
        for timeout in (0, -1, 1801):
            with self.subTest(timeout=timeout):
                self.write_job(editorial={"enabled": True, "timeout": timeout})
                with self.assertRaises(JobConfigError):
                    load_job(self.job_path)

    def test_reads_prompt_utf8(self):
        self.assertEqual(read_prompt(self.prompt_path), "Question UTF-8 médiation")

    def test_invalid_job_json_raises(self):
        self.job_path.write_text("{bad", encoding="utf-8")
        with self.assertRaises(JobConfigError):
            load_job(self.job_path)

    def test_missing_prompt_writes_failed_run_json(self):
        self.prompt_path.unlink()
        run_dir, metadata = run_job(self.job_path, output_root=self.output_root, client_factory=FakeClient)
        self.assertEqual(metadata["status"], "failed")
        self.assertTrue((run_dir / "run.json").exists())
        self.assertIn("Cannot read prompt", metadata["error"])

    def test_calls_perplexica_with_prompt_and_options(self):
        run_dir, metadata = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(metadata["status"], "completed_no_mail")
        self.assertEqual(len(FakeClient.calls), 1)
        instance, prompt, options = FakeClient.calls[0]
        self.assertEqual(instance.base_url, "https://perplexica.example")
        self.assertEqual(prompt, "Question UTF-8 médiation")
        self.assertEqual(options, {"sources": ["web"]})
        self.assertTrue(run_dir.exists())

    def test_creates_run_directory_and_result_json(self):
        run_dir, _ = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertTrue(run_dir.name.startswith("20"))
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["chat_id"], "chat-1")

    def test_mail_mode_raw_builds_only_raw_mail(self):
        self.write_job(mail_mode="raw")
        send = Mock(return_value={"sent": True})
        run_dir, metadata = run_job(
            self.job_path,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            smtp_config_factory=lambda: SMTPConfig(host="smtp.example", from_address="a@example.com", to_address="b@example.com"),
            send_mail_func=send,
        )
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue(metadata["raw_mail_built"])
        self.assertFalse(metadata["editorial_requested"])
        self.assertTrue((run_dir / "raw_mail.html").exists())
        self.assertFalse((run_dir / "editorial_mail.html").exists())
        send.assert_called_once()

    def test_mail_mode_editorial_success_builds_only_editorial_mail(self):
        send = Mock(return_value={"sent": True})
        run_dir, metadata = run_job(
            self.job_path,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            smtp_config_factory=lambda: SMTPConfig(host="smtp.example", from_address="a@example.com", to_address="b@example.com"),
            send_mail_func=send,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue(metadata["editorial_mail_built"])
        self.assertFalse((run_dir / "raw_mail.html").exists())
        send.assert_called_once()

    def test_mail_mode_editorial_error_falls_back_to_raw(self):
        builder = Mock(side_effect=mail_for_call)
        run_dir, metadata = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=builder,
            editorial_rewrite_func=Mock(side_effect=EditorialRewriteError("citation inconnue")),
        )
        self.assertEqual(metadata["editorial_status"], "fallback_raw")
        self.assertTrue(metadata["raw_mail_built"])
        self.assertFalse(metadata["editorial_mail_built"])
        self.assertTrue((run_dir / "raw_mail.html").exists())

    def test_mail_mode_both_with_editorial_success_builds_two_distinct_mails(self):
        self.write_job(mail_mode="both")
        builder = Mock(side_effect=mail_for_call)
        send = Mock(return_value={"sent": True})
        run_dir, metadata = run_job(
            self.job_path,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=builder,
            smtp_config_factory=lambda: SMTPConfig(host="smtp.example", from_address="a@example.com", to_address="b@example.com"),
            send_mail_func=send,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(len(FakeClient.calls), 1)
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue((run_dir / "raw_mail.html").exists())
        self.assertTrue((run_dir / "editorial_mail.html").exists())
        self.assertEqual(send.call_count, 2)

    def test_mail_mode_both_with_editorial_error_sends_only_raw(self):
        self.write_job(mail_mode="both")
        send = Mock(return_value={"sent": True})
        _, metadata = run_job(
            self.job_path,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            smtp_config_factory=lambda: SMTPConfig(host="smtp.example", from_address="a@example.com", to_address="b@example.com"),
            send_mail_func=send,
            editorial_rewrite_func=Mock(side_effect=EditorialRewriteError("timeout")),
        )
        self.assertEqual(metadata["editorial_status"], "fallback_raw")
        self.assertTrue(metadata["raw_mail_sent"])
        self.assertFalse(metadata["editorial_mail_sent"])
        send.assert_called_once()

    def test_dry_run_does_not_send_smtp(self):
        send = Mock(return_value={"sent": True})
        _, metadata = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            send_mail_func=send,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(metadata["status"], "completed_no_mail")
        send.assert_not_called()

    def test_no_mail_overrides_job(self):
        send = Mock(return_value={"sent": True})
        _, metadata = run_job(
            self.job_path,
            no_mail=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            send_mail_func=send,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(metadata["status"], "completed_no_mail")
        self.assertFalse(metadata["mail_requested"])
        send.assert_not_called()

    def test_perplexica_error_writes_run_json(self):
        run_dir, metadata = run_job(self.job_path, output_root=self.output_root, client_factory=FailingClient)
        self.assertEqual(metadata["status"], "perplexica_failed")
        self.assertTrue((run_dir / "run.json").exists())
        self.assertFalse((run_dir / "result.json").exists())

    def test_builder_error_keeps_result_json(self):
        def fail_builder(result, subject=None, editorial=None, display_title=None):
            raise MailBuildError("cannot build")

        run_dir, metadata = run_job(
            self.job_path,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=fail_builder,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(metadata["status"], "build_failed")
        self.assertTrue((run_dir / "result.json").exists())

    def test_smtp_error_keeps_result_and_mail_files(self):
        def fail_send(mail, config, dry_run=False):
            raise MailSendError("SMTP failed")

        run_dir, metadata = run_job(
            self.job_path,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            smtp_config_factory=lambda: SMTPConfig(host="smtp.example", from_address="a@example.com", to_address="b@example.com"),
            send_mail_func=fail_send,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(metadata["status"], "mail_failed")
        self.assertTrue((run_dir / "editorial_mail.html").exists())

    def test_run_json_has_no_secret_values(self):
        secret = "super-secret-password"

        def fail_send(mail, config, dry_run=False):
            raise MailSendError("SMTP failed")

        run_dir, _ = run_job(
            self.job_path,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            smtp_config_factory=lambda: SMTPConfig(
                host="smtp.example",
                username="user@example.com",
                password=secret,
                from_address="a@example.com",
                to_address="b@example.com",
            ),
            send_mail_func=fail_send,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        raw = (run_dir / "run.json").read_text(encoding="utf-8")
        self.assertNotIn(secret, raw)
        self.assertNotIn("user@example.com", raw)

    def test_relative_paths_do_not_depend_on_current_directory(self):
        original_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            run_dir, metadata = run_job(
                self.job_path,
                dry_run=True,
                output_root=self.output_root,
                client_factory=FakeClient,
                build_mail_func=mail_for_call,
                editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
            )
        finally:
            os.chdir(original_cwd)
        self.assertEqual(metadata["status"], "completed_no_mail")
        self.assertTrue((run_dir / "result.json").exists())

    def test_windows_style_absolute_prompt_path_is_accepted_when_current_platform_resolves_it(self):
        self.write_job(prompt_file=str(self.prompt_path))
        job = load_job(self.job_path)
        self.assertEqual(resolve_prompt_path(self.job_path, job).resolve(), self.prompt_path.resolve())

    def test_run_json_includes_editorial_fields_without_secret(self):
        secret = "editorial-secret-value"
        os.environ["PERPLEXICA_EDITORIAL_API_KEY"] = secret
        try:
            run_dir, metadata = run_job(
                self.job_path,
                dry_run=True,
                output_root=self.output_root,
                client_factory=FakeClient,
                build_mail_func=mail_for_call,
                editorial_rewrite_func=Mock(side_effect=EditorialRewriteError("offline")),
            )
        finally:
            os.environ.pop("PERPLEXICA_EDITORIAL_API_KEY", None)
        raw = (run_dir / "run.json").read_text(encoding="utf-8")
        self.assertTrue(metadata["editorial_requested"])
        self.assertNotIn(secret, raw)
        self.assertIn("editorial_status", raw)

    def test_classifies_zero_source_result_as_empty(self):
        self.assertEqual(classify_search_result(result_for("empty", sources=0)), "empty")

    def test_multisearch_executes_six_independent_searches_once(self):
        self.write_multi_job()
        run_dir, metadata = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(len(FakeClient.calls), 6)
        self.assertEqual(len(FakeClient.instances), 6)
        for name in ("expertise_justice", "expertise_construction", "mediation", "mard_textes", "jurisprudence", "institutionnelle"):
            self.assertTrue((run_dir / "searches" / name / "result.json").exists())
        self.assertEqual(metadata["search_count"], 6)
        self.assertEqual(metadata["completed_search_count"], 6)
        self.assertEqual(metadata["status"], "completed_no_mail")

    def test_multisearch_continues_after_empty_and_failed_searches(self):
        self.write_multi_job()
        FakeClient.results = [
            result_for("a", source_url="https://a.example"),
            result_for("empty", sources=0),
            PerplexicaClientError("boom"),
            result_for("d", source_url="https://d.example"),
            result_for("e", source_url="https://e.example"),
            result_for("f", source_url="https://f.example"),
        ]
        run_dir, metadata = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(len(FakeClient.calls), 6)
        self.assertEqual(metadata["completed_search_count"], 4)
        self.assertEqual(metadata["empty_search_count"], 1)
        self.assertEqual(metadata["failed_search_count"], 1)
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        self.assertIn("Aucune actualité significative identifiée", result["answer_markdown"])
        self.assertIn("Recherche indisponible pour cet axe", result["answer_markdown"])
        self.assertNotIn("I could not find any relevant information", result["answer_markdown"])

    def test_multisearch_all_empty_returns_no_results_and_no_mail(self):
        self.write_multi_job()
        FakeClient.results = [result_for(str(i), sources=0) for i in range(6)]
        run_dir, metadata = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertEqual(metadata["status"], "no_results")
        self.assertFalse((run_dir / "raw_mail.html").exists())
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "no_results")
        self.assertEqual(result["empty_search_count"], 6)

    def test_aggregate_renumbers_same_local_citation_for_different_urls(self):
        job = {"name": "job", "display_title": "Titre"}
        aggregate = aggregate_search_results(
            job,
            [
                {"name": "a", "title": "A", "status": "completed", "result": result_for("a", source_url="https://a.example", local_index=1)},
                {"name": "b", "title": "B", "status": "completed", "result": result_for("b", source_url="https://b.example", local_index=1)},
            ],
        )
        self.assertEqual([s["index"] for s in aggregate["cited_sources"]], [1, 2])
        self.assertIn("Réponse a [1].", aggregate["answer_markdown"])
        self.assertIn("Réponse b [2].", aggregate["answer_markdown"])

    def test_aggregate_deduplicates_same_url_with_different_local_numbers(self):
        job = {"name": "job", "display_title": "Titre"}
        aggregate = aggregate_search_results(
            job,
            [
                {"name": "a", "title": "A", "status": "completed", "result": result_for("a", source_url="https://same.example/path/", local_index=1)},
                {"name": "b", "title": "B", "status": "completed", "result": result_for("b", source_url="https://same.example/path", local_index=42, answer="Réponse b [42].")},
            ],
        )
        self.assertEqual(len(aggregate["cited_sources"]), 1)
        self.assertIn("Réponse a [1].", aggregate["answer_markdown"])
        self.assertIn("Réponse b [1].", aggregate["answer_markdown"])
        self.assertEqual(aggregate["citation_numbers"], [1])

    def test_aggregate_metrics_and_semantics(self):
        aggregate = aggregate_search_results(
            {"name": "job", "display_title": "Titre"},
            [
                {"name": "a", "title": "A", "status": "completed", "result": result_for("a", source_url="https://a.example")},
                {"name": "b", "title": "B", "status": "empty", "result": result_for("b", sources=0)},
                {"name": "c", "title": "C", "status": "failed", "error": "offline"},
            ],
        )
        self.assertEqual(aggregate["source_count"], 2)
        self.assertEqual(aggregate["cited_source_count"], 1)
        self.assertEqual(aggregate["search_count"], 3)
        self.assertEqual(aggregate["completed_search_count"], 1)
        self.assertEqual(aggregate["empty_search_count"], 1)
        self.assertEqual(aggregate["failed_search_count"], 1)
        self.assertIn("all_sources contains the global deduplicated cited sources only", aggregate["all_sources_semantics"])

    def test_multisearch_run_json_contains_search_summaries(self):
        self.write_multi_job()
        run_dir, metadata = run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertIn("searches", run_json)
        self.assertIn("expertise_justice", run_json["searches"])
        self.assertEqual(run_json["search_count"], 6)
        self.assertEqual(metadata["cited_source_count"], 6)

    def test_multisearch_raw_mail_is_multi_section(self):
        self.write_multi_job()
        captured = {}

        def builder(result, subject=None, editorial=None, display_title=None):
            captured["answer"] = result["answer_markdown"]
            return mail_for_call(result, subject=subject, editorial=editorial, display_title=display_title)

        run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=builder,
            editorial_rewrite_func=lambda result, config: (editorial_payload(), "raw llm"),
        )
        self.assertIn("## Expertise de justice", captured["answer"])
        self.assertIn("## Expertise construction", captured["answer"])
        self.assertIn("## Médiation", captured["answer"])
        self.assertIn("## MARD / textes", captured["answer"])
        self.assertIn("## Jurisprudence", captured["answer"])
        self.assertIn("## Actualité institutionnelle", captured["answer"])

    def test_editorial_receives_aggregated_cited_sources_only(self):
        self.write_multi_job()
        captured = {}

        def rewriter(result, config):
            captured["result"] = result
            return editorial_payload(), "raw llm"

        run_job(
            self.job_path,
            dry_run=True,
            output_root=self.output_root,
            client_factory=FakeClient,
            build_mail_func=mail_for_call,
            editorial_rewrite_func=rewriter,
        )
        result = captured["result"]
        self.assertIn("editorial_answer_markdown", result)
        self.assertEqual(result["all_sources"], result["cited_sources"])
        self.assertEqual(len(result["cited_sources"]), 6)
        for source in result["cited_sources"]:
            self.assertNotIn("content", source)


if __name__ == "__main__":
    unittest.main()
