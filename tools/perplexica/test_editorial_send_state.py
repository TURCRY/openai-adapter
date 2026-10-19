import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mail_sender import MailSendError, SMTPConfig
from run_perplexica_mail_job import JobRunError, resume_editorial_run


def result_payload():
    return {
        "chat_id": "chat-1",
        "message_id": "msg-1",
        "question": "Question",
        "answer_markdown": "Reponse [1].",
        "all_sources": [{"index": 1, "title": "Source", "url": "https://example.com/a"}],
        "cited_sources": [
            {"index": 1, "title": "Source", "url": "https://example.com/a", "citation_count": 1}
        ],
        "citation_numbers": [1],
        "unresolved_citations": [],
        "status": "completed",
    }


def editorial_payload():
    return {
        "status": "completed",
        "model": "local-gemma-4",
        "title": "Titre",
        "body_markdown": "Corps [1].",
        "citation_numbers": [1],
    }


def mail_payload(result, subject=None, editorial=None, display_title=None):
    label = "editorial" if editorial else "raw"
    return {
        "subject": subject or f"Sujet {label}",
        "text": f"Texte {label}",
        "html": f"<html><body>Texte {label}</body></html>",
        "metadata": {"subject": subject or f"Sujet {label}"},
    }


class EditorialSendStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        prompts = self.root / "prompts"
        prompts.mkdir()
        (prompts / "prompt.md").write_text("Prompt", encoding="utf-8")
        (prompts / "editorial.md").write_text("Editorial", encoding="utf-8")
        self.job_path = self.root / "job.json"
        self.job_path.write_text(
            json.dumps(
                {
                    "name": "veille_test",
                    "prompt_file": "prompts/prompt.md",
                    "base_url": "https://perplexica.example",
                    "send_mail": True,
                    "subject": "Sujet test",
                    "mail_mode": "both",
                    "editorial": {
                        "enabled": True,
                        "prompt_file": "prompts/editorial.md",
                        "model": "local-gemma-4",
                    },
                }
            ),
            encoding="utf-8",
        )
        self.marker_dir = self.root / "markers"
        self.marker_dir.mkdir()
        self.run_dir = self.root / "jobs" / "20261019T010004Z_test"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "result.json").write_text(json.dumps(result_payload()), encoding="utf-8")
        self.write_run({})
        (self.marker_dir / "2026-10-19.partial").write_text(
            json.dumps(
                {
                    "date": "2026-10-19",
                    "run_id": self.run_dir.name,
                    "raw_mail_sent": True,
                    "editorial_mail_sent": False,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def write_run(self, overrides):
        payload = {
            "job_name": "veille_test",
            "started_at": "2026-10-19T01:00:04+00:00",
            "status": "partial",
            "mail_requested": True,
            "mail_sent": True,
            "mail_mode": "both",
            "raw_mail_built": True,
            "raw_mail_sent": True,
            "editorial_requested": True,
            "editorial_status": "fallback_raw",
            "editorial_mail_built": False,
            "editorial_mail_sent": False,
        }
        payload.update(overrides)
        (self.run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def smtp_config():
        return SMTPConfig(
            host="smtp.example",
            from_address="a@example.com",
            to_address="b@example.com",
        )

    @staticmethod
    def rewrite(result, config, diagnostics_dir=None):
        return editorial_payload(), "raw"

    def resume(self, send_mail_func):
        return resume_editorial_run(
            self.job_path,
            self.run_dir,
            build_mail_func=mail_payload,
            smtp_config_factory=self.smtp_config,
            send_mail_func=send_mail_func,
            editorial_rewrite_func=self.rewrite,
            marker_dir=self.marker_dir,
        )

    def test_legacy_run_without_send_state_is_resumed(self):
        send = Mock(return_value={"sent": True, "message_id": "<legacy@test>"})
        _, metadata = self.resume(send)
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["editorial_send_state"], "sent")
        self.assertTrue(metadata["editorial_mail_sent"])
        self.assertEqual(metadata["editorial_message_id"], "<legacy@test>")
        self.assertTrue((self.marker_dir / "2026-10-19.ok").exists())
        self.assertFalse((self.marker_dir / "2026-10-19.partial").exists())

    def test_send_state_sending_refuses_resume(self):
        self.write_run(
            {
                "editorial_send_state": "sending",
                "editorial_send_attempted_at": "2026-10-19T02:00:00+00:00",
            }
        )
        send = Mock()
        with self.assertRaises(JobRunError):
            self.resume(send)
        send.assert_not_called()
        metadata = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["editorial_send_state"], "sending")
        self.assertTrue((self.marker_dir / "2026-10-19.partial").exists())

    def test_send_state_sent_refuses_resume(self):
        self.write_run({"editorial_send_state": "sent"})
        send = Mock()
        with self.assertRaises(JobRunError):
            self.resume(send)
        send.assert_not_called()

    def test_unknown_send_state_refuses_resume(self):
        self.write_run({"editorial_send_state": "unexpected"})
        send = Mock()
        with self.assertRaises(JobRunError):
            self.resume(send)
        send.assert_not_called()

    def test_smtp_failure_leaves_sending_and_partial(self):
        def fail_send(mail, config, dry_run=False):
            raise MailSendError("SMTP timeout")

        _, metadata = self.resume(fail_send)
        self.assertEqual(metadata["status"], "partial")
        self.assertEqual(metadata["editorial_send_state"], "sending")
        self.assertFalse(metadata["editorial_mail_sent"])
        self.assertTrue(metadata["editorial_message_id"])
        self.assertTrue((self.marker_dir / "2026-10-19.partial").exists())
        self.assertFalse((self.marker_dir / "2026-10-19.ok").exists())

    def test_send_state_is_persisted_as_sending_before_smtp_call(self):
        observed = {}

        def send(mail, config, dry_run=False):
            metadata = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
            observed["state"] = metadata.get("editorial_send_state")
            observed["attempted_at"] = metadata.get("editorial_send_attempted_at")
            observed["message_id"] = mail.get("smtp_message_id")
            return {"sent": True, "message_id": mail.get("smtp_message_id")}

        _, metadata = self.resume(send)
        self.assertEqual(observed["state"], "sending")
        self.assertTrue(observed["attempted_at"])
        self.assertTrue(observed["message_id"])
        self.assertEqual(metadata["editorial_message_id"], observed["message_id"])
        self.assertEqual(metadata["editorial_send_state"], "sent")

    def test_build_email_message_uses_prepared_message_id(self):
        from mail_sender import build_email_message

        message = build_email_message(
            {
                "subject": "Sujet",
                "text": "Texte",
                "html": "<p>Texte</p>",
                "smtp_message_id": "<prepared@example.com>",
            },
            self.smtp_config(),
        )
        self.assertEqual(message["Message-ID"], "<prepared@example.com>")