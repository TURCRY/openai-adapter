import argparse
import io
import json
import os
import smtplib
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from mail_sender import (
    MailSendError,
    SMTPConfig,
    build_email_message,
    load_mail_json,
    main,
    send_mail,
    smtp_config_from_env_and_args,
)


class MailSenderTests(unittest.TestCase):
    def sample_mail(self):
        return {
            "subject": "[Perplexica] État français",
            "text": "Réponse en texte brut avec accents.",
            "html": "<html><body><p>Réponse en <strong>HTML</strong> avec accents.</p></body></html>",
            "metadata": {"chat_id": "chat123"},
        }

    def sample_config(self, **overrides):
        values = {
            "host": "smtp.example.test",
            "port": 587,
            "username": "user",
            "password": "secret-password",
            "from_address": "from@example.test",
            "to_address": "to@example.test",
            "starttls": True,
            "timeout": 10,
        }
        values.update(overrides)
        return SMTPConfig(**values)

    def test_builds_multipart_text_html_message(self):
        message = build_email_message(self.sample_mail(), self.sample_config())
        self.assertEqual(message["Subject"], "[Perplexica] État français")
        self.assertTrue(message.is_multipart())
        self.assertIn("Réponse en texte", message.get_body(preferencelist=("plain",)).get_content())
        self.assertIn("<strong>HTML</strong>", message.get_body(preferencelist=("html",)).get_content())

    def test_utf8_content_is_preserved(self):
        message = build_email_message(self.sample_mail(), self.sample_config())
        self.assertIn("État français", message["Subject"])
        self.assertIn("accents", message.get_body(preferencelist=("plain",)).get_content())

    def test_send_uses_starttls_and_login(self):
        smtp = MagicMock()
        smtp.__enter__.return_value = smtp
        with patch("mail_sender.smtplib.SMTP", return_value=smtp) as smtp_cls:
            result = send_mail(self.sample_mail(), self.sample_config())
        smtp_cls.assert_called_once_with("smtp.example.test", 587, timeout=10)
        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with("user", "secret-password")
        smtp.send_message.assert_called_once()
        self.assertTrue(result["sent"])

    def test_send_without_login(self):
        smtp = MagicMock()
        smtp.__enter__.return_value = smtp
        config = self.sample_config(username=None, password=None)
        with patch("mail_sender.smtplib.SMTP", return_value=smtp):
            result = send_mail(self.sample_mail(), config)
        smtp.starttls.assert_called_once()
        smtp.login.assert_not_called()
        smtp.send_message.assert_called_once()
        self.assertTrue(result["sent"])

    def test_dry_run_does_not_connect(self):
        with patch("mail_sender.smtplib.SMTP") as smtp_cls:
            result = send_mail(self.sample_mail(), self.sample_config(), dry_run=True)
        smtp_cls.assert_not_called()
        self.assertFalse(result["sent"])
        self.assertTrue(result["dry_run"])
        self.assertGreater(result["text_size"], 0)
        self.assertGreater(result["html_size"], 0)

    def test_missing_environment_values_raise(self):
        args = argparse.Namespace(
            smtp_host=None,
            smtp_port=None,
            smtp_username=None,
            smtp_password=None,
            from_address=None,
            to=None,
            smtp_starttls=None,
            timeout=10,
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(MailSendError, "SMTP host"):
                smtp_config_from_env_and_args(args)

    def test_password_never_in_result(self):
        result = send_mail(self.sample_mail(), self.sample_config(), dry_run=True)
        self.assertNotIn("secret-password", json.dumps(result))

    def test_smtp_error_is_wrapped(self):
        smtp = MagicMock()
        smtp.__enter__.return_value = smtp
        smtp.send_message.side_effect = smtplib.SMTPException("boom")
        with patch("mail_sender.smtplib.SMTP", return_value=smtp):
            with self.assertRaisesRegex(MailSendError, "SMTP send failed"):
                send_mail(self.sample_mail(), self.sample_config())

    def test_connection_error_is_wrapped(self):
        with patch("mail_sender.smtplib.SMTP", side_effect=OSError("cannot connect")):
            with self.assertRaisesRegex(MailSendError, "OSError"):
                send_mail(self.sample_mail(), self.sample_config())

    def test_load_full_mail_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mail.json"
            path.write_text(json.dumps(self.sample_mail()), encoding="utf-8")
            loaded = load_mail_json(path)
        self.assertEqual(loaded["subject"], self.sample_mail()["subject"])
        self.assertIn("html", loaded)

    def test_load_metadata_json_with_sibling_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "mail_job"
            (base.with_suffix(".json")).write_text(json.dumps({"subject": "Sujet", "chat_id": "c"}), encoding="utf-8")
            (base.with_suffix(".txt")).write_text("Texte", encoding="utf-8")
            (base.with_suffix(".html")).write_text("<p>HTML</p>", encoding="utf-8")
            loaded = load_mail_json(base.with_suffix(".json"))
        self.assertEqual(loaded["subject"], "Sujet")
        self.assertEqual(loaded["text"], "Texte")
        self.assertEqual(loaded["html"], "<p>HTML</p>")

    def test_cli_dry_run_prints_no_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mail.json"
            path.write_text(json.dumps(self.sample_mail()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        str(path),
                        "--dry-run",
                        "--smtp-host",
                        "smtp.example.test",
                        "--from-address",
                        "from@example.test",
                        "--to",
                        "to@example.test",
                        "--smtp-username",
                        "user",
                        "--smtp-password",
                        "secret-password",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertIn("status: dry_run", output.getvalue())
        self.assertNotIn("secret-password", output.getvalue())


if __name__ == "__main__":
    unittest.main()
