import argparse
from email import policy
import io
import json
import os
import smtplib
import tempfile
import unittest
from contextlib import redirect_stdout
from email.parser import BytesParser
from pathlib import Path
from unittest.mock import MagicMock, patch

from mail_sender import (
    MailSendError,
    SMTPConfig,
    build_email_message,
    load_mail_json,
    main,
    parse_env_file,
    parse_security,
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
            "security": "starttls",
            "timeout": 10,
        }
        values.update(overrides)
        return SMTPConfig(**values)

    def empty_args(self, **overrides):
        values = {
            "smtp_host": None,
            "smtp_port": None,
            "smtp_username": None,
            "smtp_password": None,
            "from_address": None,
            "to": None,
            "smtp_security": None,
            "smtp_starttls": None,
            "env_file": Path("missing.env"),
            "timeout": 10,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

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

    def test_professional_subject_round_trips_after_bytes_serialization(self):
        subjects = [
            "Veille expertise et m\u00e9diation \u2014 Source Perplexica",
            "Veille expertise et m\u00e9diation \u2014 Synth\u00e8se \u00e9ditoriale",
            "\u00e9 \u00e8 \u00e0 \u0153 \u2014 \u2019",
        ]
        for subject in subjects:
            with self.subTest(subject=subject):
                mail = {
                    "subject": subject,
                    "text": "Texte avec accents \u00e9 \u00e0.",
                    "html": "<p>Texte avec accents \u00e9 \u00e0.</p>",
                    "metadata": {},
                }
                message = build_email_message(mail, self.sample_config())
                self.assertEqual(message["Subject"], subject)
                parsed = BytesParser(policy=policy.default).parsebytes(message.as_bytes())
                self.assertEqual(parsed["Subject"], subject)

    def test_ssl_port_465_uses_smtp_ssl(self):
        smtp = MagicMock()
        smtp.__enter__.return_value = smtp
        config = self.sample_config(port=465, security="ssl")
        with patch("mail_sender.smtplib.SMTP_SSL", return_value=smtp) as ssl_cls, patch("mail_sender.smtplib.SMTP") as smtp_cls:
            result = send_mail(self.sample_mail(), config)
        ssl_cls.assert_called_once_with("smtp.example.test", 465, timeout=10)
        smtp_cls.assert_not_called()
        smtp.starttls.assert_not_called()
        smtp.login.assert_called_once_with("user", "secret-password")
        smtp.send_message.assert_called_once()
        self.assertTrue(result["sent"])
        self.assertEqual(result["security"], "ssl")

    def test_starttls_uses_smtp_and_starttls(self):
        smtp = MagicMock()
        smtp.__enter__.return_value = smtp
        config = self.sample_config(port=587, security="starttls")
        with patch("mail_sender.smtplib.SMTP", return_value=smtp) as smtp_cls, patch("mail_sender.smtplib.SMTP_SSL") as ssl_cls:
            result = send_mail(self.sample_mail(), config)
        smtp_cls.assert_called_once_with("smtp.example.test", 587, timeout=10)
        ssl_cls.assert_not_called()
        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with("user", "secret-password")
        smtp.send_message.assert_called_once()
        self.assertTrue(result["sent"])

    def test_none_uses_smtp_without_tls(self):
        smtp = MagicMock()
        smtp.__enter__.return_value = smtp
        config = self.sample_config(port=25, security="none")
        with patch("mail_sender.smtplib.SMTP", return_value=smtp) as smtp_cls, patch("mail_sender.smtplib.SMTP_SSL") as ssl_cls:
            result = send_mail(self.sample_mail(), config)
        smtp_cls.assert_called_once_with("smtp.example.test", 25, timeout=10)
        ssl_cls.assert_not_called()
        smtp.starttls.assert_not_called()
        smtp.send_message.assert_called_once()
        self.assertTrue(result["sent"])

    def test_login_with_credentials(self):
        smtp = MagicMock()
        smtp.__enter__.return_value = smtp
        with patch("mail_sender.smtplib.SMTP", return_value=smtp):
            send_mail(self.sample_mail(), self.sample_config())
        smtp.login.assert_called_once_with("user", "secret-password")

    def test_without_login(self):
        smtp = MagicMock()
        smtp.__enter__.return_value = smtp
        config = self.sample_config(username=None, password=None)
        with patch("mail_sender.smtplib.SMTP", return_value=smtp):
            result = send_mail(self.sample_mail(), config)
        smtp.login.assert_not_called()
        self.assertTrue(result["sent"])

    def test_auto_security_465_ssl(self):
        self.assertEqual(parse_security(None, 465), "ssl")

    def test_auto_security_587_starttls(self):
        self.assertEqual(parse_security(None, 587), "starttls")

    def test_legacy_starttls_false_maps_to_none(self):
        self.assertEqual(parse_security(None, 587, "false"), "none")

    def test_env_file_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n# comment\nPERPLEXICA_SMTP_HOST='smtp.env.test'\nPERPLEXICA_MAIL_FROM=\"from@env.test\"\nPERPLEXICA_MAIL_TO=to@env.test\nPERPLEXICA_SMTP_PORT=465\nPERPLEXICA_SMTP_SECURITY=ssl\n",
                encoding="utf-8",
            )
            values = parse_env_file(path)
        self.assertEqual(values["PERPLEXICA_SMTP_HOST"], "smtp.env.test")
        self.assertEqual(values["PERPLEXICA_MAIL_FROM"], "from@env.test")
        self.assertEqual(values["PERPLEXICA_SMTP_SECURITY"], "ssl")

    def test_os_environ_has_priority_over_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "PERPLEXICA_SMTP_HOST=smtp.file.test\nPERPLEXICA_MAIL_FROM=from@file.test\nPERPLEXICA_MAIL_TO=to@file.test\n",
                encoding="utf-8",
            )
            args = self.empty_args(env_file=path)
            with patch.dict(os.environ, {"PERPLEXICA_SMTP_HOST": "smtp.env.test"}, clear=True):
                config = smtp_config_from_env_and_args(args)
        self.assertEqual(config.host, "smtp.env.test")
        self.assertEqual(config.from_address, "from@file.test")

    def test_cli_has_priority_over_environment(self):
        args = self.empty_args(
            smtp_host="smtp.cli.test",
            from_address="from@cli.test",
            to="to@cli.test",
            smtp_port="465",
            smtp_security="ssl",
        )
        with patch.dict(
            os.environ,
            {
                "PERPLEXICA_SMTP_HOST": "smtp.env.test",
                "PERPLEXICA_MAIL_FROM": "from@env.test",
                "PERPLEXICA_MAIL_TO": "to@env.test",
                "PERPLEXICA_SMTP_PORT": "587",
                "PERPLEXICA_SMTP_SECURITY": "starttls",
            },
            clear=True,
        ):
            config = smtp_config_from_env_and_args(args)
        self.assertEqual(config.host, "smtp.cli.test")
        self.assertEqual(config.from_address, "from@cli.test")
        self.assertEqual(config.to_address, "to@cli.test")
        self.assertEqual(config.port, 465)
        self.assertEqual(config.security, "ssl")

    def test_security_env_has_priority_over_legacy_starttls(self):
        args = self.empty_args(
            smtp_host="smtp.test",
            from_address="from@test",
            to="to@test",
            smtp_port="465",
        )
        with patch.dict(
            os.environ,
            {"PERPLEXICA_SMTP_SECURITY": "ssl", "PERPLEXICA_SMTP_STARTTLS": "false"},
            clear=True,
        ):
            config = smtp_config_from_env_and_args(args)
        self.assertEqual(config.security, "ssl")

    def test_dry_run_does_not_connect(self):
        with patch("mail_sender.smtplib.SMTP") as smtp_cls, patch("mail_sender.smtplib.SMTP_SSL") as ssl_cls:
            result = send_mail(self.sample_mail(), self.sample_config(security="ssl", port=465), dry_run=True)
        smtp_cls.assert_not_called()
        ssl_cls.assert_not_called()
        self.assertFalse(result["sent"])
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["security"], "ssl")
        self.assertGreater(result["text_size"], 0)
        self.assertGreater(result["html_size"], 0)

    def test_missing_environment_values_raise(self):
        args = self.empty_args()
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
                        "--smtp-port",
                        "465",
                        "--smtp-security",
                        "ssl",
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
        self.assertIn("security: ssl", output.getvalue())
        self.assertNotIn("secret-password", output.getvalue())


if __name__ == "__main__":
    unittest.main()
