#!/usr/bin/env python3
"""Send a built Perplexica email through generic SMTP.

No cron, no SMTP secret storage, and no hard-coded recipient. Configuration comes
from environment variables, an optional .env file, or CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path
from typing import Any


ENV_SMTP_HOST = "PERPLEXICA_SMTP_HOST"
ENV_SMTP_PORT = "PERPLEXICA_SMTP_PORT"
ENV_SMTP_USERNAME = "PERPLEXICA_SMTP_USERNAME"
ENV_SMTP_PASSWORD = "PERPLEXICA_SMTP_PASSWORD"
ENV_MAIL_FROM = "PERPLEXICA_MAIL_FROM"
ENV_MAIL_TO = "PERPLEXICA_MAIL_TO"
ENV_SMTP_SECURITY = "PERPLEXICA_SMTP_SECURITY"
ENV_SMTP_STARTTLS = "PERPLEXICA_SMTP_STARTTLS"
DEFAULT_SMTP_PORT = 587
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"
SMTP_SECURITY_VALUES = {"ssl", "starttls", "none"}


class MailSendError(Exception):
    """User-facing mail sending failure."""


@dataclass(frozen=True)
class SMTPConfig:
    host: str
    port: int = DEFAULT_SMTP_PORT
    username: str | None = None
    password: str | None = None
    from_address: str = ""
    to_address: str = ""
    security: str = "starttls"
    timeout: int = DEFAULT_TIMEOUT_SECONDS

    @property
    def recipients(self) -> list[str]:
        return [item.strip() for item in self.to_address.split(",") if item.strip()]


def parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise MailSendError(f"Invalid boolean value for {ENV_SMTP_STARTTLS}.")


def parse_port(value: str | int | None) -> int:
    if value is None or value == "":
        return DEFAULT_SMTP_PORT
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise MailSendError("SMTP port must be an integer.") from exc
    if port <= 0 or port > 65535:
        raise MailSendError("SMTP port must be between 1 and 65535.")
    return port


def parse_security(value: str | None, port: int, legacy_starttls: str | None = None) -> str:
    if value is not None and value != "":
        security = value.strip().lower()
        if security not in SMTP_SECURITY_VALUES:
            raise MailSendError(f"Invalid SMTP security value: {value!r}. Use ssl, starttls, or none.")
        return security

    if legacy_starttls is not None and legacy_starttls != "":
        return "starttls" if parse_bool(legacy_starttls) else "none"

    return "ssl" if port == 465 else "starttls"


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MailSendError(f"Cannot read env file: {path}") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def merged_config_values(args: argparse.Namespace) -> dict[str, str | None]:
    env_file = args.env_file or DEFAULT_ENV_FILE
    file_values = parse_env_file(env_file)

    def pick(cli_value: str | None, env_key: str) -> str | None:
        if cli_value is not None and cli_value != "":
            return cli_value
        if env_key in os.environ:
            return os.environ.get(env_key)
        return file_values.get(env_key)

    return {
        ENV_SMTP_HOST: pick(args.smtp_host, ENV_SMTP_HOST),
        ENV_SMTP_PORT: pick(args.smtp_port, ENV_SMTP_PORT),
        ENV_SMTP_USERNAME: pick(args.smtp_username, ENV_SMTP_USERNAME),
        ENV_SMTP_PASSWORD: pick(args.smtp_password, ENV_SMTP_PASSWORD),
        ENV_MAIL_FROM: pick(args.from_address, ENV_MAIL_FROM),
        ENV_MAIL_TO: pick(args.to, ENV_MAIL_TO),
        ENV_SMTP_SECURITY: pick(args.smtp_security, ENV_SMTP_SECURITY),
        ENV_SMTP_STARTTLS: pick(args.smtp_starttls, ENV_SMTP_STARTTLS),
    }


def smtp_config_from_env_and_args(args: argparse.Namespace) -> SMTPConfig:
    values = merged_config_values(args)
    port = parse_port(values[ENV_SMTP_PORT])
    security = parse_security(values[ENV_SMTP_SECURITY], port, values[ENV_SMTP_STARTTLS])

    config = SMTPConfig(
        host=values[ENV_SMTP_HOST] or "",
        port=port,
        username=values[ENV_SMTP_USERNAME] or None,
        password=values[ENV_SMTP_PASSWORD] or None,
        from_address=values[ENV_MAIL_FROM] or "",
        to_address=values[ENV_MAIL_TO] or "",
        security=security,
        timeout=args.timeout,
    )
    validate_smtp_config(config)
    return config


def validate_smtp_config(config: SMTPConfig) -> None:
    if not config.host:
        raise MailSendError(f"Missing SMTP host. Set {ENV_SMTP_HOST} or use --smtp-host.")
    if not config.from_address:
        raise MailSendError(f"Missing From address. Set {ENV_MAIL_FROM} or use --from-address.")
    if not config.recipients:
        raise MailSendError(f"Missing To address. Set {ENV_MAIL_TO} or use --to.")
    if config.security not in SMTP_SECURITY_VALUES:
        raise MailSendError("SMTP security must be one of: ssl, starttls, none.")
    if (config.username and not config.password) or (config.password and not config.username):
        raise MailSendError("SMTP username and password must be provided together.")
    if config.timeout <= 0:
        raise MailSendError("SMTP timeout must be a positive integer.")


def normalize_mail_payload(payload: dict[str, Any], input_path: Path | None = None) -> dict[str, Any]:
    if all(isinstance(payload.get(key), str) for key in ("subject", "text", "html")):
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return {
            "subject": payload["subject"],
            "text": payload["text"],
            "html": payload["html"],
            "metadata": metadata,
        }

    if input_path is not None and isinstance(payload.get("subject"), str):
        text_path = input_path.with_suffix(".txt")
        html_path = input_path.with_suffix(".html")
        try:
            text = text_path.read_text(encoding="utf-8")
            html_body = html_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MailSendError(
                "Metadata JSON was provided, but sibling .txt/.html files could not be read."
            ) from exc
        return {
            "subject": payload["subject"],
            "text": text,
            "html": html_body,
            "metadata": payload,
        }

    raise MailSendError(
        "Invalid mail JSON. Expected subject/text/html/metadata or mail_builder metadata with sibling .txt/.html files."
    )


def load_mail_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MailSendError(f"Cannot read mail JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MailSendError(f"Invalid mail JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MailSendError("Invalid mail JSON: top-level value must be an object.")
    return normalize_mail_payload(payload, input_path=path)


def build_email_message(mail: dict[str, Any], smtp_config: SMTPConfig) -> EmailMessage:
    validate_smtp_config(smtp_config)
    normalized = normalize_mail_payload(mail)
    message = EmailMessage()
    message_id = make_msgid()
    message["Subject"] = normalized["subject"]
    message["From"] = smtp_config.from_address
    message["To"] = ", ".join(smtp_config.recipients)
    message["Message-ID"] = message_id
    message.set_content(normalized["text"], charset="utf-8")
    message.add_alternative(normalized["html"], subtype="html", charset="utf-8")
    return message


def send_mail(mail: dict[str, Any], smtp_config: SMTPConfig, dry_run: bool = False) -> dict[str, Any]:
    message = build_email_message(mail, smtp_config)
    text_payload = message.get_body(preferencelist=("plain",)).get_content()
    html_payload = message.get_body(preferencelist=("html",)).get_content()
    result = {
        "sent": False,
        "message_id": message["Message-ID"],
        "smtp_host": smtp_config.host,
        "smtp_port": smtp_config.port,
        "security": smtp_config.security,
        "recipients": smtp_config.recipients,
        "subject": message["Subject"],
        "text_size": len(text_payload.encode("utf-8")),
        "html_size": len(html_payload.encode("utf-8")),
        "dry_run": dry_run,
    }

    if dry_run:
        return result

    try:
        if smtp_config.security == "ssl":
            smtp_factory = smtplib.SMTP_SSL
        else:
            smtp_factory = smtplib.SMTP

        with smtp_factory(smtp_config.host, smtp_config.port, timeout=smtp_config.timeout) as smtp:
            if smtp_config.security == "starttls":
                smtp.starttls()
            if smtp_config.username and smtp_config.password:
                smtp.login(smtp_config.username, smtp_config.password)
            smtp.send_message(message)
    except (smtplib.SMTPException, OSError) as exc:
        raise MailSendError(f"SMTP send failed: {exc.__class__.__name__}: {exc}") from exc

    result["sent"] = True
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send or dry-run a built Perplexica email via SMTP.")
    parser.add_argument("mail_json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--smtp-host", help=f"SMTP host. Overrides {ENV_SMTP_HOST}.")
    parser.add_argument("--smtp-port", help=f"SMTP port. Overrides {ENV_SMTP_PORT}.")
    parser.add_argument("--smtp-username", help=f"SMTP username. Overrides {ENV_SMTP_USERNAME}.")
    parser.add_argument("--smtp-password", help=f"SMTP password. Overrides {ENV_SMTP_PASSWORD}.")
    parser.add_argument("--from-address", help=f"From address. Overrides {ENV_MAIL_FROM}.")
    parser.add_argument("--to", help=f"Recipient address, comma-separated. Overrides {ENV_MAIL_TO}.")
    parser.add_argument("--smtp-security", choices=sorted(SMTP_SECURITY_VALUES), help=f"Overrides {ENV_SMTP_SECURITY}.")
    parser.add_argument("--smtp-starttls", choices=["true", "false", "1", "0", "yes", "no"])
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mail = load_mail_json(args.mail_json)
        config = smtp_config_from_env_and_args(args)
        result = send_mail(mail, config, dry_run=args.dry_run)
        status = "dry_run" if args.dry_run else "sent"
        print(f"status: {status}")
        print(f"subject: {result['subject']}")
        print(f"from: {config.from_address}")
        print(f"to: {', '.join(result['recipients'])}")
        print(f"smtp_host: {result['smtp_host']}")
        print(f"smtp_port: {result['smtp_port']}")
        print(f"security: {result['security']}")
        print(f"text_size: {result['text_size']}")
        print(f"html_size: {result['html_size']}")
        print(f"message_id: {result['message_id']}")
        return 0
    except MailSendError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
