import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from monday_veille_light import cmd_resume_candidate, compute_business_context


def run_candidate(marker_dir, run_root, date_str="2026-10-19"):
    output = io.StringIO()
    with redirect_stdout(output):
        rc = cmd_resume_candidate(Path(marker_dir), Path(run_root), date_str)
    fields = {}
    for line in output.getvalue().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return rc, fields


class ResumeCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.marker_dir = self.root / "markers"
        self.marker_dir.mkdir()
        self.run_root = self.root / "runs"
        self.run_dir = self.run_root / "20261019T010004Z_test"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "result.json").write_text("{}", encoding="utf-8")
        self.write_run({})
        self.write_marker()

    def tearDown(self):
        self.tmp.cleanup()

    def write_run(self, overrides):
        payload = {
            "started_at": "2026-10-19T01:00:04+00:00",
            "status": "partial",
            "mail_mode": "both",
            "raw_mail_sent": True,
            "editorial_requested": True,
            "editorial_mail_sent": False,
        }
        payload.update(overrides)
        (self.run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    def write_marker(self, overrides=None):
        payload = {
            "date": "2026-10-19",
            "run_id": self.run_dir.name,
            "raw_mail_sent": True,
            "editorial_mail_sent": False,
        }
        if overrides:
            payload.update(overrides)
        (self.marker_dir / "2026-10-19.partial").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_valid_legacy_run_is_resumable(self):
        rc, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(rc, 0)
        self.assertEqual(fields["action"], "resume")
        self.assertEqual(fields["run_id"], self.run_dir.name)
        self.assertEqual(fields["run_dir"], str(self.run_dir.resolve()))

    def test_send_state_sending_is_unknown(self):
        self.write_run({"editorial_send_state": "sending"})
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "smtp_outcome_unknown")
        self.assertEqual(fields["reason"], "send_state_sending")

    def test_send_state_sent_with_flag_is_already_sent(self):
        self.write_run({"editorial_send_state": "sent", "editorial_mail_sent": True})
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "already_sent")

    def test_send_state_sent_without_flag_is_unknown(self):
        self.write_run({"editorial_send_state": "sent", "editorial_mail_sent": False})
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "smtp_outcome_unknown")
        self.assertEqual(fields["reason"], "send_state_sent_without_sent_flag")

    def test_unknown_send_state_is_unsafe(self):
        self.write_run({"editorial_send_state": "unexpected"})
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "unsafe")
        self.assertEqual(fields["reason"], "unknown_send_state")

    def test_old_run_date_is_unsafe(self):
        self.write_run({"started_at": "2026-10-12T01:00:04+00:00"})
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "unsafe")
        self.assertEqual(fields["reason"], "run_date_mismatch")

    def test_invalid_run_id_is_unsafe(self):
        self.write_marker({"run_id": "../outside"})
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "unsafe")
        self.assertEqual(fields["reason"], "invalid_run_id")

    def test_marker_date_mismatch_is_unsafe(self):
        self.write_marker({"date": "2026-10-12"})
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "unsafe")
        self.assertEqual(fields["reason"], "marker_date_mismatch")

    def test_missing_result_json_is_unsafe(self):
        (self.run_dir / "result.json").unlink()
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "unsafe")
        self.assertEqual(fields["reason"], "result_json_missing")

    def test_no_marker_returns_none(self):
        (self.marker_dir / "2026-10-19.partial").unlink()
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "none")

    def test_marker_editorial_sent_true_is_already_sent(self):
        self.write_marker({"editorial_mail_sent": True})
        _, fields = run_candidate(self.marker_dir, self.run_root)
        self.assertEqual(fields["action"], "already_sent")
        self.assertEqual(fields["reason"], "marker_editorial_mail_sent_true")


class BusinessContextTests(unittest.TestCase):
    def test_monday_uses_current_local_date(self):
        context = compute_business_context("2026-10-19", 1, 3)
        self.assertEqual(context["mode"], "normal")
        self.assertEqual(context["business_date"], "2026-10-19")
        self.assertEqual(context["today"], "2026-10-19")

    def test_tuesday_early_slot_uses_previous_monday(self):
        context = compute_business_context("2026-10-20", 2, 3)
        self.assertEqual(context["mode"], "recovery")
        self.assertEqual(context["business_date"], "2026-10-19")

    def test_tuesday_last_recovery_slot_uses_previous_monday(self):
        context = compute_business_context("2026-10-20", 2, 6)
        self.assertEqual(context["mode"], "recovery")
        self.assertEqual(context["business_date"], "2026-10-19")

    def test_tuesday_after_window_is_closed(self):
        context = compute_business_context("2026-10-20", 2, 7)
        self.assertEqual(context["mode"], "recovery_closed")
        self.assertEqual(context["business_date"], "2026-10-19")

    def test_tuesday_before_window_is_closed(self):
        context = compute_business_context("2026-10-20", 2, 2)
        self.assertEqual(context["mode"], "recovery_closed")
        self.assertEqual(context["business_date"], "2026-10-19")

    def test_winter_tuesday_uses_previous_monday(self):
        context = compute_business_context("2026-01-20", 2, 4)
        self.assertEqual(context["mode"], "recovery")
        self.assertEqual(context["business_date"], "2026-01-19")

    def test_tuesday_after_dst_change_uses_previous_monday(self):
        context = compute_business_context("2026-10-27", 2, 4)
        self.assertEqual(context["mode"], "recovery")
        self.assertEqual(context["business_date"], "2026-10-26")

    def test_other_weekdays_keep_normal_mode(self):
        context = compute_business_context("2026-10-21", 3, 3)
        self.assertEqual(context["mode"], "normal")
        self.assertEqual(context["business_date"], "2026-10-21")

    def test_weekday_must_match_date(self):
        with self.assertRaises(ValueError):
            compute_business_context("2026-10-20", 1, 3)

    def test_invalid_today_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_business_context("not-a-date", 1, 3)
