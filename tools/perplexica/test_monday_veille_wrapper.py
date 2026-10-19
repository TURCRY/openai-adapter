import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
WRAPPER = TOOLS_DIR / "monday_veille_wrapper.sh"
HELPER = TOOLS_DIR / "monday_veille_light.py"


@unittest.skipUnless(
    shutil.which("sh") and shutil.which("flock") and shutil.which("python3"),
    "POSIX shell context required",
)
class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.script_dir = self.root / "tools"
        self.script_dir.mkdir()
        shutil.copy2(WRAPPER, self.script_dir / "monday_veille_wrapper.sh")
        shutil.copy2(HELPER, self.script_dir / "monday_veille_light.py")
        self.marker_dir = self.script_dir / "output" / "veille_wrapper" / "markers"
        self.marker_dir.mkdir(parents=True)
        self.run_root = self.script_dir / "output" / "jobs" / "veille_expertise_mediation"
        self.run_root.mkdir(parents=True)
        self.calls = self.root / "calls.txt"
        fake = self.script_dir / "run_veille_expertise_mediation.sh"
        fake.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$*\" >> \"$FAKE_CALLS_FILE\"\n"
            "if [ \"${FAKE_RESUME_OK:-0}\" = \"1\" ]; then\n"
            "  mkdir -p \"$FAKE_MARKER_DIR\"\n"
            "  printf '{}\\n' > \"$FAKE_MARKER_DIR/$VEILLE_DATE.ok\"\n"
            "fi\n"
            "exit \"${FAKE_RESUME_RC:-0}\"\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            {
                "PERPLEXICA_PYTHON": shutil.which("python3"),
                "VEILLE_DATE": "2026-10-19",
                "VEILLE_SLOT": "09:00",
                "FAKE_CALLS_FILE": str(self.calls),
                "FAKE_MARKER_DIR": str(self.marker_dir),
            }
        )

    def tearDown(self):
        self.tmp.cleanup()

    def write_marker(self, kind="partial", **overrides):
        payload = {
            "date": "2026-10-19",
            "slot": "03:00",
            "status": kind,
            "run_id": "20261019T010004Z_test",
            "run_status": "partial",
            "raw_mail_sent": True,
            "editorial_mail_sent": False,
            "mail_sent": True,
        }
        payload.update(overrides)
        (self.marker_dir / f"2026-10-19.{kind}").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def write_run(self, **overrides):
        run_dir = self.run_root / "20261019T010004Z_test"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": "2026-10-19T01:00:04+00:00",
            "status": "partial",
            "mail_mode": "both",
            "raw_mail_sent": True,
            "editorial_requested": True,
            "editorial_mail_sent": False,
        }
        payload.update(overrides)
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
        (run_dir / "result.json").write_text("{}", encoding="utf-8")
        return run_dir

    def run_wrapper(self, slot="09:00", extra_env=None):
        env = dict(self.env)
        env["VEILLE_SLOT"] = slot
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["sh", str(self.script_dir / "monday_veille_wrapper.sh")],
            cwd=str(self.script_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def call_lines(self):
        if not self.calls.exists():
            return []
        return [line for line in self.calls.read_text(encoding="utf-8").splitlines() if line]

    def test_partial_resume_calls_job_once(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper("09:00")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.call_lines()
        self.assertEqual(len(calls), 1)
        self.assertIn("--resume-editorial-run", calls[0])
        self.assertIn("20261019T010004Z_test", calls[0])

    def test_partial_resume_success_creates_ok(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper("09:00", {"FAKE_RESUME_OK": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.marker_dir / "2026-10-19.ok").exists())

    def test_partial_resume_failure_keeps_partial(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper("09:00", {"FAKE_RESUME_RC": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.marker_dir / "2026-10-19.partial").exists())
        self.assertFalse((self.marker_dir / "2026-10-19.ok").exists())

    def test_two_successive_failures_one_attempt_each(self):
        self.write_marker()
        self.write_run()
        first = self.run_wrapper("09:00", {"FAKE_RESUME_RC": "1"})
        second = self.run_wrapper("12:00", {"FAKE_RESUME_RC": "1"})
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(len(self.call_lines()), 2)

    def test_late_slot_without_marker_does_not_run_job(self):
        result = self.run_wrapper("09:00")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_late_slot_with_ok_does_nothing(self):
        self.write_marker("ok", run_status="completed", editorial_mail_sent=True)
        result = self.run_wrapper("12:00")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_late_slot_with_smtp_unknown_does_not_resume(self):
        self.write_marker("partial")
        self.write_run(editorial_send_state="sending")
        result = self.run_wrapper("15:00")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_late_slot_with_failed_does_nothing(self):
        self.write_marker("failed", run_status="no_results")
        result = self.run_wrapper("18:00")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_recovery_only_flag_without_marker_does_not_run_job(self):
        result = self.run_wrapper("09:07", {"VEILLE_RECOVERY_ONLY": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_recovery_only_flag_with_partial_still_resumes(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper("09:07", {"VEILLE_RECOVERY_ONLY": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.call_lines()
        self.assertEqual(len(calls), 1)
        self.assertIn("--resume-editorial-run", calls[0])