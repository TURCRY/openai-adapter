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

MONDAY = "2026-10-19"
TUESDAY = "2026-10-20"
OLD_MONDAY = "2026-10-12"
MONDAY_RUN_ID = "20261019T010004Z_test"
OLD_RUN_ID = "20261012T010004Z_test"


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
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        fake_date = self.bin_dir / "date"
        fake_date.write_text(
            '#!/bin/sh\n'
            'case "$1" in\n'
            "  +%F) printf '%s\\n' $FAKE_DATE_TODAY ;;\n"
            "  +%u) printf '%s\\n' $FAKE_DATE_WEEKDAY ;;\n"
            "  +%H) printf '%s\\n' $FAKE_DATE_HOUR ;;\n"
            "  +%H:%M) printf '%s\\n' $FAKE_DATE_SLOT ;;\n"
            "  +%Y-%m-%dT%H:%M:%S%z) printf '%s\\n' '2026-10-20T03:00:00+0200' ;;\n"
            "  *) printf '%s\\n' '' ;;\n"
            'esac\n',
            encoding="utf-8",
        )
        fake_date.chmod(0o755)
        fake = self.script_dir / "run_veille_expertise_mediation.sh"
        fake.write_text(
            '#!/bin/sh\n'
            "echo $* >> $FAKE_CALLS_FILE\n"
            'if [ ${FAKE_RESUME_OK:-0} = 1 ]; then\n'
            '  mkdir -p $FAKE_MARKER_DIR/history\n'
            '  if [ -f $FAKE_MARKER_DIR/$FAKE_OK_DATE.partial ]; then\n'
            '    mv $FAKE_MARKER_DIR/$FAKE_OK_DATE.partial $FAKE_MARKER_DIR/history/$FAKE_OK_DATE.partial.test.json\n'
            '  fi\n'
            "  printf '{}\\n' > $FAKE_MARKER_DIR/$FAKE_OK_DATE.ok\n"
            'fi\n'
            'exit ${FAKE_RESUME_RC:-0}\n',
            encoding='utf-8',
        )
        fake.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            {
                "PERPLEXICA_PYTHON": shutil.which("python3"),
                "VEILLE_TODAY": MONDAY,
                "VEILLE_WEEKDAY": "1",
                "VEILLE_HOUR": "3",
                "VEILLE_SLOT": "03:00",
                "VEILLE_SEARXNG_URL": "http://127.0.0.1:9/search?format=json",
                "VEILLE_SEARXNG_TIMEOUT": "1",
                "FAKE_CALLS_FILE": str(self.calls),
                "FAKE_MARKER_DIR": str(self.marker_dir),
                "FAKE_OK_DATE": MONDAY,
            }
        )

    def tearDown(self):
        self.tmp.cleanup()

    def write_marker(self, date_str=MONDAY, kind="partial", **overrides):
        payload = {
            "date": date_str,
            "slot": "03:00",
            "status": kind,
            "run_id": MONDAY_RUN_ID if date_str == MONDAY else OLD_RUN_ID,
            "run_status": "partial",
            "raw_mail_sent": True,
            "editorial_mail_sent": False,
            "mail_sent": True,
        }
        payload.update(overrides)
        (self.marker_dir / f"{date_str}.{kind}").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def write_run(self, date_str=MONDAY, run_id=MONDAY_RUN_ID, **overrides):
        run_dir = self.run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": f"{date_str}T01:00:04+00:00",
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

    def run_wrapper(
        self,
        *,
        today=MONDAY,
        weekday="1",
        hour="3",
        slot="03:00",
        use_local_date=False,
        extra_env=None,
    ):
        env = dict(self.env)
        if use_local_date:
            for key in ("VEILLE_TODAY", "VEILLE_WEEKDAY", "VEILLE_HOUR", "VEILLE_SLOT"):
                env.pop(key, None)
        else:
            env.update(
                {
                    "VEILLE_TODAY": today,
                    "VEILLE_WEEKDAY": weekday,
                    "VEILLE_HOUR": hour,
                    "VEILLE_SLOT": slot,
                }
            )
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

    def log_text(self, date_str):
        return (self.script_dir / "output" / "veille_wrapper" / "logs" / f"{date_str}.log").read_text(
            encoding="utf-8"
        )

    def test_monday_partial_resumes_same_monday(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.call_lines()
        self.assertEqual(len(calls), 1)
        self.assertIn("--resume-editorial-run", calls[0])
        self.assertIn(MONDAY_RUN_ID, calls[0])
        self.assertFalse((self.marker_dir / f"{TUESDAY}.partial").exists())

    def test_tuesday_partial_resumes_previous_monday(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper(today=TUESDAY, weekday="2", hour="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.call_lines()
        self.assertEqual(len(calls), 1)
        self.assertIn("--resume-editorial-run", calls[0])
        self.assertIn(MONDAY_RUN_ID, calls[0])
        self.assertTrue(self.log_text(TUESDAY))
        self.assertFalse(
            (self.script_dir / "output" / "veille_wrapper" / "logs" / f"{MONDAY}.log").exists()
        )

    def test_wrapper_uses_local_date_for_tuesday_transition(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper(
            use_local_date=True,
            extra_env={
                "PATH": str(self.bin_dir) + os.pathsep + self.env["PATH"],
                "VEILLE_SEARXNG_TIMEOUT": "1",
                "FAKE_DATE_TODAY": TUESDAY,
                "FAKE_DATE_WEEKDAY": "2",
                "FAKE_DATE_HOUR": "03",
                "FAKE_DATE_SLOT": "03:00",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.call_lines()
        self.assertEqual(len(calls), 1)
        self.assertIn(MONDAY_RUN_ID, calls[0])
        self.assertTrue(self.log_text(TUESDAY))

    def test_tuesday_success_promotes_ok_and_archives_partial(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper(
            today=TUESDAY, weekday="2", hour="4", extra_env={"FAKE_RESUME_OK": "1"}
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.marker_dir / f"{MONDAY}.ok").exists())
        self.assertFalse((self.marker_dir / f"{MONDAY}.partial").exists())
        self.assertTrue((self.marker_dir / "history" / f"{MONDAY}.partial.test.json").exists())

    def test_tuesday_without_partial_does_not_run_job(self):
        result = self.run_wrapper(today=TUESDAY, weekday="2", hour="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])
        for suffix in ("ok", "partial", "failed"):
            self.assertFalse((self.marker_dir / f"{MONDAY}.{suffix}").exists())
            self.assertFalse((self.marker_dir / f"{TUESDAY}.{suffix}").exists())

    def test_tuesday_with_ok_does_nothing(self):
        self.write_marker(kind="ok", run_status="completed", editorial_mail_sent=True)
        result = self.run_wrapper(today=TUESDAY, weekday="2", hour="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_tuesday_with_failed_does_nothing(self):
        self.write_marker(kind="failed", run_status="no_results")
        result = self.run_wrapper(today=TUESDAY, weekday="2", hour="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_tuesday_with_old_partial_does_not_resume(self):
        self.write_marker(date_str=OLD_MONDAY)
        self.write_run(date_str=OLD_MONDAY, run_id=OLD_RUN_ID)
        result = self.run_wrapper(today=TUESDAY, weekday="2", hour="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])
        self.assertTrue((self.marker_dir / f"{OLD_MONDAY}.partial").exists())
        self.assertFalse((self.marker_dir / f"{MONDAY}.partial").exists())

    def test_tuesday_with_wrong_run_date_is_not_resumed(self):
        self.write_marker()
        self.write_run(date_str=OLD_MONDAY, run_id=MONDAY_RUN_ID)
        result = self.run_wrapper(today=TUESDAY, weekday="2", hour="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])
        self.assertIn("run_date_mismatch", self.log_text(TUESDAY))
        self.assertTrue((self.marker_dir / f"{MONDAY}.partial").exists())

    def test_tuesday_with_smtp_unknown_does_not_resume(self):
        self.write_marker()
        self.write_run(editorial_send_state="sending")
        result = self.run_wrapper(today=TUESDAY, weekday="2", hour="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])
        self.assertIn("manual verification required", self.log_text(TUESDAY))
        self.assertTrue((self.marker_dir / f"{MONDAY}.partial").exists())

    def test_tuesday_after_recovery_window_does_nothing(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper(today=TUESDAY, weekday="2", hour="7", slot="07:00")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])
        self.assertIn("outside recovery window", self.log_text(TUESDAY))

    def test_tuesday_two_successive_failures_one_attempt_each(self):
        self.write_marker()
        self.write_run()
        first = self.run_wrapper(
            today=TUESDAY, weekday="2", hour="3", extra_env={"FAKE_RESUME_RC": "1"}
        )
        second = self.run_wrapper(
            today=TUESDAY, weekday="2", hour="4", extra_env={"FAKE_RESUME_RC": "1"}
        )
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(len(self.call_lines()), 2)
        self.assertTrue((self.marker_dir / f"{MONDAY}.partial").exists())
        self.assertFalse((self.marker_dir / f"{MONDAY}.ok").exists())

    def test_invalid_business_context_fails_closed(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper(today=TUESDAY, weekday="1", hour="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])
        self.assertIn("business-context failed", self.log_text(TUESDAY))

    def test_monday_ok_does_nothing(self):
        self.write_marker(kind="ok", run_status="completed", editorial_mail_sent=True)
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_monday_without_marker_does_not_resume(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_legacy_recovery_only_flag_without_marker_does_not_run_job(self):
        result = self.run_wrapper(
            today=MONDAY, weekday="1", hour="9", slot="09:00", extra_env={"VEILLE_RECOVERY_ONLY": "1"}
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_lines(), [])

    def test_legacy_recovery_only_flag_with_partial_still_resumes(self):
        self.write_marker()
        self.write_run()
        result = self.run_wrapper(
            today=MONDAY, weekday="1", hour="9", slot="09:00", extra_env={"VEILLE_RECOVERY_ONLY": "1"}
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.call_lines()
        self.assertEqual(len(calls), 1)
        self.assertIn("--resume-editorial-run", calls[0])
