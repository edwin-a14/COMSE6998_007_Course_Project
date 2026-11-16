"""Tests for `run_eval.main --resume` on hand-made trajectories. Standard library only."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aftraj import run_eval
from aftraj.tests.test_adapter import math_trajectory


def trajectory(conv_id: str, label: str):
    t = math_trajectory(label=label)
    t.conv_id = conv_id
    return t


TRAJECTORIES = [trajectory("math_a_safe", "safe"), trajectory("math_b_injected", "unsafe")]


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "run"

    def run_main(self, *extra: str, trajectories=TRAJECTORIES):
        argv = ["--auditor", "allow-all", "--output", str(self.output), *extra]
        stderr = io.StringIO()
        with mock.patch.object(run_eval, "load", return_value=list(trajectories)) as load, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr), \
                self.assertRaises(SystemExit):
            run_eval.main(argv)
        return load, stderr.getvalue()

    def test_unknown_domain_is_refused_before_loading(self):
        load, stderr = self.run_main("--domains", "math,maths")
        load.assert_not_called()
        self.assertIn("unknown --domains maths", stderr)
        self.assertFalse(self.output.exists())

    def test_empty_selection_writes_nothing(self):
        _, stderr = self.run_main("--domains", "coding", trajectories=[])
        self.assertIn("no trajectories selected", stderr)
        self.assertFalse(self.output.exists())


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "run"
        self.per_sample = self.output / "per_sample.jsonl"

    def run_main(self, *extra: str, trajectories=TRAJECTORIES):
        argv = ["--auditor", "allow-all", "--output", str(self.output), *extra]
        with mock.patch.object(run_eval, "load", return_value=list(trajectories)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_eval.main(argv)

    def records(self):
        return [json.loads(line) for line in self.per_sample.read_text(encoding="utf-8").splitlines()]

    def test_resume_of_a_finished_run_changes_nothing(self):
        first = self.run_main()
        before = self.per_sample.read_bytes()
        self.assertEqual(self.run_main("--resume"), first)
        self.assertEqual(self.per_sample.read_bytes(), before)

    def test_resume_with_other_settings_is_refused(self):
        self.run_main()
        before = self.per_sample.read_bytes()
        for extra in (["--auditor", "rules"], ["--limit", "1"], ["--full-scan"], ["--domains", "math"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.run_main("--resume", *extra)
        self.assertEqual(self.per_sample.read_bytes(), before)

    def test_summary_counts_only_the_current_selection(self):
        self.run_main()
        foreign = dict(self.records()[-1], conv_id="math_z_injected")
        with self.per_sample.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(foreign) + "\n")
            handle.write(json.dumps(self.records()[-1]) + "\n")  # a repeated row
        summary = self.run_main("--resume")
        self.assertEqual((summary["unsafe"], summary["audit_calls"]), (1, 6))

    def test_partial_last_line_is_dropped_and_rerun(self):
        self.run_main()
        complete = self.per_sample.read_bytes()
        self.per_sample.write_bytes(complete[:-20])  # crash in the middle of the last row
        self.run_main("--resume")
        records = self.records()
        self.assertEqual([r.get("_record_type") for r in records], ["config", None, None])
        self.assertEqual([r["conv_id"] for r in records[1:]], [t.conv_id for t in TRAJECTORIES])

    def test_partial_config_line_starts_over(self):
        self.output.mkdir()
        self.per_sample.write_text('{"_record_type": "con', encoding="utf-8")
        self.run_main("--resume")
        self.assertEqual([r.get("_record_type") for r in self.records()], ["config", None, None])

    def test_broken_line_before_the_end_is_refused(self):
        self.run_main()
        lines = self.per_sample.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[1] = "{broken\n"
        self.per_sample.write_text("".join(lines), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_main("--resume")

    def test_rows_without_config_are_refused(self):
        self.run_main()
        self.per_sample.write_text("".join(self.per_sample.read_text(encoding="utf-8").splitlines(True)[1:]),
                                   encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_main("--resume")


if __name__ == "__main__":
    unittest.main()
