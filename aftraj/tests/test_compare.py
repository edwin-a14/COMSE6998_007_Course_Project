"""Ensure comparison rejects mismatches and shares the completed cohort."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from aftraj.compare import compare
from aftraj.run_eval import evaluate, summarize
from aftraj.tests.test_adapter import math_trajectory
from audit_core import AllowAllAuditor


def fixture():
    trajectories = [math_trajectory(label=label) for label in ("safe", "unsafe")]
    for index, trajectory in enumerate(trajectories):
        trajectory.conv_id = f"fixture-{index}"
    rows = [evaluate(AllowAllAuditor(), item) for item in trajectories]
    report = {
        "config": {"auditor": "allow-all", "trajectories": 2},
        **summarize(rows, trajectories),
    }
    return report, {row["conv_id"]: row for row in rows}


class CompareTest(unittest.TestCase):
    def test_uses_common_completed_cohort(self):
        first = fixture()
        report, rows = deepcopy(first)
        rows["fixture-1"]["status"] = "error"
        report["failed_trajectories"] = 1
        with patch("aftraj.compare.load_run", side_effect=[first, (report, rows)]):
            text = compare(["a", "b"])
        self.assertIn("Completed in every mode: 1", text)
        self.assertIn("| allow-all |", text)

    def test_refuses_a_different_selection(self):
        first = fixture()
        second = deepcopy(first)
        second[0]["config"]["sample"] = {"sha256": "changed"}
        with patch("aftraj.compare.load_run", side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, "sample"):
                compare(["a", "b"])

    def test_refuses_incomplete_runs(self):
        first = fixture()
        second = deepcopy(first)
        second[1].pop("fixture-1")
        with patch("aftraj.compare.load_run", side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, "complete trajectory selection"):
                compare(["a", "b"])
