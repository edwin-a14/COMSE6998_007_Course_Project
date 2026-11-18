"""Checks on the real AFTraj files. Skipped unless data/aftraj/ is downloaded and
pandas with pyarrow>=20 is installed (see aftraj/requirements.txt)."""

import unittest

from aftraj.adapter import audit_points, classify_action
from aftraj.data import DEFAULT_DATA_DIR, FILES, load, source_id


def _available() -> bool:
    if not all((DEFAULT_DATA_DIR / name).exists() for name in FILES):
        return False
    try:
        import pandas  # noqa: F401
        import pyarrow
    except ImportError:
        return False
    return int(pyarrow.__version__.split(".")[0]) >= 20


@unittest.skipUnless(_available(), "AFTraj data or pyarrow>=20 not available")
class RealDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all = load(split="all")
        cls.test = load(split="test")
        cls.dev = load(split="dev")

    def test_counts_match_the_dataset_card(self):
        self.assertEqual(len(self.all), 2272)
        self.assertEqual(sum(t.label == "safe" for t in self.all), 1158)
        self.assertEqual((sum(t.label == "safe" for t in self.test), sum(t.label == "unsafe" for t in self.test)), (169, 163))

    def test_dev_shares_no_trajectory_source_or_task_with_test(self):
        test_ids = {t.conv_id for t in self.test}
        self.assertFalse(test_ids & {t.conv_id for t in self.dev})
        self.assertFalse({source_id(t) for t in test_ids} & {source_id(t.conv_id) for t in self.dev})
        self.assertFalse({(t.domain, t.task) for t in self.test} & {(t.domain, t.task) for t in self.dev})

    def test_every_tool_action_parses(self):
        for trajectory in self.all:
            for turn in trajectory.turns:
                self.assertNotEqual(classify_action(turn["action"]), "unparsed", trajectory.conv_id)

    def test_mistake_steps_are_inside_trajectories(self):
        for trajectory in self.all:
            if trajectory.label == "unsafe":
                self.assertTrue(0 <= trajectory.mistake_step < len(trajectory.turns), trajectory.conv_id)

    def test_audit_points_cover_every_agent_turn(self):
        trajectory = self.test[0]
        agent_steps = [i for i, t in enumerate(trajectory.turns) if t["role"] not in ("user", "environment")]
        self.assertEqual([p.step for p in audit_points(trajectory)], agent_steps)


if __name__ == "__main__":
    unittest.main()
