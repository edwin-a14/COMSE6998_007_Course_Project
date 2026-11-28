"""Verify matched sampling and resume protection without dataset or model calls."""

from copy import deepcopy
import unittest

from aftraj.data import DOMAINS, Trajectory
from aftraj.run_eval import _config_mismatch, RESUME_KEYS
from aftraj.sampling import balanced_sample


def fixtures():
    return [
        Trajectory(conv_id=f"{domain}-{label}-{i}", domain=domain, label=label,
                   task="Fixture task", turns=[])
        for domain in DOMAINS for label in ("safe", "unsafe") for i in range(5)
    ]


class SamplingTest(unittest.TestCase):
    def test_balance_and_input_order_independence(self):
        data = fixtures()
        sample, config = balanced_sample(data, 2, 43)
        again, again_config = balanced_sample(list(reversed(data)), 2, 43)
        self.assertEqual((sample, config), (again, again_config))
        self.assertEqual(len(sample), 12)
        self.assertEqual(len(set(config["conv_ids"])), 12)
        self.assertEqual(set(config["counts"].values()), {2})
        self.assertEqual(sum(item.label == "unsafe" for item in sample), 6)

    def test_fingerprint_changes_when_evidence_changes(self):
        data = fixtures()
        _, config = balanced_sample(data, 2, 43)
        changed = deepcopy(data)
        for item in changed:
            item.task += " changed"
        _, updated = balanced_sample(changed, 2, 43)
        self.assertEqual(config["conv_ids"], updated["conv_ids"])
        self.assertNotEqual(config["sha256"], updated["sha256"])
        old = dict.fromkeys(RESUME_KEYS)
        self.assertTrue(_config_mismatch({**old, "sample": config}, {**old, "sample": updated}))

    def test_refuses_small_groups_and_invalid_sizes(self):
        for count in (0, 6):
            with self.subTest(count=count), self.assertRaises(ValueError):
                balanced_sample(fixtures(), count, 43)
        data = [item for item in fixtures() if (item.domain, item.label) != ("coding", "unsafe")]
        with self.assertRaisesRegex(ValueError, "coding/unsafe"):
            balanced_sample(data, 1, 43)
