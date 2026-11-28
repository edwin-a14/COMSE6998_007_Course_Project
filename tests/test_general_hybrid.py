"""Public mode construction and short-circuit behavior, no API calls."""

import unittest

from audit_core import ToolCall, ToolSpec, Turn
from audit_core.auditors.rules import DOMAIN_CONFIGS
from audit_core.model import ModelResponse
from audit_core.modes import build_model_auditor


class Backend:
    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return ModelResponse('{"stage":"uncertain","stage_evidence":"Fixture stage","error_evidence":[],"decision":"allow","score":0,"reason":"Fixture","rule_ids":[]}', 10, 5)


class GeneralHybridTest(unittest.TestCase):
    def test_rules_block_saves_model_call_against_semantic_control(self):
        turn = Turn(kind="agent", role="assistant", tool_calls=(ToolCall("write", {"id": "UNSEEN"}),))
        tools = [ToolSpec("write", side_effect=True)]
        for mode, calls, decision in (("general-hybrid", 0, "block"), ("selective-semantic", 1, "allow")):
            with self.subTest(mode=mode):
                backend = Backend()
                auditor = build_model_auditor(mode, model="fake", backend=backend)
                verdict = auditor.audit([], turn, tools)
                self.assertEqual((backend.calls, verdict.decision), (calls, decision))

    def test_selective_modes_skip_reads_but_continuous_checks(self):
        turn = Turn(kind="agent", role="assistant", tool_calls=(ToolCall("lookup"),))
        for mode, count in (("general-hybrid", 0), ("selective-semantic", 0), ("continuous-semantic", 1)):
            backend = Backend()
            build_model_auditor(mode, model="fake", backend=backend).audit([], turn, [ToolSpec("lookup", side_effect=False)])
            self.assertEqual(backend.calls, count)

    def test_retail_configuration_is_explicit(self):
        turn = Turn(kind="agent", role="assistant", tool_calls=(ToolCall("write"),))
        backend = Backend()
        auditor = build_model_auditor("general-hybrid", model="fake", backend=backend,
                                      rules_config=DOMAIN_CONFIGS["retail"])
        verdict = auditor.audit([], turn, [ToolSpec("write", side_effect=True)])
        self.assertIn("D1_authenticated", verdict.rule_ids)
        self.assertEqual(backend.calls, 0)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model_auditor("typo", model="fake", backend=Backend())


if __name__ == "__main__":
    unittest.main()
