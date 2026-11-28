"""Selection and composition contracts; no model calls or benchmark imports."""

from copy import deepcopy
import unittest

from audit_core import AuditContext, Auditor, CompositeAuditor, ToolCall, ToolSpec, Turn, Verdict
from audit_core.auditors import RulesAuditor
from audit_core.auditors.selective import SelectiveSemanticAuditor


class RecordingAuditor(Auditor):
    name = "fake-semantic"

    def __init__(self, verdict=None):
        self.verdict = verdict if verdict is not None else Verdict.allow()
        self.calls = []

    def audit(self, history, proposal, tools, context=None):
        self.calls.append((history, proposal, tools, context))
        return self.verdict


READ = ToolSpec("lookup", side_effect=False)
WRITE = ToolSpec("update", side_effect=True)
UNKNOWN = ToolSpec("request", side_effect=None)
TOOLS = [READ, WRITE, UNKNOWN]


def proposal(*names, **kwargs):
    return Turn(kind="agent", role="assistant",
                tool_calls=tuple(ToolCall(name) for name in names), **kwargs)


class SelectiveSemanticTest(unittest.TestCase):
    def test_selected_proposals_reach_semantic_once(self):
        cases = [
            (proposal(content="Done."), "text"),
            (proposal(thought="The answer must be 42."), "thought"),
            (proposal("update"), "write_tool"),
            (proposal("request"), "unknown_tool"),
            (proposal("unlisted"), "unknown_tool"),
            (proposal(raw_action="<code>run()</code>"), "raw_action"),
            (proposal("lookup", content="Your order is cancelled."), "text"),
            (proposal("lookup", thought="Assume the cancellation worked."), "thought"),
            (proposal("lookup", raw_action="unparsed action"), "raw_action"),
            (proposal("lookup", "update"), "write_tool"),
            (proposal("lookup", "request"), "unknown_tool"),
        ]
        for turn, reason in cases:
            with self.subTest(reason=reason, turn=turn):
                model = RecordingAuditor()
                verdict = SelectiveSemanticAuditor(model).audit([], turn, TOOLS)
                self.assertEqual(len(model.calls), 1)
                self.assertTrue(verdict.details["semantic_selection"]["selected"])
                self.assertEqual(verdict.details["semantic_selection"]["reason"], reason)

    def test_read_only_batches_skip_semantic(self):
        model = RecordingAuditor()
        turn = proposal("lookup", "lookup", content=" \n", thought=" ")
        verdict = SelectiveSemanticAuditor(model).audit([], turn, TOOLS)
        self.assertEqual(model.calls, [])
        self.assertEqual(verdict.decision, "allow")
        self.assertEqual(verdict.details["semantic_selection"], {
            "policy": "writes-text-unknown-v1", "selected": False,
            "reason": "read_only_tools", "semantic_invocations": 0,
        })

    def test_empty_turn_skips_without_claiming_it_was_checked(self):
        model = RecordingAuditor()
        verdict = SelectiveSemanticAuditor(model).audit([], proposal(), [])
        self.assertEqual(model.calls, [])
        self.assertFalse(verdict.details["semantic_selection"]["selected"])
        self.assertEqual(verdict.details["semantic_selection"]["reason"], "empty_proposal")

    def test_missing_tool_catalog_is_not_treated_as_read_only(self):
        model = RecordingAuditor()
        SelectiveSemanticAuditor(model).audit([], proposal("lookup"), [])
        self.assertEqual(len(model.calls), 1)

    def test_passes_complete_inputs_and_preserves_verdict_and_usage(self):
        history = [Turn(kind="user", role="user", content="Update A only.")]
        turn = proposal("lookup", "update")
        context = AuditContext(task="Update A", policy="Ask first")
        expected = Verdict("block", 0.8, "Not authorized", ["intent"],
                           {"input_tokens": 120, "output_tokens": 20})
        before = deepcopy((history, turn, TOOLS, context, expected))
        model = RecordingAuditor(expected)
        result = SelectiveSemanticAuditor(model).audit(history, turn, TOOLS, context)
        for actual, original in zip(model.calls[0], (history, turn, TOOLS, context)):
            self.assertIs(actual, original)
        self.assertEqual((result.decision, result.score, result.reason, result.rule_ids),
                         ("block", 0.8, "Not authorized", ["intent"]))
        self.assertEqual(result.details["input_tokens"], 120)
        self.assertEqual(result.details["output_tokens"], 20)
        self.assertEqual(result.details["semantic_selection"]["semantic_invocations"], 1)
        self.assertEqual((history, turn, TOOLS, context, expected), before)

    def test_rules_block_prevents_semantic_dispatch(self):
        model = RecordingAuditor()
        auditor = CompositeAuditor(RulesAuditor(), SelectiveSemanticAuditor(model))
        turn = Turn(kind="agent", role="assistant", tool_calls=(
            ToolCall("update", {"account_id": "UNSEEN"}),
        ))
        verdict = auditor.audit([], turn, TOOLS)
        self.assertTrue(verdict.blocked)
        self.assertIn("D4_grounded_ids", verdict.rule_ids)
        self.assertEqual(model.calls, [])

    def test_rules_warning_does_not_skip_semantic_or_become_block(self):
        model = RecordingAuditor()
        auditor = CompositeAuditor(RulesAuditor(), SelectiveSemanticAuditor(model))
        verdict = auditor.audit([], proposal("lookup", "update"), TOOLS)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(verdict.decision, "warn")
        self.assertIn("D5_one_call_at_a_time", verdict.rule_ids)

    def test_text_failure_claim_can_be_blocked_after_rules_allow(self):
        history = [Turn(kind="tool", role="tool", content="Cancellation failed", error=True)]
        model = RecordingAuditor(Verdict("block", 0.9, "Contradicts tool result"))
        auditor = CompositeAuditor(RulesAuditor(), SelectiveSemanticAuditor(model))
        verdict = auditor.audit(history, proposal(content="Cancelled successfully"), TOOLS)
        self.assertTrue(verdict.blocked)
        self.assertEqual(len(model.calls), 1)

    def test_backend_error_is_not_reported_as_agent_mistake(self):
        class FailingAuditor(RecordingAuditor):
            def audit(self, *args, **kwargs):
                raise TimeoutError("Model unavailable")

        with self.assertRaises(TimeoutError):
            SelectiveSemanticAuditor(FailingAuditor()).audit([], proposal(content="Done"), [])


if __name__ == "__main__":
    unittest.main()
