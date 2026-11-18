"""Unit tests for the shared audit interface and general auditors. Standard
library only: no τ²-bench, no AFTraj data, no model calls."""

import unittest

from audit_core import (
    AllowAllAuditor,
    AuditContext,
    Auditor,
    CompositeAuditor,
    ToolCall,
    ToolSpec,
    Turn,
    Verdict,
)
from audit_core.auditors import GENERIC_CONFIG, RulesAuditor
from audit_core.auditors.rules import DOMAIN_CONFIGS, build_facts

PAY = ToolSpec("pay", "Send money", side_effect=True)
READ = ToolSpec("lookup", "Read a record", side_effect=False)
UNKNOWN = ToolSpec("http", "Send a request", side_effect=None)
TOOLS = [PAY, READ, UNKNOWN]


def agent(*calls: ToolCall, role: str = "assistant", content: str = "") -> Turn:
    return Turn(kind="agent", role=role, content=content, tool_calls=calls)


def result(call_id: str | None, content: str, error: bool | None = False) -> Turn:
    return Turn(kind="tool", role="tool", content=content, tool_call_id=call_id, error=error)


class InterfaceTest(unittest.TestCase):
    def test_rejects_unknown_turn_kind_and_decision(self):
        with self.assertRaises(ValueError):
            Turn(kind="robot", role="x")
        with self.assertRaises(ValueError):
            Verdict("maybe", 0.5)
        with self.assertRaises(ValueError):
            Verdict("block", 1.5)

    def test_tool_calls_become_a_tuple(self):
        turn = Turn(kind="agent", role="a", tool_calls=[ToolCall("pay")])
        self.assertIsInstance(turn.tool_calls, tuple)

    def test_call_key_ignores_id_and_key_order(self):
        first = ToolCall("pay", {"a": 1, "b": 2}, id="x")
        second = ToolCall("pay", {"b": 2, "a": 1}, id="y")
        self.assertEqual(first.key(), second.key())

    def test_raw_text_arguments_have_no_kwargs(self):
        self.assertEqual(ToolCall("python", "print(1)").kwargs, {})

    def test_auditor_is_abstract(self):
        with self.assertRaises(TypeError):
            Auditor()


class RulesAuditorTest(unittest.TestCase):
    def setUp(self):
        self.auditor = RulesAuditor(GENERIC_CONFIG)

    def test_text_only_proposal_is_allowed(self):
        verdict = self.auditor.audit([], agent(content="The answer is 42."), TOOLS)
        self.assertEqual((verdict.decision, verdict.score), ("allow", 0.0))

    def test_read_and_unknown_tools_are_not_checked_by_default(self):
        proposal = agent(ToolCall("lookup", {"order_id": "X9"}), ToolCall("http", {"order_id": "X9"}))
        self.assertEqual(self.auditor.audit([], proposal, TOOLS).details["checked_calls"], [])

    def test_unknown_tools_are_checked_on_request(self):
        auditor = RulesAuditor(GENERIC_CONFIG, audit_unknown=True)
        verdict = auditor.audit([], agent(ToolCall("http", {"order_id": "X9"})), TOOLS)
        self.assertEqual(verdict.rule_ids, ["D4_grounded_ids"])

    def test_tool_missing_from_specs_is_not_checked(self):
        verdict = self.auditor.audit([], agent(ToolCall("mystery", {"order_id": "X9"})), [])
        self.assertEqual(verdict.decision, "allow")

    def test_generic_config_skips_authentication_rules(self):
        # No auth tools configured: D1 and D2's account check must not fire.
        history = [Turn(kind="user", role="user", content="Pay order A1 for user_7")]
        proposal = agent(ToolCall("pay", {"order_id": "A1", "user_id": "user_7"}))
        self.assertEqual(self.auditor.audit(history, proposal, TOOLS).decision, "allow")

    def test_d4_blocks_an_ungrounded_id(self):
        verdict = self.auditor.audit([], agent(ToolCall("pay", {"account_id": "ACC1"})), TOOLS)
        self.assertEqual((verdict.decision, verdict.score), ("block", 1.0))
        self.assertEqual(verdict.rule_ids, ["D4_grounded_ids"])
        self.assertIn("ACC1", verdict.reason)

    def test_own_text_is_not_evidence_but_other_agents_text_is(self):
        proposal = agent(ToolCall("pay", {"account_id": "ACC1"}), role="Payer")
        own = [agent(role="Payer", content="Account ACC1 looks right.")]
        other = [agent(role="Researcher", content="The account is ACC1.")]
        self.assertTrue(self.auditor.audit(own, proposal, TOOLS).blocked)
        self.assertFalse(self.auditor.audit(other, proposal, TOOLS).blocked)

    def test_unlinked_tool_result_is_evidence(self):
        history = [agent(ToolCall("lookup", {}, id="c1"), ToolCall("lookup", {}, id="c2")),
                   result(None, '{"account_id": "ACC1"}', error=None)]
        verdict = self.auditor.audit(history, agent(ToolCall("pay", {"account_id": "ACC1"})), TOOLS)
        self.assertEqual(verdict.decision, "allow")

    def test_result_for_a_call_never_made_is_ignored(self):
        history = [result("ghost", '{"account_id": "ACC1"}')]
        verdict = self.auditor.audit(history, agent(ToolCall("pay", {"account_id": "ACC1"})), TOOLS)
        self.assertTrue(verdict.blocked)

    def test_d5_warns_on_a_batched_write(self):
        proposal = agent(ToolCall("lookup", {}), ToolCall("pay", {"amount": 5}))
        verdict = self.auditor.audit([], proposal, TOOLS)
        self.assertEqual((verdict.decision, verdict.score), ("warn", 0.5))
        self.assertEqual(verdict.rule_ids, ["D5_one_call_at_a_time"])

    def test_d6_blocks_the_third_identical_attempt(self):
        call = ToolCall("pay", {"amount": 5})
        history = []
        for attempt in range(2):
            call_id = f"c{attempt}"
            history += [agent(ToolCall("pay", {"amount": 5}, id=call_id)), result(call_id, "Error", error=True)]
        self.assertEqual(self.auditor.audit(history, agent(call), TOOLS).rule_ids, ["D6_repeated_failure"])

    def test_unknown_error_status_is_not_a_failure(self):
        history = [agent(ToolCall("pay", {"amount": 5}, id="c1")), result("c1", "Error", error=None)] * 2
        self.assertEqual(self.auditor.audit(history, agent(ToolCall("pay", {"amount": 5})), TOOLS).decision, "allow")

    def test_raw_text_arguments_do_not_crash(self):
        verdict = self.auditor.audit([], agent(ToolCall("pay", "send everything")), TOOLS)
        self.assertEqual(verdict.decision, "allow")

    def test_does_not_modify_inputs(self):
        history = [Turn(kind="user", role="user", content="hi")]
        proposal = agent(ToolCall("pay", {"account_id": "ACC1"}))
        before = (list(history), proposal, list(TOOLS))
        self.auditor.audit(history, proposal, TOOLS)
        self.assertEqual((history, proposal, TOOLS), before)

    def test_retail_config_still_requires_authentication(self):
        auditor = RulesAuditor(DOMAIN_CONFIGS["retail"])
        cancel = ToolSpec("cancel_pending_order", side_effect=True)
        history = [Turn(kind="user", role="user", content="Cancel #W1")]
        verdict = auditor.audit(history, agent(ToolCall("cancel_pending_order", {"order_id": "#W1"})), [cancel])
        self.assertEqual(verdict.rule_ids, ["D1_authenticated"])

    def test_facts_author_defaults_to_assistant(self):
        facts = build_facts([agent(content="ACC1")], GENERIC_CONFIG)
        self.assertFalse(facts.grounded("ACC1"))


class _Fixed(Auditor):
    def __init__(self, name, verdict):
        self.name, self.verdict, self.calls = name, verdict, 0

    def audit(self, history, proposal, tools, context=None):
        self.calls += 1
        return self.verdict


class CompositeAuditorTest(unittest.TestCase):
    def test_stops_at_first_block(self):
        rules = _Fixed("rules", Verdict("block", 1.0, reason="bad id", rule_ids=["D4"]))
        model = _Fixed("model", Verdict.allow())
        verdict = CompositeAuditor(rules, model).audit([], agent(), [])
        self.assertEqual((verdict.decision, verdict.reason, verdict.rule_ids), ("block", "bad id", ["D4"]))
        self.assertEqual(model.calls, 0)

    def test_takes_most_severe_decision_and_highest_score(self):
        warn = _Fixed("a", Verdict("warn", 0.5, reason="batched"))
        allow = _Fixed("b", Verdict("allow", 0.3))
        verdict = CompositeAuditor(warn, allow).audit([], agent(), [])
        self.assertEqual((verdict.decision, verdict.score, verdict.reason), ("warn", 0.5, "batched"))
        self.assertEqual([item["name"] for item in verdict.details["auditors"]], ["a", "b"])
        self.assertEqual(CompositeAuditor(warn, allow).name, "a+b")

    def test_allow_all(self):
        verdict = AllowAllAuditor().audit([], agent(), [], AuditContext(task="t"))
        self.assertEqual((verdict.decision, verdict.score), ("allow", 0.0))


if __name__ == "__main__":
    unittest.main()
