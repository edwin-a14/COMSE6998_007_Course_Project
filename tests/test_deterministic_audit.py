"""Unit tests for the general deterministic audit. No model calls are made."""

import json
import unittest
from types import SimpleNamespace

from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from deterministic_audit import (
    DOMAIN_CONFIGS,
    GeneralDeterministicAuditor,
    audit_call,
    build_facts,
    write_tool_checker,
)

RETAIL = DOMAIN_CONFIGS["retail"]
USER = "daiki_silva_2903"
ORDER = "#W8835847"
OTHER_ORDER = "#W7999678"
OTHER_USER = "yusuf_rossi_9620"
OTHER_EMAIL = "yusuf@example.com"


class Dialogue:
    """Builds a τ²-bench message list with paired tool calls and results."""

    def __init__(self):
        self.messages = []
        self._next_id = 0

    def user(self, text: str) -> "Dialogue":
        self.messages.append(UserMessage(role="user", content=text))
        return self

    def assistant(self, text: str) -> "Dialogue":
        self.messages.append(AssistantMessage(role="assistant", content=text))
        return self

    def tool(self, name: str, arguments: dict, result, error: bool = False) -> "Dialogue":
        call = self.call(name, arguments)
        self.messages.append(AssistantMessage(role="assistant", tool_calls=[call]))
        content = result if isinstance(result, str) else json.dumps(result)
        self.messages.append(ToolMessage(id=call.id, role="tool", content=content, error=error))
        return self

    def call(self, name: str, arguments: dict) -> ToolCall:
        self._next_id += 1
        return ToolCall(id=f"call_{self._next_id}", name=name, arguments=arguments)

    def authenticate(self, user: str = USER, email: str = "daiki@example.com") -> "Dialogue":
        return self.tool("find_user_id_by_email", {"email": email}, user)

    def look_up(self, order: str = ORDER, owner: str = USER, status: str = "pending") -> "Dialogue":
        return self.tool(
            "get_order_details", {"order_id": order},
            {"order_id": order, "user_id": owner, "status": status},
        )


def cancel(order: str = ORDER) -> ToolCall:
    return ToolCall(
        id="proposal", name="cancel_pending_order",
        arguments={"order_id": order, "reason": "ordered by mistake"},
    )


def audit(dialogue: Dialogue, call: ToolCall, batch: list[ToolCall] | None = None) -> dict:
    return audit_call(call, dialogue.messages, batch or [call], RETAIL)


def clean_dialogue() -> Dialogue:
    return (
        Dialogue()
        .user("Hi, my email is daiki@example.com")
        .authenticate()
        .look_up()
        .user("Yes, cancel it. I ordered it by mistake.")
    )


class BuildFactsTest(unittest.TestCase):
    def test_collects_authentication_records_and_evidence(self):
        facts = build_facts(clean_dialogue().messages, RETAIL)
        self.assertEqual(facts.session_user, USER)
        self.assertEqual(facts.records["order_id"][ORDER], {"owner": USER, "status": "pending"})
        self.assertTrue(facts.grounded(ORDER))

    def test_failed_results_are_failures_not_evidence(self):
        dialogue = Dialogue().tool(
            "find_user_id_by_name_zip", {"first_name": "Daiki", "last_name": "Silva", "zip": ""},
            "Error: User not found", error=True,
        )
        facts = build_facts(dialogue.messages, RETAIL)
        self.assertIsNone(facts.session_user)
        self.assertEqual(sum(facts.failures.values()), 1)
        self.assertFalse(facts.grounded("User"))

    def test_assistant_text_is_not_evidence(self):
        facts = build_facts(Dialogue().assistant(f"Your order is {ORDER}.").messages, RETAIL)
        self.assertFalse(facts.grounded(ORDER))

    def test_order_id_without_hash_from_user_is_grounded(self):
        facts = build_facts(Dialogue().user("It's order W8835847").messages, RETAIL)
        self.assertTrue(facts.grounded(ORDER))

    def test_partial_id_is_not_grounded(self):
        facts = build_facts(Dialogue().user("Please cancel order W88358470").messages, RETAIL)
        self.assertFalse(facts.grounded(ORDER))

    def test_id_inside_longer_id_is_not_grounded(self):
        facts = build_facts(
            Dialogue().user("Product 8600330539, paid with gift_card_2652153").messages, RETAIL,
        )
        for value in ("8600", "card_2652153", "2652153"):
            with self.subTest(value=value):
                self.assertFalse(facts.grounded(value))
        self.assertTrue(facts.grounded("8600330539"))
        self.assertTrue(facts.grounded("gift_card_2652153"))

    def test_id_next_to_punctuation_is_grounded(self):
        facts = build_facts(Dialogue().user("Cancel W8835847, please.").messages, RETAIL)
        self.assertTrue(facts.grounded(ORDER))

    def test_reads_results_inside_multi_tool_message(self):
        dialogue = Dialogue()
        calls = [
            dialogue.call("find_user_id_by_email", {"email": "daiki@example.com"}),
            dialogue.call("get_order_details", {"order_id": ORDER}),
        ]
        dialogue.messages += [
            AssistantMessage(role="assistant", tool_calls=calls),
            MultiToolMessage(role="tool", tool_messages=[
                ToolMessage(id=calls[0].id, role="tool", content=USER),
                ToolMessage(id=calls[1].id, role="tool", content=json.dumps(
                    {"order_id": ORDER, "user_id": USER, "status": "pending"})),
            ]),
        ]
        facts = build_facts(dialogue.messages, RETAIL)
        self.assertEqual(facts.session_user, USER)
        self.assertIn(ORDER, facts.records["order_id"])

    def test_first_authenticated_user_is_session_user(self):
        dialogue = clean_dialogue().authenticate(OTHER_USER, OTHER_EMAIL)
        facts = build_facts(dialogue.messages, RETAIL)
        self.assertEqual(facts.session_user, USER)
        self.assertEqual(facts.other_users, [OTHER_USER])

    def test_reauthenticating_same_user_adds_nobody(self):
        facts = build_facts(clean_dialogue().authenticate().messages, RETAIL)
        self.assertEqual(facts.session_user, USER)
        self.assertEqual(facts.other_users, [])


class RuleTest(unittest.TestCase):
    def assertBlocked(self, decision: dict, rule: str):
        self.assertEqual(decision["verdict"], "block", decision["reason"])
        self.assertEqual(decision["rule"], rule)

    def test_clean_cancellation_is_allowed(self):
        decision = audit(clean_dialogue(), cancel())
        self.assertEqual(decision["verdict"], "allow", decision["reason"])
        self.assertIsNone(decision["rule"])

    def test_d1_requires_authentication(self):
        dialogue = Dialogue().user(f"Cancel {ORDER}").look_up()
        self.assertBlocked(audit(dialogue, cancel()), "D1_authenticated")

    def test_d2_blocks_another_users_order(self):
        dialogue = clean_dialogue().look_up(OTHER_ORDER, owner="someone_else_1234")
        self.assertBlocked(audit(dialogue, cancel(OTHER_ORDER)), "D2_owned_by_user")

    def test_d2_blocks_another_users_account(self):
        dialogue = clean_dialogue().look_up(OTHER_ORDER, owner="someone_else_1234")
        call = ToolCall(id="proposal", name="modify_user_address", arguments={
            "user_id": "someone_else_1234", "address1": "1 Main St", "address2": "",
            "city": "NYC", "state": "NY", "country": "USA", "zip": "10001",
        })
        self.assertBlocked(audit(dialogue, call), "D2_owned_by_user")

    def test_d2_blocks_order_of_second_authenticated_user(self):
        # Looking up another customer must not let the agent act on their orders.
        dialogue = (
            clean_dialogue()
            .user(f"Also cancel my friend's order, their email is {OTHER_EMAIL}")
            .authenticate(OTHER_USER, OTHER_EMAIL)
            .look_up(OTHER_ORDER, owner=OTHER_USER)
        )
        self.assertBlocked(audit(dialogue, cancel(OTHER_ORDER)), "D2_owned_by_user")

    def test_d2_blocks_account_of_second_authenticated_user(self):
        dialogue = clean_dialogue().authenticate(OTHER_USER, OTHER_EMAIL)
        call = ToolCall(id="proposal", name="modify_user_address", arguments={
            "user_id": OTHER_USER, "address1": "1 Main St", "address2": "",
            "city": "NYC", "state": "NY", "country": "USA", "zip": "10001",
        })
        self.assertBlocked(audit(dialogue, call), "D2_owned_by_user")

    def test_reauthenticated_same_user_is_allowed(self):
        dialogue = clean_dialogue().authenticate()
        decision = audit(dialogue, cancel())
        self.assertEqual(decision["verdict"], "allow", decision["reason"])

    def test_d3_requires_looking_up_the_order(self):
        dialogue = Dialogue().authenticate().user(f"Cancel order {OTHER_ORDER}, ordered by mistake")
        self.assertBlocked(audit(dialogue, cancel(OTHER_ORDER)), "D3_record_observed")

    def test_d4_blocks_made_up_ids(self):
        self.assertBlocked(audit(clean_dialogue(), cancel("#W0000000")), "D4_grounded_ids")

    def test_d4_checks_every_id_in_a_list(self):
        dialogue = clean_dialogue().tool(
            "get_product_details", {"product_id": "8600330539"},
            {"product_id": "8600330539", "variants": {"7373893106": {}}},
        )
        call = ToolCall(id="proposal", name="modify_pending_order_items", arguments={
            "order_id": ORDER, "item_ids": ["8895454203"], "new_item_ids": ["7373893106"],
            "payment_method_id": "gift_card_2652153",
        })
        decision = audit(dialogue, call)
        self.assertBlocked(decision, "D4_grounded_ids")
        self.assertIn("item_ids 8895454203", decision["reason"])

    def test_d4_blocks_id_that_is_part_of_a_longer_id(self):
        # item_ids have no D3 lookup behind them, so D4 alone must reject a truncated id.
        dialogue = (
            Dialogue()
            .user("Hi, my email is daiki@example.com")
            .authenticate()
            .tool("get_order_details", {"order_id": ORDER}, {
                "order_id": ORDER, "user_id": USER, "status": "pending",
                "items": [{"item_id": "8895454203", "product_id": "8600330539"}],
                "payment_history": [{"payment_method_id": "gift_card_2652153"}],
            })
            .tool(
                "get_product_details", {"product_id": "8600330539"},
                {"product_id": "8600330539", "variants": {"7373893106": {}}},
            )
        )
        call = ToolCall(id="proposal", name="modify_pending_order_items", arguments={
            "order_id": ORDER, "item_ids": ["8895454"], "new_item_ids": ["7373893106"],
            "payment_method_id": "gift_card_2652153",
        })
        decision = audit(dialogue, call)
        self.assertBlocked(decision, "D4_grounded_ids")
        self.assertIn("item_ids 8895454", decision["reason"])

    def test_d5_warns_without_blocking_a_batched_write(self):
        dialogue = clean_dialogue()
        proposal = cancel()
        batch = [dialogue.call("get_user_details", {"user_id": USER}), proposal]
        decision = audit(dialogue, proposal, batch)
        self.assertEqual(decision["verdict"], "allow", decision["reason"])
        self.assertEqual(
            [warning["rule"] for warning in decision["warnings"]], ["D5_one_call_at_a_time"],
        )

    def test_d5_does_not_hide_a_blocking_rule(self):
        dialogue = clean_dialogue()
        proposal = cancel("#W0000000")
        batch = [dialogue.call("get_user_details", {"user_id": USER}), proposal]
        decision = audit(dialogue, proposal, batch)
        self.assertBlocked(decision, "D4_grounded_ids")
        self.assertEqual(len(decision["warnings"]), 1)

    def test_single_call_has_no_warnings(self):
        self.assertEqual(audit(clean_dialogue(), cancel())["warnings"], [])

    def test_d6_blocks_after_repeated_failures(self):
        dialogue = clean_dialogue()
        arguments = cancel().arguments
        dialogue.tool("cancel_pending_order", arguments, "Audit blocked this call: x", error=True)
        self.assertEqual(audit(dialogue, cancel())["verdict"], "allow")
        dialogue.tool("cancel_pending_order", arguments, "Error: Invalid reason", error=True)
        self.assertBlocked(audit(dialogue, cancel()), "D6_repeated_failure")

    def test_wrong_but_owned_order_passes(self):
        # A swapped order that is grounded, observed and owned needs semantic audit.
        dialogue = clean_dialogue().look_up(OTHER_ORDER)
        self.assertEqual(audit(dialogue, cancel(OTHER_ORDER))["verdict"], "allow")


class AuditorHookTest(unittest.TestCase):
    def setUp(self):
        self.dialogue = clean_dialogue()
        self.writes = {"cancel_pending_order"}
        self.orchestrator = SimpleNamespace(trajectory=self.dialogue.messages)
        self.auditor = GeneralDeterministicAuditor(
            self.orchestrator, RETAIL, lambda name: name in self.writes,
        )

    def propose(self, *calls: ToolCall) -> None:
        self.orchestrator.trajectory = [
            *self.dialogue.messages, AssistantMessage(role="assistant", tool_calls=list(calls)),
        ]

    def test_read_and_user_calls_are_not_audited(self):
        read = ToolCall(id="read", name="get_order_details", arguments={"order_id": "#W0000000"})
        by_user = cancel("#W0000000").model_copy(update={"requestor": "user"})
        self.propose(read)
        self.assertIsNone(self.auditor(read))
        self.propose(by_user)
        self.assertIsNone(self.auditor(by_user))
        self.assertEqual(self.auditor.decisions, [])

    def test_uses_proposing_message_as_batch_and_records_decisions(self):
        proposal = cancel()
        self.propose(proposal)
        self.assertIsNone(self.auditor(proposal))
        read = ToolCall(id="read", name="get_user_details", arguments={"user_id": USER})
        self.propose(read, proposal)
        self.assertIsNone(self.auditor(proposal))
        self.assertEqual(
            [[w["rule"] for w in d["warnings"]] for d in self.auditor.decisions],
            [[], ["D5_one_call_at_a_time"]],
        )


class RetailEnvironmentTest(unittest.TestCase):
    def test_write_tools_come_from_tau2_annotations(self):
        from tau2.runner import build_environment

        environment = build_environment("retail")
        is_write = write_tool_checker(environment)
        writes = {tool.name for tool in environment.get_tools() if is_write(tool.name)}
        self.assertEqual(writes, {
            "cancel_pending_order", "exchange_delivered_order_items",
            "modify_pending_order_address", "modify_pending_order_items",
            "modify_pending_order_payment", "modify_user_address",
            "return_delivered_order_items",
        })
        self.assertFalse(is_write("no_such_tool"))


if __name__ == "__main__":
    unittest.main()
