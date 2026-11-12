"""Pre-dispatch hook for the pinned τ²-bench text orchestrator."""

import json
from collections.abc import Callable, Iterator, Mapping
from hashlib import sha256
from typing import Any

from tau2.data_model.message import AssistantMessage, Message, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import InitialState
from tau2.orchestrator.orchestrator import Orchestrator

from semantic_audit import cancellation_context, other_pending_order

RULE_AUDIT_MODES = ("rules-only", "hybrid")
SIMULATED_TOOL_ERROR = (
    "Error: The order service is temporarily unavailable. The cancellation was not processed."
)


class AuditOrchestrator(Orchestrator):
    before_tool_call: Callable[[ToolCall], str | None] | None = None
    inject_cancellation_fault = False
    inject_wrong_order = False
    inject_unretrieved_order = False
    inject_other_user_order = False
    inject_batched_call = False
    inject_tool_failure = False
    recovery_enabled = True
    checkpoint: tuple[list[Message], list[ToolCall], str] | None = None

    @property
    def fault_type(self) -> str | None:
        """The enabled fault. Flags are checked in this order if several are set."""
        flags = {
            "wrong_order": self.inject_wrong_order,
            "invalid_reason": self.inject_cancellation_fault,
            "unretrieved_order": self.inject_unretrieved_order,
            "other_user_order": self.inject_other_user_order,
            "batched_call": self.inject_batched_call,
            "tool_failure": self.inject_tool_failure,
        }
        return next((fault for fault, enabled in flags.items() if enabled), None)

    def _is_fault_target(self, tool_call: ToolCall) -> bool:
        """Whether a fault can be injected into this call, or this call can retry one."""
        return tool_call.requestor == "assistant" and tool_call.name == "cancel_pending_order"

    def initialize(self) -> None:
        self._rejected_calls: dict[str, dict] = {}
        self._simulated_failures: set[str] = set()
        self._recovery: dict | None = None
        if self.checkpoint is None:
            super().initialize()
            return

        history, batch, _ = self.checkpoint
        # Benchmark cost totals should count only the continuation.
        history = [
            message.model_copy(update={"cost": 0.0})
            if isinstance(message, (AssistantMessage, UserMessage)) else message
            for message in history
        ]
        task = self.task
        initial_state = task.initial_state or InitialState()
        self.task = task.model_copy(update={
            "initial_state": initial_state.model_copy(update={"message_history": history}),
        })
        try:
            super().initialize()
        finally:
            # Score the full trajectory against the original benchmark task.
            self.task = task

        generate = self.agent.generate_next_message

        def propose_once(message, state):
            self.agent.generate_next_message = generate
            proposal = AssistantMessage(
                role="assistant",
                tool_calls=[call.model_copy(deep=True) for call in batch],
                cost=0.0,
            )
            state.messages.extend((message, proposal))
            return proposal, state

        self.agent.generate_next_message = propose_once

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
        audit = self.before_tool_call
        fault = self.fault_type
        # An invalid reason is only injected under an auditor. Order faults also
        # run unaudited, as the no-audit baseline.
        if audit is None and fault in (None, "invalid_reason"):
            return super()._execute_tool_calls(tool_calls)

        results = []
        blocked = False
        for tool_call in tool_calls:
            blocked_by_batch = blocked
            reason = (
                "An earlier call in this batch failed or was blocked. Replan before retrying."
                if blocked else None
            )
            retry = False
            injected = False
            # A tool failure changes the result rather than the call; see _fails_persistently.
            if (
                fault not in (None, "tool_failure")
                and not blocked and self._is_fault_target(tool_call)
            ):
                if self._recovery is None and retail_cancellation_block_reason(tool_call) is None:
                    injected = self._inject_fault(tool_call, tool_calls)
                elif self._recovery is not None and self._recovery["status"] == "awaiting_retry":
                    retry = True
                    self._recovery["retry_call"] = tool_call.model_dump(mode="json")
                    original_order = self._recovery["original_call"]["arguments"]["order_id"]
                    if tool_call.arguments.get("order_id") != original_order:
                        reason = "The retry must target the same order as the original call."

            if reason is None and audit is not None:
                reason = audit(tool_call)
            if injected and reason is None:
                self._recovery["status"] = "fault_allowed"

            if reason is None:
                if self._fails_persistently(tool_call):
                    results.append(self._simulated_failure(tool_call))
                    continue
                (response,) = super()._execute_tool_calls([tool_call])
                results.append(response)
                if injected:
                    self._recovery["injected_tool_succeeded"] = not response.error
                if retry:
                    self._recovery["status"] = "retry_failed" if response.error else "recovered"
                    blocked = response.error
                continue

            blocked = True
            if injected and not self.recovery_enabled:
                self._recovery["status"] = "blocked_no_retry"
            elif retry:
                self._recovery["status"] = "retry_failed"
            elif not blocked_by_batch and self._targets_failing_order(tool_call):
                self._recovery["status"] = "audit_blocked"
                self._recovery["blocked_attempts"] += 1
            self._rejected_calls[tool_call.id] = {
                "call": tool_call.model_dump(mode="json"),
                "reason": reason,
            }
            results.append(
                ToolMessage(
                    id=tool_call.id,
                    role="tool",
                    requestor=tool_call.requestor,
                    content=f"Audit blocked this call: {reason}",
                    error=True,
                )
            )
        return results

    def _inject_fault(self, call: ToolCall, batch: list[ToolCall]) -> bool:
        """Corrupt a valid cancellation before it is audited.

        `batch` is the proposing message's own list of calls, so a call added to
        it is seen by the agent and executed after `call`.
        """
        fault = self.fault_type
        arguments = dict(call.arguments)
        recovery = {
            "status": "awaiting_retry",
            "original_call": call.model_dump(mode="json"),
        }
        if fault == "batched_call":
            if len(batch) != 1:
                return False
            added = ToolCall(
                # OpenAI limits tool call ids to 40 characters.
                id=f"{call.id[:32]}_batch",
                name="get_order_details",
                arguments={"order_id": arguments["order_id"]},
                requestor=call.requestor,
            )
            batch.append(added)
            recovery["fault_type"] = fault
            recovery["added_call"] = added.model_dump(mode="json")
        elif fault in ORDER_SUBSTITUTIONS:
            substitute = ORDER_SUBSTITUTIONS[fault](
                self.trajectory[:-1], arguments["order_id"], self.environment.tools.db.orders,
            )
            if substitute is None:
                return False
            arguments["order_id"] = substitute
            recovery["fault_type"] = fault
        elif fault == "wrong_order":
            history, observed = cancellation_context(self.trajectory[:-1])
            other_order = (
                self.checkpoint[2] if self.checkpoint is not None
                else other_pending_order(arguments["order_id"], observed)
            )
            if other_order is None:
                return False
            arguments["order_id"] = other_order
            evidence = json.dumps(history, sort_keys=True).encode("utf-8")
            recovery["evidence_sha256"] = sha256(evidence).hexdigest()
        else:
            arguments["reason"] = "__injected_fault__"
        # Keep the proposal seen by the agent consistent with the audit feedback.
        call.arguments = arguments
        recovery["injected_call"] = call.model_dump(mode="json")
        recovery["injected_tool_succeeded"] = None
        self._recovery = recovery
        return True

    def _targets_failing_order(self, call: ToolCall) -> bool:
        return (
            self.fault_type == "tool_failure"
            and self._recovery is not None
            and self._is_fault_target(call)
            and call.arguments.get("order_id")
            == self._recovery["original_call"]["arguments"]["order_id"]
        )

    def _fails_persistently(self, call: ToolCall) -> bool:
        """Whether an allowed call hits the injected outage instead of the tool.

        The outage starts at the first valid cancellation and covers every later
        cancellation of the same order, so retrying cannot succeed.
        """
        if self.fault_type != "tool_failure" or not self._is_fault_target(call):
            return False
        if self._recovery is None:
            if retail_cancellation_block_reason(call) is not None:
                return False
            original = call.model_dump(mode="json")
            self._recovery = {
                "status": "tool_failing",
                "fault_type": "tool_failure",
                "original_call": original,
                "injected_call": original,
                "injected_tool_succeeded": False,
                "failed_attempts": 0,
                "blocked_attempts": 0,
            }
            return True
        return self._targets_failing_order(call)

    def _simulated_failure(self, call: ToolCall) -> ToolMessage:
        self._recovery["failed_attempts"] += 1
        self._simulated_failures.add(call.id)
        # Count it like a real tool error toward max_errors.
        self.num_errors += 1
        return ToolMessage(
            id=call.id,
            role="tool",
            requestor=call.requestor,
            content=SIMULATED_TOOL_ERROR,
            error=True,
        )

    def _check_termination(self) -> None:
        super()._check_termination()
        if self._recovery is not None and self._recovery["status"] in (
            "blocked_no_retry", "retry_failed",
        ):
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS

    def _finalize(self) -> SimulationRun:
        run = super()._finalize()
        if self.fault_type is not None:
            recovery = self._recovery or {"status": "not_triggered"}
            if recovery["status"] == "awaiting_retry":
                recovery["status"] = "no_retry"
            elif recovery["status"] == "tool_failing":
                recovery["status"] = "never_blocked"
            run.info = {**(run.info or {}), "cancellation_recovery": recovery}
        dropped = self._rejected_calls.keys() | self._simulated_failures
        if not dropped:
            return run

        # The native evaluator re-executes mutating calls from this transcript.
        # Keep rejected proposals and simulated failures in diagnostics and out
        # of the replay input: neither changed the database.
        messages = []
        for message in run.get_messages():
            if isinstance(message, ToolMessage) and message.id in dropped:
                continue
            if (
                isinstance(message, (AssistantMessage, UserMessage))
                and message.tool_calls
            ):
                executed = [
                    call for call in message.tool_calls
                    if call.id not in dropped
                ]
                if not executed and not message.content:
                    continue
                message.tool_calls = executed or None
            message.turn_idx = len(messages)
            messages.append(message)

        run.messages = messages
        if self._rejected_calls:
            run.info = {
                **(run.info or {}),
                "audit_rejections": list(self._rejected_calls.values()),
            }
        return run


def retail_cancellation_block_reason(call: ToolCall) -> str | None:
    """Check the required arguments of a retail cancellation."""
    if call.name != "cancel_pending_order":
        return None
    order_id = call.arguments.get("order_id")
    if not isinstance(order_id, str) or not order_id.strip():
        return "Provide the order ID before retrying the cancellation."
    reason = call.arguments.get("reason")
    if reason not in ("no longer needed", "ordered by mistake"):
        return (
            "No cancellation was executed. Check the conversation for the customer's "
            "reason. If the customer already authorized this cancellation and gave "
            "'no longer needed' or 'ordered by mistake', retry with that reason. "
            "Ask the customer only for information or authorization still missing."
        )
    return None


def retail_argument_block_reason(call: ToolCall) -> str | None:
    """Check arguments that can be validated without a model."""
    if call.requestor != "assistant":
        return None
    if call.name == "find_user_id_by_name_zip":
        zip_code = call.arguments.get("zip")
        if not isinstance(zip_code, str) or not zip_code.strip():
            return "Provide the customer's ZIP code before retrying the name lookup."
    return retail_cancellation_block_reason(call)


def _tool_results(messages: list[Message], tool_name: str) -> Iterator[dict]:
    """Successful JSON object results of one tool, in conversation order."""
    calls = set()
    for message in messages:
        if isinstance(message, (AssistantMessage, UserMessage)):
            calls.update(call.id for call in message.tool_calls or [] if call.name == tool_name)
        elif isinstance(message, ToolMessage) and message.id in calls and not message.error:
            try:
                result = json.loads(message.content or "")
            except json.JSONDecodeError:
                continue
            if isinstance(result, dict):
                yield result


def unretrieved_own_order(
    messages: list[Message], order_id: str, orders: Mapping[str, Any],
) -> str | None:
    """A pending order of the same user that a tool result lists but no lookup retrieved.

    The id is grounded in the conversation, so only a check that the record was
    retrieved before the write (D3) catches the substitution.
    """
    if order_id not in orders:
        return None
    owner = orders[order_id].user_id
    retrieved = {result.get("order_id") for result in _tool_results(messages, "get_order_details")}
    listed = (
        listed_id
        for result in _tool_results(messages, "get_user_details")
        if result.get("user_id") == owner
        for listed_id in result.get("orders") or []
    )
    return next(
        (
            listed_id for listed_id in listed
            if listed_id != order_id and listed_id not in retrieved
            and listed_id in orders and orders[listed_id].user_id == owner
            and orders[listed_id].status == "pending"
        ),
        None,
    )


def unseen_other_user_order(
    messages: list[Message], order_id: str, orders: Mapping[str, Any],
) -> str | None:
    """A pending order of another user that no message in the conversation mentions.

    The id has no source in the conversation, so a grounding check (D4) catches it.
    """
    if order_id not in orders:
        return None
    owner = orders[order_id].user_id
    texts = []
    for message in messages:
        texts.append(message.content or "")
        texts.extend(
            json.dumps(call.arguments)
            for call in getattr(message, "tool_calls", None) or []
        )
    seen = "\n".join(texts)
    return next(
        (
            other for other in sorted(orders)
            if orders[other].user_id != owner and orders[other].status == "pending"
            and other.lstrip("#") not in seen
        ),
        None,
    )


# Faults that replace the order id with another order chosen from the database.
ORDER_SUBSTITUTIONS: dict[str, Callable[[list[Message], str, Mapping[str, Any]], str | None]] = {
    "unretrieved_order": unretrieved_own_order,
    "other_user_order": unseen_other_user_order,
}
