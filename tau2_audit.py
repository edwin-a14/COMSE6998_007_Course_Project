"""Pre-dispatch hook for the pinned τ²-bench text orchestrator."""

from collections.abc import Callable

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.orchestrator.orchestrator import Orchestrator

from semantic_audit import cancellation_context, other_pending_order


class AuditOrchestrator(Orchestrator):
    before_tool_call: Callable[[ToolCall], str | None] | None = None
    inject_cancellation_fault = False
    inject_wrong_order = False

    def initialize(self) -> None:
        self._rejected_calls: dict[str, dict] = {}
        self._recovery: dict | None = None
        super().initialize()

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
        audit = self.before_tool_call
        if audit is None:
            return super()._execute_tool_calls(tool_calls)

        results = []
        blocked = False
        for tool_call in tool_calls:
            reason = (
                "An earlier call in this batch failed or was blocked. Replan before retrying."
                if blocked else None
            )
            retry = False
            injected = False
            if (
                (self.inject_cancellation_fault or self.inject_wrong_order)
                and not blocked
                and tool_call.requestor == "assistant"
                and tool_call.name == "cancel_pending_order"
            ):
                if self._recovery is None and retail_cancellation_block_reason(tool_call) is None:
                    injected = self._inject_fault(tool_call)
                elif self._recovery is not None and self._recovery["status"] == "awaiting_retry":
                    retry = True
                    self._recovery["retry_call"] = tool_call.model_dump(mode="json")
                    original_order = self._recovery["original_call"]["arguments"]["order_id"]
                    if tool_call.arguments.get("order_id") != original_order:
                        reason = "The retry must target the same order as the original call."

            if reason is None:
                reason = audit(tool_call)
            if injected and reason is None:
                self._recovery["status"] = "fault_allowed"

            if reason is None:
                (response,) = super()._execute_tool_calls([tool_call])
                results.append(response)
                if retry:
                    self._recovery["status"] = "retry_failed" if response.error else "recovered"
                    blocked = response.error
                continue

            blocked = True
            if retry:
                self._recovery["status"] = "retry_failed"
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

    def _inject_fault(self, call: ToolCall) -> bool:
        arguments = dict(call.arguments)
        if self.inject_wrong_order:
            _, observed = cancellation_context(self.trajectory[:-1])
            other_order = other_pending_order(arguments["order_id"], observed)
            if other_order is None:
                return False
            arguments["order_id"] = other_order
        else:
            arguments["reason"] = "__injected_fault__"
        self._recovery = {
            "status": "awaiting_retry",
            "original_call": call.model_dump(mode="json"),
        }
        # Keep the proposal seen by the agent consistent with the audit feedback.
        call.arguments = arguments
        return True

    def _check_termination(self) -> None:
        super()._check_termination()
        if self._recovery is not None and self._recovery["status"] == "retry_failed":
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS

    def _finalize(self) -> SimulationRun:
        run = super()._finalize()
        if self.inject_cancellation_fault or self.inject_wrong_order:
            recovery = self._recovery or {"status": "not_triggered"}
            if recovery["status"] == "awaiting_retry":
                recovery["status"] = "no_retry"
            run.info = {**(run.info or {}), "cancellation_recovery": recovery}
        if not self._rejected_calls:
            return run

        # The native evaluator re-executes mutating calls from this transcript.
        # Keep rejected proposals in diagnostics and out of the replay input.
        messages = []
        for message in run.get_messages():
            if isinstance(message, ToolMessage) and message.id in self._rejected_calls:
                continue
            if (
                isinstance(message, (AssistantMessage, UserMessage))
                and message.tool_calls
            ):
                executed = [
                    call for call in message.tool_calls
                    if call.id not in self._rejected_calls
                ]
                if not executed and not message.content:
                    continue
                message.tool_calls = executed or None
            message.turn_idx = len(messages)
            messages.append(message)

        run.messages = messages
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
