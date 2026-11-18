"""τ²-bench hook for the general deterministic rules D1–D6.

The rules live in `audit_core/auditors/rules.py` and work on the shared audit
interface, so the same code also runs on AFTraj. This module converts τ²
messages with `tau2_adapter` and keeps the τ²-facing API: `build_facts` and
`audit_call` take τ² messages, and `GeneralDeterministicAuditor` is the
`before_tool_call` hook.

Each time the assistant proposes a write tool call, the auditor rebuilds a fact
table from the conversation before that proposal and checks policy rules that
the domain tools do not enforce themselves. Checks the tools already make, such
as order status or a valid cancellation reason, are left to the tools.
"""

from collections.abc import Callable, Iterable
from typing import Any

from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType

from audit_core.auditors import rules
from audit_core.auditors.rules import (  # noqa: F401  re-exported for existing callers
    DOMAIN_CONFIGS,
    ID_TOKEN,
    MAX_FAILURES,
    RULES,
    WARNING_RULES,
    DomainConfig,
    Facts,
)
from tau2_adapter import to_tool_call, to_turns


def build_facts(messages: Iterable[Message], config: DomainConfig) -> Facts:
    """Build the fact table from the τ² messages before a proposed call."""
    return rules.build_facts(to_turns(messages), config, author="assistant")


def audit_call(
    call: ToolCall, history: list[Message], batch: list[ToolCall], config: DomainConfig,
) -> dict:
    """Audit one write call.

    `history` holds the messages before the assistant message that proposed the
    call, and `batch` holds every call in that message.
    """
    facts = build_facts(history, config)
    check = rules.check_write_call(
        to_tool_call(call), facts, [to_tool_call(item) for item in batch], config,
    )
    return {"call": call.model_dump(mode="json"), **check, "facts": facts.snapshot()}


def write_tool_checker(environment: Environment) -> Callable[[str], bool]:
    """Classify tools with the WRITE annotation of the τ²-bench environment."""

    def is_write(name: str) -> bool:
        try:
            return environment.tools.tool_type(name) == ToolType.WRITE
        except (AttributeError, KeyError):
            return False

    return is_write


class GeneralDeterministicAuditor:
    """`before_tool_call` hook that audits the assistant's write tool calls."""

    def __init__(self, orchestrator: Any, config: DomainConfig, is_write: Callable[[str], bool]):
        self.orchestrator = orchestrator
        self.config = config
        self.is_write = is_write
        self.decisions: list[dict] = []

    @classmethod
    def for_orchestrator(cls, orchestrator: Any, domain: str) -> "GeneralDeterministicAuditor":
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"No deterministic audit config for domain {domain!r}")
        return cls(orchestrator, DOMAIN_CONFIGS[domain], write_tool_checker(orchestrator.environment))

    def __call__(self, call: ToolCall) -> str | None:
        if call.requestor != "assistant" or not self.is_write(call.name):
            return None
        # During dispatch the last trajectory message is the proposing assistant message.
        *history, current = self.orchestrator.trajectory
        batch = list(current.tool_calls or []) if isinstance(current, AssistantMessage) else []
        if not any(item is call or item.id == call.id for item in batch):
            batch = [call]
        decision = audit_call(call, history, batch, self.config)
        self.decisions.append(decision)
        return decision["reason"]
