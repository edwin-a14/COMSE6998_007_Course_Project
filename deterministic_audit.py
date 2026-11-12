"""Tool-agnostic deterministic audit for τ²-bench write tool calls.

Each time the assistant proposes a write tool call, the auditor rebuilds a fact
table from the conversation before that proposal and checks policy rules that
the domain tools do not enforce themselves. Checks the tools already make, such
as order status or a valid cancellation reason, are left to the tools.
"""

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType

MAX_FAILURES = 2


@dataclass(frozen=True)
class DomainConfig:
    """Domain-specific inputs to the otherwise tool-agnostic rules."""

    auth_tools: frozenset[str]
    """Read tools whose successful result is the authenticated user id."""
    record_lookups: dict[str, str]
    """Write argument that names a record -> read tool that returns the record."""
    owner_field: str = "user_id"
    """Field of a looked-up record that names its owner."""
    user_arg: str = "user_id"
    """Write argument that names a user account directly."""


DOMAIN_CONFIGS = {
    "retail": DomainConfig(
        auth_tools=frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"}),
        record_lookups={"order_id": "get_order_details"},
    ),
}


@dataclass
class Facts:
    """What the conversation establishes, from tool results and user messages."""

    authenticated_users: set[str] = field(default_factory=set)
    records: dict[str, dict[str, dict]] = field(default_factory=dict)
    """Write argument -> record id -> {"owner", "status"}."""
    failures: Counter = field(default_factory=Counter)
    """(tool name, canonical arguments) -> failed or blocked attempts."""
    evidence: list[str] = field(default_factory=list)
    """Successful tool results and user messages. Assistant text is excluded."""

    def grounded(self, value: str) -> bool:
        """Whether an identifier appears in a tool result or a user message."""
        value = value.strip()
        forms = {value, value.lstrip("#")} - {""}
        return any(form in text for form in forms for text in self.evidence)

    def snapshot(self) -> dict:
        return {
            "authenticated_users": sorted(self.authenticated_users),
            "records": self.records,
            "failures": [
                {"tool": name, "arguments": json.loads(arguments), "count": count}
                for (name, arguments), count in self.failures.items()
            ],
        }


def call_key(call: ToolCall) -> tuple[str, str]:
    return call.name, json.dumps(call.arguments, sort_keys=True)


def build_facts(messages: Iterable[Message], config: DomainConfig) -> Facts:
    """Build the fact table from the messages before a proposed call.

    Failed and audit-blocked results count as failures and are not evidence.
    An identifier the assistant only states itself is not evidence either.
    """
    facts = Facts()
    calls: dict[str, ToolCall] = {}
    for message in _flatten(messages):
        if isinstance(message, (AssistantMessage, UserMessage)):
            calls.update((call.id, call) for call in message.tool_calls or [])
            if isinstance(message, UserMessage) and message.content:
                facts.evidence.append(message.content)
        elif isinstance(message, ToolMessage) and message.id in calls:
            call = calls[message.id]
            if message.error:
                facts.failures[call_key(call)] += 1
            elif message.content:
                facts.evidence.append(message.content)
                _record_result(facts, config, call, _parse(message.content))
    return facts


def _flatten(messages: Iterable[Message]) -> Iterable[Message]:
    for message in messages:
        if isinstance(message, MultiToolMessage):
            yield from message.tool_messages
        else:
            yield message


def _parse(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _record_result(facts: Facts, config: DomainConfig, call: ToolCall, result: Any) -> None:
    if call.name in config.auth_tools and isinstance(result, str):
        facts.authenticated_users.add(result)
    for arg, lookup in config.record_lookups.items():
        if call.name != lookup or not isinstance(result, dict):
            continue
        record_id = result.get(arg) or call.arguments.get(arg)
        if isinstance(record_id, str):
            facts.records.setdefault(arg, {})[record_id] = {
                "owner": result.get(config.owner_field),
                "status": result.get("status"),
            }


# Each rule returns feedback for the agent when the call must be blocked.
Rule = Callable[[ToolCall, Facts, list[ToolCall], DomainConfig], str | None]


def repeated_failure(call: ToolCall, facts: Facts, batch: list[ToolCall], config: DomainConfig) -> str | None:
    count = facts.failures[call_key(call)]
    if count < MAX_FAILURES:
        return None
    return (
        f"This exact {call.name} call has already failed or been blocked {count} times, "
        "so it was not executed again. Do not retry it unchanged. Re-check the details "
        "with the user. If the request cannot be completed, tell the user, or transfer "
        "them as the policy allows."
    )


def one_call_at_a_time(call: ToolCall, facts: Facts, batch: list[ToolCall], config: DomainConfig) -> str | None:
    if len(batch) <= 1:
        return None
    return (
        f"{call.name} was proposed together with {len(batch) - 1} other tool call(s), "
        "but the policy allows one tool call at a time. This call was not executed. "
        "Make this call on its own."
    )


def authenticated(call: ToolCall, facts: Facts, batch: list[ToolCall], config: DomainConfig) -> str | None:
    if facts.authenticated_users:
        return None
    tools = " or ".join(sorted(config.auth_tools))
    return (
        "No user has been authenticated in this conversation, so this call was not "
        f"executed. Authenticate the user with {tools} before changing the database."
    )


def grounded_ids(call: ToolCall, facts: Facts, batch: list[ToolCall], config: DomainConfig) -> str | None:
    for name, value in call.arguments.items():
        if not _is_id_arg(name):
            continue
        values = value if isinstance(value, list) else [value]
        missing = [str(item) for item in values if not facts.grounded(str(item))]
        if missing:
            return (
                f"{name} {', '.join(missing)} did not appear in any tool result or user "
                "message, so this call was not executed. Do not guess identifiers: look "
                "them up with a tool or ask the user."
            )
    return None


def record_observed(call: ToolCall, facts: Facts, batch: list[ToolCall], config: DomainConfig) -> str | None:
    for arg, lookup in config.record_lookups.items():
        value = call.arguments.get(arg)
        if isinstance(value, str) and value not in facts.records.get(arg, {}):
            return (
                f"{arg} {value} has not been retrieved with {lookup} in this "
                f"conversation, so this call was not executed. Call {lookup} first "
                "to confirm its current details and status."
            )
    return None


def owned_by_user(call: ToolCall, facts: Facts, batch: list[ToolCall], config: DomainConfig) -> str | None:
    for arg in config.record_lookups:
        value = call.arguments.get(arg)
        record = facts.records.get(arg, {}).get(value) if isinstance(value, str) else None
        owner = record["owner"] if record else None
        if owner and owner not in facts.authenticated_users:
            return (
                f"{arg} {value} belongs to a different user than the authenticated "
                "one, so this call was not executed. Only act on the authenticated "
                "user's records."
            )
    user = call.arguments.get(config.user_arg)
    if user is not None and user not in facts.authenticated_users:
        return (
            f"{config.user_arg} {user} is not the authenticated user, so this call was "
            "not executed. Only act on the authenticated user's account."
        )
    return None


def _is_id_arg(name: str) -> bool:
    return name == "id" or name.endswith(("_id", "_ids"))


# Evaluated in order; the first rule that fires is reported.
RULES: tuple[tuple[str, Rule], ...] = (
    ("D6_repeated_failure", repeated_failure),
    ("D5_one_call_at_a_time", one_call_at_a_time),
    ("D1_authenticated", authenticated),
    ("D4_grounded_ids", grounded_ids),
    ("D3_record_observed", record_observed),
    ("D2_owned_by_user", owned_by_user),
)


def audit_call(
    call: ToolCall, history: list[Message], batch: list[ToolCall], config: DomainConfig,
) -> dict:
    """Audit one write call.

    `history` holds the messages before the assistant message that proposed the
    call, and `batch` holds every call in that message.
    """
    facts = build_facts(history, config)
    rule_id, reason = next(
        ((rule_id, reason) for rule_id, rule in RULES
         if (reason := rule(call, facts, batch, config)) is not None),
        (None, None),
    )
    return {
        "call": call.model_dump(mode="json"),
        "verdict": "allow" if reason is None else "block",
        "rule": rule_id,
        "reason": reason,
        "facts": facts.snapshot(),
    }


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
