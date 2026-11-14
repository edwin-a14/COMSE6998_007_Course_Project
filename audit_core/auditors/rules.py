"""Deterministic rules D1–D6 over write tool calls, on the shared interface.

Before each proposal that calls a write tool (`ToolSpec.side_effect` is True),
the auditor rebuilds a fact table from the history and checks rules the tools
do not enforce themselves.

Two kinds of rules:
- D4, D5, D6 need nothing domain-specific and run everywhere.
- D1, D2, D3 need to know which tools authenticate a user and which tools look
  up a record (`DomainConfig`). They are skipped when the config does not name
  any, as in `GENERIC_CONFIG`.

Evidence for D4 comes from tool results, user turns, and turns of agents other
than the one proposing. The proposing agent's own text is not evidence: an
identifier it only states itself may be made up. In τ²-bench there is one
agent ("assistant"), so this is the original rule.

Writes may only touch the session user's records: the first user an
authentication tool returns. Authenticating someone else later grants nothing,
since the agent may help only one user per conversation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from audit_core.interface import AuditContext, Auditor, ToolCall, ToolSpec, Turn, Verdict

MAX_FAILURES = 2

# Identifiers are runs of these characters; "#", spaces and punctuation separate
# them. This holds for every id in the τ²-bench retail, airline, telecom and
# banking databases once a leading "#" is removed.
ID_TOKEN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class DomainConfig:
    """Domain-specific inputs to the otherwise tool-agnostic rules."""

    auth_tools: frozenset[str] = frozenset()
    """Read tools whose successful result is the authenticated user id.
    Empty: D1 and D2's account check are skipped."""
    record_lookups: dict[str, str] = field(default_factory=dict)
    """Write argument that names a record -> read tool that returns the record.
    Empty: D2's record check and D3 are skipped."""
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

GENERIC_CONFIG = DomainConfig()
"""No authentication or lookup tools: only D4, D5 and D6 apply."""


@dataclass
class Facts:
    """What the history establishes, from tool results and other parties' turns."""

    session_user: str | None = None
    """First user id an authentication tool returned. Fixed for the conversation."""
    other_users: list[str] = field(default_factory=list)
    """Other user ids authenticated later. Recorded only; they grant nothing."""
    records: dict[str, dict[str, dict]] = field(default_factory=dict)
    """Write argument -> record id -> {"owner", "status"}."""
    failures: Counter = field(default_factory=Counter)
    """(tool name, canonical arguments) -> failed or blocked attempts."""
    tokens: set[str] = field(default_factory=set)
    """Identifier tokens of successful tool results, user turns and other
    agents' turns. The proposing agent's text is excluded."""

    def grounded(self, value: str) -> bool:
        """Whether an identifier appears whole in the evidence.

        A leading "#" is optional on either side, so a customer may omit it. Part
        of a longer identifier does not count: W8835847 is not grounded by
        W88358470.
        """
        core = value.strip().lstrip("#")
        return bool(core) and core in self.tokens

    def snapshot(self) -> dict:
        return {
            "session_user": self.session_user,
            "other_users": self.other_users,
            "records": self.records,
            "failures": [
                {"tool": name, "arguments": json.loads(arguments), "count": count}
                for (name, arguments), count in self.failures.items()
            ],
        }


def build_facts(history: Iterable[Turn], config: DomainConfig, author: str = "assistant") -> Facts:
    """Build the fact table from the turns before a proposal by `author`.

    Failed and audit-blocked results count as failures and are not evidence.
    A tool result that names a call id the history never made is ignored; one
    with no call id at all (the benchmark could not link it) is still evidence.
    """
    facts = Facts()
    calls: dict[str, ToolCall] = {}
    for turn in history:
        calls.update((call.id, call) for call in turn.tool_calls if call.id is not None)
        if turn.kind == "tool":
            call = calls.get(turn.tool_call_id) if turn.tool_call_id is not None else None
            if turn.tool_call_id is not None and call is None:
                continue
            if turn.error:
                if call is not None:
                    facts.failures[call.key()] += 1
            elif turn.content:
                facts.tokens.update(ID_TOKEN.findall(turn.content))
                if call is not None:
                    _record_result(facts, config, call, _parse(turn.content))
        elif turn.kind == "user" or (turn.kind == "agent" and turn.role != author):
            if turn.content:
                facts.tokens.update(ID_TOKEN.findall(turn.content))
    return facts


def _parse(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _record_result(facts: Facts, config: DomainConfig, call: ToolCall, result: Any) -> None:
    if call.name in config.auth_tools and isinstance(result, str):
        if facts.session_user is None:
            facts.session_user = result
        elif result != facts.session_user and result not in facts.other_users:
            facts.other_users.append(result)
    for arg, lookup in config.record_lookups.items():
        if call.name != lookup or not isinstance(result, dict):
            continue
        record_id = result.get(arg) or call.kwargs.get(arg)
        if isinstance(record_id, str):
            facts.records.setdefault(arg, {})[record_id] = {
                "owner": result.get(config.owner_field),
                "status": result.get("status"),
            }


# Each rule returns feedback for the agent when the call must be blocked.
Rule = Callable[[ToolCall, Facts, Sequence[ToolCall], DomainConfig], "str | None"]


def repeated_failure(call: ToolCall, facts: Facts, batch: Sequence[ToolCall], config: DomainConfig) -> str | None:
    count = facts.failures[call.key()]
    if count < MAX_FAILURES:
        return None
    return (
        f"This exact {call.name} call has already failed or been blocked {count} times, "
        "so it was not executed again. Do not retry it unchanged. Re-check the details "
        "with the user. If the request cannot be completed, tell the user, or transfer "
        "them as the policy allows."
    )


def one_call_at_a_time(call: ToolCall, facts: Facts, batch: Sequence[ToolCall], config: DomainConfig) -> str | None:
    if len(batch) <= 1:
        return None
    return (
        f"{call.name} was proposed together with {len(batch) - 1} other tool call(s), "
        "but the policy allows one tool call at a time."
    )


def authenticated(call: ToolCall, facts: Facts, batch: Sequence[ToolCall], config: DomainConfig) -> str | None:
    if not config.auth_tools or facts.session_user is not None:
        return None
    tools = " or ".join(sorted(config.auth_tools))
    return (
        "No user has been authenticated in this conversation, so this call was not "
        f"executed. Authenticate the user with {tools} before changing the database."
    )


def grounded_ids(call: ToolCall, facts: Facts, batch: Sequence[ToolCall], config: DomainConfig) -> str | None:
    for name, value in call.kwargs.items():
        if not _is_id_arg(name):
            continue
        values = value if isinstance(value, list) else [value]
        missing = [str(item) for item in values if not facts.grounded(str(item))]
        if missing:
            return (
                f"{name} {', '.join(missing)} did not appear as a complete identifier in "
                "any tool result or user message, so this call was not executed. Do not "
                "guess identifiers: look them up with a tool or ask the user."
            )
    return None


def record_observed(call: ToolCall, facts: Facts, batch: Sequence[ToolCall], config: DomainConfig) -> str | None:
    for arg, lookup in config.record_lookups.items():
        value = call.kwargs.get(arg)
        if isinstance(value, str) and value not in facts.records.get(arg, {}):
            return (
                f"{arg} {value} has not been retrieved with {lookup} in this "
                f"conversation, so this call was not executed. Call {lookup} first "
                "to confirm its current details and status."
            )
    return None


def owned_by_user(call: ToolCall, facts: Facts, batch: Sequence[ToolCall], config: DomainConfig) -> str | None:
    for arg in config.record_lookups:
        value = call.kwargs.get(arg)
        record = facts.records.get(arg, {}).get(value) if isinstance(value, str) else None
        owner = record["owner"] if record else None
        if owner and owner != facts.session_user:
            return (
                f"{arg} {value} belongs to a different user than the one authenticated "
                "first in this conversation, so this call was not executed. Only one "
                "user can be helped per conversation: act only on that user's records."
            )
    if not config.auth_tools:
        return None
    user = call.kwargs.get(config.user_arg)
    if user is not None and user != facts.session_user:
        return (
            f"{config.user_arg} {user} is not the user authenticated first in this "
            "conversation, so this call was not executed. Only one user can be helped "
            "per conversation: act only on that user's account."
        )
    return None


def _is_id_arg(name: str) -> bool:
    return name == "id" or name.endswith(("_id", "_ids"))


# Evaluated in order; the first rule that fires blocks the call and is reported.
RULES: tuple[tuple[str, Rule], ...] = (
    ("D6_repeated_failure", repeated_failure),
    ("D1_authenticated", authenticated),
    ("D4_grounded_ids", grounded_ids),
    ("D3_record_observed", record_observed),
    ("D2_owned_by_user", owned_by_user),
)

# Recorded on the decision without blocking. Batching broke the policy in live
# runs without harm, while blocking it cost extra turns and hid the rules above.
WARNING_RULES: tuple[tuple[str, Rule], ...] = (
    ("D5_one_call_at_a_time", one_call_at_a_time),
)


def check_write_call(
    call: ToolCall, facts: Facts, batch: Sequence[ToolCall], config: DomainConfig,
) -> dict:
    """Run the rules on one write call. `batch` holds every call of its proposal."""
    rule_id, reason = next(
        ((rule_id, reason) for rule_id, rule in RULES
         if (reason := rule(call, facts, batch, config)) is not None),
        (None, None),
    )
    warnings = [
        {"rule": warning_id, "reason": warning}
        for warning_id, rule in WARNING_RULES
        if (warning := rule(call, facts, batch, config)) is not None
    ]
    return {
        "verdict": "allow" if reason is None else "block",
        "rule": rule_id,
        "reason": reason,
        "warnings": warnings,
    }


class RulesAuditor(Auditor):
    """D1–D6 as an `Auditor`.

    Only calls to tools with `side_effect=True` are checked; a proposal without
    one is allowed with score 0. Set `audit_unknown=True` to also check tools
    whose side effect is unknown (`side_effect=None`).

    Scores: block 1.0, warn (D5 only) 0.5, allow 0.0.
    """

    name = "rules"

    def __init__(self, config: DomainConfig = GENERIC_CONFIG, audit_unknown: bool = False):
        self.config = config
        self.audit_unknown = audit_unknown

    def is_write(self, name: str, specs: dict[str, ToolSpec]) -> bool:
        spec = specs.get(name)
        side_effect = spec.side_effect if spec is not None else None
        return side_effect is True or (side_effect is None and self.audit_unknown)

    def audit(
        self,
        history: Sequence[Turn],
        proposal: Turn,
        tools: Sequence[ToolSpec],
        context: AuditContext | None = None,
    ) -> Verdict:
        specs = {spec.name: spec for spec in tools}
        writes = [call for call in proposal.tool_calls if self.is_write(call.name, specs)]
        if not writes:
            return Verdict.allow(checked_calls=[])
        facts = build_facts(history, self.config, author=proposal.role)
        checks = [
            {"call": call.to_dict(), **check_write_call(call, facts, proposal.tool_calls, self.config)}
            for call in writes
        ]
        details = {"checked_calls": checks, "facts": facts.snapshot()}
        blocked = [check for check in checks if check["verdict"] == "block"]
        if blocked:
            return Verdict(
                "block", 1.0, reason=blocked[0]["reason"],
                rule_ids=[check["rule"] for check in blocked], details=details,
            )
        warnings = [warning for check in checks for warning in check["warnings"]]
        if warnings:
            return Verdict(
                "warn", 0.5, reason=warnings[0]["reason"],
                rule_ids=sorted({warning["rule"] for warning in warnings}), details=details,
            )
        return Verdict("allow", 0.0, details=details)
