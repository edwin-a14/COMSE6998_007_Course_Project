"""Benchmark-independent audit interface.

Every auditor implements one method:

    audit(history, proposal, tools, context=None) -> Verdict

`history` is everything that happened before the proposal, `proposal` is the
agent turn about to take effect (its text and every tool call it makes, possibly
none), `tools` are the tools the agent may choose from, and `context` holds the
task and, when the benchmark has one, a policy. Benchmark adapters
(`tau2_adapter.py`, `aftraj/adapter.py`) build these values; auditors never see
benchmark objects, labels or answers.

This module imports nothing from τ²-bench or AFTraj.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

TurnKind = Literal["system", "user", "agent", "tool"]
Decision = Literal["allow", "warn", "block"]

TURN_KINDS: tuple[str, ...] = ("system", "user", "agent", "tool")
DECISIONS: tuple[str, ...] = ("allow", "warn", "block")


@dataclass(frozen=True)
class ToolCall:
    """One tool call an agent makes."""

    name: str
    arguments: dict[str, Any] | str = field(default_factory=dict)
    """Parsed keyword arguments, or the raw argument text when it is not a JSON
    object (for example Python source passed to an interpreter)."""
    id: str | None = None
    """Links the call to its result turn (`Turn.tool_call_id`). Unique within
    one conversation."""

    @property
    def kwargs(self) -> dict[str, Any]:
        """The arguments as a dict; empty when they are raw text."""
        return self.arguments if isinstance(self.arguments, dict) else {}

    def key(self) -> tuple[str, str]:
        """Identity of the call ignoring its id: same name and same arguments."""
        return self.name, json.dumps(self.arguments, sort_keys=True, default=str)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "id": self.id}


@dataclass(frozen=True)
class Turn:
    """One step of a conversation, in history or as the proposal.

    kind     who produced it: "system", "user", "agent" or "tool" (a tool or
             environment result).
    role     the producer's name as the benchmark gives it: "assistant",
             "MathSolver", "search_agent", "user", "environment", ...
             Several agents can share one conversation.
    """

    kind: TurnKind
    role: str
    content: str = ""
    thought: str = ""
    """The agent's reasoning shown before acting, when the benchmark records it."""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    """For a "tool" turn: the id of the call it answers, or None when unknown."""
    error: bool | None = None
    """For a "tool" turn: whether the call failed. None when the benchmark does
    not say."""
    raw_action: str = ""
    """An action the adapter could not turn into tool calls, such as a code
    block, kept verbatim so model-based auditors can still read it."""

    def __post_init__(self) -> None:
        if self.kind not in TURN_KINDS:
            raise ValueError(f"unknown turn kind {self.kind!r}; expected one of {TURN_KINDS}")
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "role": self.role,
            "content": self.content,
            "thought": self.thought,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "error": self.error,
            "raw_action": self.raw_action,
        }


@dataclass(frozen=True)
class ToolSpec:
    """A tool the agent may call."""

    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None
    """JSON schema of the arguments, or None when the benchmark gives none."""
    side_effect: bool | None = None
    """True when the tool changes state outside the conversation (a write),
    False when it only reads, None when unknown."""


@dataclass(frozen=True)
class AuditContext:
    """Task-level information that is not part of the conversation."""

    domain: str = ""
    task: str = ""
    """What the user asked for, when the benchmark states it separately."""
    policy: str | None = None
    """Rules the agent must follow, when the benchmark has them."""


@dataclass
class Verdict:
    """An auditor's judgment of one proposal.

    decision  "block": stop the proposal. "warn": let it run but record a
              concern. "allow": no concern.
    score     risk in [0, 1]; higher is riskier. Used to rank proposals and
              sweep thresholds, so it should be comparable across proposals of
              one auditor.
    reason    feedback the agent can act on when the proposal is blocked.
    rule_ids  which checks fired, for ablations.
    details   anything else the auditor wants recorded (JSON-serializable).
    """

    decision: Decision
    score: float
    reason: str = ""
    rule_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown decision {self.decision!r}; expected one of {DECISIONS}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    @classmethod
    def allow(cls, **details: Any) -> "Verdict":
        return cls("allow", 0.0, details=details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "score": self.score,
            "reason": self.reason,
            "rule_ids": list(self.rule_ids),
            "details": self.details,
        }


class Auditor(ABC):
    """Base class of every auditor.

    An auditor must:
    - use only its arguments, never benchmark internals, so the same object runs
      on τ²-bench and AFTraj;
    - accept a proposal without tool calls (a plain text answer);
    - cope with missing information (no tools, no parameter schemas, no policy,
      unknown tool errors) by allowing or lowering its score, not by raising;
    - not modify its arguments.
    """

    name: str = "auditor"

    @abstractmethod
    def audit(
        self,
        history: Sequence[Turn],
        proposal: Turn,
        tools: Sequence[ToolSpec],
        context: AuditContext | None = None,
    ) -> Verdict:
        """Judge `proposal` given everything before it."""
