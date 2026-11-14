"""Convert τ²-bench objects to the shared audit interface (audit_core.interface).

The only module on the τ² side that auditors' inputs pass through: τ² messages
become `Turn`s, τ² tools become `ToolSpec`s, and the domain policy goes into
`AuditContext`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall as Tau2ToolCall,
    ToolMessage,
    UserMessage,
)

from audit_core.interface import AuditContext, ToolCall, ToolSpec, Turn


def to_tool_call(call: Tau2ToolCall) -> ToolCall:
    return ToolCall(name=call.name, arguments=dict(call.arguments), id=call.id)


def to_turns(messages: Iterable[Message]) -> list[Turn]:
    """One `Turn` per message; the results inside a MultiToolMessage become one
    turn each. Other message types (ticks, audio control) are dropped."""
    turns: list[Turn] = []
    for message in messages:
        if isinstance(message, MultiToolMessage):
            turns.extend(_tool_turn(item) for item in message.tool_messages)
        elif isinstance(message, ToolMessage):
            turns.append(_tool_turn(message))
        elif isinstance(message, AssistantMessage):
            turns.append(to_proposal(message))
        elif isinstance(message, UserMessage):
            turns.append(Turn(
                kind="user", role="user", content=message.content or "",
                tool_calls=tuple(to_tool_call(call) for call in message.tool_calls or []),
            ))
        elif isinstance(message, SystemMessage):
            turns.append(Turn(kind="system", role="system", content=message.content or ""))
    return turns


def to_proposal(message: AssistantMessage) -> Turn:
    return Turn(
        kind="agent", role="assistant", content=message.content or "",
        tool_calls=tuple(to_tool_call(call) for call in message.tool_calls or []),
    )


def _tool_turn(message: ToolMessage) -> Turn:
    return Turn(
        kind="tool", role="tool", content=message.content or "",
        tool_call_id=message.id, error=bool(message.error),
    )


def tool_specs(environment, is_write: Callable[[str], bool] | None = None) -> list[ToolSpec]:
    """The agent's tools, with τ²'s WRITE annotation as `side_effect`."""
    if is_write is None:
        from deterministic_audit import write_tool_checker

        is_write = write_tool_checker(environment)
    specs = []
    for tool in environment.get_tools():
        function = tool.openai_schema["function"]
        specs.append(ToolSpec(
            name=tool.name,
            description=function.get("description", ""),
            parameters=function.get("parameters"),
            side_effect=is_write(tool.name),
        ))
    return specs


def audit_context(environment, domain: str) -> AuditContext:
    return AuditContext(domain=domain, policy=environment.get_policy())
