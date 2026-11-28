"""Select proposals for a supplied semantic auditor, without calling a model here.

Known read-only calls may skip review unless they compute, verify or carry
executable arguments. Empty turns also skip. Selection is not a verdict.
Compose this wrapper after RulesAuditor to avoid semantic work on rule blocks.
"""

from collections.abc import Sequence
from dataclasses import replace

from audit_core.interface import AuditContext, Auditor, ToolSpec, Turn, Verdict


# Routing heuristics, independent of benchmark labels or task correctness.
# A sandboxed computation can still contain a consequential argument error.
COMPUTATION_TOOLS = frozenset({
    "compute", "calculator", "verify_answer", "run_tests", "python_interpreter",
})
EXECUTABLE_ARGUMENTS = frozenset({
    "code", "solution_code", "test_code", "expression", "command", "commands", "script",
})


class SelectiveSemanticAuditor(Auditor):
    """Apply the v2 selection policy to any shared-interface semantic auditor.

    Unknown or unlisted tools are reviewed conservatively. Text includes
    otherwise read-only turns with a textual claim or recorded reasoning.
    The first matching reason is recorded; the entire proposal is forwarded.

    `semantic_invocations` counts delegate calls, NOT provider API requests:
    token usage and actual model retries must be recorded by the delegate.
    Delegate exceptions propagate so infrastructure failures cannot silently
    become allow/block judgments about the agent.
    """

    name = "selective-semantic"
    policy = "writes-text-computation-v2"

    def __init__(self, semantic: Auditor):
        self.semantic = semantic

    def _selection(self, proposal: Turn, tools: Sequence[ToolSpec]) -> tuple[bool, str]:
        if proposal.raw_action.strip():
            return True, "raw_action"
        if proposal.content.strip():
            return True, "text"
        if proposal.thought.strip():
            return True, "thought"
        if not proposal.tool_calls:
            return False, "empty_proposal"

        specs = {tool.name: tool for tool in tools}
        effects = [
            specs[call.name].side_effect if call.name in specs else None
            for call in proposal.tool_calls
        ]
        if any(effect is True for effect in effects):
            return True, "write_tool"
        if any(effect is None for effect in effects):
            return True, "unknown_tool"
        if any(call.name in COMPUTATION_TOOLS for call in proposal.tool_calls):
            return True, "computation_tool"
        if any(EXECUTABLE_ARGUMENTS.intersection(call.kwargs) for call in proposal.tool_calls):
            return True, "executable_arguments"
        return False, "read_only_tools"

    def audit(
        self,
        history: Sequence[Turn],
        proposal: Turn,
        tools: Sequence[ToolSpec],
        context: AuditContext | None = None,
    ) -> Verdict:
        selected, reason = self._selection(proposal, tools)
        selection = {
            "policy": self.policy,
            "selected": selected,
            "reason": reason,
            "semantic_invocations": int(selected),
        }
        if not selected:
            return Verdict.allow(semantic_selection=selection)

        verdict = self.semantic.audit(history, proposal, tools, context)
        # Do not mutate a delegate's verdict, which may be cached or reused.
        return replace(verdict, details={**verdict.details, "semantic_selection": selection})
