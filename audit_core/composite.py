"""Auditors built from other auditors, and trivial baselines."""

from __future__ import annotations

from collections.abc import Sequence

from audit_core.interface import DECISIONS, AuditContext, Auditor, ToolSpec, Turn, Verdict


class AllowAllAuditor(Auditor):
    """Never objects. The no-audit baseline: FAR 0, recall 0."""

    name = "allow-all"

    def audit(
        self,
        history: Sequence[Turn],
        proposal: Turn,
        tools: Sequence[ToolSpec],
        context: AuditContext | None = None,
    ) -> Verdict:
        return Verdict.allow()


class CompositeAuditor(Auditor):
    """Runs auditors in order and stops at the first block.

    Put cheap auditors first (rules before models): a proposal the rules already
    block never reaches the expensive auditor. The result takes the most severe
    decision and the highest score; `details["auditors"]` lists what each
    auditor that ran returned.
    """

    def __init__(self, *auditors: Auditor, name: str | None = None):
        if not auditors:
            raise ValueError("CompositeAuditor needs at least one auditor")
        self.auditors = auditors
        self.name = name or "+".join(auditor.name for auditor in auditors)

    def audit(
        self,
        history: Sequence[Turn],
        proposal: Turn,
        tools: Sequence[ToolSpec],
        context: AuditContext | None = None,
    ) -> Verdict:
        ran: list[tuple[str, Verdict]] = []
        for auditor in self.auditors:
            verdict = auditor.audit(history, proposal, tools, context)
            ran.append((auditor.name, verdict))
            if verdict.blocked:
                break
        _, worst = max(ran, key=lambda item: (DECISIONS.index(item[1].decision), item[1].score))
        return Verdict(
            worst.decision,
            max(verdict.score for _, verdict in ran),
            reason=worst.reason,
            rule_ids=[rule for _, verdict in ran for rule in verdict.rule_ids],
            details={"auditors": [{"name": name, **verdict.to_dict()} for name, verdict in ran]},
        )
