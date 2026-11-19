"""Shared model auditors at the tau2 tool-batch dispatch boundary.

Text-only proposals and cancellation fault/checkpoint experiments are not hooked
here yet. A blocked batch has no side effects, so recovery means returning tool
feedback and allowing a fresh agent proposal, not rolling back executed actions.
"""
from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.data_model.simulation import TerminationReason
from tau2.orchestrator.orchestrator import Orchestrator

from audit_core.auditors import SemanticAuditError
from tau2_adapter import to_proposal, to_turns
from tau2_audit import AuditOrchestrator


class ModelAuditOrchestrator(AuditOrchestrator):
    def configure_audit(self, auditor, tools, context, max_retries=2):
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')
        self.model_auditor = auditor
        self.audit_tools = tools
        self.audit_context = context
        self.audit_max_retries = max_retries
        self.audit_records = []
        self.audit_stop_reason = None
        self._consecutive_blocks = 0

    def initialize(self):
        if self.fault_type is not None or self.checkpoint is not None or self.before_tool_call is not None:
            raise ValueError('Model batch auditing cannot be combined with legacy hooks, faults or checkpoints')
        self.audit_records = []
        self.audit_stop_reason = None
        self._consecutive_blocks = 0
        super().initialize()

    def _execute_tool_calls(self, tool_calls):
        if not tool_calls or all(c.requestor == 'user' for c in tool_calls):
            return Orchestrator._execute_tool_calls(self, tool_calls)
        proposal = self.trajectory[-1]
        if (not isinstance(proposal, AssistantMessage) or proposal.tool_calls != tool_calls
                or any(c.requestor != 'assistant' for c in tool_calls)):
            raise ValueError('Expected the current complete assistant tool batch')
        if self.audit_stop_reason is not None:
            raise RuntimeError('Audit session already stopped')
        record = {'proposal': to_proposal(proposal).to_dict()}
        try:
            verdict = self.model_auditor.audit(
                to_turns(self.trajectory[:-1]), to_proposal(proposal),
                self.audit_tools, self.audit_context)
        except SemanticAuditError as exc:
            self.audit_stop_reason = 'auditor_error'
            record.update(status='error', error=exc.kind, details=exc.details)
            feedback = 'Auditor unavailable or invalid output. No tools executed; this run is stopping.'
        else:
            record.update(status='ok', verdict=verdict.to_dict())
            if not verdict.blocked:
                self.audit_records.append(record)
                self._consecutive_blocks = 0
                return Orchestrator._execute_tool_calls(self, tool_calls)
            self._consecutive_blocks += 1
            if self._consecutive_blocks > self.audit_max_retries:
                self.audit_stop_reason = 'retry_limit'
            feedback = f'Audit blocked the entire batch; no tools executed. {verdict.reason} '
            feedback += ('Retry limit reached; this run is stopping.' if self.audit_stop_reason
                         else 'Review the request and evidence, then propose a corrected action.')
        self.audit_records.append(record)
        results = []
        for call in tool_calls:
            self._rejected_calls[call.id] = {
                'call': call.model_dump(mode='json'), 'reason': feedback,
                'status': record['status'],
            }
            results.append(ToolMessage(id=call.id, role='tool', requestor=call.requestor,
                                       content=feedback, error=True))
        return results

    def _check_termination(self):
        super()._check_termination()
        if self.audit_stop_reason is not None:
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS

    def _finalize(self):
        run = super()._finalize()
        run.info = {**(run.info or {}), 'model_audit': {
            'scope': 'assistant-tool-batches', 'max_retries': self.audit_max_retries,
            'stop_reason': self.audit_stop_reason, 'records': self.audit_records,
            'evaluation_valid': self.audit_stop_reason != 'auditor_error',
        }}
        return run
