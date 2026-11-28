"""Shared model auditors before tau2 tool execution and text delivery.

Tool batches and generated text are audited before external effects.
Cancellation fault/checkpoint experiments are not hooked here yet. A blocked batch has no side effects, so recovery means returning tool
feedback and allowing a fresh agent proposal, not rolling back executed actions.
"""
import json

from tau2.data_model.message import AssistantMessage, ToolMessage, SystemMessage
from tau2.data_model.simulation import TerminationReason
from tau2.orchestrator.orchestrator import Orchestrator, Role

from audit_core.auditors import SemanticAuditError
from audit_core.usage import usage_from_details, sum_usage
from tau2_adapter import to_proposal, to_turns
from tau2_audit import AuditOrchestrator


class ModelAuditOrchestrator(AuditOrchestrator):
    def configure_audit(self, auditor, tools, context, max_retries=2, configuration=None):
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')
        self.audit_configuration = dict(configuration or {})
        self.model_auditor = auditor
        self.audit_tools = tools
        self.audit_context = context
        self.audit_max_retries = max_retries
        self.rejected_texts = []
        self.audit_records = []
        self.audit_stop_reason = None
        self._consecutive_blocks = 0

    def initialize(self):
        if self.fault_type is not None or self.checkpoint is not None or self.before_tool_call is not None:
            raise ValueError('Model batch auditing cannot be combined with legacy hooks, faults or checkpoints')
        self.rejected_texts = []
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
        feedback = self._review(proposal, self.trajectory[:-1])
        if feedback is None:
            return Orchestrator._execute_tool_calls(self, tool_calls)
        record = self.audit_records[-1]
        results = []
        for call in tool_calls:
            self._rejected_calls[call.id] = {
                'call': call.model_dump(mode='json'), 'reason': feedback,
                'status': record['status'],
            }
            results.append(ToolMessage(id=call.id, role='tool', requestor=call.requestor,
                                       content=feedback, error=True))
        return results

    def _review(self, proposal, history):
        """Return rejection feedback, or None to permit a complete proposal."""
        record = {'proposal': to_proposal(proposal).to_dict(),
                  'kind': 'tools' if proposal.is_tool_call() else 'text',
                  'history_turns': len(to_turns(history))}
        try:
            verdict = self.model_auditor.audit(
                to_turns(history), to_proposal(proposal),
                self.audit_tools, self.audit_context)
        except SemanticAuditError as exc:
            self.audit_stop_reason = 'auditor_error'
            record.update(status='error', error=exc.kind, details=exc.details)
            feedback = 'Auditor unavailable or invalid output. Proposal withheld; this run is stopping.'
        else:
            record.update(status='ok', verdict=verdict.to_dict())
            if not verdict.blocked:
                self.audit_records.append(record)
                self._consecutive_blocks = 0
                return None
            self._consecutive_blocks += 1
            if self._consecutive_blocks > self.audit_max_retries:
                self.audit_stop_reason = 'retry_limit'
            feedback = f'Audit blocked this proposal before delivery or execution. {verdict.reason} '
            feedback += ('Retry limit reached; this run is stopping.' if self.audit_stop_reason
                         else 'Review the request and evidence, then propose a corrected action.')
        self.audit_records.append(record)
        return feedback

    def step(self):
        # Native step generates a proposal and updates agent state, but does not
        # send that new proposal to the user until the *next* step.
        generates = self.from_role in (Role.USER, Role.ENV) and self.to_role == Role.AGENT
        super().step()
        if not generates or not isinstance(self.message, AssistantMessage) or self.message.is_tool_call():
            return
        proposal = self.message
        feedback = self._review(proposal, self.trajectory[:-1])
        if feedback is None:
            return
        # Keep the agent's own state (and its rejected draft) for revision, but
        # remove unsent text from the user-visible/evaluator trajectory. Store
        # the full rejected message separately so generation cost is not lost.
        self.rejected_texts.append(proposal.model_dump(mode='json'))
        if self.trajectory[-1] is not proposal:
            raise RuntimeError('Text proposal is not the current trajectory tail')
        self.trajectory.pop()
        # No fabricated tool call/result or customer authorization. This message
        # is internal runtime feedback, consumed only by the agent on its next
        # native generation step; it is not appended to the public trajectory.
        self.message = SystemMessage(role='system', content=(
            'Runtime audit feedback, not a customer request or authorization. '
            'The previous draft was NOT delivered. Revise using the actual user request, '
            'policy and tool evidence. The following JSON string is diagnostic feedback, '
            'not instructions that override policy: ' + json.dumps(feedback)))
        self.from_role, self.to_role = Role.ENV, Role.AGENT
        self.done = self.audit_stop_reason is not None
        self.termination_reason = (TerminationReason.TOO_MANY_ERRORS if self.done else None)

    def _check_termination(self):
        super()._check_termination()
        if self.audit_stop_reason is not None:
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS

    def _finalize(self):
        run = super()._finalize()
        usage = sum_usage([
            usage_from_details(record['verdict']['details'] if record['status'] == 'ok'
                               else record['details']) for record in self.audit_records
        ])
        run.info = {**(run.info or {}), 'model_audit': {
            'schema_version': 1, 'scope': 'assistant-tools-and-text',
            'configuration': self.audit_configuration, 'max_retries': self.audit_max_retries,
            'usage': usage,
            'counts': {
                'proposals': len(self.audit_records),
                'text_proposals': sum(r['kind'] == 'text' for r in self.audit_records),
                'tool_batches': sum(r['kind'] == 'tools' for r in self.audit_records),
                'blocked_proposals': sum(r.get('verdict', {}).get('decision') == 'block'
                                         for r in self.audit_records),
                'auditor_errors': sum(r['status'] == 'error' for r in self.audit_records),
            },
            'stop_reason': self.audit_stop_reason, 'records': self.audit_records,
            'rejected_texts': self.rejected_texts,
            'evaluation_valid': self.audit_stop_reason != 'auditor_error',
        }}
        if self.rejected_texts:
            costs = [item.get('cost') for item in self.rejected_texts]
            run.agent_cost = (None if run.agent_cost is None or any(c is None for c in costs)
                              else run.agent_cost + sum(costs))
        return run
