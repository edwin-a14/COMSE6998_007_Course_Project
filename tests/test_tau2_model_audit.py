"""Tool dispatch contracts, with real tau2 messages and a fake auditor/environment."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from audit_core import AuditContext, ToolSpec, Verdict
from audit_core.auditors import SemanticAuditError
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2_model_audit import ModelAuditOrchestrator


def make_orchestrator(*outcomes):
    orch = ModelAuditOrchestrator.__new__(ModelAuditOrchestrator)
    auditor = Mock()
    auditor.audit.side_effect = outcomes
    orch.configure_audit(auditor, [ToolSpec('write', side_effect=True)], AuditContext(domain='retail'))
    orch._rejected_calls = {}
    orch._simulated_failures = set()
    orch._recovery = None
    orch.trajectory = [UserMessage(role='user', content='Update the record')]
    orch.num_errors = 0
    orch.environment = Mock()
    orch.environment.get_response.side_effect = lambda c: ToolMessage(id=c.id, role='tool', content='done')
    return orch, auditor


def dispatch(orch, *ids):
    message = AssistantMessage(role='assistant', content='Proposed update', tool_calls=[
        ToolCall(id=i, name='write', arguments={}) for i in ids])
    orch.trajectory.append(message)
    results = orch._execute_tool_calls(message.tool_calls)
    orch.trajectory.extend(results)
    return results


class ModelDispatchTest(unittest.TestCase):
    def test_allow_and_warn_execute_entire_batch_once(self):
        for decision in ('allow', 'warn'):
            orch, auditor = make_orchestrator(Verdict(decision, 0.1))
            results = dispatch(orch, 'a', 'b')
            self.assertEqual(orch.environment.get_response.call_count, 2)
            self.assertEqual(auditor.audit.call_count, 1)
            history, proposal, tools, context = auditor.audit.call_args.args
            self.assertEqual(len(history), 1)
            self.assertEqual(len(proposal.tool_calls), 2)
            self.assertEqual(proposal.content, 'Proposed update')
            self.assertTrue(all(not r.error for r in results))

    def test_block_feedback_then_corrected_proposal_executes(self):
        orch, auditor = make_orchestrator(Verdict('block', 1, 'Wrong record'), Verdict.allow())
        results = dispatch(orch, 'a', 'b')
        orch.environment.get_response.assert_not_called()
        self.assertEqual([r.id for r in results], ['a', 'b'])
        self.assertTrue(all(r.error and 'Wrong record' in r.content for r in results))
        dispatch(orch, 'corrected')
        self.assertEqual(orch.environment.get_response.call_count, 1)
        self.assertTrue(any('Wrong record' in t.content for t in auditor.audit.call_args.args[0]))
        self.assertIsNone(orch.audit_stop_reason)

    def test_native_step_delivers_feedback_to_agent_and_executes_correction(self):
        orch, _ = make_orchestrator(Verdict('block', 1, 'Wrong record'), Verdict.allow())
        original = AssistantMessage(role='assistant', tool_calls=[
            ToolCall(id='bad', name='write', arguments={})])
        correction = AssistantMessage(role='assistant', tool_calls=[
            ToolCall(id='fixed', name='write', arguments={})])
        orch.trajectory.append(original)
        orch.message = original
        orch.from_role, orch.to_role = Role.AGENT, Role.ENV
        orch.done = False
        orch.step_count = 0
        orch.validate_communication = False
        orch.agent_state = None
        orch.agent = Mock()
        orch.agent.is_stop.return_value = False
        orch.agent.generate_next_message.return_value = correction, None
        orch.step()  # tool dispatch rejects
        orch.environment.get_response.assert_not_called()
        orch.step()  # actual benchmark routing delivers feedback to agent
        feedback = orch.agent.generate_next_message.call_args.args[0]
        self.assertIn('Wrong record', str(feedback))
        orch.step()  # corrected proposal executes
        self.assertEqual(orch.environment.get_response.call_args.args[0].id, 'fixed')

    def test_repeated_blocks_stop_at_retry_limit(self):
        orch, _ = make_orchestrator(*[Verdict('block', 1, 'Wrong') for _ in range(3)])
        for i in range(3): dispatch(orch, str(i))
        self.assertEqual(orch.audit_stop_reason, 'retry_limit')
        with patch.object(Orchestrator, '_check_termination'):
            orch._check_termination()
        self.assertTrue(orch.done)
        orch.environment.get_response.assert_not_called()

    def test_backend_error_stops_without_agent_error_verdict_or_retry(self):
        orch, auditor = make_orchestrator(SemanticAuditError('backend_error', {'model_calls': 1}))
        result = dispatch(orch, 'a')[0]
        self.assertTrue(result.error)
        self.assertEqual(orch.audit_stop_reason, 'auditor_error')
        self.assertEqual(orch.audit_records[0]['status'], 'error')
        self.assertNotIn('verdict', orch.audit_records[0])
        orch.environment.get_response.assert_not_called()
        self.assertEqual(auditor.audit.call_count, 1)

    def test_cli_rejects_unsupported_fault_before_building_environment(self):
        import contextlib
        import io
        import run_retail
        with patch.object(run_retail, 'build_environment') as build:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                run_retail.main(['--audit', 'general-hybrid', '--inject-wrong-order',
                                 '--output', '/tmp/unused-model-audit-test-output.json'])
            self.assertEqual(caught.exception.code, 2)
            build.assert_not_called()

    def test_auditor_error_finalization_marks_evaluation_invalid(self):
        orch, _ = make_orchestrator(SemanticAuditError('invalid_output', {'input_tokens': 10}))
        dispatch(orch, 'a')
        run = SimpleNamespace(info={}, get_messages=lambda: list(orch.trajectory))
        with patch.object(Orchestrator, '_finalize', return_value=run):
            result = orch._finalize()
        self.assertFalse(result.info['model_audit']['evaluation_valid'])
        self.assertEqual(result.info['model_audit']['stop_reason'], 'auditor_error')

    def test_user_tools_bypass_auditor(self):
        orch, auditor = make_orchestrator()
        call = ToolCall(id='u', name='write', arguments={}, requestor='user')
        orch._execute_tool_calls([call])
        auditor.audit.assert_not_called()
        orch.environment.get_response.assert_called_once()

    def test_finalize_excludes_rejected_calls_from_evaluator_replay(self):
        orch, _ = make_orchestrator(Verdict('block', 1, 'Wrong'))
        dispatch(orch, 'a', 'b')
        run = SimpleNamespace(info={}, get_messages=lambda: list(orch.trajectory))
        with patch.object(Orchestrator, '_finalize', return_value=run):
            result = orch._finalize()
        self.assertFalse(any(getattr(m, 'tool_calls', None) for m in result.messages))
        self.assertFalse(any(isinstance(m, ToolMessage) for m in result.messages))
        self.assertEqual(len(result.info['model_audit']['records']), 1)
