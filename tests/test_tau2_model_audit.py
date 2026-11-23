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

    def test_finalize_aggregates_usage_without_double_counting(self):
        details = {'model_calls': 1, 'input_tokens': 12, 'output_tokens': 3, 'latency_ms': 4,
                   'semantic_selection': {'selected': True}}
        nested = {'auditors': [{'details': {}}, {'details': details}]}
        orch, _ = make_orchestrator(Verdict('warn', 0.2, details=nested),
                                    SemanticAuditError('backend_error', {'model_calls': 1}))
        orch.audit_configuration = {'mode': 'general-hybrid', 'model': 'fake'}
        dispatch(orch, 'allowed')
        dispatch(orch, 'withheld')
        run = SimpleNamespace(info={}, get_messages=lambda: list(orch.trajectory))
        with patch.object(Orchestrator, '_finalize', return_value=run):
            result = orch._finalize()
        audit = result.info['model_audit']
        self.assertEqual(audit['usage']['model_calls'], 2)
        self.assertIsNone(audit['usage']['input_tokens'])
        self.assertEqual(audit['counts']['auditor_errors'], 1)
        self.assertEqual(audit['counts']['blocked_proposals'], 0)
        self.assertEqual(audit['counts']['tool_batches'], 2)
        self.assertEqual(audit['configuration']['model'], 'fake')

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


class TextDispatchTest(unittest.TestCase):
    def setup_text(self, *outcomes):
        orch, auditor = make_orchestrator(*outcomes)
        orch.message = orch.trajectory[-1]
        orch.from_role, orch.to_role = Role.USER, Role.AGENT
        orch.done = False
        orch.step_count = 0
        orch.validate_communication = False
        orch.solo_mode = False
        orch.agent_state = None
        orch.agent = Mock()
        orch.agent.is_stop.return_value = False
        orch.user = Mock()
        orch.user_state = None
        return orch, auditor

    def test_rejected_text_never_reaches_user_and_agent_gets_feedback(self):
        orch, auditor = self.setup_text(Verdict('block', 1, 'Tool failed'), Verdict.allow())
        bad = AssistantMessage(role='assistant', content='Cancelled successfully', cost=0.2)
        good = AssistantMessage(role='assistant', content='Cancellation failed', cost=0.1)
        orch.agent.generate_next_message.side_effect = [(bad, None), (good, None)]
        orch.step()
        orch.user.generate_next_message.assert_not_called()
        self.assertEqual(orch.to_role, Role.AGENT)
        self.assertNotIn(bad, orch.trajectory)
        orch.step()
        self.assertIn('Tool failed', orch.agent.generate_next_message.call_args.args[0].content)
        self.assertEqual(orch.message.content, 'Cancellation failed')
        self.assertEqual(orch.to_role, Role.USER)
        self.assertEqual(auditor.audit.call_count, 2)
        self.assertEqual(orch.rejected_texts[0]['cost'], 0.2)

    def test_warn_text_is_delivered_without_retry(self):
        orch, auditor = self.setup_text(Verdict('warn', 0.2))
        orch.agent.generate_next_message.return_value = AssistantMessage(role='assistant', content='Hello'), None
        orch.step()
        self.assertEqual(orch.to_role, Role.USER)
        self.assertEqual(auditor.audit.call_count, 1)

    def test_text_error_and_retry_exhaustion_stop_without_delivery(self):
        for outcomes, reason in (([SemanticAuditError('backend_error', {})], 'auditor_error'),
                                 ([Verdict('block', 1, 'Wrong')] * 3, 'retry_limit')):
            orch, _ = self.setup_text(*outcomes)
            orch.agent.generate_next_message.side_effect = [
                (AssistantMessage(role='assistant', content='Wrong'), None) for _ in outcomes]
            for _ in outcomes: orch.step()
            self.assertTrue(orch.done)
            self.assertEqual(orch.audit_stop_reason, reason)
            self.assertEqual(orch.to_role, Role.AGENT)
            orch.user.generate_next_message.assert_not_called()

    def test_stop_text_is_audited_before_termination(self):
        orch, auditor = self.setup_text(Verdict('block', 1, 'False completion'))
        orch.agent.is_stop.return_value = True
        orch.agent.generate_next_message.return_value = AssistantMessage(role='assistant', content='Done'), None
        orch.step()
        self.assertFalse(orch.done)
        self.assertEqual(orch.to_role, Role.AGENT)
        auditor.audit.assert_called_once()

    def test_native_llm_agent_keeps_feedback_private_and_delivers_only_revision(self):
        from tau2.agent.llm_agent import LLMAgent
        orch, _ = self.setup_text(Verdict('block', 1, 'Tool failed'), Verdict.allow())
        orch.agent = LLMAgent(tools=[], domain_policy='Be truthful', llm='fake')
        orch.agent_state = orch.agent.get_init_state()
        bad = AssistantMessage(role='assistant', content='Cancelled successfully', cost=0.2)
        good = AssistantMessage(role='assistant', content='Cancellation failed', cost=0.1)
        orch.user.generate_next_message.return_value = UserMessage(role='user', content='Thanks'), None
        with patch('tau2.agent.llm_agent.generate', side_effect=[bad, good]) as generate:
            orch.step()
            orch.step()
        prompt = generate.call_args.kwargs['messages']
        self.assertTrue(any('Runtime audit feedback' in (m.content or '') for m in prompt))
        self.assertIn(bad, orch.agent_state.messages)
        self.assertNotIn(bad, orch.trajectory)
        orch.step()
        self.assertEqual(orch.user.generate_next_message.call_args.args[0].content, 'Cancellation failed')

    def test_rejected_generation_cost_is_retained_on_finalize(self):
        orch, _ = self.setup_text(Verdict('block', 1, 'Wrong'), Verdict.allow())
        orch.agent.generate_next_message.side_effect = [
            (AssistantMessage(role='assistant', content='Wrong', cost=0.2), None),
            (AssistantMessage(role='assistant', content='Corrected', cost=0.1), None)]
        orch.step()
        orch.step()
        run = SimpleNamespace(info={}, agent_cost=0.1, get_messages=lambda: list(orch.trajectory))
        with patch.object(Orchestrator, '_finalize', return_value=run):
            result = orch._finalize()
        self.assertAlmostEqual(result.agent_cost, 0.3)
        self.assertEqual(result.info['model_audit']['rejected_texts'][0]['content'], 'Wrong')
