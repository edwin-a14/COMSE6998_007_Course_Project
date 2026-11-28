"""Offline CLI integration with fake transport and hand-made trajectories."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aftraj import run_eval
from aftraj.tests.test_adapter import math_trajectory
from audit_core.model import ModelResponse
from audit_core.modes import build_model_auditor


class Backend:
    def __init__(self, response=None, error=None):
        self.calls = 0
        self.response = response or ModelResponse(
            '{"stage":"uncertain","stage_evidence":"Fixture stage","error_evidence":["Fixture conflict"],"decision":"block","score":0.9,"reason":"Fixture error","rule_ids":["evidence_conflict"]}', 10, 5)
        self.error = error

    def complete(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


class ModelModesTest(unittest.TestCase):
    def test_evaluation_reviews_compute_arguments_and_records_nested_usage(self):
        backend = Backend()
        auditor = build_model_auditor("general-hybrid", model="fake", backend=backend)
        row = run_eval.evaluate(auditor, math_trajectory())
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["pred_step"], 1)
        self.assertEqual(row["num_audits"], 1)
        self.assertEqual(row["usage"]["model_calls"], 1)
        self.assertEqual(row["usage"]["input_tokens"], 10)
        self.assertEqual(row["usage"]["output_tokens"], 5)
        self.assertEqual(row["usage"]["semantic_skips"], 0)
        self.assertIn("details", row["steps"][-1])

    def test_model_failure_is_excluded_from_detection_metrics_but_counted(self):
        auditor = build_model_auditor("general-hybrid", model="fake", backend=Backend(error=TimeoutError()))
        t = math_trajectory()
        row = run_eval.evaluate(auditor, t)
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["pred_step"])
        self.assertEqual(row["steps"][-1]["error"], "backend_error")
        summary = run_eval.summarize([row], [t])
        self.assertEqual(summary["evaluated_trajectories"], 0)
        self.assertEqual(summary["failed_trajectories"], 1)
        self.assertEqual(summary["usage"]["model_calls"], 1)
        self.assertIsNone(summary["usage"]["input_tokens"])
        self.assertEqual(summary["by_domain"]["overall"]["n"], 0)

    def test_invalid_output_retains_billed_tokens(self):
        auditor = build_model_auditor("general-hybrid", model="fake", backend=Backend(ModelResponse("bad", 20, 3)))
        row = run_eval.evaluate(auditor, math_trajectory())
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["usage"]["input_tokens"], 20)

    def test_missing_usage_is_not_counted_as_zero(self):
        response = ModelResponse('{"stage":"uncertain","stage_evidence":"Fixture stage","error_evidence":[],"decision":"allow","score":0,"reason":"Fixture","rule_ids":[]}')
        row = run_eval.evaluate(build_model_auditor("general-hybrid", model="fake", backend=Backend(response)), math_trajectory())
        self.assertIsNone(row["usage"]["input_tokens"])

    def run_cli(self, path, backend, *extra):
        def factory(mode, **kwargs):
            return build_model_auditor(mode, backend=backend, **kwargs)
        with patch.object(run_eval, "load", return_value=[math_trajectory()]), \
                patch("importlib.util.find_spec", return_value=object()), \
                patch.object(run_eval, "build_model_auditor", side_effect=factory), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_eval.main(["--auditor", "general-hybrid", "--model", "fake",
                                  "--output", str(path), *extra])

    def test_cli_resume_preserves_results_and_rejects_model_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            backend = Backend()
            first = self.run_cli(path, backend)
            self.assertEqual(self.run_cli(path, backend, "--resume"), first)
            self.assertEqual(backend.calls, 1)
            before = (path / "per_sample.jsonl").read_bytes()
            for extra in (("--model", "another"), ("--max-tokens", "100"), ("--timeout", "10")):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    self.run_cli(path, backend, "--resume", *extra)
            self.assertEqual((path / "per_sample.jsonl").read_bytes(), before)
            config = json.loads(before.splitlines()[0])
            self.assertEqual(config["model_config"]["selection_policy"], "writes-text-computation-v2")

    def test_resume_rejects_prompt_selection_and_adapter_version_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            backend = Backend()
            self.run_cli(path, backend)
            for cls, attr in ((run_eval.SemanticAuditor, "prompt_version"),
                              (run_eval.SelectiveSemanticAuditor, "policy"),
                              (run_eval, "ADAPTER_VERSION")):
                with patch.object(cls, attr, "changed"), self.assertRaises(SystemExit):
                    self.run_cli(path, backend, "--resume")
            self.assertEqual(backend.calls, 1)

    def test_failed_run_can_be_summarized_and_resume_does_not_retry_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = Backend(error=TimeoutError())
            first = self.run_cli(Path(tmp), backend)
            self.assertEqual(first["failed_trajectories"], 1)
            self.assertEqual(self.run_cli(Path(tmp), backend, "--resume"), first)
            self.assertEqual(backend.calls, 1)

    def test_model_required_before_loading_or_writing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run_eval, "load") as load:
            path = Path(tmp) / "not-created"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_eval.main(["--auditor", "general-hybrid", "--output", str(path)])
            load.assert_not_called()
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
