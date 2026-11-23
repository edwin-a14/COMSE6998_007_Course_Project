# General-hybrid: team handoff and integration contract

Status: implementation handoff, 2026-11-23. This is a runnable research prototype,
not evidence that hybrid auditing improves benchmark accuracy or task success.

## 1. Scope and architecture

The shared pipeline is:

```
normalized history + complete proposal + tool catalog + context
  -> RulesAuditor
     block: return immediately, no semantic request
     allow/warn: continue
  -> SelectiveSemanticAuditor
     known read-only tool batch without text: skip semantic
     write / text / thought / raw action / unknown tool: call semantic once
  -> SemanticAuditor -> LiteLLMBackend -> structured response -> Verdict
```

There is no third hybrid model. Hybrid composes existing rule and semantic
components. The selector sees the whole proposal; a read tool accompanied by
text can still require semantic auditing. Missing metadata is not proof of safety.

| Mode | Rules | Selector | Semantic |
|---|---|---|---|
| `general-hybrid` | yes | yes | selected proposals |
| `selective-semantic` | no | same selector | selected proposals |
| `continuous-semantic` | no | no | every proposal presented by the runner |

The three modes use the same semantic prompt/schema and model settings. The old
`semantic`, `hybrid`, and `continuous` CLI modes use the legacy implementation;
they are **not aliases** for these new modes.

Key modules:

| Module | Responsibility |
|---|---|
| `audit_core/interface.py` | Turn, ToolCall, ToolSpec, AuditContext, Verdict |
| `audit_core/auditors/rules.py` | deterministic D1–D6, domain configurations |
| `audit_core/auditors/selective.py` | decide whether semantic evaluation runs |
| `audit_core/auditors/semantic.py` | serialize evidence, prompt/schema, parse output |
| `audit_core/model.py` | replaceable model transport; no automatic API retries |
| `audit_core/modes.py` | construct the three comparable modes |
| `audit_core/usage.py` | nested usage aggregation shared by both runners |
| `tau2_adapter.py` | normalize benchmark messages, tool schemas and policy |
| `tau2_model_audit.py` | pre-execution tool and pre-delivery text control |
| `run_retail.py` | retail CLI/configuration/output |
| `aftraj/adapter.py`, `aftraj/run_eval.py` | static trajectory detection evaluation |

## 2. Auditor interface (for teammates)

```python
from audit_core.modes import build_model_auditor
from audit_core.auditors.rules import DOMAIN_CONFIGS

# Retail runtime: explicitly supply retail rules.
auditor = build_model_auditor(
    "general-hybrid", model="openai/gpt-4.1-mini",
    rules_config=DOMAIN_CONFIGS["retail"], max_tokens=512, timeout=30,
)
verdict = auditor.audit(history, proposal, tools, context)
```

- `history`: prior normalized Turns; exclude the current proposal and future turns.
- `proposal`: one complete Turn, including text and all tool calls, if any.
- `tools`: ToolSpecs including name, description, schema and side-effect metadata.
- `context`: domain/policy/task known to the agent. Never supply benchmark gold
  answers, labels, future outcomes or hidden user scenario instructions.
- `Verdict`: shared dataclass with `decision`, `score`, `reason`, `rule_ids`, `details`.
- `decision`: `allow` and `warn` permit execution/delivery; `block` prevents it.
- `score`: model risk score, not a calibrated probability.
- `SemanticAuditError`: auditor infrastructure/output failure, not an agent verdict.

AFTraj deliberately uses GENERIC_CONFIG, not retail-specific authentication and
ownership rules. Generic D1–D3 are disabled without a domain configuration. D4–D6
remain available where their prerequisites apply; do not expect retail rules to
identify arbitrary mathematical reasoning errors.

The v3 model response has required fields:

```json
{
  "stage": "intermediate",
  "stage_evidence": "The proposal is an action before a completion claim.",
  "error_evidence": [],
  "decision": "allow",
  "score": 0.1,
  "reason": "Consistent with the evidence.",
  "rule_ids": []
}
```

Stage is `intermediate`, `handoff`, `final`, or `uncertain`. The parser puts the
three diagnostic fields under `Verdict.details.semantic_assessment`; the shared
Verdict interface does not change. Missing fields, malformed JSON, unsupported
stage, or a block with an empty error-evidence list are auditor errors. Parsing
checks structure, not factual sufficiency. Stage never automatically overrides
the verdict. Keep prompt_version updated when changing the prompt or schema.

## 3. Runtime behavior and recovery semantics

Tools: audit the complete assistant batch once before any operation executes.
Allow/warn delegates to normal tau2 execution. Block executes none and returns a
ToolMessage with matching call ID for every proposed tool. The agent receives
feedback on its next normal generation step. User-side tools bypass this audit.

Text: after generating a text-only draft, audit before its next delivery step.
A blocked draft is removed from the delivered/evaluator trajectory. The agent
keeps its draft privately and receives labeled internal runtime feedback, encoded
as a SystemMessage, not as a fake customer request or a fabricated tool result.
The revised proposal is audited again. The fixed bootstrap greeting is excluded.
A generated stop/completion message is not exempt from auditing.

Retry policy:

- Default `--audit-max-retries 2`: up to two additional attempts after the first
  block; the third consecutive blocked proposal stops the run.
- Counter is shared by tool and text proposals; an allowed/warned proposal resets it.
- Native `--max-steps` still bounds the conversation; revisions consume steps.
- `--audit-max-retries 0` stops on the first block (a useful no-revision control).
- SemanticAuditError stops immediately without model retry or proposal execution.
- Benchmark termination uses TOO_MANY_ERRORS; inspect `model_audit.stop_reason`
  to distinguish `retry_limit` from `auditor_error`.

This is bounded pre-action/pre-delivery revision, **not arbitrary goal-stage
checkpoint rollback**. Already executed operations are not undone. Allowed batch
execution is also not transactional: a later tool failure does not roll back an
earlier successful tool. Legacy cancellation fault/checkpoint flags are rejected
with the new model modes. General fault integration remains separate work.

## 4. Output schema and experiment accounting

Retail JSON: `simulations[i].info.model_audit` contains:

- `schema_version`, `scope` (`assistant-tools-and-text`), `configuration` with mode,
  model, prompt version, selector policy, rule domain, temperature and limits.
- `records`: each has `kind`, `history_turns`, complete normalized `proposal`,
  `status`, and either `verdict` or `error` plus available model `details`.
- `counts`: proposal, text, tool-batch, block and auditor-error counts.
- `usage`: model_calls, input_tokens, output_tokens, model_latency_ms, semantic_skips.
- `rejected_texts`: original rejected text messages, including available generation
  usage and cost. They are diagnostics, not messages delivered to the user.
- `max_retries`, `stop_reason`, `evaluation_valid`.

Composite verdicts retain child verdicts in `details.auditors`; find semantic
stage/model details in the semantic child. Rule short-circuit is not a selector
skip. One tool batch is one proposal, even if it contains several calls. Audit
invocations and actual model requests are different counts.

Usage is summed without counting wrapper details twice. Zero means no model call;
missing provider usage is null/unknown, not zero. Calls include failed requests
and are not proof of billing. Auditor token totals are separate from tau2's
`agent_cost`/`user_cost`; adding those two dollar costs alone omits auditor cost.
Rejected text generation cost is included in agent_cost (unknown stays unknown).

On auditor error, results are saved, evaluation_valid is false, and retail CLI
exits 1. Do not count its benchmark reward as an ordinary agent failure: exclude
from quality metrics and report invalid-run counts/coverage separately. Retry
exhaustion is a valid policy outcome, not infrastructure failure.

Rejected tool calls are omitted from evaluator replay. For debugging, combine
public messages with audit records/rejected-text diagnostics; the public transcript
alone is intentionally not a complete record of the agent's private history.
Raw model request snapshots are not persisted.

AFTraj output: config + trajectory rows in `per_sample.jsonl`, aggregate results
in `results.json`. First blocked step is the prediction unless `--full-scan` is
requested. This is static detection, not live recovery. Error rows are excluded
from quality metrics, their usage retained. Resume requires matching settings
including prompt version; existing error rows are not automatically retried.

## 5. Environment and free tests

All commands below run from the repository root on branch
`michael/feature/general-hybrid`. Use the existing sibling tau2-bench Python 3.12
environment; do not upgrade benchmark dependencies just to run these tests.
Keep provider keys in your local environment / `../tau2-bench/.env`, never in Git.

Core tests with no provider calls:

```sh
../tau2-bench/.venv/bin/python -B -m unittest -v \
  tests.test_general_hybrid tests.test_semantic_auditor \
  tests.test_tau2_model_audit aftraj.tests.test_model_modes
```

Full offline regression (leave both LIVE_* variables unset):

```sh
env -u LIVE_SEMANTIC_MODEL -u LIVE_TEXT_RECOVERY_MODEL \
  ../tau2-bench/.venv/bin/python -B -m unittest discover -s . -t .
```

Five real-data checks may skip if PyArrow/data are unavailable. When the local
AFTraj files already exist, run them in the dedicated dependency environment:

```sh
uv run --no-project --python 3.12 --with-requirements aftraj/requirements.txt \
  python -B -m unittest -v aftraj.tests.test_data
```

Test responsibilities:

- Fake backend tests: serialization, strict parsing, selection, rule short-circuit,
  usage, error distinction, resume settings.
- Runtime tests: block prevents side effects/delivery, actual native step routing
  conveys feedback, native LLMAgent state accepts feedback (generation mocked),
  retry limits, stop handling, rejected costs and evaluator replay.
- Neither category proves model accuracy. Do not weaken a failing live semantic
  expectation just to make a test green.

## 6. Opt-in paid validation (not executed automatically)

Controlled text-recovery smoke: one injected false draft after a failed tool
result, real auditor, real agent revisions, fake recipient/environment. At most
three auditor calls and two agent calls; no real order operation executes.

```sh
LIVE_TEXT_RECOVERY_MODEL=openai/gpt-4.1-mini \
  ../tau2-bench/.venv/bin/python -B -m unittest -v \
  tests.test_text_recovery_live
```

Inspect its printed diagnostics to confirm the revision is truthful. A model
allowing its own revision is not independent proof of correctness. The original
six semantic cases are separately opt-in with LIVE_SEMANTIC_MODEL; v3 had a known
failure on an incorrect intermediate result and should not be advertised as green.

One real retail task (agent, user and auditor may all incur costs):

```sh
../tau2-bench/.venv/bin/python -B run_retail.py \
  --audit general-hybrid --audit-llm openai/gpt-4.1-mini \
  --audit-max-tokens 512 --audit-timeout 30 --audit-max-retries 2 \
  --task-id 88 --task-split train --seed 43 --max-steps 30 \
  --output results/my-general-hybrid-retail.json
```

Use a fresh output filename. For comparisons, change only `--audit` and output
path across the three new modes while keeping task IDs, seeds, model settings,
step/retry limits and scope identical. Seed/temperature do not guarantee identical
provider responses. Legacy deterministic/off baselines have different dispatch
semantics; document this rather than claiming a perfectly isolated ablation.

AFTraj small smoke (first three dev trajectories, not a balanced evaluation):

```sh
uv run --no-project --python 3.12 --with-requirements aftraj/requirements-model.txt \
  python -B -m aftraj.run_eval --auditor general-hybrid \
  --model openai/gpt-4.1-mini --env-file ../tau2-bench/.env \
  --split dev --limit 3 --output results/aftraj/my-general-hybrid-smoke
```

`--limit` counts trajectories, not API calls. Default data order is safe-first;
these three are not suitable for estimating recall. Use a fixed representative
safe/unsafe development set before drawing conclusions; tune on dev, reserve test.
Never resume v1/v2 results with a v3 prompt. Avoid interpreting fewer calls caused
by early false blocks as an efficiency improvement.

## 7. Evidence and remaining research work

Observed development checks (not broad performance claims):

- v3 passed 5/6 synthetic semantic cases: one concrete wrong intermediate result
  got warn instead of block. Three safe math trajectories had zero blocks but
  still produced unnecessary warnings.
- Tool-only retail task 88 completed with reward 1, one semantic cancellation
  audit, and no block. The shorter 15-step run reached only read operations.
- Controlled text smoke blocked the injected false success claim; a real agent
  revision truthfully reported failure and passed the second audit. This used an
  explicit truthfulness policy and a fake recipient, not a full retail recovery run.

Remaining work is experimental, plus any separately agreed extensions: select a
representative evaluation set; compare quality/cost and report error coverage;
integrate controlled faults with the new runtime if required; implement genuine
goal-stage checkpoints if the final project claims them; prepare plots/report/demo.
Do not claim that current tests prove hybrid superiority, general recovery success,
or production readiness. The framework handoff is complete within the stated scope.


Handoff verification on 2026-11-23: standard discovery ran 165 tests, with 158
passing and 7 skipped (5 data/PyArrow-dependent checks and 2 explicitly opt-in
live tests). No paid model requests were made during this handoff. CLI help and
`git diff --check` passed. Re-run using your environment rather than treating
these counts as a permanent acceptance threshold.
