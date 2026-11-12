# Selective Auditing and Local Recovery for LLM Agent Tool-Calling Loops

COMSE6998_007 Course Project

## Team
- Tzu-An Cheng (tc3497@columbia.edu)
- Binyan Sun (bs3654@columbia.edu)
- Edwin Argudo (eaa2193@columbia.edu)

## Run the retail pilot

Use the sibling `tau2-bench` checkout at `v1.0.1` (commit `fc0055d`). Run
`uv sync --frozen` there and set `OPENAI_API_KEY` in your shell or the benchmark's
local `.env` file. From `tau2-bench`, run one trial with each audit mode:

```sh
uv run --no-sync python ../COMSE6998_007_Course_Project/run_retail.py \
  --audit off --output data/simulations/project-baseline-88/results.json

uv run --no-sync python ../COMSE6998_007_Course_Project/run_retail.py \
  --audit deterministic --output data/simulations/project-deterministic-88/results.json

uv run --no-sync tau2 view \
  --file data/simulations/project-deterministic-88/results.json
```

Both commands default to training task `88`, seed `43`, and the same models and
step limit as the baseline smoke run. The deterministic mode currently checks
cancellation arguments. Blocked proposals appear under `info.audit_rejections`
inside the simulation record. Use a fresh output path for each run.

### Test live cancellation recovery

From `tau2-bench`, run the fault-injection diagnostic:

```sh
uv run --no-sync python -B ../COMSE6998_007_Course_Project/run_retail.py \
  --audit deterministic --inject-cancellation-fault \
  --output data/simulations/project-recovery-88/results.json
```

The runner replaces the first valid cancellation reason with an invalid one.
After receiving the audit feedback, the agent can propose one cancellation retry
for the same order. A failed retry ends the run. The result records the
original proposal, the retry if any, and the outcome in
`info.cancellation_recovery`. Check both the recovery status and the native
reward. Keep injected runs separate from clean benchmark results.

### Replay the semantic auditor

From `tau2-bench`, compare the current argument rule and the semantic auditor
using the successful seed-43 baseline trace:

```sh
uv run --no-sync python -B ../COMSE6998_007_Course_Project/replay_cancellation_audit.py \
  data/simulations/retail-baseline-88-seed43/results.json
```

The replay sends two proposals against the same pre-cancellation history. It
expects `allow` for the confirmed order and `block` for a different pending
order. It prints each verdict and latency as JSON, along with token usage for
the semantic auditor. The command exits nonzero if either semantic verdict
differs from its expected verdict. Add `--dry-run` to inspect the cases and the
deterministic verdicts without making API calls.

### Run the general deterministic auditor

`--audit deterministic-general` audits every assistant call to a tool that
τ²-bench marks as `WRITE`, not only cancellations. Before each such call it
rebuilds facts from the conversation so far and checks policy rules the domain
tools do not enforce:

| Rule | Blocks a write call when |
| --- | --- |
| `D1_authenticated` | no user has been authenticated |
| `D2_owned_by_user` | the target order or account belongs to another user |
| `D3_record_observed` | the target order was never retrieved |
| `D4_grounded_ids` | an `*_id` / `*_ids` argument appears in no tool result or user message |
| `D5_one_call_at_a_time` | the write is proposed together with other calls |
| `D6_repeated_failure` | the identical call already failed or was blocked twice |

Facts come only from tool results and user messages, not from assistant text.
Every audited call, with its verdict, rule, and fact snapshot, is saved under
`info.deterministic_checks`. The original `deterministic` mode is unchanged.

From `tau2-bench`:

```sh
uv run --no-sync python -B ../COMSE6998_007_Course_Project/run_retail.py \
  --audit deterministic-general --output data/simulations/project-general-88/results.json

# Replay the auditor over saved results without calling any model
uv run --no-sync python -B ../COMSE6998_007_Course_Project/replay_deterministic_audit.py \
  data/simulations/*/results.json

# Unit tests
uv run --no-sync python -B -m unittest discover \
  -s ../COMSE6998_007_Course_Project/tests -t ../COMSE6998_007_Course_Project
```

### Compare the same wrong-order fault across audit modes

From `tau2-bench`:

```sh
for mode in off deterministic semantic; do
  output="data/simulations/project-wrong-order-88-seed43-$mode/results.json"
  if [ -f "$output" ]; then
    echo "Keeping existing result: $output"
    continue
  fi
  uv run --no-sync python -B ../COMSE6998_007_Course_Project/run_retail.py \
    --task-id 88 --seed 43 --audit "$mode" --inject-wrong-order \
    --output "$output" || break
done
```

The loop skips existing result files. The diagnostic substitutes a different
observed pending order. A block gives the agent one cancellation retry for
the original order. Semantic mode calls
the model only for assistant cancellations and blocks on abstention.

Compare the injected arguments, `injected_tool_succeeded`, and recovery status in
`info.cancellation_recovery`, alongside the native reward and database match.
`fault_allowed` means the injected proposal reached dispatch.
`injected_tool_succeeded` is null when the proposal never reached the tool.
If no eligible cancellation was proposed, the status is `not_triggered`.
Semantic verdicts, token usage, and latency are in `info.semantic_checks`.

The shared seed holds the run configuration fixed, but live conversations can
differ. Compare `evidence_sha256` to see whether the dialogue and order status
presented to the auditor were identical across runs.

### Run the cancellation suite

From `tau2-bench`, inspect the two clean baselines, then collect the other
clean runs with auditing off:

```sh
uv run --no-sync python -B ../COMSE6998_007_Course_Project/run_retail_suite.py \
  --tasks 66 59 --conditions clean --modes off --summary-only

uv run --no-sync python -B ../COMSE6998_007_Course_Project/run_retail_suite.py \
  --conditions clean --modes off
```

The full suite uses tasks 66, 69, 59, 78, and 98. Omit the mode and condition
filters to run all 30 trials. The runner reuses saved results with matching
settings. Use a separate `--output-root` to rerun the same seed with a different
step limit. Use `--summary-only` to see completed and missing trials without
calling any models.

Review clean rejections against the conversation before classifying them as
false blocks. On injected trials, keep `not_triggered` separate from runs where
the fault was injected. The table reports native reward, termination, tool and
recovery outcomes, and semantic audit tokens and latency.
