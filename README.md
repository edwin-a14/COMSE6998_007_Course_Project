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

### Compare audit modes from a saved checkpoint

From `tau2-bench`, start each mode from the same task 69 conversation just before
the cancellation. The runner forces one wrong-order proposal and uses live
participants for the following turns.

```sh
for mode in off deterministic semantic; do
  uv run --no-sync python -B ../COMSE6998_007_Course_Project/run_retail.py \
    --task-id 69 --seed 43 --max-steps 60 --audit "$mode" \
    --checkpoint data/simulations/project-suite-clean-69-seed43-off/results.json \
    --output "data/simulations/project-checkpoint-69-seed43-$mode/results.json" || break
done
```

Check that `evidence_sha256` and the injected arguments in
`info.cancellation_recovery` match across the three files. Compare the audit
verdicts, executed writes, recovery status, and reward. The step limit and
agent and user model costs apply to the continuation. Semantic audit tokens
and latency are recorded in `info.semantic_checks`.

### Compare selective and continuous auditing

From `tau2-bench`, run both modes on full retail conversations:

```sh
uv run --no-sync python -B ../COMSE6998_007_Course_Project/run_retail_suite.py \
  --tasks 69 88 --modes semantic continuous --conditions clean
```

Both modes use the same cancellation prompt, audit model, and recovery
behavior. Continuous mode also reviews lookups and other tool proposals before
dispatch. It receives the proposed tool's definition and earlier dialogue and
tool results, with each result capped at 4,000 characters. The suite reports
reward, audit call counts, tokens, and latency.

To include continuous auditing in the checkpoint comparison, add
`continuous` to the loop's mode list. Only calls proposed after the checkpoint
contribute to that run's audit counts.

### Replay both modes on the same tool proposals

From `tau2-bench`, pass saved retail traces to the replay command:

```sh
uv run --no-sync python -B ../COMSE6998_007_Course_Project/replay_tool_audit.py \
  data/simulations/project-suite-clean-69-seed43-off/results.json \
  data/simulations/retail-baseline-88-seed43/results.json
```

Each proposal uses the recorded dialogue and tool results from before its
batch. The report includes each verdict, audit call counts, input and output
tokens, and summed audit latency. The trace's recorded tool errors provide
context for interpreting verdicts. Replay checks proposals without executing
tools or changing the saved conversation. Add `--dry-run` to see the planned
checks without model calls.

### Compare hybrid auditing

From `tau2-bench`, compare the three model-based modes on the same recorded
tool proposals and on the cancellation replay with the substituted order:

```sh
uv run --no-sync python -B ../COMSE6998_007_Course_Project/replay_tool_audit.py \
  --modes semantic continuous hybrid -- \
  data/simulations/project-suite-clean-69-seed43-off/results.json \
  data/simulations/retail-baseline-88-seed43/results.json

uv run --no-sync python -B ../COMSE6998_007_Course_Project/replay_cancellation_audit.py \
  data/simulations/project-suite-clean-69-seed43-off/results.json \
  --modes semantic continuous hybrid
```

Hybrid and `rules-only` both block missing or blank ZIP codes on name lookups
and check cancellation arguments. Hybrid sends cancellations that pass the
rules to the semantic auditor. Add `rules-only` to either replay's mode list
to include the rules-only baseline. Replay reports rule blocks alongside
model verdicts. Audit calls, tokens, and latency count only model checks.
The cancellation replay exits with status 1 when any selected verdict
differs from the expected verdict, including when `rules-only` allows the
wrong order.

Run the live hybrid checkpoint with the same task, seed, and trace used for
the earlier controls:

```sh
uv run --no-sync python -B ../COMSE6998_007_Course_Project/run_retail.py \
  --task-id 69 --seed 43 --max-steps 60 --audit hybrid \
  --checkpoint data/simulations/project-suite-clean-69-seed43-off/results.json \
  --output data/simulations/project-checkpoint-69-seed43-hybrid/results.json
```

For the rules-only checkpoint, use `--audit rules-only` and the output path
`data/simulations/project-checkpoint-69-seed43-rules-only/results.json`.
