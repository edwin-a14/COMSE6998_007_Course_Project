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

