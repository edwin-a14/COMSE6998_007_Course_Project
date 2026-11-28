# Frozen AFTraj balanced development comparison

This comparison measures detection and usage after prompt, routing and coding
context development. The implementation is frozen at semantic-evidence-v6,
writes-text-computation-v2 and aftraj-coding-workflow-v3. Known false alarms
and unstable judgments remain documented in the
[development log](aftraj-development-pilot.md). This is development evidence,
not the reserved test evaluation or a claim of improved model quality.

## Sample and settings

The [committed manifest](../experiments/aftraj/dev-balanced60-seed45.json)
records all 60 IDs, their ordered content fingerprint, settings and routing
coverage. The sample was drawn before inspecting its model outcomes and was
not filtered to remove known difficult cases.

| Setting | Frozen value |
| --- | --- |
| Split | dev |
| Domains | agentic, coding, math |
| Sample | 10 safe + 10 unsafe per domain; 60 total |
| Sample seed | 45 |
| Model | openai/gpt-4.1-mini |
| Temperature | 0 |
| Maximum output tokens | 512 per audit |
| Request timeout | 30 seconds |
| Prompt | semantic-evidence-v6 |
| Selective/hybrid routing | writes-text-computation-v2 |
| Adapter/context | aftraj-coding-workflow-v3 |
| Hybrid rules | generic |
| Scan | stop at first block; no full scan |

There are 60 distinct trajectory sources. One trajectory and task,
math_0384_unsafe_injected_static, appeared in the earlier development pilot.
It remains in this sample and is listed in the manifest. The loader's dev
split excludes sources and tasks shared with the reserved test split.

Before early stopping, routing eligibility is:

| Domain | Agent proposals | Selected proposals | Labeled agent mistakes selected |
| --- | --- | --- | --- |
| Agentic | 84 | 79 | 10/10 |
| Coding | 128 | 111 | 10/10 |
| Math | 208 | 208 | 10/10 |
| Total | 420 | 398 | 30/30 |

Coverage means a proposal reaches semantic review, not that the model catches
its mistake. Only 22 proposals are skipped on a full traversal, so routing
alone offers limited call savings on this sample. Early stopping can reduce
actual calls substantially, including through false blocks. Report that
effect alongside quality. The three model runs make at most 1,216 requests
in total under the frozen default transport with no retries; early stopping
and hybrid rule blocks can reduce that count. Request counts are not a price
or latency estimate.

## Run from the repository root

Use the downloaded data/aftraj files described in the
[AFTraj setup guide](../aftraj/README.md#setup). Model runs need credentials
in the environment or ../tau2-bench/.env; existing environment values take
precedence over that file.

The local allow-all and rules baselines have completed: 60/60 trajectories
each, zero failures, zero blocks, zero Exact-F1 and zero false-alarm rate.
Both visited all 420 agent proposals. Outputs are ignored by Git and can be
reproduced with:

```sh
for mode in allow-all rules; do
  uv run --no-project --python 3.12 \
    --default-index https://pypi.org/simple \
    --with-requirements aftraj/requirements.txt \
    python -B -m aftraj.run_eval --auditor "$mode" \
    --split dev --sample-per-group 10 --sample-seed 45 --resume \
    --output "results/aftraj/dev-balanced60-seed45-v6-routing-v2-context-v3/$mode" || break
done
```

Use these flags and versioned components for the three model modes. The
execution status below records completed attempts separately:

```sh
for mode in selective-semantic general-hybrid continuous-semantic; do
  uv run --no-project --python 3.12 \
    --default-index https://pypi.org/simple \
    --with-requirements aftraj/requirements-model.txt \
    python -B -m aftraj.run_eval --auditor "$mode" \
    --model openai/gpt-4.1-mini --env-file ../tau2-bench/.env \
    --max-tokens 512 --timeout 30 \
    --split dev --sample-per-group 10 --sample-seed 45 --resume \
    --output "results/aftraj/dev-balanced60-seed45-v6-routing-v2-context-v3/$mode" || break
done
```

`--resume` skips already saved rows with matching settings and continues an
interrupted run. It also skips saved error rows. If any trajectory fails, the
runner saves its diagnostics and exits nonzero, stopping the loop. Inspect
the error before an explicit rerun in a fresh output root. Do not delete or
overwrite original failed attempts or silently merge their usage away.
Changing versions, the sample, or model settings requires separate outputs.

## Compare after all five modes finish

```sh
uv run --no-project --python 3.12 \
  python -B -m aftraj.compare \
  results/aftraj/dev-balanced60-seed45-v6-routing-v2-context-v3/{allow-all,rules,selective-semantic,general-hybrid,continuous-semantic}
```

The comparison checks cohort/settings compatibility and uses the common
completed set for detection metrics. Usage includes failed attempts. Each
mode's results.json also contains domain-specific metrics and failure counts.
Inspect individual first blocks in per_sample.jsonl before interpreting
aggregate differences between modes. Preserve the dataset's original labels.

At implementation handoff, only the offline baselines and routing scan were
complete. Live evaluation and analysis are experiment work after that handoff.

## First selective-semantic attempt

The November 28 run saved all 60 rows: 59 completed and one auditor error.
gaia_0039_safe failed at Manager step 1 with incomplete_response,
finish_reason=length and 512 output tokens, the frozen maximum. This was a
truncated audit response. It was correctly excluded from detection metrics,
with its usage retained. No agent mistake verdict is inferred from the error.

| Domain | Completed | Exact-F1 % | Exact recall % | FAR % |
| --- | --- | --- | --- | --- |
| Math | 20 | 52.63 | 50.00 | 30.00 |
| Coding | 20 | 52.63 | 50.00 | 20.00 |
| Agentic | 19 | 0.00 | 0.00 | 22.22 |
| Overall | 59 | 40.00 | 33.33 | 24.14 |

Usage including the failed request: 331 model calls, 636,978 input tokens,
37,754 output tokens and 594.79 summed model seconds, with 18 selector skips.
Of the 30 unsafe trajectories, ten were blocked at the labeled step, three
were blocked at a different step, and seventeen had no block. Seven of the
29 completed safe trajectories were blocked. All ten unsafe agentic
trajectories had no block.

Hybrid and continuous have not started because the shell loop stopped on
selective's nonzero exit. Continue them with the same frozen settings:

```sh
for mode in general-hybrid continuous-semantic; do
  uv run --no-project --python 3.12 \
    --default-index https://pypi.org/simple \
    --with-requirements aftraj/requirements-model.txt \
    python -B -m aftraj.run_eval --auditor "$mode" \
    --model openai/gpt-4.1-mini --env-file ../tau2-bench/.env \
    --max-tokens 512 --timeout 30 \
    --split dev --sample-per-group 10 --sample-seed 45 --resume \
    --output "results/aftraj/dev-balanced60-seed45-v6-routing-v2-context-v3/$mode" || break
done
```

The original three-mode loop would stop again at selective because resume
retains saved errors. Keep this output intact. The final comparison will
use the common completed cohort, at most 59 trajectories given this error,
and retain all attempted usage. Changing the token limit would be a separate
experiment with fresh outputs and matching settings across model modes.
