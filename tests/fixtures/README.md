# Coding prefix controls

`coding_prefixes.json` contains six normalized development prefixes extracted
from [ZBox008003/AFTraj](https://huggingface.co/datasets/ZBox008003/AFTraj),
distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution: the AFTraj dataset contributors and the
[AgentForesight authors](https://arxiv.org/abs/2605.08715).

We parsed tool calls, linked available results, and supplied the reconstructed
coding workflow context through `aftraj.adapter`, version
`aftraj-coding-workflow-v2`. Original source IDs and zero-based turn indices
are retained for reproducibility. See `docs/aftraj-development-pilot.md` for
context provenance and limitations.

Expected decisions and stages are our development checks, separate from
benchmark metrics. Only `inputs` reaches the auditor. Source IDs, split,
expectations and adapter version are test metadata. Each input contains only
the current proposal, preceding turns, tool catalog and task/policy context.
The verified-completion control includes prior independent review evidence.

These controls were selected after examining development failures. They are
neither an independent evaluation set nor evidence of generalization. No
reserved test trajectories are included. Passing fake-backend tests verifies
serialization and routing, not semantic correctness.
