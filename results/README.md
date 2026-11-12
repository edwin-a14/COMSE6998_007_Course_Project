# Results

Each folder holds one experiment: a write-up and the τ²-bench `results.json`
traces it cites, so every number can be checked without rerunning a model.

| Folder | What it shows |
| --- | --- |
| [`general_deterministic_audit/`](general_deterministic_audit/README.md) | Pilot runs of `--audit deterministic-general`: D4 stops a cancellation of another user's order, D3 catches an agent cancelling an order it never retrieved, and D5 warnings no longer cost runs |

Results for the cancellation audit modes are in
[`../RETAIL_RESULTS.md`](../RETAIL_RESULTS.md).
