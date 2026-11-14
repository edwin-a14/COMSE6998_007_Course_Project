# Results

Each folder holds one experiment: a write-up and the τ²-bench `results.json`
traces it cites, so every number can be checked without rerunning a model.

| Folder | What it shows |
| --- | --- |
| [`general_deterministic_audit/`](general_deterministic_audit/README.md) | Pilot runs of `--audit deterministic-general`: D4 stops a cancellation of another user's order, D3 catches an agent cancelling an order it never retrieved, and D5 warnings no longer cost runs |
| [`clean_run_false_positives/`](clean_run_false_positives/README.md) | All 40 retail test tasks with and without `deterministic-general`: 3 of 49 write calls blocked, none of them correct calls |

The traces were recorded before D2 was tied to the session user, the first
user authenticated in a conversation. Their fact snapshots still list
`authenticated_users` instead of `session_user` and `other_users`. Every
snapshot names exactly one user, and replaying all traces with the current
auditor gives the same verdicts and warnings.

Results for the cancellation audit modes are in
[`../RETAIL_RESULTS.md`](../RETAIL_RESULTS.md).
