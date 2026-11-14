# General deterministic audit on clean retail test runs

Tasks: 40 unaudited runs, 40 audited runs.

## Blocks by rule

| Rule | Live blocks | of which expected calls | Replay blocks on `off` | of which expected calls |
| --- | --- | --- | --- | --- |
| `D3_record_observed` | 1 | 0 | 0 | 0 |
| `D4_grounded_ids` | 2 | 0 | 0 | 0 |

Warnings in audited runs: {'D5_one_call_at_a_time': 24}

## Blocks in audited runs

| Task | Rule | Blocked call | Expected call | Retried and allowed | Run reward |
| --- | --- | --- | --- | --- | --- |
| 36 | `D4_grounded_ids` | `modify_pending_order_items(order_id=#W9348897)` | no | yes | 0.0 |
| 39 | `D3_record_observed` | `modify_pending_order_address(order_id=#W5285031)` | no | no | 0.0 |
| 55 | `D4_grounded_ids` | `return_delivered_order_items(order_id=#W4597054)` | no | yes | 1.0 |

## Replay over unaudited runs

| Task | Rule | Executed call | Expected call | First block in run | Run reward |
| --- | --- | --- | --- | --- | --- |

## Rewards on the same tasks

| Mode | Tasks | Mean reward | Terminations |
| --- | --- | --- | --- |
| `off` | 40 | 0.575 | {'user_stop': 40} |
| `deterministic-general` | 40 | 0.625 | {'user_stop': 40} |

Tasks whose reward differs (off → audited), and whether the audit blocked anything:

- task 33: 1.0 → 0.0, no block
- task 53: 1.0 → 0.0, no block
- task 60: 0.0 → 1.0, no block
- task 64: 0.0 → 1.0, no block
- task 70: 1.0 → 0.0, no block
- task 77: 0.0 → 1.0, no block
- task 90: 0.0 → 1.0, no block
- task 97: 0.0 → 1.0, no block
- task 100: 0.0 → 1.0, no block
- task 101: 0.0 → 1.0, no block
- task 102: 1.0 → 0.0, no block
- task 108: 1.0 → 0.0, no block
