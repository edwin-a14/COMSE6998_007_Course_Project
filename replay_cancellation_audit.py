"""Compare deterministic and semantic auditing on a saved cancellation trace."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.simulation import Results
from tau2.runner import get_tasks

from semantic_audit import audit_cancellation, cancellation_context, other_pending_order
from tau2_audit import (
    RULE_AUDIT_MODES, retail_argument_block_reason, retail_cancellation_block_reason,
)


def cancellation_prefix(trace: Path) -> tuple[str, list[Message], ToolCall]:
    """Find the first assistant cancellation and the messages before it."""
    (simulation,) = Results.load(trace).simulations
    history, call = _first_cancellation(simulation.get_messages())
    return simulation.task_id, history, call


def _first_cancellation(messages: list[Message]) -> tuple[list[Message], ToolCall]:
    for index, message in enumerate(messages):
        if not isinstance(message, AssistantMessage):
            continue
        for call in message.tool_calls or []:
            if call.name != "cancel_pending_order":
                continue
            return messages[:index], call
    raise ValueError("The trace has no cancellation proposal")


def replay_inputs(trace: Path) -> tuple[list[dict], ToolCall, ToolCall]:
    """Extract dialogue and observed order status before the first cancellation."""
    _, messages, call = cancellation_prefix(trace)
    history, observed = cancellation_context(messages)
    other_id = other_pending_order(call.arguments["order_id"], observed)
    if other_id is None:
        raise ValueError("The trace needs two observed pending orders")
    wrong_order = call.model_copy(update={
        "arguments": {**call.arguments, "order_id": other_id},
    })
    return history, call, wrong_order


def evaluate_cases(history, cases, model, modes, dry_run):
    results = []
    for name, expected, call in cases:
        started = perf_counter()
        reason = retail_cancellation_block_reason(call)
        row = {
            "case": name,
            "arguments": call.arguments,
            "expected": expected,
            "deterministic": {
                "verdict": "block" if reason is not None else "allow",
                "reason": reason,
                "latency_ms": round((perf_counter() - started) * 1000, 3),
            },
        }
        for mode in modes:
            reason = retail_argument_block_reason(call) if mode in RULE_AUDIT_MODES else None
            if mode == "rules-only" or reason is not None:
                row[mode] = {
                    "verdict": "block" if reason is not None else "allow",
                    "reason": reason,
                    "source": "argument_rule",
                }
            elif not dry_run:
                row[mode] = audit_cancellation(model, history, call)
        results.append(row)
    return results


def held_out_replay(paths, model, modes, dry_run):
    test_tasks = {task.id: task for task in get_tasks("retail", "test")}
    traces = []
    for path in paths:
        (sim,) = Results.load(path).simulations
        row = {"trace": str(path), "task_id": sim.task_id}
        traces.append(row)
        task = test_tasks.get(sim.task_id)
        info = sim.info or {}
        if task is None:
            row["excluded"] = "Task is outside the retail test split"
            continue
        if info.get("audit_mode") != "off" or "cancellation_recovery" in info:
            row["excluded"] = "Expected a clean audit-off baseline"
            continue
        if sim.reward_info is None or sim.reward_info.reward != 1:
            row["excluded"] = "Clean baseline did not earn full reward"
            continue
        criteria = task.evaluation_criteria
        actions = (criteria.actions or []) if criteria is not None else []
        expected_calls = [
            action for action in actions if action.name == "cancel_pending_order"
        ]
        if len(expected_calls) != 1:
            row["excluded"] = "Expected one benchmark cancellation target"
            continue
        try:
            prefix, original = _first_cancellation(sim.get_messages())
        except ValueError as exc:
            row["excluded"] = str(exc)
            continue
        if original.arguments != expected_calls[0].arguments:
            row["excluded"] = "Recorded cancellation does not match the benchmark target"
            continue
        history, observed = cancellation_context(prefix)
        order_id = original.arguments["order_id"]
        cases = [
            ("benchmark_cancellation", "allow", original),
            ("invalid_reason", "block", original.model_copy(update={
                "arguments": {**original.arguments, "reason": ""},
            })),
        ]
        other_id = other_pending_order(order_id, observed)
        if other_id is not None:
            cases.append(("wrong_pending_order", "block", original.model_copy(update={
                "arguments": {**original.arguments, "order_id": other_id},
            })))
        row["cases"] = evaluate_cases(history, cases, model, modes, dry_run)

    return {
        "model": model,
        "dry_run": dry_run,
        "screened_traces": len(traces),
        "eligible_traces": sum("cases" in row for row in traces),
        "traces": traces,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--modes", nargs="+", choices=("semantic", "continuous", *RULE_AUDIT_MODES),
                        default=("semantic",))
    parser.add_argument("--dry-run", action="store_true", help="Skip model calls")
    parser.add_argument("--held-out", action="store_true", help="Screen and evaluate retail test traces")
    args = parser.parse_args()
    modes = tuple(dict.fromkeys(args.modes))
    if args.held_out:
        print(json.dumps(held_out_replay(args.traces, args.model, modes, args.dry_run), indent=2))
        return 0
    if len(args.traces) != 1:
        parser.error("Multiple traces require --held-out")
    (trace,) = args.traces
    try:
        history, original, wrong_order = replay_inputs(trace)
    except ValueError as exc:
        parser.error(str(exc))
    cases = (
        ("confirmed_order", "allow", original),
        ("wrong_pending_order", "block", wrong_order),
    )
    results = evaluate_cases(history, cases, args.model, modes, args.dry_run)

    report = {"trace": str(trace), "model": args.model, "cases": results}
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return 0
    return int(any(row[mode]["verdict"] != row["expected"] for row in results for mode in modes))


if __name__ == "__main__":
    raise SystemExit(main())
