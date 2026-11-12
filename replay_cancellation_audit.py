"""Compare deterministic and semantic auditing on a saved cancellation trace."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.simulation import Results

from semantic_audit import audit_cancellation, cancellation_context, other_pending_order
from tau2_audit import retail_cancellation_block_reason


def cancellation_prefix(trace: Path) -> tuple[str, list[Message], ToolCall]:
    """Find the first assistant cancellation and the messages before it."""
    (simulation,) = Results.load(trace).simulations
    messages = simulation.get_messages()
    for index, message in enumerate(messages):
        if not isinstance(message, AssistantMessage):
            continue
        for call in message.tool_calls or []:
            if call.name != "cancel_pending_order":
                continue
            return simulation.task_id, messages[:index], call
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--dry-run", action="store_true", help="Skip model calls")
    args = parser.parse_args()
    try:
        history, original, wrong_order = replay_inputs(args.trace)
    except ValueError as exc:
        parser.error(str(exc))

    results = []
    for name, expected, call in (
        ("confirmed_order", "allow", original),
        ("wrong_pending_order", "block", wrong_order),
    ):
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
        if not args.dry_run:
            row["semantic"] = audit_cancellation(args.model, history, call)
        results.append(row)

    report = {"trace": str(args.trace), "model": args.model, "cases": results}
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return 0
    return int(any(row["semantic"]["verdict"] != row["expected"] for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
