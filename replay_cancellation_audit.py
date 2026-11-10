"""Compare deterministic and semantic auditing on a saved cancellation trace."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import Results

from semantic_audit import audit_cancellation
from tau2_audit import retail_cancellation_block_reason


def replay_inputs(trace: Path) -> tuple[list[dict], ToolCall, ToolCall]:
    """Extract dialogue and observed order status before the first cancellation."""
    (simulation,) = Results.load(trace).simulations
    history = []
    order_lookups = set()
    observed_status = {}
    for message in simulation.get_messages():
        if isinstance(message, (AssistantMessage, UserMessage)):
            for call in message.tool_calls or []:
                if call.name == "cancel_pending_order":
                    order_id = call.arguments["order_id"]
                    if observed_status.get(order_id) != "pending":
                        raise ValueError("The original order must be observed as pending")
                    other_id = next(
                        (
                            key for key, status in observed_status.items()
                            if key != order_id and status == "pending"
                        ),
                        None,
                    )
                    if other_id is None:
                        raise ValueError("The trace needs another observed pending order")
                    wrong_order = call.model_copy(update={
                        "arguments": {**call.arguments, "order_id": other_id},
                    })
                    return history, call, wrong_order
                if call.name == "get_order_details":
                    order_lookups.add(call.id)
            if message.content:
                history.append({"role": message.role, "content": message.content})
        elif (
            isinstance(message, ToolMessage)
            and message.id in order_lookups
            and not message.error
        ):
            order = json.loads(message.content)
            observed_status[order["order_id"]] = order["status"]
            history.append({
                "role": "tool",
                "tool": "get_order_details",
                "order_id": order["order_id"],
                "status": order["status"],
            })
    raise ValueError("The trace has no cancellation proposal")


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
