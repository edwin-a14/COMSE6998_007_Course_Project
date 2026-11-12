"""Run or summarize the retail cancellation pilot from the tau2-bench checkout."""

import argparse
import json
from itertools import product
from pathlib import Path
import subprocess
import sys

from replay_cancellation_audit import replay_inputs
from tau2_audit import retail_cancellation_block_reason


TASKS = (66, 69, 59, 78, 98)
MODES = ("off", "deterministic", "semantic")
CONDITIONS = ("clean", "injected")
MODEL = "openai/gpt-4.1-mini"
RUNNER = Path(__file__).with_name("run_retail.py").resolve()


def load_result(path, task, mode, condition, seed, max_steps, task_split="train"):
    data = json.loads(path.read_text())
    (simulation,) = data["simulations"]
    config = data["info"]
    info = simulation.get("info") or {}
    if (
        simulation["task_id"] != str(task)
        or info.get("audit_mode") != mode
        or info.get("task_split", "train") != task_split
        or config["seed"] != seed
        or config["max_steps"] != max_steps
        or config["agent_info"]["llm"] != MODEL
        or config["user_info"]["llm"] != MODEL
        or (mode in ("semantic", "continuous", "hybrid") and info.get("audit_model") != MODEL)
        or ("cancellation_recovery" in info) != (condition == "injected")
    ):
        raise ValueError(f"Existing result has different trial settings: {path}")
    return simulation


def clean_outcome(path, simulation):
    calls = (
        call for message in simulation["messages"]
        for call in message.get("tool_calls") or []
    )
    if not any(call["name"] == "cancel_pending_order" for call in calls):
        return "No dispatched cancellation"
    try:
        _, call, _ = replay_inputs(path)
    except ValueError:
        return "No eligible pending-order pair"
    if retail_cancellation_block_reason(call) is not None:
        return "Invalid cancellation arguments"
    return "Pending-order pair observed"


def summarize(path, simulation, condition):
    info = simulation["info"]
    reward = simulation.get("reward_info") or {}
    db_match = (reward.get("db_check") or {}).get("db_match")
    recovery = info.get("cancellation_recovery") or {}
    checks = info.get("semantic_checks") or []
    tokens = [check.get(key) for check in checks for key in ("input_tokens", "output_tokens")]
    return (
        reward.get("reward"),
        db_match,
        simulation["termination_reason"],
        clean_outcome(path, simulation) if condition == "clean" else recovery["status"],
        recovery.get("injected_tool_succeeded"),
        len(info.get("audit_rejections") or []),
        len(checks),
        sum(tokens) if all(token is not None for token in tokens) else None,
        round(sum(check["latency_ms"] for check in checks), 2),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", type=int, default=TASKS)
    parser.add_argument("--task-split", choices=("train", "test", "base"), default="train")
    parser.add_argument(
        "--modes", nargs="+", choices=(*MODES, "continuous", "rules-only", "hybrid"),
        default=MODES,
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--output-root", type=Path, default=Path("data/simulations"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    columns = (
        "Task", "Condition", "Audit", "Reward", "DB match", "Termination",
        "Outcome", "Injected tool succeeded", "Rejected calls", "Audit calls", "Audit tokens", "Audit ms",
    )
    print("| " + " | ".join(columns) + " |", flush=True)
    print("| " + " | ".join("---" for _ in columns) + " |", flush=True)
    for task, condition, mode in product(
        dict.fromkeys(args.tasks), dict.fromkeys(args.conditions), dict.fromkeys(args.modes)
    ):
        path = args.output_root / f"project-suite-{condition}-{task}-seed{args.seed}-{mode}" / "results.json"
        if not path.exists() and not args.summary_only:
            command = [
                sys.executable, "-B", str(RUNNER), "--task-id", str(task),
                "--task-split", args.task_split,
                "--audit", mode, "--seed", str(args.seed), "--max-steps", str(args.max_steps),
                "--agent-llm", MODEL, "--user-llm", MODEL, "--audit-llm", MODEL,
                "--output", str(path),
            ]
            if condition == "injected":
                command.append("--inject-wrong-order")
            subprocess.run(command, stdout=sys.stderr, check=True)
        if path.exists():
            try:
                simulation = load_result(
                    path, task, mode, condition, args.seed, args.max_steps, args.task_split,
                )
                metrics = summarize(path, simulation, condition)
            except (KeyError, ValueError) as exc:
                parser.exit(1, f"Cannot use {path}: {exc}\n")
        else:
            metrics = (None, None, None, "Missing", None, None, None, None, None)
        row = (task, condition, mode, *metrics)
        print("| " + " | ".join("—" if value is None else str(value) for value in row) + " |", flush=True)


if __name__ == "__main__":
    main()
