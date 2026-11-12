"""Summarize the matched task 69 and 74 retail checkpoint experiments."""

import argparse
from pathlib import Path

from tau2.data_model.message import ToolMessage
from tau2.data_model.simulation import Results
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.registry import registry


def load_group(root: Path, task_id: str, split: str, variants: tuple[str, ...]):
    runs = {}
    reference = None
    audit_models = set()
    for variant in variants:
        path = root / f"project-checkpoint-{task_id}-seed43-{variant}" / "results.json"
        data = Results.load(path)
        (task,) = data.tasks
        (sim,) = data.simulations
        info = sim.info or {}
        recovery = info["cancellation_recovery"]
        mode = variant.removesuffix("-no-recovery")
        recovery_enabled = variant == mode
        if (
            sim.task_id != task_id or task.id != task_id or data.info.seed != 43
            or info["audit_mode"] != mode or info.get("task_split", "train") != split
            or info.get("recovery_enabled", True) != recovery_enabled
        ):
            raise ValueError(f"Unexpected trial settings in {path}")
        settings = (
            task, data.info, sim.seed, info["checkpoint_trace"],
            recovery["evidence_sha256"], recovery["original_call"]["arguments"],
            recovery["injected_call"]["arguments"],
        )
        if reference is not None and settings != reference:
            raise ValueError(f"Checkpoint, fault, or trial settings differ in {path}")
        reference = settings
        if mode in ("semantic", "hybrid", "continuous"):
            audit_models.add(info["audit_model"])
        runs[variant] = (task, sim)
    if len(audit_models) != 1:
        raise ValueError(f"Expected one shared audit model for task {task_id}")
    return runs


def metrics(task, sim):
    messages = sim.get_messages()
    info = sim.info
    recovery = info["cancellation_recovery"]
    results = {msg.id: msg for msg in messages if isinstance(msg, ToolMessage)}
    canceled = {
        call.arguments["order_id"]
        for msg in messages
        for call in getattr(msg, "tool_calls", None) or []
        if call.name == "cancel_pending_order" and not results[call.id].error
    }
    checks = info.get("semantic_checks", [])
    tokens = [check[key] for check in checks for key in ("input_tokens", "output_tokens")]
    reward = sim.reward_info
    db_check = reward.db_check
    if db_check is None:
        db_check = EnvironmentEvaluator.calculate_reward(
            environment_constructor=registry.get_env_constructor("retail"),
            task=task, full_trajectory=messages, solo_mode=False,
        ).db_check
    return (
        recovery["status"],
        "Yes" if recovery["injected_call"]["arguments"]["order_id"] in canceled else "No",
        "Yes" if recovery["original_call"]["arguments"]["order_id"] in canceled else "No",
        f"{reward.reward:g}",
        "Yes" if db_check.db_match else "No",
        len(checks),
        sum(tokens) if all(token is not None for token in tokens) else "—",
        f'{sum(check["latency_ms"] for check in checks) / 1000:.2f}',
    )


def table(labels, rows):
    columns = (*labels, "Outcome", "Wrong order canceled", "Authorized order canceled",
               "Reward", "DB match", "LLM audit calls", "Audit tokens", "Audit seconds")
    lines = (columns, ("---",) * len(columns), *rows)
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("data/simulations"))
    args = parser.parse_args()
    modes = ("off", "rules-only", "semantic", "hybrid", "continuous")
    ablation = ("hybrid-no-recovery", "hybrid")
    try:
        task69 = load_group(args.results_root, "69", "train", ablation)
        task74 = load_group(
            args.results_root / "project-heldout-retail",
            "74", "test", (*modes, ablation[0]),
        )
        rows69 = {name: metrics(*run) for name, run in task69.items()}
        rows74 = {name: metrics(*run) for name, run in task74.items()}
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot summarize checkpoint results: {exc}\n")
    print("## Five-mode comparison on task 74\n")
    print(table(("Audit mode",), [(mode, *rows74[mode]) for mode in modes]))
    print("\n## Hybrid recovery ablation\n")
    print(table(("Task", "Recovery"), [
        (task_id, label, *rows[variant])
        for task_id, rows in ((69, rows69), (74, rows74))
        for label, variant in zip(("Disabled", "Enabled"), ablation)
    ]))


if __name__ == "__main__":
    main()
