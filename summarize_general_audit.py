"""Measure how often the general deterministic audit blocks correct calls.

Reads clean runs (no injected fault) laid out as `<root>/<mode>/task<ID>/results.json`,
where `<mode>` is `off` or `deterministic-general`. A blocked call that matches one of
the task's expected actions in τ²-bench is counted as a false positive candidate.

Three views are printed as Markdown:

1. Blocks in the audited runs, by rule, and whether each blocked call was expected.
2. A replay of the auditor over the unaudited runs, without calling any model: which
   writes the agent executed there would have been blocked. Only the first block in a
   run is a faithful counterfactual, since a block changes what the agent does next.
3. Rewards of the two modes on the same tasks.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import Results
from tau2.runner import build_environment, get_tasks

from deterministic_audit import DOMAIN_CONFIGS, audit_call, write_tool_checker

AUDITED = "deterministic-general"
TARGET_ARGS = ("order_id", "user_id")


def load_runs(root: Path, mode: str) -> dict[str, dict]:
    runs = {}
    for path in sorted((root / mode).glob("task*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        (simulation,) = data["simulations"]
        if (simulation.get("info") or {}).get("cancellation_recovery") is not None:
            raise ValueError(f"{path} has an injected fault; this summary needs clean runs")
        runs[simulation["task_id"]] = {"path": path, "simulation": simulation}
    return runs


def expected(call: dict | ToolCall, actions) -> bool:
    tool_call = call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
    return any(action.compare_with_tool_call(tool_call) for action in actions)


def describe(call: dict) -> str:
    arguments = call["arguments"]
    target = ", ".join(f"{key}={arguments[key]}" for key in TARGET_ARGS if key in arguments)
    return f"{call['name']}({target})"


def reward(simulation: dict) -> float | None:
    return (simulation.get("reward_info") or {}).get("reward")


def audited_blocks(runs: dict[str, dict], actions: dict) -> list[dict]:
    blocks = []
    for task_id, run in runs.items():
        decisions = run["simulation"]["info"].get("deterministic_checks") or []
        for index, decision in enumerate(decisions):
            if decision["verdict"] != "block":
                continue
            call = decision["call"]
            target = {key: call["arguments"].get(key) for key in TARGET_ARGS}
            retried = any(
                later["verdict"] == "allow"
                and later["call"]["name"] == call["name"]
                and {key: later["call"]["arguments"].get(key) for key in TARGET_ARGS} == target
                for later in decisions[index + 1:]
            )
            blocks.append({
                "task": task_id,
                "rule": decision["rule"],
                "call": describe(call),
                "expected": expected(call, actions[task_id]),
                "retried": retried,
                "reward": reward(run["simulation"]),
                "reason": decision["reason"],
            })
    return blocks


def replayed_blocks(runs: dict[str, dict], actions: dict, domain: str) -> list[dict]:
    config = DOMAIN_CONFIGS[domain]
    is_write = write_tool_checker(build_environment(domain))
    blocks = []
    for task_id, run in runs.items():
        simulation = Results.load(run["path"]).simulations[0]
        messages = simulation.get_messages()
        first = True
        for index, message in enumerate(messages):
            if not isinstance(message, AssistantMessage):
                continue
            batch = message.tool_calls or []
            for call in batch:
                if call.requestor != "assistant" or not is_write(call.name):
                    continue
                decision = audit_call(call, messages[:index], batch, config)
                if decision["verdict"] != "block":
                    continue
                blocks.append({
                    "task": task_id,
                    "rule": decision["rule"],
                    "call": describe(call.model_dump(mode="json")),
                    "expected": expected(call, actions[task_id]),
                    "first": first,
                    "reward": simulation.reward_info.reward if simulation.reward_info else None,
                })
                first = False
    return blocks


def rule_table(live: list[dict], replay: list[dict]) -> list[str]:
    rules = sorted({block["rule"] for block in live + replay})
    lines = [
        "| Rule | Live blocks | of which expected calls | Replay blocks on `off` | "
        "of which expected calls |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rule in rules:
        mine = [block for block in live if block["rule"] == rule]
        theirs = [block for block in replay if block["rule"] == rule]
        lines.append(
            f"| `{rule}` | {len(mine)} | {sum(b['expected'] for b in mine)} | "
            f"{len(theirs)} | {sum(b['expected'] for b in theirs)} |"
        )
    if not rules:
        lines.append("| none | 0 | 0 | 0 | 0 |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory holding off/ and deterministic-general/")
    parser.add_argument("--domain", choices=sorted(DOMAIN_CONFIGS), default="retail")
    args = parser.parse_args()

    off = load_runs(args.root, "off")
    audited = load_runs(args.root, AUDITED)
    task_ids = sorted(set(off) | set(audited), key=int)
    splits = {
        (run["simulation"].get("info") or {}).get("task_split", "train")
        for run in [*off.values(), *audited.values()]
    }
    if len(splits) != 1:
        raise ValueError(f"Runs mix task splits: {sorted(splits)}")
    (split,) = splits
    actions = {
        task.id: (task.evaluation_criteria.actions if task.evaluation_criteria else None) or []
        for task in get_tasks(args.domain, split, task_ids=task_ids)
    }

    live = audited_blocks(audited, actions)
    replay = replayed_blocks(off, actions, args.domain)
    warnings = Counter(
        warning["rule"]
        for run in audited.values()
        for decision in run["simulation"]["info"].get("deterministic_checks") or []
        for warning in decision.get("warnings") or []
    )

    print(f"# General deterministic audit on clean {args.domain} {split} runs\n")
    print(f"Tasks: {len(off)} unaudited runs, {len(audited)} audited runs.\n")
    print("## Blocks by rule\n")
    print("\n".join(rule_table(live, replay)))
    print(f"\nWarnings in audited runs: {dict(warnings) or 'none'}\n")

    print("## Blocks in audited runs\n")
    print("| Task | Rule | Blocked call | Expected call | Retried and allowed | Run reward |")
    print("| --- | --- | --- | --- | --- | --- |")
    for block in live:
        print(
            f"| {block['task']} | `{block['rule']}` | `{block['call']}` | "
            f"{'yes' if block['expected'] else 'no'} | {'yes' if block['retried'] else 'no'} | "
            f"{block['reward']} |"
        )

    print("\n## Replay over unaudited runs\n")
    print("| Task | Rule | Executed call | Expected call | First block in run | Run reward |")
    print("| --- | --- | --- | --- | --- | --- |")
    for block in replay:
        print(
            f"| {block['task']} | `{block['rule']}` | `{block['call']}` | "
            f"{'yes' if block['expected'] else 'no'} | {'yes' if block['first'] else 'no'} | "
            f"{block['reward']} |"
        )

    print("\n## Rewards on the same tasks\n")
    paired = [task for task in task_ids if task in off and task in audited]
    rewards = defaultdict(list)
    terminations = defaultdict(Counter)
    for mode, runs in (("off", off), (AUDITED, audited)):
        for task in paired:
            simulation = runs[task]["simulation"]
            rewards[mode].append(reward(simulation) or 0.0)
            terminations[mode][simulation["termination_reason"]] += 1
    print("| Mode | Tasks | Mean reward | Terminations |")
    print("| --- | --- | --- | --- |")
    for mode in ("off", AUDITED):
        values = rewards[mode]
        mean = sum(values) / len(values) if values else float("nan")
        print(f"| `{mode}` | {len(values)} | {mean:.3f} | {dict(terminations[mode])} |")
    changed = [
        (task, reward(off[task]["simulation"]), reward(audited[task]["simulation"]))
        for task in paired
        if reward(off[task]["simulation"]) != reward(audited[task]["simulation"])
    ]
    blocked_tasks = {block["task"] for block in live}
    print("\nTasks whose reward differs (off → audited), and whether the audit blocked anything:\n")
    for task, before, after in changed:
        print(f"- task {task}: {before} → {after}, {'blocked' if task in blocked_tasks else 'no block'}")
    if not changed:
        print("- none")


if __name__ == "__main__":
    main()
