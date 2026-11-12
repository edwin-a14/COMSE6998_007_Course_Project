"""Compare audit modes on recorded retail tool proposals."""

import argparse
import json
from collections import Counter
from pathlib import Path

from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.data_model.simulation import Results, SimulationRun
from tau2.runner import build_environment

from semantic_audit import audit_tool_call, should_audit_tool
from tau2_audit import RULE_AUDIT_MODES, retail_argument_block_reason


MODES = ("semantic", "continuous")


def replay(
    simulation: SimulationRun, model: str, tool_definitions: dict, dry_run: bool, modes=MODES,
) -> dict:
    messages = simulation.get_messages()
    outcomes = {message.id: message.error for message in messages if isinstance(message, ToolMessage)}
    proposals = []
    for index, message in enumerate(messages):
        if not isinstance(message, AssistantMessage) or not message.tool_calls:
            continue
        # The live runner uses the prefix from before the whole proposal batch.
        history = messages[:index]
        for call in message.tool_calls:
            checks = {}
            rule_blocks = {}
            for mode in modes:
                if mode in RULE_AUDIT_MODES:
                    reason = retail_argument_block_reason(call)
                    if reason is not None:
                        rule_blocks[mode] = reason
                        continue
                if should_audit_tool(mode, call):
                    checks[mode] = {"verdict": "planned"} if dry_run else audit_tool_call(
                        model, history, call, tool_definitions.get(call.name),
                    )
            proposals.append({
                "message_index": index,
                "call": call.model_dump(mode="json"),
                "recorded_tool_error": outcomes.get(call.id),
                "checks": checks,
                "rule_blocks": rule_blocks,
            })

    summary = {}
    for mode in modes:
        checks = [row["checks"][mode] for row in proposals if mode in row["checks"]]
        totals = {}
        for field in ("input_tokens", "output_tokens", "latency_ms"):
            values = [check.get(field) for check in checks]
            totals[field] = (
                None if dry_run or any(value is None for value in values)
                else round(sum(values), 2)
            )
        summary[mode] = {
            "audit_calls": len(checks),
            "rule_blocks": sum(mode in row["rule_blocks"] for row in proposals),
            "verdicts": dict(Counter(check["verdict"] for check in checks)),
            **totals,
        }
    return {
        "task_id": simulation.task_id,
        "simulation_id": simulation.id,
        "summary": summary,
        "proposals": proposals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--modes", nargs="+", choices=(*MODES, *RULE_AUDIT_MODES), default=MODES)
    parser.add_argument("--dry-run", action="store_true", help="Show planned checks without model calls")
    args = parser.parse_args()
    modes = tuple(dict.fromkeys(args.modes))

    environment = build_environment("retail")
    tool_definitions = {tool.name: tool.openai_schema["function"] for tool in environment.get_tools()}
    traces = []
    for path in args.traces:
        results = Results.load(path)
        if results.info.environment_info.domain_name != "retail":
            parser.error(f"Expected a retail trace: {path}")
        traces.append({
            "trace": str(path),
            "simulations": [
                replay(simulation, args.model, tool_definitions, args.dry_run, modes)
                for simulation in results.simulations
            ],
        })
    print(json.dumps({"model": args.model, "dry_run": args.dry_run, "traces": traces}, indent=2))


if __name__ == "__main__":
    main()
