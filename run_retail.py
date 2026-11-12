"""Run and score one τ²-bench retail simulation."""

import argparse
from pathlib import Path
from random import Random

from tau2.data_model.message import ToolCall
from tau2.data_model.simulation import Results, TextRunConfig
from tau2.runner import (
    build_agent,
    build_environment,
    build_user,
    get_info,
    get_tasks,
    run_simulation,
)

from deterministic_audit import GeneralDeterministicAuditor
from semantic_audit import audit_cancellation, cancellation_context
from tau2_audit import AuditOrchestrator, retail_cancellation_block_reason


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", choices=("off", "deterministic", "deterministic-general", "semantic"),
        default="off",
    )
    parser.add_argument("--audit-llm", default="openai/gpt-4.1-mini")
    faults = parser.add_mutually_exclusive_group()
    faults.add_argument(
        "--inject-cancellation-fault", action="store_true",
        help="Corrupt the first valid cancellation reason and allow one retry",
    )
    faults.add_argument(
        "--inject-wrong-order", action="store_true",
        help="Substitute another observed pending order and allow one retry",
    )
    parser.add_argument("--task-id", default="88")
    parser.add_argument("--task-split", choices=("train", "test", "base"), default="train")
    parser.add_argument("--agent-llm", default="openai/gpt-4.1-mini")
    parser.add_argument("--user-llm", default="openai/gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True, help="New results JSON file")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.inject_cancellation_fault and args.audit != "deterministic":
        parser.error("--inject-cancellation-fault requires --audit deterministic")
    config = TextRunConfig(
        domain="retail",
        task_split_name=args.task_split,
        num_trials=1,
        seed=args.seed,
        max_steps=args.max_steps,
        llm_agent=args.agent_llm,
        llm_user=args.user_llm,
    )
    try:
        (task,) = get_tasks(config.domain, config.task_split_name, task_ids=[args.task_id])
    except ValueError as exc:
        parser.error(str(exc))

    environment = build_environment(config.domain)
    agent = build_agent(
        config.effective_agent,
        environment,
        llm=config.llm_agent,
        llm_args=config.llm_args_agent,
        task=task,
    )
    user = build_user(
        config.effective_user,
        environment,
        task,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
    )
    # Match the seed of the first trial in the stock benchmark CLI.
    trial_seed = Random(config.seed).randint(0, 1_000_000)
    orchestrator = AuditOrchestrator(
        domain=config.domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        seed=trial_seed,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
    )
    semantic_checks = []

    def semantic_block_reason(call: ToolCall) -> str | None:
        if call.requestor != "assistant" or call.name != "cancel_pending_order":
            return None
        history, _ = cancellation_context(orchestrator.trajectory[:-1])
        verdict = audit_cancellation(args.audit_llm, history, call)
        semantic_checks.append({"call": call.model_dump(mode="json"), **verdict})
        if verdict["verdict"] == "allow":
            return None
        return (
            f"Semantic audit {verdict['verdict']}: {verdict['reason']} "
            "No cancellation was executed. Check which order and reason the customer "
            "already authorized, and ask only for anything still missing."
        )

    general_auditor = None
    if args.audit == "deterministic":
        orchestrator.before_tool_call = retail_cancellation_block_reason
    elif args.audit == "deterministic-general":
        general_auditor = GeneralDeterministicAuditor.for_orchestrator(orchestrator, config.domain)
        orchestrator.before_tool_call = general_auditor
    elif args.audit == "semantic":
        orchestrator.before_tool_call = semantic_block_reason
    orchestrator.inject_cancellation_fault = args.inject_cancellation_fault
    orchestrator.inject_wrong_order = args.inject_wrong_order

    args.output.parent.mkdir(parents=True, exist_ok=True)
    info = get_info(config)
    simulation = run_simulation(orchestrator)
    simulation.trial = 0
    simulation.info = {**(simulation.info or {}), "audit_mode": args.audit}
    if args.audit == "semantic":
        simulation.info.update(audit_model=args.audit_llm, semantic_checks=semantic_checks)
    if general_auditor is not None:
        simulation.info["deterministic_checks"] = general_auditor.decisions
    results = Results(info=info, tasks=[task], simulations=[simulation])
    with args.output.open("x", encoding="utf-8") as output:
        output.write(results.model_dump_json(indent=2))

    rejections = len(simulation.info.get("audit_rejections", []))
    print(
        f"Task {task.id}: reward={simulation.reward_info.reward:g}, "
        f"termination={simulation.termination_reason.value}, "
        f"rejected proposals={rejections}"
    )
    if args.inject_cancellation_fault or args.inject_wrong_order:
        print(f'Cancellation recovery: {simulation.info["cancellation_recovery"]["status"]}')
    if semantic_checks:
        print("Semantic verdicts: " + ", ".join(check["verdict"] for check in semantic_checks))
    if general_auditor is not None:
        decisions = general_auditor.decisions
        rules = [decision["rule"] for decision in decisions if decision["rule"]]
        print(
            f"Deterministic checks: {len(decisions)} write calls audited, "
            f"blocked by: {', '.join(rules) or 'none'}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
