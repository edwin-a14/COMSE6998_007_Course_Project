"""Evaluate auditors on AFTraj-2K with the project's pre-action protocol.

    python -m aftraj.run_eval --auditor rules --split dev
    python -m aftraj.run_eval --auditor rules --split test --output results/aftraj/rules-test

For each trajectory the auditor sees every agent turn in order, as a proposal
with the turns before it as history. The first blocked step is the predicted
mistake step (`pred_step`); a trajectory with no block is predicted safe (-1).
Metrics are AgentForesight's Exact-F1, ASS, FAR and Step-Acc (aftraj/metrics.py).

Output directory:
    per_sample.jsonl   first line is the run config, then one row per trajectory
    results.json       metrics by domain and overall, plus audit-call counts
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aftraj.adapter import audit_context, audit_points, is_auditable, tool_specs
from aftraj.data import DEFAULT_DATA_DIR, DOMAINS, SPLITS, Trajectory, load
from aftraj.metrics import aggregate_by_domain, format_report, per_sample_metrics
from aftraj.sampling import balanced_sample
from audit_core import AllowAllAuditor, Auditor
from audit_core.auditors import GENERIC_CONFIG, RulesAuditor, SemanticAuditError, SemanticAuditor, SelectiveSemanticAuditor
from audit_core.modes import MODEL_MODES, build_model_auditor
from audit_core.usage import usage_from_details as _usage, sum_usage as _sum_usage

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "aftraj"

# Name -> factory. Register new auditors here.
AUDITORS: dict[str, Callable[[], Auditor]] = {
    "allow-all": AllowAllAuditor,
    "rules": lambda: RulesAuditor(GENERIC_CONFIG),
    "rules-unknown": lambda: RulesAuditor(GENERIC_CONFIG, audit_unknown=True),
}


def evaluate(auditor: Auditor, trajectory: Trajectory, full_scan: bool = False) -> dict[str, Any]:
    """Run the online protocol on one trajectory and score it.

    Stops at the first block unless `full_scan`, which audits every agent turn
    so thresholds on `score` can be swept offline afterwards.
    """
    tools = tool_specs(trajectory)
    context = audit_context(trajectory)
    steps, pred_step = [], -1
    usage, status = [], "ok"
    started = time.perf_counter()
    for point in audit_points(trajectory):
        try:
            verdict = auditor.audit(point.history, point.proposal, tools, context)
        except SemanticAuditError as exc:
            status = "error"
            steps.append({"step": point.step, "role": point.proposal.role,
                          "error": exc.kind, "details": exc.details})
            usage.append(_usage(exc.details))
            break
        step = {
            "step": point.step, "role": point.proposal.role,
            "decision": verdict.decision, "score": verdict.score, "rule_ids": verdict.rule_ids,
            "details": verdict.details,
        }
        usage.append(_usage(verdict.details))
        if verdict.decision != "allow":
            step["reason"] = verdict.reason
        steps.append(step)
        if verdict.blocked and pred_step == -1:
            pred_step = point.step
            if not full_scan:
                break
    return {
        "conv_id": trajectory.conv_id,
        "domain": trajectory.domain,
        "label": trajectory.label,
        "unsafe_source": trajectory.unsafe_source,
        "gt_step": trajectory.mistake_step,
        "pred_step": pred_step if status == "ok" else None,
        "status": status,
        "usage": _sum_usage(usage),
        "auditable": is_auditable(trajectory),
        "num_turns": len(trajectory.turns),
        "num_audits": len(steps),
        "audit_seconds": round(time.perf_counter() - started, 4),
        "steps": steps,
        **(per_sample_metrics(pred_step, trajectory.mistake_step) if status == "ok" else {}),
    }


def summarize(rows: list[dict[str, Any]], trajectories: list[Trajectory]) -> dict[str, Any]:
    """Metrics plus cost: audit calls made vs agent turns that could be audited."""
    unsafe = [row for row in rows if row["gt_step"] >= 0]
    completed = [row for row in rows if row.get("status", "ok") == "ok"]
    return {
        "by_domain": aggregate_by_domain(completed, macro=True),
        "evaluated_trajectories": len(completed),
        "failed_trajectories": len(rows) - len(completed),
        "usage": _sum_usage([row.get("usage", _sum_usage([])) for row in rows]),
        "audit_calls": sum(row["num_audits"] for row in rows),
        "agent_turns": sum(sum(1 for _ in audit_points(t)) for t in trajectories),
        "unsafe_auditable": sum(row["auditable"] for row in unsafe),
        "unsafe": len(unsafe),
    }


# Settings that change the rows a run produces; a resumed run must match them.
RESUME_KEYS = ("auditor", "split", "domains", "limit", "full_scan", "model_config", "result_version", "sample")


def _load_previous(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return the config and rows of an earlier run in `path`.

    A row counts only once its newline is written. A trailing partial line, left
    by a crash mid-write, is cut from the file so new rows start on a clean
    line, and that trajectory is run again. A broken line before the end is
    real corruption and raises ValueError.
    """
    if not path.exists():
        return None, []
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    if len(complete) < len(data):
        print(f"dropping a partial last line ({len(data) - len(complete)} bytes) from {path}")
        with path.open("r+b") as handle:
            handle.truncate(len(complete))
    config, rows = None, []
    for number, line in enumerate(complete.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} line {number} is not valid JSON: {error}") from None
        if record.get("_record_type") == "config":
            config = config or record
        else:
            rows.append(record)
    return config, rows


def _config_mismatch(previous: dict[str, Any], config: dict[str, Any]) -> list[str]:
    return [
        f"{key}: {previous.get(key)!r} in file, {config[key]!r} now"
        for key in RESUME_KEYS
        if previous.get(key) != config[key]
    ]


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auditor", choices=sorted((*AUDITORS, *MODEL_MODES)), required=True)
    parser.add_argument("--model", help="Required for model-based auditors; provider/model name")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum model output tokens per audit")
    parser.add_argument("--timeout", type=float, default=30, help="Model request timeout in seconds")
    parser.add_argument("--env-file", type=Path, help="Optional local dotenv file for model credentials")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--domains", help=f"comma-separated: {','.join(DOMAINS)}")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="first N trajectories (smoke test)")
    selection.add_argument("--sample-per-group", type=int, help="Sample N safe and N unsafe trajectories per domain")
    parser.add_argument("--sample-seed", type=int, default=43)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, help="default: results/aftraj/<auditor>-<split>")
    parser.add_argument("--full-scan", action="store_true", help="audit every agent turn, not only up to the first block")
    parser.add_argument("--resume", action="store_true", help="skip trajectories already in per_sample.jsonl")
    args = parser.parse_args(argv)
    if args.sample_per_group is not None and args.sample_per_group < 1:
        parser.error("--sample-per-group must be positive")

    uses_model = args.auditor in MODEL_MODES
    if uses_model and (not args.model or not args.model.strip()):
        parser.error("--model is required for model-based auditors")
    if not uses_model and (args.model or args.env_file):
        parser.error("--model and --env-file require a model-based auditor")
    # Validate settings before creating output or making a model request.
    try:
        auditor = (build_model_auditor(args.auditor, model=args.model,
                                      max_tokens=args.max_tokens, timeout=args.timeout)
                   if uses_model else AUDITORS[args.auditor]())
    except ValueError as exc:
        parser.error(str(exc))
    if uses_model:
        try:
            import importlib.util
            if importlib.util.find_spec("litellm") is None:
                parser.error("Install LiteLLM (aftraj/requirements-model.txt) for model-based modes")
            if args.env_file is not None:
                if not args.env_file.is_file():
                    parser.error("--env-file must name an existing local file")
                from dotenv import load_dotenv
                load_dotenv(args.env_file, override=False)
        except ImportError:
            parser.error("Install aftraj/requirements-model.txt for model-based modes")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else None
    unknown = sorted(set(domains or ()) - set(DOMAINS))
    if unknown:
        parser.error(f"unknown --domains {', '.join(unknown)}; choose from {', '.join(DOMAINS)}")
    trajectories = load(args.data_dir, args.split, domains=domains, limit=args.limit)
    # Check before anything is written, so a bad selection leaves no output behind.
    if not trajectories:
        parser.error(f"no trajectories selected (split={args.split}, domains={domains}, limit={args.limit})")
    sample = None
    if args.sample_per_group is not None:
        try:
            trajectories, sample = balanced_sample(trajectories, args.sample_per_group, args.sample_seed)
        except ValueError as exc:
            parser.error(str(exc))
    output = args.output or RESULTS_ROOT / f"{args.auditor}-{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    per_sample = output / "per_sample.jsonl"
    if per_sample.exists() and not args.resume:
        parser.error(f"{per_sample} exists; pass --resume to continue it or choose another --output")

    config = {
        "auditor": args.auditor, "split": args.split, "domains": domains, "limit": args.limit,
        "full_scan": args.full_scan, "trajectories": len(trajectories),
        "sample": sample,
        "result_version": 2,
        "model_config": ({
            "model": args.model, "max_tokens": args.max_tokens, "timeout": args.timeout,
            "temperature": 0, "prompt_version": SemanticAuditor.prompt_version,
            "selection_policy": ("all-agent-turns-v1" if args.auditor == "continuous-semantic"
                                 else SelectiveSemanticAuditor.policy),
            "rules_config": "generic" if args.auditor == "general-hybrid" else None,
        } if uses_model else None),
    }
    try:
        previous, rows = _load_previous(per_sample)
    except ValueError as error:
        parser.error(str(error))
    if previous is None and rows:
        parser.error(f"{per_sample} has rows but no config line; cannot check it matches this run")
    if previous is not None:
        mismatch = _config_mismatch(previous, config)
        if mismatch:
            parser.error(
                f"{per_sample} was written with other settings; use the same settings or another --output\n  "
                + "\n  ".join(mismatch)
            )
    # Score only the current selection, once per trajectory.
    selected = {t.conv_id for t in trajectories}
    kept = {row["conv_id"]: row for row in rows if row["conv_id"] in selected}
    if len(kept) < len(rows):
        print(f"ignoring {len(rows) - len(kept)} rows in {per_sample} that are outside this selection or repeated")
    rows = list(kept.values())
    done = set(kept)
    with per_sample.open("a", encoding="utf-8") as handle:
        if previous is None:
            handle.write(json.dumps({"_record_type": "config", **config}) + "\n")
        for index, trajectory in enumerate(trajectories, 1):
            if trajectory.conv_id in done:
                continue
            row = evaluate(auditor, trajectory, full_scan=args.full_scan)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            if index % 100 == 0:
                print(f"{index}/{len(trajectories)}")

    summary = {"config": config, **summarize(rows, trajectories)}
    (output / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if summary["evaluated_trajectories"]:
        print(format_report(summary["by_domain"]))
    else:
        print("No completed trajectories to score.")
    print(f"Completed {summary['evaluated_trajectories']}; failed {summary['failed_trajectories']} "
          "(failed trajectories excluded from detection metrics).")
    print("Model usage: " + json.dumps(summary["usage"]))
    print(
        f"\naudit calls {summary['audit_calls']} of {summary['agent_turns']} agent turns; "
        f"{summary['unsafe_auditable']}/{summary['unsafe']} unsafe mistake steps are agent turns"
    )
    print(f"results in {output}")
    return summary


if __name__ == "__main__":
    result = main()
    raise SystemExit(1 if result["failed_trajectories"] else 0)
