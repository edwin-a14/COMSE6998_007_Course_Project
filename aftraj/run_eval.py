"""Evaluate an auditor on AFTraj-2K with the paper's online protocol.

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
from audit_core import AllowAllAuditor, Auditor
from audit_core.auditors import GENERIC_CONFIG, RulesAuditor

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
    started = time.perf_counter()
    for point in audit_points(trajectory):
        verdict = auditor.audit(point.history, point.proposal, tools, context)
        step = {
            "step": point.step, "role": point.proposal.role,
            "decision": verdict.decision, "score": verdict.score, "rule_ids": verdict.rule_ids,
        }
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
        "pred_step": pred_step,
        "auditable": is_auditable(trajectory),
        "num_turns": len(trajectory.turns),
        "num_audits": len(steps),
        "audit_seconds": round(time.perf_counter() - started, 4),
        "steps": steps,
        **per_sample_metrics(pred_step, trajectory.mistake_step),
    }


def summarize(rows: list[dict[str, Any]], trajectories: list[Trajectory]) -> dict[str, Any]:
    """Metrics plus cost: audit calls made vs agent turns that could be audited."""
    unsafe = [row for row in rows if row["gt_step"] >= 0]
    return {
        "by_domain": aggregate_by_domain(rows, macro=True),
        "audit_calls": sum(row["num_audits"] for row in rows),
        "agent_turns": sum(sum(1 for _ in audit_points(t)) for t in trajectories),
        "unsafe_auditable": sum(row["auditable"] for row in unsafe),
        "unsafe": len(unsafe),
    }


# Settings that change the rows a run produces; a resumed run must match them.
RESUME_KEYS = ("auditor", "split", "domains", "limit", "full_scan")


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
    parser.add_argument("--auditor", choices=sorted(AUDITORS), required=True)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--domains", help=f"comma-separated: {','.join(DOMAINS)}")
    parser.add_argument("--limit", type=int, help="first N trajectories (smoke test)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, help="default: results/aftraj/<auditor>-<split>")
    parser.add_argument("--full-scan", action="store_true", help="audit every agent turn, not only up to the first block")
    parser.add_argument("--resume", action="store_true", help="skip trajectories already in per_sample.jsonl")
    args = parser.parse_args(argv)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else None
    unknown = sorted(set(domains or ()) - set(DOMAINS))
    if unknown:
        parser.error(f"unknown --domains {', '.join(unknown)}; choose from {', '.join(DOMAINS)}")
    trajectories = load(args.data_dir, args.split, domains=domains, limit=args.limit)
    # Check before anything is written, so a bad selection leaves no output behind.
    if not trajectories:
        parser.error(f"no trajectories selected (split={args.split}, domains={domains}, limit={args.limit})")
    output = args.output or RESULTS_ROOT / f"{args.auditor}-{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    per_sample = output / "per_sample.jsonl"
    if per_sample.exists() and not args.resume:
        parser.error(f"{per_sample} exists; pass --resume to continue it or choose another --output")

    config = {
        "auditor": args.auditor, "split": args.split, "domains": domains, "limit": args.limit,
        "full_scan": args.full_scan, "trajectories": len(trajectories),
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
    auditor = AUDITORS[args.auditor]()
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
    print(format_report(summary["by_domain"]))
    print(
        f"\naudit calls {summary['audit_calls']} of {summary['agent_turns']} agent turns; "
        f"{summary['unsafe_auditable']}/{summary['unsafe']} unsafe mistake steps are agent turns"
    )
    print(f"results in {output}")
    return summary


if __name__ == "__main__":
    main()
