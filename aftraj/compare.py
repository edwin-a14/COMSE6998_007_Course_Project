"""Compare saved AFTraj runs on the same sampled trajectories."""

import argparse
import json
from pathlib import Path

from aftraj.metrics import aggregate


def load_run(path: Path):
    report = json.loads((path / "results.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line) for line in (path / "per_sample.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = {row["conv_id"]: row for row in records if row.get("_record_type") != "config"}
    return report, rows


def compare(paths: list[Path]) -> str:
    runs = [load_run(path) for path in paths]
    reference = runs[0][0]["config"]
    cohort = {
        key: (row["domain"], row["label"], row["gt_step"])
        for key, row in runs[0][1].items()
    }
    model_settings = None
    for report, rows in runs:
        config = report["config"]
        for key in ("split", "domains", "limit", "sample", "full_scan", "result_version", "adapter_version"):
            if config.get(key) != reference.get(key):
                raise ValueError(f"Runs differ on {key}")
        labels = {key: (row["domain"], row["label"], row["gt_step"]) for key, row in rows.items()}
        if labels != cohort or len(rows) != config["trajectories"]:
            raise ValueError("Runs must contain the same complete trajectory selection")
        if config.get("model_config") is not None:
            settings = {key: config["model_config"][key] for key in (
                "model", "max_tokens", "timeout", "temperature", "prompt_version",
            )}
            if model_settings is not None and settings != model_settings:
                raise ValueError("Runs use different model settings")
            model_settings = settings
    completed = set.intersection(*[
        {key for key, row in rows.items() if row.get("status", "ok") == "ok"}
        for _, rows in runs
    ])
    lines = [
        f"Matched trajectories: {len(cohort)}. Completed in every mode: {len(completed)}.",
        "Detection metrics use the common completed set. Usage includes failed attempts.",
        "",
        "| Auditor | Exact-F1 % | Exact precision % | Exact recall % | FAR % | Model calls | Audit tokens | Model seconds | Failed trajectories |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report, rows in runs:
        scores = aggregate([row for key, row in rows.items() if key in completed])
        usage = report["usage"]
        token_counts = (usage["input_tokens"], usage["output_tokens"])
        tokens = sum(token_counts) if all(value is not None for value in token_counts) else "—"
        seconds = usage["model_latency_ms"]
        seconds = f"{seconds / 1000:.2f}" if seconds is not None else "—"
        metrics = [f"{scores[key]:.2f}" if completed else "—" for key in (
            "exact_f1", "precision", "recall", "far",
        )]
        values = (
            report["config"]["auditor"], *metrics, usage["model_calls"], tokens,
            seconds, report["failed_trajectories"],
        )
        lines.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Directories containing saved evaluation outputs")
    args = parser.parse_args()
    try:
        print(compare(args.runs))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
