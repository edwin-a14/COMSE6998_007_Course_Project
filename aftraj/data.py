"""Download and load AFTraj-2K (HF dataset ZBox008003/AFTraj, CC BY 4.0).

    python -m aftraj.data download          # into data/aftraj/ (git-ignored)
    python -m aftraj.data stats [--split test]

Splits:
- "test": the paper's held-out 169 safe + 163 unsafe trajectories. Report
  final numbers here only, once.
- "dev":  every other trajectory, minus those that share a source trajectory
  (an injected trajectory is a copy of a safe one with one step changed) or a
  task with a test trajectory (enterprise scenarios were run up to five times).
  Tune prompts and thresholds here.
- "all":  everything.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HF_REPO = "ZBox008003/AFTraj"
FILES = ("aftraj_safe.parquet", "aftraj_unsafe.parquet", "splits_test.json")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "aftraj"
SPLITS = ("test", "dev", "all")

# Suffixes that mark the variant of one source trajectory, e.g.
# math_0000_safe / math_0000_injected, gaia_0051_unsafe_diagnosed.
_VARIANT_SUFFIX = re.compile(r"_(safe|injected|unsafe_diagnosed|diagnosed|unsafe)$")


@dataclass
class Trajectory:
    """One AFTraj record. Label fields are for scoring only and must never reach
    an auditor; `aftraj.adapter` builds auditor inputs without them."""

    conv_id: str
    domain: str
    label: str
    """"safe" or "unsafe"."""
    task: str
    turns: list[dict]
    """Raw {role, thought, action, content} records; step i is turns[i]."""
    tools: list[dict] = field(default_factory=list)
    gold_answer: str = ""
    mistake_step: int = -1
    mistake_agent: str = ""
    mistake_reason: str = ""
    unsafe_source: str = ""


def download(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = data_dir / name
        if target.exists() and not force:
            print(f"have {target}")
            continue
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{name}"
        print(f"downloading {url}")
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
    return data_dir


def source_id(conv_id: str) -> str:
    """The source trajectory a record was derived from (its id without the
    variant suffix)."""
    return _VARIANT_SUFFIX.sub("", conv_id)


def load(
    data_dir: Path = DEFAULT_DATA_DIR,
    split: str = "test",
    domains: list[str] | None = None,
    limit: int | None = None,
) -> list[Trajectory]:
    """Load trajectories in a fixed order: safe first, then unsafe, file order."""
    import pandas as pd  # needs pyarrow>=20; see aftraj/requirements.txt

    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    data_dir = Path(data_dir)
    missing = [name for name in FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {data_dir}; run `python -m aftraj.data download`",
        )
    test_ids = json.loads((data_dir / "splits_test.json").read_text(encoding="utf-8"))
    test = set(test_ids["test_safe"]) | set(test_ids["test_unsafe"])

    rows = [
        (row, label)
        for label in ("safe", "unsafe")
        for row in pd.read_parquet(data_dir / f"aftraj_{label}.parquet").to_dict("records")
    ]
    test_rows = [row for row, _ in rows if str(row["conv_id"]) in test]
    test_sources = {source_id(str(row["conv_id"])) for row in test_rows}
    test_tasks = {(row["domain"], row["task"]) for row in test_rows}

    records: list[Trajectory] = []
    for row, label in rows:
        conv_id = str(row["conv_id"])
        if split == "test" and conv_id not in test:
            continue
        if split == "dev" and (
            conv_id in test
            or source_id(conv_id) in test_sources
            or (row["domain"], row["task"]) in test_tasks
        ):
            continue
        if domains and row["domain"] not in domains:
            continue
        records.append(_to_trajectory(row, label))
    return records[:limit] if limit is not None else records


def _to_trajectory(row: dict[str, Any], label: str) -> Trajectory:
    def text(value: Any) -> str:
        return "" if value is None else str(value)

    turns = [
        {key: text(turn.get(key)) for key in ("role", "thought", "action", "content")}
        for turn in _as_list(row.get("turns"))
    ]
    tools = [
        {key: text(value) for key, value in dict(tool).items()}
        for tool in _as_list(row.get("tools"))
    ]
    mistake_step = row.get("mistake_step")
    return Trajectory(
        conv_id=str(row["conv_id"]),
        domain=str(row["domain"]),
        label=label,
        task=text(row.get("task")),
        turns=turns,
        tools=tools,
        gold_answer=text(row.get("gold_answer")),
        mistake_step=-1 if mistake_step is None or label == "safe" else int(mistake_step),
        mistake_agent=text(row.get("mistake_agent")),
        mistake_reason=text(row.get("mistake_reason")),
        unsafe_source=text(row.get("unsafe_source")),
    )


def _as_list(value: Any) -> list:
    return [] if value is None else list(value)


def stats(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Counts by domain and label, and how well the adapter parses actions."""
    from aftraj.adapter import ACTION_FORMATS, classify_action, is_auditable

    by_domain = Counter((t.domain, t.label) for t in trajectories)
    formats = Counter()
    for trajectory in trajectories:
        for turn in trajectory.turns:
            if turn["action"]:
                formats[classify_action(turn["action"])] += 1
    unsafe = [t for t in trajectories if t.label == "unsafe"]
    return {
        "trajectories": len(trajectories),
        "by_domain_label": {f"{d}/{l}": n for (d, l), n in sorted(by_domain.items())},
        "action_formats": {name: formats.get(name, 0) for name in ACTION_FORMATS},
        "unsafe_with_auditable_mistake_step": sum(is_auditable(t) for t in unsafe),
        "unsafe": len(unsafe),
        "max_turns": max((len(t.turns) for t in trajectories), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("download", "stats"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=SPLITS, default="all")
    parser.add_argument("--force", action="store_true", help="download again")
    args = parser.parse_args()
    if args.command == "download":
        download(args.data_dir, force=args.force)
    else:
        print(json.dumps(stats(load(args.data_dir, args.split)), indent=2))


if __name__ == "__main__":
    main()
