"""Choose a reproducible development sample across domains and labels."""

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from random import Random

from aftraj.data import Trajectory


def balanced_sample(
    trajectories: list[Trajectory], per_group: int, seed: int,
) -> tuple[list[Trajectory], dict]:
    if per_group < 1:
        raise ValueError("--sample-per-group must be positive")
    groups = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.domain, trajectory.label].append(trajectory)
    domains = sorted({trajectory.domain for trajectory in trajectories})
    rng = Random(seed)
    selected = []
    counts = {}
    for domain in domains:
        for label in ("safe", "unsafe"):
            candidates = sorted(groups[domain, label], key=lambda item: item.conv_id)
            if len(candidates) < per_group:
                raise ValueError(f"Need {per_group} {domain}/{label} trajectories, found {len(candidates)}")
            selected.extend(rng.sample(candidates, per_group))
            counts[f"{domain}/{label}"] = per_group
    rng.shuffle(selected)
    fingerprint = sha256(json.dumps(
        [asdict(item) for item in selected], sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    return selected, {
        "policy": "domain-label-v1", "per_group": per_group, "seed": seed,
        "counts": counts, "conv_ids": [item.conv_id for item in selected],
        "sha256": fingerprint,
    }
