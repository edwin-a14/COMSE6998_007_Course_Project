"""Replay the general deterministic audit over saved τ²-bench results.

Each assistant write call in a saved transcript is audited against the messages
before it. No model is called. Calls an earlier audit rejected were already
removed from the transcript, so only executed proposals are replayed.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import Results
from tau2.runner import build_environment

from deterministic_audit import DOMAIN_CONFIGS, audit_call, write_tool_checker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="results.json files")
    parser.add_argument("--domain", choices=sorted(DOMAIN_CONFIGS), default="retail")
    args = parser.parse_args()

    config = DOMAIN_CONFIGS[args.domain]
    is_write = write_tool_checker(build_environment(args.domain))
    verdicts = Counter()
    for path in args.results:
        for simulation in Results.load(path).simulations:
            messages = simulation.get_messages()
            for index, message in enumerate(messages):
                if not isinstance(message, AssistantMessage):
                    continue
                batch = message.tool_calls or []
                for call in batch:
                    if call.requestor != "assistant" or not is_write(call.name):
                        continue
                    decision = audit_call(call, messages[:index], batch, config)
                    verdicts[decision["rule"] or "allow"] += 1
                    print(json.dumps({
                        "trace": str(path),
                        "reward": simulation.reward_info.reward if simulation.reward_info else None,
                        "call": {"name": call.name, "arguments": call.arguments},
                        "verdict": decision["verdict"],
                        "rule": decision["rule"],
                        "reason": decision["reason"],
                    }))
    print(json.dumps({"summary": dict(verdicts)}))


if __name__ == "__main__":
    main()
