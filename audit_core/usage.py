"""Shared audit usage accounting; unknown usage is never silently zero."""

def usage_from_details(details: dict) -> dict:
    """Read nested composite results without counting a delegate twice."""
    if "auditors" in details:
        return sum_usage([usage_from_details(item["details"]) for item in details["auditors"]])
    calls = details.get("model_calls", 0)
    selection = details.get("semantic_selection", {})
    return {
        "model_calls": calls,
        "input_tokens": details.get("input_tokens") if calls else 0,
        "output_tokens": details.get("output_tokens") if calls else 0,
        "model_latency_ms": details.get("latency_ms") if calls else 0,
        "semantic_skips": int(selection.get("selected") is False),
    }


def sum_usage(items: list[dict]) -> dict:
    totals = {}
    for key in ("model_calls", "input_tokens", "output_tokens", "model_latency_ms", "semantic_skips"):
        values = [item[key] for item in items]
        totals[key] = None if any(value is None for value in values) else sum(values)
    return totals


