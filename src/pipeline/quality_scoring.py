from __future__ import annotations

from typing import Any, Dict


def weighted_average(metrics: Dict[str, Any], weights: Dict[str, float]) -> float:
    total = 0.0
    denom = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        if isinstance(value, dict) and "overall" in value:
            score = float(value["overall"]) * 100.0 if value["overall"] <= 1 else float(value["overall"])
        elif isinstance(value, (int, float)):
            score = float(value)
        else:
            score = 0.0
        total += score * weight
        denom += weight
    return round(total / denom, 2) if denom else 0.0



