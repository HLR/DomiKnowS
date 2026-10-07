"""Paired assistance-ablation evaluation for the VLABench controller.

Execution assistance (scripted orientation, approach, skills and grasp
latching) is training scaffolding. This module measures how much each family
contributes to simulator success by evaluating the *same* checkpoint on the
*same* fixed-seed scenes under several assistance modes, and reports Wilson
confidence intervals plus an exact McNemar test against the first mode.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .program import ASSIST_COMPONENTS, parse_assist_components

# mode name -> (assistance enabled, component selection)
ASSIST_MODES: dict[str, tuple[bool, tuple[str, ...]]] = {
    "unassisted": (False, ()),
    "orientation_only": (True, ("orientation",)),
    "approach_only": (True, ("approach",)),
    "orientation_approach": (True, ("orientation", "approach")),
    "no_orientation": (True, tuple(name for name in ASSIST_COMPONENTS if name != "orientation")),
    "no_latch": (True, tuple(name for name in ASSIST_COMPONENTS if name != "latch")),
    "full": (True, tuple(ASSIST_COMPONENTS)),
}

DEFAULT_ASSIST_MODES = ("unassisted", "orientation_only", "full")


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""

    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(only_first, only_second) + 1))
    return min(1.0, 2.0 * tail / 2 ** discordant)


def parse_assist_modes(value: str | Sequence[str] | None) -> list[tuple[str, bool, frozenset[str]]]:
    """Resolve mode names (or ``a+b`` component lists) to (name, assist, components)."""

    if value is None:
        names = list(DEFAULT_ASSIST_MODES)
    elif isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in value]
    if not names:
        raise ValueError("at least one assistance mode is required")
    modes: list[tuple[str, bool, frozenset[str]]] = []
    for name in names:
        if name in ASSIST_MODES:
            assist, components = ASSIST_MODES[name]
            modes.append((name, assist, frozenset(components)))
        else:
            # A custom mode such as "orientation+approach".
            modes.append((name, True, parse_assist_components(name)))
    return modes


def _proportion(successes: int, trials: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, trials)
    return {
        "count": int(successes),
        "n": int(trials),
        "rate": successes / trials if trials else 0.0,
        "wilson95": [low, high],
    }


def summarize_mode(result: Mapping[str, Any]) -> dict[str, Any]:
    """Condense one ``evaluate_rollouts`` result with confidence intervals."""

    episodes = result["episode_diagnostics"]
    n = len(episodes)
    successes = sum(bool(item["success"]) for item in episodes)
    positive = sum(float(item["return"]) > 1e-6 for item in episodes)
    per_task: dict[str, dict[str, Any]] = {}
    for item in episodes:
        entry = per_task.setdefault(item["task"], {"n": 0, "successes": 0, "return": 0.0})
        entry["n"] += 1
        entry["successes"] += int(bool(item["success"]))
        entry["return"] += float(item["return"])
    for entry in per_task.values():
        entry["return"] /= max(1, entry["n"])
        entry["success"] = _proportion(entry["successes"], entry["n"])
    return {
        "success": _proportion(successes, n),
        "positive_return": _proportion(positive, n),
        "mean_return": sum(float(item["return"]) for item in episodes) / max(1, n),
        "unassisted_success_rate": result.get("unassisted_success_rate"),
        "assisted_episode_rate": result.get("assisted_episode_rate"),
        "ik_truncation_rate": result.get("ik_truncation_rate"),
        "per_task": per_task,
    }


def paired_comparison(reference: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Compare per-episode success on identical scenes (same evaluation seed)."""

    first = [bool(item["success"]) for item in reference["episode_diagnostics"]]
    second = [bool(item["success"]) for item in other["episode_diagnostics"]]
    if len(first) != len(second):
        raise ValueError("paired comparison requires equally sized evaluations")
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    return {
        "only_reference_success": only_first,
        "only_other_success": only_second,
        "both_success": sum(a and b for a, b in zip(first, second)),
        "mcnemar_exact_p": mcnemar_exact_p(only_first, only_second),
    }


def run_assist_ablation(
    program: Any,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    rollouts_per_task: int,
    seed: int,
    modes: Sequence[tuple[str, bool, frozenset[str]]],
    progress=None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate each mode on identical scenes and compare against the first."""

    results: dict[str, Any] = {}
    raw: dict[str, Mapping[str, Any]] = {}
    for name, assist, components in modes:
        if progress is not None:
            progress(
                f"assist ablation mode={name} assistance={assist} "
                f"components={sorted(components)} rollouts/task={rollouts_per_task}"
            )
        evaluation = program.evaluate_rollouts(
            descriptors,
            rollouts_per_task=rollouts_per_task,
            seed=seed,
            assistance=assist,
            assist_components=components,
        )
        raw[name] = evaluation
        results[name] = {
            "assistance": assist,
            "components": sorted(components),
            **summarize_mode(evaluation),
        }
    reference_name = modes[0][0]
    for name, _assist, _components in modes[1:]:
        results[name]["paired_vs_" + reference_name] = paired_comparison(raw[reference_name], raw[name])
    report = {
        "seed": int(seed),
        "rollouts_per_task": int(rollouts_per_task),
        "reference_mode": reference_name,
        "modes": results,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
