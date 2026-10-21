"""Bounded terminal reward of a simulator rollout, from VLABench's success, progress and intention signals."""
from __future__ import annotations

from typing import Any

from ..reward import RewardBreakdown, _clamp


class RolloutRewardAccumulator:
    """Compute the bounded terminal reward from VLABench environment signals."""

    def __init__(self, max_steps: int):
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.progress = 0.0
        self.intention = 0.0
        self.valid = True

    def update(self, *, progress: Any = None, intention: Any = None, valid: bool = True, steps: int = 1) -> None:
        self.steps = min(self.max_steps, self.steps + max(0, int(steps)))
        if progress is not None:
            self.progress = _clamp(progress)
        if intention is not None:
            self.intention = _clamp(intention)
        self.valid = self.valid and bool(valid)

    def finalize(
        self,
        success: Any,
        *,
        progress: Any = None,
        intention: Any = None,
        steps: int | None = None,
        valid: bool | None = None,
        mode: str = "composite",
    ) -> RewardBreakdown:
        if mode not in {"composite", "success"}:
            raise ValueError("rollout reward mode must be 'composite' or 'success'")
        final_steps = self.steps if steps is None else max(0, min(self.max_steps, int(steps)))
        final_progress = self.progress if progress is None else _clamp(progress)
        final_intention = self.intention if intention is None else _clamp(intention)
        final_valid = self.valid if valid is None else self.valid and bool(valid)
        final_success = _clamp(success)
        efficiency = final_success * _clamp(1.0 - final_steps / self.max_steps)
        if mode == "success":
            total = final_success
        else:
            total = (
                0.60 * final_success
                + 0.25 * final_progress
                + 0.10 * final_intention
                + 0.05 * efficiency
            )
        if not final_valid:
            total = 0.0
        return RewardBreakdown(
            total=_clamp(total),
            success=final_success,
            progress=final_progress,
            intention=final_intention,
            efficiency=efficiency,
            valid=final_valid,
            steps=final_steps,
            max_steps=self.max_steps,
        )
