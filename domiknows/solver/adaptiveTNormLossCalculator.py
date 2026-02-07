"""
Adaptive T-Norm Loss Calculator for DomiKnowS.

Tracks per-constraint-type loss/gradient history and adaptively selects
the best t-norm for each constraint type based on training dynamics.

Integrates with LossCalculator by updating LossCalculator.TNORM_CONFIG
so that the next epoch uses the recommended t-norms automatically.
"""

import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TNormType(Enum):
    LUKASIEWICZ = "L"
    PRODUCT = "P"
    SIMPLIFIED_PRODUCT = "SP"
    GODEL = "G"


COUNTING_TYPES = frozenset([
    'sumL', 'atLeastL', 'atLeastAL', 'atMostL', 'atMostAL', 'exactL', 'exactAL'
])

GLOBAL_LC_PREFIX = "LC"
EXECUTABLE_LC_PREFIX = "ELC"


def is_global_constraint(lc_name: str) -> bool:
    """Global constraints have names like LC0, LC1 (not ELC0)."""
    return lc_name.startswith(GLOBAL_LC_PREFIX) and not lc_name.startswith(EXECUTABLE_LC_PREFIX)


def get_constraint_type(lc) -> str:
    """Extract constraint type name from LC object."""
    if lc is None:
        return 'other'
    if hasattr(lc, 'innerLC') and lc.innerLC is not None:
        lc = lc.innerLC
    type_name = type(lc).__name__
    # Normalize AL/L variants for grouping
    normalize = {
        'atMostL': 'atMostAL', 'atLeastL': 'atLeastAL',
        'exactL': 'exactAL',
    }
    return normalize.get(type_name, type_name)


@dataclass
class ConstraintMetrics:
    """Metrics tracked for a single constraint."""
    losses: List[float] = field(default_factory=list)
    gradients: List[float] = field(default_factory=list)
    tnorms_used: List[str] = field(default_factory=list)

    loss_by_tnorm: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    grad_by_tnorm: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    current_tnorm: str = "L"
    tnorm_scores: Dict[str, float] = field(default_factory=lambda: {"L": 0.0, "P": 0.0, "SP": 0.0, "G": 0.0})

    constraint_type: str = ""
    constraint_formula: str = ""

    def add_observation(self, loss: float, grad_norm: float, tnorm: str):
        self.losses.append(loss)
        self.gradients.append(grad_norm)
        self.tnorms_used.append(tnorm)
        self.loss_by_tnorm[tnorm].append(loss)
        self.grad_by_tnorm[tnorm].append(grad_norm)

    def get_best_tnorm(self) -> Tuple[str, float]:
        best_tnorm = self.current_tnorm
        best_loss = float('inf')
        for tnorm, losses in self.loss_by_tnorm.items():
            if losses:
                avg = np.mean(losses[-10:])
                if avg < best_loss:
                    best_loss = avg
                    best_tnorm = tnorm
        return best_tnorm, best_loss

    @property
    def avg_loss(self) -> float:
        return np.mean(self.losses[-10:]) if self.losses else float('inf')

    @property
    def avg_gradient(self) -> float:
        return np.mean(self.gradients[-10:]) if self.gradients else 0.0

    @property
    def loss_trend(self) -> float:
        if len(self.losses) < 2:
            return 0.0
        recent = self.losses[-5:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)

    @property
    def gradient_health(self) -> str:
        avg = self.avg_gradient
        if avg < 1e-7:
            return "vanishing"
        elif avg > 100:
            return "exploding"
        return "healthy"


@dataclass
class EpochTypeMetrics:
    """Per-epoch, per-constraint-type metrics. Reset each epoch."""
    losses: List[float] = field(default_factory=list)
    grads: List[float] = field(default_factory=list)
    by_tnorm: Dict[str, List[float]] = field(default_factory=lambda: {t: [] for t in ["L", "P", "SP", "G"]})


class AdaptiveTNormLossCalculator:
    """
    Tracks per-constraint-type loss/gradient history and adaptively selects
    the best t-norm. At epoch end, pushes recommendations into
    LossCalculator.TNORM_CONFIG so they take effect on next epoch.
    """

    def __init__(
        self,
        solver=None,
        tnorms: List[str] = None,
        adaptation_interval: int = 50,
        warmup_steps: int = 100,
        selection_strategy: str = "gradient_weighted",
        auto_apply: bool = True,
        min_observations: int = 20,
    ):
        self.solver = solver
        self.tnorms = tnorms or ["L", "P", "SP", "G"]
        self.adaptation_interval = adaptation_interval
        self.warmup_steps = warmup_steps
        self.selection_strategy = selection_strategy
        self.auto_apply = auto_apply
        self.min_observations = min_observations

        # Per-constraint tracking (cumulative, for individual constraint view)
        self.constraint_metrics: Dict[str, ConstraintMetrics] = defaultdict(ConstraintMetrics)

        # Per-epoch, per-type tracking (reset each epoch)
        self._epoch_type_metrics: Dict[str, EpochTypeMetrics] = {}

        # Cumulative per-type tracking (across epochs, for stable recommendations)
        self._cumulative_type_metrics: Dict[str, EpochTypeMetrics] = {}

        self.step_count = 0
        self.epoch_count = 0

        # Track which t-norm is currently active per type
        self.active_tnorms: Dict[str, str] = {}

        # History of recommendations per epoch
        self.recommendation_history: List[Dict[str, str]] = []

        # Default t-norms per constraint type
        self.default_tnorms = {
            "sumL": "L",
            "atLeastAL": "L",
            "atMostAL": "L",
            "exactAL": "L",
            "andL": "SP",
            "orL": "SP",
            "ifL": "P",
            "existsL": "L",
            "notL": "SP",
        }

    def _get_or_create_epoch_metrics(self, ctype: str) -> EpochTypeMetrics:
        if ctype not in self._epoch_type_metrics:
            self._epoch_type_metrics[ctype] = EpochTypeMetrics()
        return self._epoch_type_metrics[ctype]

    def _get_or_create_cumulative_metrics(self, ctype: str) -> EpochTypeMetrics:
        if ctype not in self._cumulative_type_metrics:
            self._cumulative_type_metrics[ctype] = EpochTypeMetrics()
        return self._cumulative_type_metrics[ctype]

    def record_observation(self, lc_name: str, lc, loss_val: float, grad_norm: float, tnorm: str):
        """Record a single constraint observation during training."""
        ctype = get_constraint_type(lc)

        # Individual constraint tracking
        metrics = self.constraint_metrics[lc_name]
        metrics.constraint_type = ctype
        metrics.add_observation(loss_val, grad_norm, tnorm)

        # Per-epoch type tracking
        epoch_m = self._get_or_create_epoch_metrics(ctype)
        epoch_m.losses.append(loss_val)
        epoch_m.grads.append(grad_norm)
        epoch_m.by_tnorm[tnorm].append(loss_val)

        # Cumulative type tracking
        cum_m = self._get_or_create_cumulative_metrics(ctype)
        cum_m.losses.append(loss_val)
        cum_m.grads.append(grad_norm)
        cum_m.by_tnorm[tnorm].append(loss_val)

    def record_tnorm_comparison(self, lc_name: str, lc, tnorm: str, loss_val: float):
        """Record an alternative t-norm loss for comparison."""
        ctype = get_constraint_type(lc)

        # Individual
        self.constraint_metrics[lc_name].loss_by_tnorm[tnorm].append(loss_val)

        # Per-epoch type
        self._get_or_create_epoch_metrics(ctype).by_tnorm[tnorm].append(loss_val)

        # Cumulative type
        self._get_or_create_cumulative_metrics(ctype).by_tnorm[tnorm].append(loss_val)

    def get_recommendations(self, use_cumulative: bool = True) -> Dict[str, str]:
        """Get best t-norm per constraint type based on tracked metrics."""
        source = self._cumulative_type_metrics if use_cumulative else self._epoch_type_metrics
        recommendations = {}

        for ctype, data in source.items():
            if len(data.losses) < self.min_observations:
                # Not enough data — keep default or current
                recommendations[ctype] = self.active_tnorms.get(
                    ctype, self.default_tnorms.get(ctype, 'L'))
                continue

            best_tnorm = self.active_tnorms.get(ctype, 'L')
            best_score = -float('inf')

            for tnorm in self.tnorms:
                tnorm_losses = data.by_tnorm.get(tnorm, [])
                if len(tnorm_losses) < max(5, self.min_observations // 4):
                    continue

                score = self._score_tnorm(tnorm_losses, data.grads)
                if score > best_score:
                    best_score = score
                    best_tnorm = tnorm

            recommendations[ctype] = best_tnorm

        return recommendations

    def _score_tnorm(self, losses: List[float], grads: List[float]) -> float:
        """Score a t-norm. Higher = better."""
        if not losses:
            return -float('inf')

        recent_losses = losses[-20:]
        avg_loss = np.mean(recent_losses)

        # Lower average loss is better
        loss_score = -avg_loss * 10.0

        # Loss trend (improving = good)
        if len(recent_losses) >= 4:
            first_half = np.mean(recent_losses[:len(recent_losses)//2])
            second_half = np.mean(recent_losses[len(recent_losses)//2:])
            improvement = first_half - second_half  # positive = improving
            loss_score += improvement * 5.0

        if self.selection_strategy == "gradient_weighted" and grads:
            recent_grads = grads[-20:]
            avg_grad = np.mean(recent_grads)
            if avg_grad < 1e-6:
                loss_score -= 10.0  # Penalize vanishing gradients
            elif avg_grad > 100:
                loss_score -= 5.0   # Penalize exploding gradients

        return loss_score

    def _score_loss_based(self, losses: List[float]) -> float:
        if not losses:
            return -float('inf')
        return -np.mean(losses[-10:])

    def apply_recommendations(self, recommendations: Dict[str, str]):
        """Push recommended t-norms into LossCalculator.TNORM_CONFIG."""
        from domiknows.solver.lossCalculator import LossCalculator

        changed = {}
        for ctype, tnorm in recommendations.items():
            old = LossCalculator.TNORM_CONFIG.get(ctype, 'L')
            if old != tnorm:
                LossCalculator.set_tnorm_for_type(ctype, tnorm)
                changed[ctype] = (old, tnorm)

        self.active_tnorms.update(recommendations)
        return changed

    def get_constraint_coverage(self) -> Tuple[int, int]:
        """Count global vs executable constraints seen this epoch."""
        global_count = 0
        exec_count = 0
        for lc_name in self.constraint_metrics:
            if is_global_constraint(lc_name):
                global_count += 1
            else:
                exec_count += 1
        return global_count, exec_count

    def on_epoch_end(self, apply: Optional[bool] = None) -> Dict[str, str]:
        """
        End-of-epoch processing:
        1. Print analysis
        2. Compute recommendations
        3. Optionally apply to LossCalculator
        4. Reset epoch metrics
        
        Returns the recommendations dict.
        """
        self.epoch_count += 1
        should_apply = apply if apply is not None else self.auto_apply

        print(f"\n[Epoch {self.epoch_count}] Adaptive T-Norm Analysis")
        print("=" * 75)

        # --- Aggregated by constraint type ---
        print("\n📊 AGGREGATED BY CONSTRAINT TYPE (what the model actually learns):")
        print("-" * 75)
        print(f"{'Type':<12} {'Count':>6} {'AvgLoss':>8} {'AvgGrad':>8} │ {'L':>7} {'P':>7} {'SP':>7} {'G':>7} │ Best")
        print("-" * 75)

        recommendations = self.get_recommendations(use_cumulative=True)

        for ctype in sorted(self._epoch_type_metrics.keys()):
            data = self._epoch_type_metrics[ctype]
            if not data.losses:
                continue

            count = len(data.losses)
            avg_loss = np.mean(data.losses)
            avg_grad = np.mean(data.grads) if data.grads else 0

            best_tnorm = recommendations.get(ctype, 'L')

            tnorm_strs = []
            for tnorm in ["L", "P", "SP", "G"]:
                tnorm_losses = data.by_tnorm.get(tnorm, [])
                if tnorm_losses:
                    avg = np.mean(tnorm_losses)
                    marker = "✓" if tnorm == best_tnorm else " "
                    tnorm_strs.append(f"{avg:>6.3f}{marker}")
                else:
                    tnorm_strs.append(f"{'---':>7}")

            print(f"{ctype:<12} {count:>6} {avg_loss:>8.4f} {avg_grad:>8.2f} │ {' '.join(tnorm_strs)} │ {best_tnorm}")

        print("-" * 75)

        # --- Constraint coverage ---
        global_count, exec_count = self.get_constraint_coverage()
        print(f"\n📈 CONSTRAINT COVERAGE:")
        print(f"   Global constraints (graph-level):    {global_count}")
        print(f"   Executable constraints (per-sample): {exec_count}")

        # --- Recommendations ---
        print(f"\n💡 RECOMMENDATIONS:")
        for ctype in sorted(recommendations.keys()):
            if ctype in self._epoch_type_metrics and self._epoch_type_metrics[ctype].losses:
                tnorm = recommendations[ctype]
                current = self.active_tnorms.get(ctype, '?')
                changed = " ← SWITCH" if current != tnorm and current != '?' else ""
                print(f"   {ctype}: Use t-norm '{tnorm}'{changed}")

        # --- Apply ---
        if should_apply and self.epoch_count > 1:
            changed = self.apply_recommendations(recommendations)
            if changed:
                print(f"\n🔄 APPLIED T-NORM CHANGES:")
                for ctype, (old, new) in changed.items():
                    print(f"   {ctype}: '{old}' → '{new}'")
            else:
                print(f"\n   (no t-norm changes needed)")
        elif not should_apply:
            print(f"\n   (auto-apply disabled, use recommendations manually)")

        self.recommendation_history.append(recommendations)

        print("\n" + "=" * 75)

        # Reset per-epoch metrics
        self._epoch_type_metrics.clear()

        return recommendations

    def get_tnorm_for_constraint(self, lc_name: str) -> str:
        """Get the currently selected t-norm for a constraint."""
        if lc_name in self.constraint_metrics:
            ctype = self.constraint_metrics[lc_name].constraint_type
            if ctype in self.active_tnorms:
                return self.active_tnorms[ctype]
        return "L"

    def get_adaptive_tnorm_dict(self) -> Dict[str, str]:
        return {
            lc_name: self.get_tnorm_for_constraint(lc_name)
            for lc_name in self.constraint_metrics
        }

    def get_summary_stats(self) -> Dict:
        return {
            lc_name: {
                "current_tnorm": self.get_tnorm_for_constraint(lc_name),
                "constraint_type": m.constraint_type,
                "avg_loss": m.avg_loss,
                "avg_gradient": m.avg_gradient,
                "loss_trend": m.loss_trend,
                "gradient_health": m.gradient_health,
                "observations": len(m.losses),
            }
            for lc_name, m in self.constraint_metrics.items()
        }

    def calculate_loss_with_comparison(self, dn, primary_tnorm="L", compare_all=False) -> Dict:
        """Calculate loss and optionally compare across t-norms."""
        self.step_count += 1

        results = {
            "primary_losses": {},
            "comparison": {} if compare_all else None,
            "selected_tnorms": {},
        }

        primary_losses = self._calculate_loss(dn, primary_tnorm)
        results["primary_losses"] = primary_losses

        for lc_name, loss_dict in primary_losses.items():
            loss_tensor = loss_dict.get("loss")
            if loss_tensor is not None and torch.is_tensor(loss_tensor):
                grad_norm = self._compute_gradient_norm(loss_tensor)
                loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
                lc = loss_dict.get('lc')
                self.record_observation(lc_name, lc, loss_val, grad_norm, primary_tnorm)

        if compare_all and self.step_count % self.adaptation_interval == 0:
            results["comparison"] = self._compare_tnorms(dn)

        for lc_name in primary_losses:
            results["selected_tnorms"][lc_name] = self.get_tnorm_for_constraint(lc_name)

        return results

    def _calculate_loss(self, dn, tnorm: str) -> Dict:
        from domiknows.solver.lossCalculator import LossCalculator
        calculator = LossCalculator(self.solver)
        return calculator.calculateLoss(dn, tnorm=tnorm)

    def _compute_gradient_norm(self, loss_tensor: torch.Tensor) -> float:
        if not loss_tensor.requires_grad:
            return 0.0
        try:
            grad = torch.autograd.grad(
                loss_tensor, loss_tensor,
                retain_graph=True, allow_unused=True
            )
            if grad[0] is not None:
                return grad[0].norm().item()
        except Exception:
            pass
        return 0.0

    def _compare_tnorms(self, dn) -> Dict[str, Dict[str, float]]:
        comparison = {}
        for tnorm in self.tnorms:
            try:
                losses = self._calculate_loss(dn, tnorm)
                for lc_name, loss_dict in losses.items():
                    if lc_name not in comparison:
                        comparison[lc_name] = {}
                    loss_tensor = loss_dict.get("loss")
                    if loss_tensor is not None and torch.is_tensor(loss_tensor):
                        loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
                        comparison[lc_name][tnorm] = loss_val
                        lc = loss_dict.get('lc')
                        self.record_tnorm_comparison(lc_name, lc, tnorm, loss_val)
            except Exception:
                pass
        return comparison