"""
Adaptive T-Norm Loss Calculator for DomiKnowS.

Provides unified t-norm selection for both LossCalculator and SampleLossCalculator.

Three modes:
  - Specific ("G", "P", "L", "SP"): Use that t-norm for ALL constraints, no adaptation.
  - "default": Use per-type default mapping (DEFAULT_TNORM_BY_TYPE).
  - "auto": Dynamically adapt t-norm per constraint type during training.
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


VALID_TNORMS = frozenset(["L", "P", "SP", "G"])

GLOBAL_LC_PREFIX = "LC"
EXECUTABLE_LC_PREFIX = "ELC"

# Optimal t-norm mapping based on mathematical properties
DEFAULT_TNORM_BY_TYPE = {
    # Counting
    'sumL': 'L',
    'atLeastL': 'L',
    'atLeastAL': 'L',
    # Upper bounds
    'atMostL': 'G',
    'atMostAL': 'G',
    # Exact count
    'exactL': 'L',
    'exactAL': 'L',
    # Exactly-one-of (issue #371)
    'oneOfL': 'L',
    'oneOfAL': 'L',
    # Boolean logic
    'andL': 'SP',
    'orL': 'SP',
    'nandL': 'SP',
    'norL': 'SP',
    'ifL': 'P',
    'equivalenceL': 'P',
    'iffL': 'P',
    'existsL': 'L',
    'existsL': 'P',
    'queryL': 'P',
    'iotaL': 'P',
    'notL': 'SP',
    # Fallback
    'default': 'L',
}

COUNTING_CONSTRAINTS = {'sumL', 'atLeastL', 'atLeastAL', 'atMostL', 'atMostAL', 'exactL', 'exactAL', 'oneOfL', 'oneOfAL'}

def is_global_constraint(lc_name: str) -> bool:
    """Global constraints have names like LC0, LC1 (not ELC0)."""
    return lc_name.startswith(GLOBAL_LC_PREFIX) and not lc_name.startswith(EXECUTABLE_LC_PREFIX)


def get_constraint_type(lc) -> str:
    """Extract constraint type name from LC object."""
    if lc is None:
        return 'default'
    if hasattr(lc, 'innerLC') and lc.innerLC is not None:
        lc = lc.innerLC
    type_name = type(lc).__name__
    normalize = {
        'atMostL': 'atMostAL', 'atLeastL': 'atLeastAL',
        'exactL': 'exactAL',
        'oneOfL': 'oneOfAL',
    }
    return normalize.get(type_name, type_name)


def resolve_tnorm_mode(tnorm_arg: str, counting_tnorm=None) -> str:
    """
    Classify a t-norm argument into one of three modes.

    Returns:
        "specific" if tnorm_arg is a valid t-norm code (G, P, L, SP)
        "default"  if tnorm_arg == "default" or counting_tnorm == "default"
        "auto"     if tnorm_arg == "auto" or counting_tnorm == "auto"
    
    Raises ValueError for unrecognized values.
    """
    if tnorm_arg in VALID_TNORMS and (counting_tnorm in VALID_TNORMS.union({"default", "auto"}) if counting_tnorm else True):
        return "specific"
    if tnorm_arg == "default" or counting_tnorm == "default":
        return "default"
    if tnorm_arg == "auto" or counting_tnorm == "auto":
        return "auto"
    raise ValueError(
        f"Invalid t-norm argument '{tnorm_arg}'. "
        f"Must be one of: {sorted(VALID_TNORMS)}, 'default', or 'auto'."
    )


class TNormSelector:
    """
    Unified t-norm selection logic shared by LossCalculator and SampleLossCalculator.

    Modes:
      - specific: Always returns the single specified t-norm.
      - default:  Looks up per-type defaults from tnorm_config.
      - auto:     Delegates to AdaptiveTNormLossCalculator's active_tnorms,
                  falling back to per-type defaults when no recommendation exists.
    """

    def __init__(self, tnorm_arg: str = "L", counting_tnorm=None, adaptive_tracker: Optional['AdaptiveTNormLossCalculator'] = None):
        """
        Args:
            tnorm_arg: One of "L","P","SP","G","default","auto".
            counting_tnorm: Deprecated —  but stil accepted for backward compatibility. Ignored if tnorm is a auto.
            adaptive_tracker: Required when mode is "auto". The tracker that
                              holds active_tnorms updated at epoch boundaries.
        """
        self.mode = resolve_tnorm_mode(tnorm_arg, counting_tnorm)
        self.specific_tnorm = tnorm_arg if self.mode == "specific" else None
        self.specific_counting_tnorm = counting_tnorm if self.mode == "specific" else None  
        self.tnorm_config = DEFAULT_TNORM_BY_TYPE.copy()
        self.adaptive_tracker = adaptive_tracker
        
        if self.adaptive_tracker is None and self.mode == "auto":
           self.adaptive_tracker = AdaptiveTNormLossCalculator()
                             

    def update_config(self, config: Dict[str, str]):
        """Update the per-type default config (used in 'default' and 'auto' fallback)."""
        self.tnorm_config.update(config)

    def select(self, lc=None) -> str:
        """
        Select the t-norm to use for a given logical constraint.

        The TNormSelector already knows the correct t-norm for each constraint
        type (including counting constraints) based on its mode:
          - specific: always returns the single fixed t-norm
          - default: looks up per-type defaults (counting types get their own defaults)
          - auto: uses adaptive recommendations, falls back to per-type defaults

        Args:
            lc: The logical constraint object (used for type lookup in default/auto modes).

        Returns:
            A t-norm code string: "L", "P", "SP", or "G".
        """
        # --- Specific mode: always one t-norm ---
        if self.mode == "specific":
            if lc is not None and get_constraint_type(lc) in COUNTING_CONSTRAINTS and self.specific_counting_tnorm is not None:
                return self.specific_counting_tnorm
            return self.specific_tnorm

        ctype = get_constraint_type(lc)

        # --- Auto mode: prefer adaptive recommendation, fall back to config ---
        if self.mode == "auto" and self.adaptive_tracker is not None:
            active = self.adaptive_tracker.active_tnorms
            if ctype in active:
                return active[ctype]
            # Fall through to default lookup

        # --- Default mode (or auto fallback) ---
        return self.tnorm_config.get(ctype, self.tnorm_config.get('default', 'G'))


# ---------------------------------------------------------------------------
# Metrics dataclasses
# ---------------------------------------------------------------------------

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
    global_count: int = 0
    executable_count: int = 0


# ---------------------------------------------------------------------------
# Main adaptive tracker
# ---------------------------------------------------------------------------

class AdaptiveTNormLossCalculator:
    """
    Tracks per-constraint-type loss/gradient history and adaptively selects
    the best t-norm.  At epoch end, pushes recommendations into active_tnorms
    (and optionally into LossCalculator.TNORM_CONFIG for backward compat).

    Also serves as the backing store for TNormSelector in "auto" mode.
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

        self.constraint_metrics: Dict[str, ConstraintMetrics] = defaultdict(ConstraintMetrics)
        self._epoch_type_metrics: Dict[str, EpochTypeMetrics] = {}
        self._cumulative_type_metrics: Dict[str, EpochTypeMetrics] = {}

        self.step_count = 0
        self.epoch_count = 0

        # Currently active t-norm per constraint type (read by TNormSelector)
        self.active_tnorms: Dict[str, str] = {}

        self.recommendation_history: List[Dict[str, str]] = []

        self.default_tnorms = DEFAULT_TNORM_BY_TYPE.copy()

    # --- Recording ---------------------------------------------------------------

    def _get_or_create_epoch_metrics(self, ctype: str) -> EpochTypeMetrics:
        if ctype not in self._epoch_type_metrics:
            self._epoch_type_metrics[ctype] = EpochTypeMetrics()
        return self._epoch_type_metrics[ctype]

    def _get_or_create_cumulative_metrics(self, ctype: str) -> EpochTypeMetrics:
        if ctype not in self._cumulative_type_metrics:
            self._cumulative_type_metrics[ctype] = EpochTypeMetrics()
        return self._cumulative_type_metrics[ctype]

    def record_observation(self, lc_name: str, lc, loss_val: float, grad_norm: float, tnorm: str):
        ctype = get_constraint_type(lc)
        is_global = is_global_constraint(lc_name)

        metrics = self.constraint_metrics[lc_name]
        metrics.constraint_type = ctype
        metrics.add_observation(loss_val, grad_norm, tnorm)

        epoch_m = self._get_or_create_epoch_metrics(ctype)
        epoch_m.losses.append(loss_val)
        epoch_m.grads.append(grad_norm)
        epoch_m.by_tnorm[tnorm].append(loss_val)
        if is_global:
            epoch_m.global_count += 1
        else:
            epoch_m.executable_count += 1

        cum_m = self._get_or_create_cumulative_metrics(ctype)
        cum_m.losses.append(loss_val)
        cum_m.grads.append(grad_norm)
        cum_m.by_tnorm[tnorm].append(loss_val)
        if is_global:
            cum_m.global_count += 1
        else:
            cum_m.executable_count += 1

    def record_tnorm_comparison(self, lc_name: str, lc, tnorm: str, loss_val: float):
        ctype = get_constraint_type(lc)
        self.constraint_metrics[lc_name].loss_by_tnorm[tnorm].append(loss_val)
        self._get_or_create_epoch_metrics(ctype).by_tnorm[tnorm].append(loss_val)
        self._get_or_create_cumulative_metrics(ctype).by_tnorm[tnorm].append(loss_val)

    # --- Recommendations ---------------------------------------------------------

    def get_recommendations(self, use_cumulative: bool = True) -> Dict[str, str]:
        source = self._cumulative_type_metrics if use_cumulative else self._epoch_type_metrics
        recommendations = {}

        for ctype, data in source.items():
            if len(data.losses) < self.min_observations:
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

    def get_detailed_recommendations(self, use_cumulative: bool = True) -> Dict[str, Dict]:
        source = self._cumulative_type_metrics if use_cumulative else self._epoch_type_metrics
        details = {}

        for ctype, data in source.items():
            if len(data.losses) < self.min_observations:
                continue

            tnorm_details = {}
            for tnorm in self.tnorms:
                tnorm_losses = data.by_tnorm.get(tnorm, [])
                if len(tnorm_losses) < max(5, self.min_observations // 4):
                    continue

                recent_losses = tnorm_losses[-20:]
                avg_loss = np.mean(recent_losses)

                trend = 0.0
                if len(recent_losses) >= 4:
                    first_half = np.mean(recent_losses[:len(recent_losses)//2])
                    second_half = np.mean(recent_losses[len(recent_losses)//2:])
                    trend = first_half - second_half

                grad_status = "healthy"
                if self.selection_strategy == "gradient_weighted" and data.grads:
                    recent_grads = data.grads[-20:]
                    avg_grad = np.mean(recent_grads)
                    if avg_grad < 1e-6:
                        grad_status = "vanishing"
                    elif avg_grad > 100:
                        grad_status = "exploding"

                score = self._score_tnorm(tnorm_losses, data.grads)

                tnorm_details[tnorm] = {
                    'loss': avg_loss,
                    'trend': trend,
                    'grad_status': grad_status,
                    'score': score,
                    'observations': len(tnorm_losses)
                }

            details[ctype] = tnorm_details

        return details

    def _score_tnorm(self, losses: List[float], grads: List[float]) -> float:
        if not losses:
            return -float('inf')

        recent_losses = losses[-20:]
        avg_loss = np.mean(recent_losses)
        loss_score = -avg_loss * 10.0

        if len(recent_losses) >= 4:
            first_half = np.mean(recent_losses[:len(recent_losses)//2])
            second_half = np.mean(recent_losses[len(recent_losses)//2:])
            improvement = first_half - second_half
            loss_score += improvement * 5.0

        if self.selection_strategy == "gradient_weighted" and grads:
            recent_grads = grads[-20:]
            avg_grad = np.mean(recent_grads)
            if avg_grad < 1e-6:
                loss_score -= 10.0
            elif avg_grad > 100:
                loss_score -= 5.0

        return loss_score

    def apply_recommendations(self, recommendations: Dict[str, str]):
        """Push recommended t-norms into active_tnorms and LossCalculator.TNORM_CONFIG."""
        changed = {}
        old_active = dict(self.active_tnorms)

        self.active_tnorms.update(recommendations)

        for ctype, tnorm in recommendations.items():
            old = old_active.get(ctype, 'L')
            if old != tnorm:
                changed[ctype] = (old, tnorm)

        # Backward compat: also push to LossCalculator class-level config
        try:
            from domiknows.solver.lossCalculator import LossCalculator as _LC
            for ctype, tnorm in recommendations.items():
                _LC.TNORM_CONFIG[ctype] = tnorm
        except ImportError:
            pass

        return changed

    def get_constraint_coverage(self) -> Tuple[int, int]:
        global_count = 0
        exec_count = 0
        for lc_name in self.constraint_metrics:
            if is_global_constraint(lc_name):
                global_count += 1
            else:
                exec_count += 1
        return global_count, exec_count

    # --- Epoch boundary ----------------------------------------------------------

    def on_epoch_end(self, apply: Optional[bool] = None) -> Dict[str, str]:
        self.epoch_count += 1
        should_apply = apply if apply is not None else self.auto_apply

        print(f"\n[Epoch {self.epoch_count}] Adaptive T-Norm Analysis")
        print("=" * 95)

        print("\n\u0001f4ca AGGREGATED BY CONSTRAINT TYPE (what the model actually learns):")
        print("-" * 95)
        print(f"{'Type':<12} {'Count':>6} {'Global':>7} {'Exec':>7} {'AvgLoss':>8} {'AvgGrad':>8} | {'L':>7} {'P':>7} {'SP':>7} {'G':>7} | Best")
        print("-" * 95)

        recommendations = self.get_recommendations(use_cumulative=True)

        for ctype in sorted(self._epoch_type_metrics.keys()):
            data = self._epoch_type_metrics[ctype]
            if not data.losses:
                continue

            count = len(data.losses)
            global_cnt = data.global_count
            exec_cnt = data.executable_count
            avg_loss = np.mean(data.losses)
            avg_grad = np.mean(data.grads) if data.grads else 0

            best_tnorm = recommendations.get(ctype, 'L')

            tnorm_strs = []
            for tnorm in ["L", "P", "SP", "G"]:
                tnorm_losses = data.by_tnorm.get(tnorm, [])
                if tnorm_losses:
                    avg = np.mean(tnorm_losses)
                    marker = "+" if tnorm == best_tnorm else " "
                    tnorm_strs.append(f"{avg:>6.3f}{marker}")
                else:
                    tnorm_strs.append(f"{'---':>7}")

            print(f"{ctype:<12} {count:>6} {global_cnt:>7} {exec_cnt:>7} {avg_loss:>8.4f} {avg_grad:>8.2f} | {' '.join(tnorm_strs)} | {best_tnorm}")

        print("-" * 95)

        global_count, exec_count = self.get_constraint_coverage()
        print(f"\nCONSTRAINT COVERAGE:")
        print(f"   Global constraints (graph-level):    {global_count}")
        print(f"   Executable constraints (per-sample): {exec_count}")

        print(f"\nRECOMMENDATIONS (strategy: {self.selection_strategy}):")
        detailed_info = self.get_detailed_recommendations(use_cumulative=True)

        for ctype in sorted(recommendations.keys()):
            if ctype in self._epoch_type_metrics and self._epoch_type_metrics[ctype].losses:
                tnorm = recommendations[ctype]
                current = self.active_tnorms.get(ctype, '?')
                changed = " <- SWITCH" if current != tnorm and current != '?' else ""
                print(f"   {ctype}: Use t-norm '{tnorm}'{changed}")

                if ctype in detailed_info:
                    details = detailed_info[ctype]
                    if tnorm in details:
                        chosen = details[tnorm]
                        lowest_loss_tnorm = min(details.keys(), key=lambda t: details[t]['loss'])
                        lowest_loss = details[lowest_loss_tnorm]['loss']

                        if tnorm != lowest_loss_tnorm and abs(chosen['loss'] - lowest_loss) > 0.001:
                            reasons = []
                            if chosen['trend'] > 0.01:
                                reasons.append(f"improving trend ({chosen['trend']:+.4f})")
                            if details[lowest_loss_tnorm]['grad_status'] != "healthy":
                                reasons.append(f"{lowest_loss_tnorm} has {details[lowest_loss_tnorm]['grad_status']} gradients")
                            if chosen['grad_status'] == "healthy" and details[lowest_loss_tnorm]['grad_status'] != "healthy":
                                reasons.append("healthier gradients")
                            if reasons:
                                print(f"      -> Chosen over {lowest_loss_tnorm} (loss {lowest_loss:.4f}) due to: {', '.join(reasons)}")
                        elif tnorm != lowest_loss_tnorm:
                            print(f"      -> Similar loss to best ({lowest_loss_tnorm}: {lowest_loss:.4f})")

        if should_apply and self.epoch_count > 1:
            changed = self.apply_recommendations(recommendations)
            if changed:
                print(f"\nAPPLIED T-NORM CHANGES:")
                for ctype, (old, new) in changed.items():
                    print(f"   {ctype}: '{old}' -> '{new}'")
            else:
                print(f"\n   (no t-norm changes needed)")
        elif not should_apply:
            print(f"\n   (auto-apply disabled, use recommendations manually)")

        self.recommendation_history.append(recommendations)
        print("\n" + "=" * 95)

        self._epoch_type_metrics.clear()
        return recommendations

    # --- Utility -----------------------------------------------------------------

    def get_tnorm_for_constraint(self, lc_name: str) -> str:
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

    def export_detailed_stats_to_csv(self, filename='adaptive_tnorm_details.csv'):
        import csv

        rows = []
        for constraint_name, metrics in self.constraint_metrics.items():
            if len(metrics.losses) == 0:
                continue

            row = {
                'constraint_name': constraint_name,
                'constraint_type': metrics.constraint_type or 'unknown',
                'total_observations': len(metrics.losses),
                'current_tnorm': metrics.current_tnorm,
                'avg_loss': metrics.avg_loss,
                'avg_gradient': metrics.avg_gradient,
            }

            for tnorm in self.tnorms:
                tnorm_losses = metrics.loss_by_tnorm.get(tnorm, [])
                if tnorm_losses:
                    row[f'{tnorm}_loss'] = np.mean(tnorm_losses)
                    row[f'{tnorm}_observations'] = len(tnorm_losses)
                else:
                    row[f'{tnorm}_loss'] = None
                    row[f'{tnorm}_observations'] = 0

            best_tnorm, best_loss = metrics.get_best_tnorm()
            row['best_tnorm'] = best_tnorm
            row['loss_trend'] = metrics.loss_trend
            row['gradient_health'] = metrics.gradient_health
            rows.append(row)

        if not rows:
            raise ValueError(f"No data to export to {filename}")

        fieldnames = ['constraint_name', 'constraint_type', 'total_observations',
                      'current_tnorm', 'avg_loss', 'avg_gradient']
        for tnorm in self.tnorms:
            fieldnames.extend([f'{tnorm}_loss', f'{tnorm}_observations'])
        fieldnames.extend(['best_tnorm', 'loss_trend', 'gradient_health'])

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            raise IOError(f"Failed to write CSV file {filename}: {e}")

        return len(rows)