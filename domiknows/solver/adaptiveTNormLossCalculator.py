"""
Adaptive T-Norm Loss Calculator for DomiKnowS.

Tracks per-constraint loss/gradient history and adaptively selects
the best t-norm for each constraint based on training dynamics.
"""

import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
from enum import Enum


class TNormType(Enum):
    LUKASIEWICZ = "L"        # Linear/bounded, stable for counting
    PRODUCT = "P"            # Poisson-binomial, smooth but can vanish
    GODEL = "G"              # Min/max, sparse gradients
    SIMPLIFIED_PRODUCT = "SP" # Fast product approximation


# T-norm characteristics for adaptive selection
TNORM_PROPERTIES = {
    "L": {
        "name": "Łukasiewicz",
        "gradient": "linear",
        "best_for": ["counting", "sumL", "atLeastL", "atMostL"],
        "vanishing_risk": "low",
        "computation": "fast",
        "description": "max(0, sum - (n-1)), piece-wise linear gradients"
    },
    "P": {
        "name": "Product", 
        "gradient": "multiplicative",
        "best_for": ["boolean", "andL", "orL", "impliesL"],
        "vanishing_risk": "medium",  # Can vanish with many terms
        "computation": "slow",  # Uses Poisson-binomial
        "description": "Full probabilistic, uses calc_probabilities()"
    },
    "G": {
        "name": "Gödel",
        "gradient": "sparse",
        "best_for": ["late_training", "sharp_decisions"],
        "vanishing_risk": "high",  # Only min/max gets gradient
        "computation": "fast",
        "description": "min/max operations, only extremes get signal"
    },
    "SP": {
        "name": "Simplified Product",
        "gradient": "multiplicative",
        "best_for": ["boolean", "fast_approximation"],
        "vanishing_risk": "medium",
        "computation": "fast",  # Direct torch.prod, no Poisson-binomial
        "description": "torch.prod directly, faster than P"
    }
}


@dataclass
class ConstraintMetrics:
    """Metrics tracked for a single constraint."""
    losses: List[float] = field(default_factory=list)
    gradients: List[float] = field(default_factory=list)
    tnorms_used: List[str] = field(default_factory=list)
    
    # Per t-norm tracking
    loss_by_tnorm: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    grad_by_tnorm: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Adaptive selection state
    current_tnorm: str = "L"
    tnorm_scores: Dict[str, float] = field(default_factory=lambda: {"L": 0.0, "P": 0.0, "SP": 0.0, "G": 0.0})
    
    # Constraint metadata
    constraint_type: str = ""  # e.g., "atMostAL", "atLeastAL", "sumL"
    constraint_formula: str = ""  # e.g., "atMostAL(people, 3)"
    
    def add_observation(self, loss: float, grad_norm: float, tnorm: str):
        self.losses.append(loss)
        self.gradients.append(grad_norm)
        self.tnorms_used.append(tnorm)
        self.loss_by_tnorm[tnorm].append(loss)
        self.grad_by_tnorm[tnorm].append(grad_norm)
    
    def get_best_tnorm(self) -> Tuple[str, float]:
        """Return the t-norm with lowest average loss."""
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
        """Negative = improving, Positive = worsening."""
        if len(self.losses) < 2:
            return 0.0
        recent = self.losses[-5:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)
    
    @property  
    def gradient_health(self) -> str:
        """Classify gradient health."""
        avg = self.avg_gradient
        if avg < 1e-7:
            return "vanishing"
        elif avg > 100:
            return "exploding"
        else:
            return "healthy"


class AdaptiveTNormLossCalculator:
    """
    Wraps LossCalculator with adaptive t-norm selection per constraint.
    
    Features:
    - Tracks loss/gradient history per constraint
    - Compares t-norms by computing loss with each
    - Adaptively selects best t-norm based on:
      - Gradient magnitude (avoid vanishing)
      - Loss reduction rate
      - Stability (low variance)
    """
    
    def __init__(
        self,
        solver,
        tnorms: List[str] = None,
        adaptation_interval: int = 50,
        warmup_steps: int = 100,
        selection_strategy: str = "gradient_weighted",
    ):
        """
        Args:
            solver: Reference to gurobiILPOntSolver instance
            tnorms: List of t-norms to consider (default: ["L", "P", "SP", "G"])
            adaptation_interval: Steps between t-norm re-evaluation
            warmup_steps: Steps before adaptive selection begins
            selection_strategy: One of "gradient_weighted", "loss_based", "rotating"
        """
        self.solver = solver
        self.tnorms = tnorms or ["L", "P", "SP", "G"]
        self.adaptation_interval = adaptation_interval
        self.warmup_steps = warmup_steps
        self.selection_strategy = selection_strategy
        
        # Per-constraint tracking
        self.constraint_metrics: Dict[str, ConstraintMetrics] = defaultdict(ConstraintMetrics)
        
        # Global state
        self.step_count = 0
        self.epoch_count = 0
        
        # Default t-norms per constraint type (based on mathematical properties)
        self.default_tnorms = {
            "sumL": "L",       # Counting - Łukasiewicz has stable linear gradients
            "countL": "L",     # Counting variant
            "atLeastL": "L",   # Counting - benefits from bounded gradients
            "atMostL": "L",    # Counting - benefits from bounded gradients
            "exactlyL": "L",   # Exact count - L is best
            "andL": "SP",      # Boolean - SP is fast with good gradients
            "orL": "SP",       # Boolean - SP is fast with good gradients
            "impliesL": "P",   # Implication - full Product for accuracy
            "iotaL": "L",      # Selection - L for stable selection
            "existsL": "SP",   # Existence - SP is efficient
            "notL": "SP",      # Negation - SP works well
        }
    
    def calculate_loss_with_comparison(
        self,
        dn,
        primary_tnorm: str = "L",
        compare_all: bool = False,
    ) -> Dict:
        """
        Calculate loss and optionally compare across t-norms.
        
        Args:
            dn: Data node
            primary_tnorm: T-norm to use for actual loss
            compare_all: If True, compute loss with all t-norms for comparison
            
        Returns:
            Dict with losses per constraint and comparison data
        """
        self.step_count += 1
        
        results = {
            "primary_losses": {},
            "comparison": {} if compare_all else None,
            "selected_tnorms": {},
        }
        
        # Calculate primary loss
        primary_losses = self._calculate_loss(dn, primary_tnorm)
        results["primary_losses"] = primary_losses
        
        # Track metrics and compute gradients
        for lc_name, loss_dict in primary_losses.items():
            loss_tensor = loss_dict.get("loss")
            if loss_tensor is not None and torch.is_tensor(loss_tensor):
                grad_norm = self._compute_gradient_norm(loss_tensor)
                loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
                
                self.constraint_metrics[lc_name].add_observation(
                    loss=loss_val,
                    grad_norm=grad_norm,
                    tnorm=primary_tnorm
                )
        
        # Compare t-norms if requested
        if compare_all and self.step_count % self.adaptation_interval == 0:
            results["comparison"] = self._compare_tnorms(dn)
            self._update_tnorm_selection()
        
        # Record selected t-norms
        for lc_name in primary_losses:
            results["selected_tnorms"][lc_name] = self.get_tnorm_for_constraint(lc_name)
        
        return results
    
    def _calculate_loss(self, dn, tnorm: str) -> Dict:
        """Calculate loss using standard LossCalculator."""
        from domiknows.solver.ilpOntSolverTools.lossCalculator import LossCalculator
        
        calculator = LossCalculator(self.solver)
        return calculator.calculateLoss(dn, tnorm=tnorm)
    
    def _compute_gradient_norm(self, loss_tensor: torch.Tensor) -> float:
        """Compute gradient norm for a loss tensor."""
        if not loss_tensor.requires_grad:
            return 0.0
        
        try:
            # Create computation graph for gradient
            grad = torch.autograd.grad(
                loss_tensor, 
                loss_tensor, 
                retain_graph=True,
                allow_unused=True
            )
            if grad[0] is not None:
                return grad[0].norm().item()
        except Exception:
            pass
        
        return 0.0
    
    def _compare_tnorms(self, dn) -> Dict[str, Dict[str, float]]:
        """Compare loss values across all t-norms for each constraint."""
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
                        grad_norm = self._compute_gradient_norm(loss_tensor)
                        
                        comparison[lc_name][tnorm] = {
                            "loss": loss_val,
                            "grad_norm": grad_norm,
                        }
                        
                        # Update per-tnorm tracking
                        self.constraint_metrics[lc_name].loss_by_tnorm[tnorm].append(loss_val)
                        self.constraint_metrics[lc_name].grad_by_tnorm[tnorm].append(grad_norm)
            except Exception as e:
                print(f"Warning: Failed to compute loss with t-norm {tnorm}: {e}")
        
        return comparison
    
    def _update_tnorm_selection(self):
        """Update t-norm selection for each constraint based on history."""
        if self.step_count < self.warmup_steps:
            return
        
        for lc_name, metrics in self.constraint_metrics.items():
            scores = {}
            
            for tnorm in self.tnorms:
                tnorm_losses = metrics.loss_by_tnorm.get(tnorm, [])
                tnorm_grads = metrics.grad_by_tnorm.get(tnorm, [])
                
                if not tnorm_losses or not tnorm_grads:
                    scores[tnorm] = -float('inf')
                    continue
                
                if self.selection_strategy == "gradient_weighted":
                    scores[tnorm] = self._score_gradient_weighted(tnorm_losses, tnorm_grads)
                elif self.selection_strategy == "loss_based":
                    scores[tnorm] = self._score_loss_based(tnorm_losses)
                else:
                    scores[tnorm] = 0.0
            
            # Select best t-norm
            if scores:
                best_tnorm = max(scores, key=scores.get)
                if best_tnorm != metrics.current_tnorm:
                    print(f"[Adaptive] Switching {lc_name}: {metrics.current_tnorm} → {best_tnorm}")
                metrics.current_tnorm = best_tnorm
                metrics.tnorm_scores = scores
    
    def _score_gradient_weighted(self, losses: List[float], grads: List[float]) -> float:
        """
        Score t-norm based on gradient health and loss reduction.
        
        Prefers t-norms with:
        - Healthy gradients (not vanishing or exploding)
        - Decreasing loss trend
        - Low variance (stability)
        """
        if not losses or not grads:
            return -float('inf')
        
        recent_losses = losses[-10:]
        recent_grads = grads[-10:]
        
        avg_grad = np.mean(recent_grads)
        grad_var = np.var(recent_grads) if len(recent_grads) > 1 else 0
        
        # Gradient health score (penalize vanishing/exploding)
        if avg_grad < 1e-6:
            grad_score = -10.0  # Vanishing
        elif avg_grad > 100:
            grad_score = -5.0   # Exploding
        else:
            grad_score = np.log10(avg_grad + 1e-8)  # Log scale, higher is better
        
        # Loss improvement score
        if len(recent_losses) >= 2:
            loss_improvement = recent_losses[0] - recent_losses[-1]
        else:
            loss_improvement = 0
        
        # Stability score (lower variance is better)
        stability = -np.log10(grad_var + 1e-8)
        
        # Combined score
        return grad_score * 2.0 + loss_improvement * 5.0 + stability * 0.5
    
    def _score_loss_based(self, losses: List[float]) -> float:
        """Score based purely on loss values."""
        if not losses:
            return -float('inf')
        return -np.mean(losses[-10:])  # Lower loss = higher score
    
    def get_tnorm_for_constraint(self, lc_name: str) -> str:
        """Get the currently selected t-norm for a constraint."""
        if lc_name in self.constraint_metrics:
            return self.constraint_metrics[lc_name].current_tnorm
        
        # Default based on constraint type
        for constraint_type, default_tnorm in self.default_tnorms.items():
            if constraint_type in lc_name:
                return default_tnorm
        
        return "L"  # Global default
    
    def get_adaptive_tnorm_dict(self) -> Dict[str, str]:
        """Get dict mapping constraint names to their selected t-norms."""
        return {
            lc_name: metrics.current_tnorm
            for lc_name, metrics in self.constraint_metrics.items()
        }
    
    def on_epoch_end(self):
        """Called at end of each epoch for logging and potential adjustments."""
        self.epoch_count += 1
        
        print(f"\n[Epoch {self.epoch_count}] Adaptive T-Norm Summary:")
        print("-" * 60)
        
        for lc_name, metrics in self.constraint_metrics.items():
            print(f"  {lc_name}:")
            print(f"    Current t-norm: {metrics.current_tnorm}")
            print(f"    Avg loss: {metrics.avg_loss:.4f}")
            print(f"    Gradient health: {metrics.gradient_health}")
            print(f"    Loss trend: {metrics.loss_trend:+.4f}")
            
            if metrics.tnorm_scores:
                scores_str = ", ".join(f"{t}:{s:.2f}" for t, s in metrics.tnorm_scores.items())
                print(f"    T-norm scores: {scores_str}")
        
        print("-" * 60)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for all constraints."""
        return {
            lc_name: {
                "current_tnorm": m.current_tnorm,
                "avg_loss": m.avg_loss,
                "avg_gradient": m.avg_gradient,
                "loss_trend": m.loss_trend,
                "gradient_health": m.gradient_health,
                "observations": len(m.losses),
            }
            for lc_name, m in self.constraint_metrics.items()
        }