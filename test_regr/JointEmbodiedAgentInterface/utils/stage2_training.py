"""Stage 2 parameter-policy provenance and lightweight shared-update evidence."""
import math

import torch


def resolve_parameter_policy(args, payload=None):
    requested = getattr(args, "stage2_parameter_policy", "auto")
    legacy = getattr(args, "stage2_freeze_shared_backbone", None)
    if legacy is not None:
        alias = "freeze_shared" if legacy else "freeze_inactive"
        if requested != "auto" and requested != alias:
            raise ValueError("conflicting Stage 2 parameter-policy and legacy freeze flags")
        requested = alias
    if payload and payload.get("stage") == "stage2":
        # Historical checkpoints did not record this policy; retain the old default.
        saved = payload.get("metrics", {}).get("stage2_parameter_policy", "freeze_shared")
        if requested != "auto" and requested != saved:
            raise ValueError("Stage 2 parameter policy differs from the checkpoint; "
                             "resume with its saved policy or start Stage 2 from a Stage 1 checkpoint")
        return saved
    return "freeze_inactive" if requested == "auto" else requested


def validate_shared_optimizer(planner, optimizer, policy):
    if policy == "freeze_shared":
        return
    shared = [p for p in planner.shared_parameters() if p.requires_grad]
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not shared:
        raise ValueError("Stage 2 shared training has no trainable backbone/LoRA parameters")
    if any(id(p) not in owned for p in shared):
        raise ValueError("Stage 2 planner optimizer is missing trainable backbone/LoRA parameters")


class SharedUpdateEvidence:
    """Sample 16 entries per trainable shared tensor; never clone the full backbone."""
    def __init__(self, planner, optimizer):
        owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.parameters = [p for p in planner.shared_parameters() if p.requires_grad]
        self.optimizer_count = sum(p.numel() for p in self.parameters if id(p) in owned)
        self.samples = []
        with torch.no_grad():
            for p in self.parameters:
                indices = torch.linspace(0, p.numel()-1, min(16, p.numel()), device=p.device).long()
                self.samples.append((p, indices, p.detach().reshape(-1)[indices].float().cpu().clone()))

    def report(self):
        with torch.no_grad():
            gradient_squared = sum(float(p.grad.detach().float().square().sum())
                                   for p in self.parameters if p.grad is not None)
            delta_squared = sum(float((p.detach().reshape(-1)[indices].float().cpu()-before).square().sum())
                                for p, indices, before in self.samples)
        return {"shared_trainable_parameters": sum(p.numel() for p in self.parameters),
                "shared_optimizer_parameters": self.optimizer_count,
                "shared_gradient_norm": math.sqrt(gradient_squared),
                "sampled_shared_delta_l2": math.sqrt(delta_squared),
                "shared_parameters_changed": delta_squared > 0,
                "finite": math.isfinite(gradient_squared) and math.isfinite(delta_squared)}


def eai_retention_rejections(metrics, baseline, *, max_success_regression, max_recall_regression):
    if baseline is None:
        return []
    reasons = []
    if not float(metrics["success"]) >= float(baseline["goal_success"]) - max_success_regression:
        reasons.append("eai_success_regression")
    if not float(metrics["goal_recall"]) >= float(baseline["goal_recall"]) - max_recall_regression:
        reasons.append("eai_recall_regression")
    return reasons


def selection_rejections(metrics, *, min_success, min_tasks, max_ik):
    """Explain rejection without lowering gates or substituting training rollouts."""
    reasons = []
    if metrics.get("stage2_evaluation", {}).get("source") != "fixed_seed_rollouts":
        reasons.append("fixed_seed_evaluation_missing")
    vla = metrics["vlabench"]
    if not all(math.isfinite(float(vla.get(key, 0))) for key in ("success_rate", "ik_truncation_rate")):
        reasons.append("nonfinite_rollout_metrics")
    if float(vla.get("success_rate", 0)) < min_success:
        reasons.append("vlabench_success_below_minimum")
    if int(vla.get("successful_task_count", 0)) < min_tasks:
        reasons.append("successful_task_coverage_below_minimum")
    if float(vla.get("ik_truncation_rate", 0)) > max_ik:
        reasons.append("ik_truncation_above_maximum")
    if not metrics.get("retention_eligible", True):
        reasons.append("baseline_or_retention_gate_failed")
    if any(v.get("nonfinite_updates", 0) for v in metrics.get("planner_evidence", {}).values()):
        reasons.append("nonfinite_shared_updates")
    return reasons
