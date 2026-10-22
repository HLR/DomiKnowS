"""Alternating supervised and reinforcement programs over the joint graph."""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from domiknows.generation.latent.losses import GenerationLossWeights, compute_generation_training_loss
from domiknows.generation.learners.common.losses import allowed_mass_loss
from domiknows.program import SolverPOIProgram
from domiknows.program.lossprogram import PrimalDualProgram, _project_cmodel_lambdas, reverse_sign_grad
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.gradient_manager import GradientConflictManager
from domiknows.program.metric import MacroAverageTracker
from domiknows.reinforcement.reinforcement_program import ReinforcementProgram

from test_regr.VLABenchAgentInterface.simulation.controller import controller_loss
from test_regr.VLABenchAgentInterface.simulation.program import (
    EOSMaskedCrossEntropyLoss,
    VLABenchHierarchicalReinforcementProgram,
    attach_planner_sensors,
)

from .models import JointQwenVLPlanner
from .world_graph import JointDomainRuntime


@dataclass(frozen=True)
class DomainUpdate:
    domain: str
    planner_loss: float
    controller_loss: float = 0.0
    reward: float | None = None


@dataclass(frozen=True)
class DFALossConfig:
    """Stage 1 DFA loss: allowed-mass on the DFA path plus the primal-dual constraint loss of the generation graph.

    ``pmd_weight`` is the primal-dual program's ``beta``; the multipliers take an ascent step every ``dual_every``
    rounds after ``dual_warmup`` rounds, as in ``PrimalDualProgram.train``.
    """

    mass_weight: float = 1.0
    pmd_weight: float = 1.0
    dual_lr: float = 0.05
    dual_warmup: int = 10
    dual_every: int = 10
    mass_steps: int | None = 12  # score at most this many random steps per item (the DFA walk is the cost)
    compile_lc: bool = True  # evaluate the constraint loss with the compiled (batched) formula protocol
    domains: tuple[str, ...] = ("eai", "vlabench")  # domains trained on the DFA loss
    # Domains whose allowed set also requires a reachable completion. The search for one costs seconds per EAI item
    # (up to 26 s), so EAI scores the labels valid at each step only (hundreds of times cheaper, same loss on most items).
    lookahead_domains: tuple[str, ...] = ("vlabench",)
    pmd_domains: tuple[str, ...] = ("vlabench",)  # of those, the domains that get the primal-dual constraint loss


def _attach_order_edges(bundle) -> Any:
    """Give the generation graph its ``is_before`` edges (once), so its ordered EOS constraints have groundings."""
    from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor

    if not getattr(bundle.is_before_rel, "_order_edges_attached", False):
        bundle.is_before_rel._order_edges_attached = True
        bundle.is_before_rel[bundle.first_token.reversed, bundle.second_token.reversed] = CompositionCandidateSensor(
            relations=(bundle.first_token.reversed, bundle.second_token.reversed),
            forward=lambda *_args, arg1, arg2: arg1.getAttribute("position") < arg2.getAttribute("position"),
        )
    return bundle.is_before_rel


def _emit_progress(message: str) -> None:
    """Write progress that remains visible when stdout/stderr are redirected."""

    print(f"[joint-training] {message}", file=sys.stderr, flush=True)


class _TrainingProgress:
    def __init__(
        self,
        label: str,
        total: int,
        *,
        initial: int = 0,
        interval_seconds: float = 30.0,
    ):
        self.label = label
        self.total = max(0, int(total))
        self.initial = min(max(0, int(initial)), self.total)
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.started = self.last_report = time.monotonic()
        _emit_progress(f"{self.label}: {self.initial}/{self.total} started")

    def update(self, completed: int, **metrics: float) -> None:
        now = time.monotonic()
        completed = int(completed)
        if completed < self.total and completed != 1 and now - self.last_report < self.interval_seconds:
            return
        elapsed = max(now - self.started, 1e-9)
        rate = max(0, completed - self.initial) / elapsed
        remaining = max(0, self.total - completed)
        eta = remaining / rate if rate > 0 else float("inf")
        values = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        suffix = f" {values}" if values else ""
        _emit_progress(
            f"{self.label}: {completed}/{self.total} "
            f"elapsed={elapsed / 60.0:.1f}m eta={eta / 60.0:.1f}m{suffix}"
        )
        self.last_report = now


def _context(item: Mapping[str, Any], domain: str) -> Mapping[str, Any]:
    from .utils.planner_context import planner_context
    return planner_context(domain, item)


def _labels(item: Mapping[str, Any], domain: str) -> torch.Tensor:
    key = "target_action_labels" if domain == "eai" else "target_plan_labels"
    if key not in item:
        raise KeyError(f"{domain} supervised item is missing {key!r}")
    return torch.as_tensor(item[key], dtype=torch.long)


def _cycle(values: Sequence[Any]):
    while True:
        order = list(values)
        random.shuffle(order)
        yield from order


def _runtime_adapter(runtime: JointDomainRuntime, domain: str):
    if domain == "eai":
        return SimpleNamespace(
            vocabulary=runtime.eai_vocabulary,
            generation_graph=runtime.root,
            generation_bundle=runtime.eai_generation,
            dfa=runtime.eai_dfa,
            dfa_for=lambda d, ctx=None: runtime.dfa_for(d, ctx),
            world_bundle=runtime.world.eai,
            max_operations=runtime.max_eai_steps,
            max_tokens=runtime.max_eai_steps,
        )
    return SimpleNamespace(
        vocabulary=runtime.vlabench_vocabulary,
        generation_graph=runtime.root,
        generation_bundle=runtime.vlabench_generation,
        dfa=runtime.vlabench_dfa,
        dfa_for=lambda d, ctx=None: runtime.dfa_for(d, ctx),
        world_bundle=runtime.world.vlabench,
        max_operations=runtime.max_vlabench_operations,
        max_tokens=runtime.max_vlabench_operations * 5 + 1,
    )


class _ActiveDomainEOSLoss(torch.nn.Module):
    def __init__(self, runtime: JointDomainRuntime):
        super().__init__()
        self.runtime = runtime

    def forward(self, input, target, *args, **kwargs):
        domain = self.runtime.active_domain or "eai"
        vocabulary = (
            self.runtime.eai_vocabulary if domain == "eai"
            else self.runtime.vlabench_vocabulary
        )
        return EOSMaskedCrossEntropyLoss(vocabulary.eos_label)(input, target, *args, **kwargs)


class JointSolverPOIProgram(SolverPOIProgram):
    """One SolverPOI lifecycle with both sensor branches attached once."""

    def __init__(
        self,
        runtime: JointDomainRuntime,
        planner: JointQwenVLPlanner,
        *,
        planner_optimizer: torch.optim.Optimizer,
        controller: torch.nn.Module | None = None,
        controller_optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
        eai_loss_weight: float = 3.0,
        dfa_loss: DFALossConfig | None = None,
    ):
        eai_view = planner.for_domain("eai")
        vlabench_view = planner.for_domain("vlabench")
        poi = []
        for domain, view in (("eai", eai_view), ("vlabench", vlabench_view)):
            poi.extend(attach_planner_sensors(_runtime_adapter(runtime, domain), view, device=device))
        # Preserve order while removing repeated relation/property objects.
        poi = list(dict.fromkeys(poi))
        super().__init__(
            runtime.root,
            poi=poi,
            inferTypes=["local/argmax"],
            loss=MacroAverageTracker(_ActiveDomainEOSLoss(runtime)),
            device=device,
            metric={},
        )
        self.runtime = runtime
        self.planner_head = planner
        self.eai_planner = eai_view
        self.vlabench_planner = vlabench_view
        self.planner_optimizer = planner_optimizer
        self.controller = controller
        self.controller_optimizer = controller_optimizer
        self.gradient_manager = GradientConflictManager(self.planner_head, self.planner_optimizer)
        self.round_robin_cursor = 0
        self.eai_loss_weight = float(eai_loss_weight)
        self.dfa_loss = dfa_loss
        self.pmd_programs: dict[str, PrimalDualProgram] = {}
        self.dual_optimizers: dict[str, torch.optim.Optimizer] = {}
        self.dfa_loss_rounds = 0
        if dfa_loss is not None:
            planner.capture_logits = True
            for domain, graph, view in (
                ("eai", runtime.eai_generation_graph, eai_view),
                ("vlabench", runtime.vlabench_generation_graph, vlabench_view),
            ):
                if domain not in dfa_loss.domains or domain not in dfa_loss.pmd_domains:
                    continue  # no DataNode graph is built for a domain without the constraint loss
                adapter = _runtime_adapter(runtime, domain)
                program = PrimalDualProgram(
                    graph, SolverModel,
                    poi=[*attach_planner_sensors(adapter, view, device=device),
                         _attach_order_edges(adapter.generation_bundle)],
                    inferTypes=["local/argmax"], loss=MacroAverageTracker(_ActiveDomainEOSLoss(runtime)),
                    beta=dfa_loss.pmd_weight, device=device, metric={}, tnorm="P", counting_tnorm="P",
                    compile_lc=dfa_loss.compile_lc,
                )
                self.pmd_programs[domain] = program
                params = list(program.cmodel.parameters())
                if params:
                    self.dual_optimizers[domain] = torch.optim.Adam(params, lr=dfa_loss.dual_lr)

    def _domain_loss(self, domain: str, item: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Planner loss of one item; with ``dfa_loss`` the CE is combined with the DFA terms in one forward."""
        context = _context(item, domain)
        labels = _labels(item, domain).to(self.planner_head.device)
        if self.dfa_loss is None or domain not in self.dfa_loss.domains:
            return self.planner_head.supervised_loss(domain, context=context, target_labels=labels), {}
        keep = ((labels == int(self.planner_head.vocabulary(domain).eos_label)).cumsum(dim=-1) <= 1)
        program = self.pmd_programs.get(domain)
        if program is not None:
            data = {"planner_context": context, "token_positions": torch.arange(labels.numel()),
                    "target_plan_labels": labels}
            # One planner forward: the program builds the DataNode graph for the primal-dual constraint loss and
            # the planner leaves its logits behind; the cross entropy is the same as ``supervised_loss``.
            _model_loss, _metric, _node, builder = program.model(data)
            pmd, *_ = program.cmodel(builder)
            logits = self.planner_head.last_logits
        else:
            logits = self.planner_head.sequence_logits(
                domain, context, self.planner_head.shift_right(domain, labels)).squeeze(0)
            pmd = logits.new_zeros(())
        ce = torch.nn.functional.cross_entropy(logits[keep], labels[keep])
        dfa = self.runtime.dfa_for(domain, {**dict(item), **dict(context)})
        kept = int(keep.sum())
        cap = self.dfa_loss.mass_steps
        steps = sorted(random.sample(range(kept), cap)) if cap and kept > cap else None
        mass = allowed_mass_loss(torch.softmax(logits[keep], dim=-1), dfa, path=labels[keep], score_steps=steps,
                                 lookahead=domain in self.dfa_loss.lookahead_domains)
        breakdown = compute_generation_training_loss(
            supervised_loss=ce, pmd_loss=pmd, allowed_mass_loss_value=mass,
            weights=GenerationLossWeights(supervised=1.0, pmd=self.dfa_loss.pmd_weight,
                                          allowed_mass=self.dfa_loss.mass_weight),
        )
        self._dual_grads(domain, pmd)
        return breakdown.total, {k: v for k, v in breakdown.as_float_dict().items() if k != "total_loss"}

    def _dual_grads(self, domain: str, pmd: torch.Tensor) -> None:
        optimizer = self.dual_optimizers.get(domain)
        if optimizer is None or not pmd.requires_grad:
            return
        params = [p for group in optimizer.param_groups for p in group["params"]]
        grads = torch.autograd.grad(pmd, params, retain_graph=True, allow_unused=True)
        for param, grad in zip(params, grads):
            if grad is not None:
                param.grad = grad if param.grad is None else param.grad + grad

    def _dual_step(self) -> None:
        """Gradient ascent on the multipliers (sign-reversed gradients), on the ``PrimalDualProgram`` schedule."""
        self.dfa_loss_rounds += 1
        cfg = self.dfa_loss
        if cfg is None or self.dfa_loss_rounds <= cfg.dual_warmup or self.dfa_loss_rounds % cfg.dual_every:
            return
        for domain, optimizer in self.dual_optimizers.items():
            params = [p for group in optimizer.param_groups for p in group["params"]]
            reverse_sign_grad(params)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            _project_cmodel_lambdas(self.pmd_programs[domain].cmodel)

    def _planner_step(self, domain: str, item: Mapping[str, Any]) -> float:
        # Validation routes through the same shared object and may leave it in
        # eval mode.  Prefix-level activation checkpointing is enabled only
        # during training, so restore the lifecycle explicitly before every
        # optimizer-bearing planner turn.
        self.planner_head.train()
        with self.runtime.domain_scope(domain, parameter_policy="freeze_inactive"):
            loss, _terms = self._domain_loss(domain, item)
            self.planner_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.planner_head.parameters(), 1.0)
            self.planner_optimizer.step()
            self._dual_step()
            return float(loss.detach())

    def train_eai_warmup(
        self,
        eai_examples: Sequence[Mapping[str, Any]],
        *,
        steps: int = 350,
    ) -> dict[str, float]:
        """Run pure EAI supervised training steps before alternating multi-domain rounds."""
        steps = int(steps)
        if steps <= 0:
            return {"loss": 0.0, "steps": 0}
        if not eai_examples:
            raise ValueError("eai_examples cannot be empty for warmup")
        stream = _cycle(eai_examples)
        total = 0.0
        progress = _TrainingProgress("EAI warm-up", steps)
        for index in range(steps):
            item = next(stream)
            loss = self._planner_step("eai", item)
            total += loss
            progress.update(index + 1, eai_warmup_loss=total / (index + 1))
        return {"loss": total / steps, "steps": steps}

    def _controller_step(self, batch: Mapping[str, torch.Tensor] | None) -> float:
        if batch is None or self.controller is None or self.controller_optimizer is None:
            return 0.0
        with self.runtime.domain_scope("vlabench"):
            self.controller.train()
            device = next(self.controller.parameters()).device
            controller_inputs = (
                batch["images"].to(device),
                batch["state"].to(device),
                batch["task_index"].to(device),
            )
            plan_context = batch.get("plan_context")
            if plan_context is not None:
                controller_inputs += (plan_context.to(device),)
            prediction = self.controller(*controller_inputs)
            loss = controller_loss(
                prediction,
                batch["actions"].to(device),
                state=controller_inputs[1],
                pose_step_scale=getattr(self.controller, "pose_step_scale", None),
            )[0]
            self.controller_optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                raise ValueError("controller behavior-cloning loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.controller.parameters(),
                1.0,
                error_if_nonfinite=True,
            )
            self.controller_optimizer.step()
            if not all(
                bool(torch.isfinite(parameter).all())
                for parameter in self.controller.parameters()
                if parameter.requires_grad
            ):
                raise RuntimeError("controller behavior-cloning update produced non-finite parameters")
            return float(loss.detach())

    def train_controller_warmup(
        self,
        controller_loader: Iterable[Mapping[str, torch.Tensor]],
        *,
        steps: int,
    ) -> dict[str, float]:
        """Run controller-only BC before online reinforcement learning."""

        steps = int(steps)
        if steps < 0:
            raise ValueError("controller warm-up steps cannot be negative")
        if steps == 0:
            return {"loss": 0.0, "steps": 0}
        stream = iter(controller_loader)
        total = 0.0
        progress = _TrainingProgress("Controller BC warm-up", steps)
        for index in range(steps):
            try:
                batch = next(stream)
            except StopIteration:
                stream = iter(controller_loader)
                batch = next(stream)
            total += self._controller_step(batch)
            progress.update(index + 1, controller_bc_loss=total / (index + 1))
        return {"loss": total / steps, "steps": steps}

    def train_round(
        self,
        eai_item: Mapping[str, Any],
        vlabench_item: Mapping[str, Any],
        *,
        controller_batch: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[DomainUpdate, DomainUpdate]:
        """Apply one conflict-projected, equal-domain planner update.

        The old implementation stepped the shared Qwen/LoRA once for EAI and
        then again for VLABench.  That made update order a hidden domain
        weight: the second step could immediately undo the first.  PCGrad
        projects only genuinely conflicting shared gradients and performs one
        optimizer step for the pair. Domain-only decoder parameters retain
        their own gradients.
        """

        self.planner_head.train()
        self.gradient_manager.begin_step()

        with self.runtime.domain_scope("eai", parameter_policy="freeze_inactive"):
            eai_tensor, eai_terms = self._domain_loss("eai", eai_item)
            scaled_eai = eai_tensor * self.eai_loss_weight if self.eai_loss_weight != 1.0 else eai_tensor
            self.gradient_manager.capture("eai", scaled_eai)

        with self.runtime.domain_scope("vlabench", parameter_policy="freeze_inactive"):
            vlabench_tensor, vlabench_terms = self._domain_loss("vlabench", vlabench_item)
            self.gradient_manager.capture("vlabench", vlabench_tensor)

        self.gradient_manager.step(self.planner_optimizer, method="pcgrad", clip_grad_norm=1.0)
        self._dual_step()
        self.last_dfa_terms = {"eai": eai_terms, "vlabench": vlabench_terms}
        eai_loss = float(eai_tensor.detach())
        vlabench_loss = float(vlabench_tensor.detach())
        bc_loss = self._controller_step(controller_batch)
        self.round_robin_cursor += 1
        return (
            DomainUpdate("eai", eai_loss),
            DomainUpdate("vlabench", vlabench_loss, controller_loss=bc_loss),
        )

    def train_alternating_epoch(
        self,
        eai_examples: Sequence[Mapping[str, Any]],
        vlabench_examples: Sequence[Mapping[str, Any]],
        *,
        controller_loader: Iterable[Mapping[str, torch.Tensor]] | None = None,
        rounds: int | None = None,
    ) -> dict[str, float]:
        if not eai_examples or not vlabench_examples:
            raise ValueError("joint Stage 1 requires non-empty EAI and VLABench loaders")
        rounds = int(rounds) if rounds is not None else min(len(eai_examples), len(vlabench_examples))
        eai_stream, vlabench_stream = _cycle(eai_examples), _cycle(vlabench_examples)
        controller_stream = iter(controller_loader) if controller_loader is not None else None
        totals = {"eai_loss": 0.0, "vlabench_loss": 0.0, "controller_bc_loss": 0.0}
        if self.dfa_loss is not None:
            totals |= {f"{d}_{t}": 0.0 for d in self.dfa_loss.domains for t in ("allowed_mass_loss", "constraint_loss")}
        progress = _TrainingProgress("Stage 1 rounds", rounds)
        for round_index in range(rounds):
            batch = None
            if controller_stream is not None:
                try:
                    batch = next(controller_stream)
                except StopIteration:
                    controller_stream = iter(controller_loader)
                    batch = next(controller_stream)
            eai, vlabench = self.train_round(next(eai_stream), next(vlabench_stream), controller_batch=batch)
            totals["eai_loss"] += eai.planner_loss
            totals["vlabench_loss"] += vlabench.planner_loss
            totals["controller_bc_loss"] += vlabench.controller_loss
            dfa_terms = {}
            if self.dfa_loss is not None:
                for domain, terms in self.last_dfa_terms.items():
                    for term in ("allowed_mass_loss", "constraint_loss") if terms else ():
                        totals[f"{domain}_{term}"] += terms[term]
            completed = round_index + 1
            if self.dfa_loss is not None:
                dfa_terms = {"vla_mass": totals["vlabench_allowed_mass_loss"] / completed,
                             "vla_pmd": totals["vlabench_constraint_loss"] / completed}
            progress.update(
                completed,
                eai_loss=totals["eai_loss"] / completed,
                vlabench_loss=totals["vlabench_loss"] / completed,
                controller_bc_loss=totals["controller_bc_loss"] / completed,
                **dfa_terms,
            )
        return {key: value / max(1, rounds) for key, value in totals.items()} | {"rounds": rounds}


class JointReinforcementProgram(VLABenchHierarchicalReinforcementProgram):
    """Alternating EAI REINFORCE and VLABench planner/PPO optimization.

    EAI and VLABench rewards are intentionally stored and reported separately.
    No reward scalar crosses the domain boundary. The CLI freezes the shared
    Qwen/LoRA by default during this stage; callers that leave it trainable
    explicitly opt back into cross-domain gradient coupling.
    """

    def __init__(
        self,
        runtime: JointDomainRuntime,
        planner: JointQwenVLPlanner,
        controller: torch.nn.Module,
        *,
        planner_optimizer: torch.optim.Optimizer,
        controller_optimizer: torch.optim.Optimizer,
        env_factory,
        eai_supervised_examples: Sequence[Mapping[str, Any]] = (),
        vlabench_supervised_examples: Sequence[Any] = (),
        controller_anchor_loader=None,
        eai_num_samples: int = 8,
        eai_supervised_weight: float = 0.5,
        parameter_policy: str | None = "freeze_shared",
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        self.joint_runtime = runtime
        self.joint_planner = planner
        self.eai_planner = planner.for_domain("eai")
        self.vlabench_planner = planner.for_domain("vlabench")
        self.eai_supervised_examples = tuple(eai_supervised_examples)
        self.eai_num_samples = int(eai_num_samples)
        self.eai_supervised_weight = float(eai_supervised_weight)
        self.parameter_policy = parameter_policy
        if self.parameter_policy != "freeze_shared":
            import warnings
            warnings.warn(
                f"Stage 2 JointReinforcementProgram initialized with parameter_policy={self.parameter_policy!r}. "
                "This policy is an experimental ablation: the shared Qwen/LoRA backbone remains trainable during "
                "independent domain updates without gradient conflict projection, which may lead to cross-domain policy interference.",
                UserWarning,
                stacklevel=2,
            )
        self.round_robin_cursor = 0
        kwargs.setdefault("progress_callback", _emit_progress)
        super().__init__(
            _runtime_adapter(runtime, "vlabench"),
            self.vlabench_planner,
            controller,
            planner_optimizer=planner_optimizer,
            controller_optimizer=controller_optimizer,
            env_factory=env_factory,
            supervised_examples=vlabench_supervised_examples,
            controller_anchor_loader=controller_anchor_loader,
            device=device,
            **kwargs,
        )
        # Public identity used by checkpointing and acceptance tests.
        self.planner_head = planner

    def collect_episode(self, descriptor):
        with self.joint_runtime.domain_scope("vlabench", parameter_policy=self.parameter_policy):
            # The parent implementation invokes methods through planner_head;
            # temporarily expose its non-owning VLABench view.
            planner = self.planner_head
            self.planner_head = self.vlabench_planner
            try:
                return super().collect_episode(descriptor)
            finally:
                self.planner_head = planner

    def _planner_anchor(self):
        with self.joint_runtime.domain_scope("vlabench", parameter_policy=self.parameter_policy):
            planner = self.planner_head
            self.planner_head = self.vlabench_planner
            try:
                return super()._planner_anchor()
            finally:
                self.planner_head = planner

    def _eai_anchor(self) -> torch.Tensor:
        if not self.eai_supervised_examples or not self.eai_supervised_weight:
            return torch.zeros((), device=self.joint_planner.device)
        item = random.choice(self.eai_supervised_examples)
        return self.joint_planner.supervised_loss(
            "eai",
            context=_context(item, "eai"),
            target_labels=_labels(item, "eai").to(self.joint_planner.device),
        )

    def train_eai_update(self, item: Mapping[str, Any]) -> dict[str, float]:
        self.joint_planner.train()
        reward_fn = item.get("reward_function")
        if not callable(reward_fn):
            raise ValueError("EAI reinforcement items require reward_function")
        context = _context(item, "eai")
        pairs = []
        with self.joint_runtime.domain_scope("eai", parameter_policy=self.parameter_policy):
            encoded_context = self.eai_planner.encode_context(context)
            prepared_context = self.eai_planner.prepare_replay_context(context)
            samples = []
            with torch.no_grad():
                for _ in range(self.eai_num_samples):
                    policy_dfa = self.joint_runtime.dfa_for("eai", item)
                    labels, _ = self.eai_planner.sample_labels_from_context(
                        encoded_context,
                        policy_dfa,
                        max_steps=self.joint_runtime.max_eai_steps,
                    )
                    reward = reward_fn(labels, data_item=item, policy=policy_dfa)
                    from test_regr.EmbodiedAgentInterface.reward import evaluate_goal_satisfaction
                    goal = evaluate_goal_satisfaction(
                        labels, item, self.joint_runtime.eai_vocabulary,
                        world_bundle=self.joint_runtime.world.eai,
                        policy=policy_dfa,
                    )
                    samples.append((
                        labels,
                        float(torch.as_tensor(reward).float().mean()),
                        float(goal["recall"]),
                        float(goal["is_success"]),
                        policy_dfa,
                    ))
            rewards = torch.tensor([s[1] for s in samples], device=self.joint_planner.device, dtype=torch.float32)
            advantages = rewards - rewards.mean()
            self.planner_optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            count = max(1, len(samples))
            for i, (labels, _, _, _, policy_dfa) in enumerate(samples):
                adv = advantages[i].detach()
                if adv.abs() > 1e-6:
                    logprob = self.eai_planner.replay_labels_logprob(
                        prepared_context,
                        labels,
                        policy_dfa,
                        max_steps=self.joint_runtime.max_eai_steps,
                    )
                    sample_loss = -(logprob * adv) / count
                    sample_loss.backward()
                    total_loss += float(sample_loss.detach())

            if self.parameter_policy == "pcgrad_protected_eai":
                rl_grads = []
                for p in self.joint_planner.parameters():
                    if p.grad is not None:
                        rl_grads.append(p.grad.clone())
                        p.grad.zero_()
                    else:
                        rl_grads.append(None)
            
            anchor_loss = self.eai_supervised_weight * self._eai_anchor()
            if anchor_loss.requires_grad:
                anchor_loss.backward()
                total_loss += float(anchor_loss.detach())
                
            if self.parameter_policy == "pcgrad_protected_eai":
                with torch.no_grad():
                    for p, rl_g in zip(self.joint_planner.parameters(), rl_grads):
                        if rl_g is not None and p.grad is not None:
                            anchor_g = p.grad
                            dot = torch.sum(rl_g * anchor_g)
                            if dot < 0:
                                rl_g = rl_g - (dot / (torch.sum(anchor_g * anchor_g) + 1e-8)) * anchor_g
                            p.grad.add_(rl_g)
                        elif rl_g is not None:
                            p.grad = rl_g.clone()
            torch.nn.utils.clip_grad_norm_(self.joint_planner.parameters(), 1.0)
            self.planner_optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {
            "loss": float(total_loss),
            "reward": float(rewards.mean()),
            "goal_recall": sum(s[2] for s in samples) / len(samples),
            "success": sum(s[3] for s in samples) / len(samples),
            "samples": len(samples),
        }

    def train_vlabench_update(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        *,
        rollouts_per_update: int = 8,
    ) -> dict[str, float]:
        self.joint_planner.train()
        self.controller.train()
        with self.joint_runtime.domain_scope("vlabench", parameter_policy=self.parameter_policy):
            planner = self.planner_head
            self.planner_head = self.vlabench_planner
            try:
                return super().train_joint_epoch(
                    descriptors,
                    rollouts_per_update=rollouts_per_update,
                )
            finally:
                self.planner_head = planner

    def train_alternating_epoch(
        self,
        eai_examples: Sequence[Mapping[str, Any]],
        vlabench_descriptors: Sequence[Mapping[str, Any]],
        *,
        rounds: int = 10,
        vlabench_rollouts_per_update: int = 8,
        start_round: int = 0,
        current_epoch: int = 0,
        total_epochs: int = 1,
        initial_metrics: Mapping[str, Any] | None = None,
        round_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not eai_examples or not vlabench_descriptors:
            raise ValueError("joint Stage 2 requires both domain sources")
        rounds = int(rounds)
        start_round = int(start_round)
        if rounds <= 0:
            raise ValueError("joint Stage 2 requires a positive round count")
        if start_round < 0 or start_round > rounds:
            raise ValueError("Stage 2 resume round is outside the current epoch")
        if start_round and initial_metrics is None:
            raise ValueError("Stage 2 resume requires accumulated round metrics")
        if initial_metrics is not None and int(initial_metrics.get("rounds", 0)) != start_round:
            raise ValueError("Stage 2 resume metrics do not match the next round")
        self.joint_planner.train()
        self.controller.train()
        eai_totals = {"loss": 0.0, "reward": 0.0, "goal_recall": 0.0, "success": 0.0}
        vla_totals = {
            "planner_loss": 0.0,
            "controller_loss": 0.0,
            "return": 0.0,
            "positive_return_rate": 0.0,
            "success_rate": 0.0,
            "valid_rate": 0.0,
            "steps": 0.0,
            "ik_truncation_rate": 0.0,
            "execution_complete_rate": 0.0,
        }
        vla_task_totals: dict[str, dict[str, float]] = {}
        vla_episode_count = 0
        vla_ik_failures = 0
        vla_ik_recoveries = 0
        controller_evidence = {
            "updates_attempted": 0,
            "updates_rolled_back": 0,
            "updates_with_parameter_change": 0,
            "ppo_epochs_completed": 0,
            "parameter_delta_l2": 0.0,
            "approximate_kl": 0.0,
        }
        if initial_metrics is not None:
            for key in eai_totals:
                eai_totals[key] = float(initial_metrics.get("eai", {}).get(key, 0.0)) * start_round
            for key in vla_totals:
                vla_totals[key] = float(initial_metrics.get("vlabench", {}).get(key, 0.0)) * start_round
            vla_episode_count = int(initial_metrics.get("vlabench", {}).get("episodes", 0))
            vla_ik_failures = int(initial_metrics.get("vlabench", {}).get("ik_failures", 0))
            vla_ik_recoveries = int(initial_metrics.get("vlabench", {}).get("ik_recoveries", 0))
            saved_evidence = initial_metrics.get("vlabench", {}).get("controller_evidence", {})
            for key in controller_evidence:
                controller_evidence[key] = type(controller_evidence[key])(
                    saved_evidence.get(key, controller_evidence[key])
                )
            for task_name, task_metrics in initial_metrics.get("vlabench", {}).get("per_task", {}).items():
                episodes = int(task_metrics.get("episodes", 0))
                vla_task_totals[task_name] = {
                    "episodes": float(episodes),
                    "successes": float(task_metrics.get("successes", 0)),
                    "positive_returns": (
                        float(task_metrics.get(
                            "positive_return_rate",
                            float(float(task_metrics.get("return", 0.0)) > 1e-6),
                        ))
                        * episodes
                    ),
                    "valid": float(task_metrics.get("valid_rate", 0.0)) * episodes,
                    "return": float(task_metrics.get("return", 0.0)) * episodes,
                    "steps": float(task_metrics.get("steps", 0.0)) * episodes,
                    "ik_failures": float(task_metrics.get("ik_failures", 0)),
                    "ik_recoveries": float(task_metrics.get("ik_recoveries", 0)),
                    "ik_truncations": float(task_metrics.get("ik_truncation_rate", 0.0)) * episodes,
                    "execution_complete": float(task_metrics.get("execution_complete_rate", 0.0)) * episodes,
                }

        def completed_metrics(completed: int) -> dict[str, Any]:
            count = max(1, int(completed))
            per_task = {
                task_name: {
                    "episodes": int(totals["episodes"]),
                    "successes": int(totals["successes"]),
                    "success_rate": totals["successes"] / max(1.0, totals["episodes"]),
                    "positive_return_rate": (
                        totals["positive_returns"] / max(1.0, totals["episodes"])
                    ),
                    "valid_rate": totals["valid"] / max(1.0, totals["episodes"]),
                    "return": totals["return"] / max(1.0, totals["episodes"]),
                    "steps": totals["steps"] / max(1.0, totals["episodes"]),
                    "ik_failures": int(totals["ik_failures"]),
                    "ik_recoveries": int(totals["ik_recoveries"]),
                    "ik_recovery_rate": totals["ik_recoveries"] / max(1.0, totals["ik_failures"]),
                    "ik_truncation_rate": totals["ik_truncations"] / max(1.0, totals["episodes"]),
                    "execution_complete_rate": totals["execution_complete"] / max(1.0, totals["episodes"]),
                }
                for task_name, totals in sorted(vla_task_totals.items())
            }
            return {
                "eai": {key: value / count for key, value in eai_totals.items()},
                "vlabench": {
                    **{key: value / count for key, value in vla_totals.items()},
                    "episodes": vla_episode_count,
                    "ik_failures": vla_ik_failures,
                    "ik_recoveries": vla_ik_recoveries,
                    "controller_evidence": {
                        **controller_evidence,
                        "mean_approximate_kl": (
                            controller_evidence["approximate_kl"]
                            / max(1, controller_evidence["updates_attempted"])
                        ),
                    },
                    "successful_task_count": sum(
                        int(metrics["successes"] > 0) for metrics in per_task.values()
                    ),
                    "per_task": per_task,
                },
                "rounds": int(completed),
                "round_robin_cursor": self.round_robin_cursor,
            }

        progress = _TrainingProgress("Stage 2 rounds", rounds, initial=start_round)
        total_rl_steps = max(1, total_epochs * rounds)
        for round_index in range(start_round, rounds):
            if hasattr(self, "set_curriculum_progress"):
                self.set_curriculum_progress((current_epoch * rounds + round_index) / total_rl_steps)
            _emit_progress(f"Stage 2 round {round_index + 1}/{rounds}: EAI policy update")
            eai = self.train_eai_update(random.choice(eai_examples))
            if getattr(self, "learning_progress", False):
                import math
                scores = []
                for idx, d in enumerate(vlabench_descriptors):
                    task_name = str(d.get("task", "unknown"))
                    totals = vla_task_totals.get(task_name)
                    if not totals or totals["episodes"] < 1:
                        score = 1000.0
                    else:
                        succ = totals["successes"]
                        pos = totals["positive_returns"]
                        n = max(1.0, totals["episodes"])
                        def wilson(k, n, z=1.96):
                            if n <= 0: return 0.0
                            p = k / n
                            denominator = 1.0 + z * z / n
                            center = (p + z * z / (2 * n)) / denominator
                            half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
                            return max(0.0, center - half)
                        score = wilson(pos, n) - wilson(succ, n) + math.sqrt(2 * math.log(vla_episode_count + 1) / n)
                    scores.append(score)
                max_score = max(scores)
                best_indices = [i for i, s in enumerate(scores) if s >= max_score - 1e-4]
                chosen_index = random.choice(best_indices)
                self.round_robin_cursor = chosen_index
                descriptor = vlabench_descriptors[chosen_index]
            else:
                descriptor = vlabench_descriptors[
                    self.round_robin_cursor % len(vlabench_descriptors)
                ]
            _emit_progress(
                f"Stage 2 round {round_index + 1}/{rounds}: "
                f"VLABench task={descriptor.get('task', 'unknown')} "
                f"rollouts={vlabench_rollouts_per_update}"
            )
            vla = self.train_vlabench_update(
                [descriptor],
                rollouts_per_update=vlabench_rollouts_per_update,
            )
            for key in eai_totals:
                eai_totals[key] += float(eai[key])
            for key in vla_totals:
                vla_totals[key] += float(vla.get(key, 0.0))
            vla_episode_count += int(vla["episodes"])
            vla_ik_failures += int(vla.get("ik_failures", 0))
            vla_ik_recoveries += int(vla.get("ik_recoveries", 0))
            update = vla.get("controller_update", {})
            controller_evidence["updates_attempted"] += int(
                bool(update.get("actor_update_attempted", False))
            )
            controller_evidence["updates_rolled_back"] += int(
                bool(update.get("rolled_back", False))
            )
            controller_evidence["updates_with_parameter_change"] += int(
                bool(update.get("parameters_changed", False))
            )
            controller_evidence["ppo_epochs_completed"] += int(
                update.get("ppo_epochs_completed", 0)
            )
            controller_evidence["parameter_delta_l2"] += float(
                update.get("parameter_delta_l2", 0.0)
            )
            controller_evidence["approximate_kl"] += float(
                update.get("approximate_kl", 0.0)
            )
            for task_name, task_metrics in vla.get("per_task", {}).items():
                episodes = int(task_metrics["episodes"])
                totals = vla_task_totals.setdefault(
                    task_name,
                    {
                        "episodes": 0.0,
                        "successes": 0.0,
                        "positive_returns": 0.0,
                        "valid": 0.0,
                        "return": 0.0,
                        "steps": 0.0,
                        "ik_failures": 0.0,
                        "ik_recoveries": 0.0,
                        "ik_truncations": 0.0,
                        "execution_complete": 0.0,
                    },
                )
                totals["episodes"] += episodes
                totals["successes"] += int(task_metrics["successes"])
                totals["positive_returns"] += (
                    float(task_metrics.get(
                        "positive_return_rate",
                        float(float(task_metrics.get("return", 0.0)) > 1e-6),
                    ))
                    * episodes
                )
                totals["valid"] += float(task_metrics["valid_rate"]) * episodes
                totals["return"] += float(task_metrics["return"]) * episodes
                totals["steps"] += float(task_metrics["steps"]) * episodes
                totals["ik_failures"] += float(task_metrics.get("ik_failures", 0))
                totals["ik_recoveries"] += float(task_metrics.get("ik_recoveries", 0))
                totals["ik_truncations"] += float(task_metrics.get("ik_truncation_rate", 0.0)) * episodes
                totals["execution_complete"] += float(task_metrics.get("execution_complete_rate", 0.0)) * episodes
            self.round_robin_cursor += 1
            completed = round_index + 1
            progress.update(
                completed,
                eai_reward=eai_totals["reward"] / completed,
                vlabench_return=vla_totals["return"] / completed,
            )
            if round_callback is not None:
                round_callback(completed, completed_metrics(completed))
        return completed_metrics(rounds)


assert issubclass(JointReinforcementProgram, ReinforcementProgram)
