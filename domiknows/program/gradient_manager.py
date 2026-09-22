"""Gradient conflict management and projection utilities for multi-domain learning."""

from collections.abc import Sequence
from contextlib import contextmanager
import threading
import torch


class GradientConflictManager:
    """Coordinates gradients across multiple active concept scopes to prevent interference.

    Supports recording gradients from multiple active scopes (e.g. Domain A and Domain B),
    projecting conflicting gradients on shared parameters using PCGrad, and executing
    a single synchronized optimizer step.

    Sequential-execution and transaction requirements:
        A complete step transaction spans ``begin_step()``, one or more ``capture()`` calls,
        and finally ``step()``. The manager uses a transaction lock and validates thread
        ownership across this sequence so concurrent worker threads cannot interleave
        or overwrite domain gradient buffers.

    Recommended workflow:
        gradient_manager.begin_step()

        with root.active_scope(eai_concepts, parameter_policy="freeze_inactive"):
            eai_loss = ...
            gradient_manager.capture("eai", eai_loss)

        with root.active_scope(vlabench_concepts, parameter_policy="freeze_inactive"):
            vlabench_loss = ...
            gradient_manager.capture("vlabench", vlabench_loss)

        gradient_manager.step(optimizer)
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None):
        self.model = model
        self.optimizer = optimizer
        self._recorded_gradients: dict[str, list[torch.Tensor | None]] = {}
        self._parameters: list[torch.nn.Parameter] = [
            p for p in model.parameters() if p.requires_grad
        ]
        self._transaction_lock = threading.RLock()
        self._owner_thread: int | None = None
        self._in_step = False

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return self._parameters

    def refresh_parameters(self) -> list[torch.nn.Parameter]:
        """Refresh the tracked trainable parameter list from the model."""
        self._parameters = [p for p in self.model.parameters() if p.requires_grad]
        return self._parameters

    @contextmanager
    def step_context(self):
        """Transaction context manager ensuring transaction lock release on exit or exception."""
        self.begin_step()
        try:
            yield self
        finally:
            if self._in_step:
                self.abort_step()

    def begin_step(self, timeout: float | None = 0.0) -> None:
        """Begin a multi-scope training step.
        
        Acquires the step transaction lock and ensures model parameter gradients
        are reset so stale gradients from preceding operations are not applied.
        Rejects nested begin_step() calls on the same thread as well as concurrent threads.
        """
        if self._in_step:
            raise RuntimeError(
                f"Cannot begin step; step already in progress (owned by thread {self._owner_thread}). "
                "Complete the current step with step() or cancel it with abort_step() before starting a new step."
            )
        current_thread = threading.get_ident()
        blocking = timeout is None or timeout < 0
        kwargs = {} if blocking else {"timeout": timeout}
        acquired = self._transaction_lock.acquire(**kwargs)
        if not acquired:
            raise RuntimeError(
                f"Cannot begin step; step transaction lock held by another thread ({self._owner_thread})."
            )
        self._owner_thread = current_thread
        self.clear()
        self._zero_model_grads()
        self._in_step = True

    def abort_step(self) -> None:
        """Abort an in-progress step transaction and release the transaction lock."""
        if not self._in_step:
            return
        current_thread = threading.get_ident()
        if self._owner_thread is not None and self._owner_thread != current_thread:
            raise RuntimeError(
                f"Cannot abort step; transaction owned by thread {self._owner_thread}, "
                f"not current thread {current_thread}."
            )
        self.clear()
        self._zero_model_grads()
        self._in_step = False
        self._owner_thread = None
        try:
            self._transaction_lock.release()
        except RuntimeError:
            pass

    def capture(
        self,
        scope_name: str,
        loss: torch.Tensor,
        parameters: Sequence[torch.nn.Parameter] | None = None,
    ) -> list[torch.Tensor | None]:
        """Capture gradients for a domain scope without applying an optimizer step.
        
        Ensures newly computed gradients are safely isolated and any parameter.grad
        attributes on the model are cleared so subsequent domain forward/loss
        evaluations are completely free from stale gradient state.
        """
        if not self._in_step:
            raise RuntimeError(
                "gradient_manager.capture() called without an active step transaction. "
                "Call begin_step() or use with gradient_manager.step_context(): first."
            )
        current_thread = threading.get_ident()
        if self._owner_thread != current_thread:
            raise RuntimeError(
                f"GradientConflictManager step transaction owned by thread {self._owner_thread}, "
                f"not current thread {current_thread}."
            )
        grads = self.record_scope_gradients(scope_name, loss, parameters=parameters)
        self._zero_model_grads()
        return grads

    def record_scope_gradients(
        self,
        scope_name: str,
        loss: torch.Tensor,
        parameters: Sequence[torch.nn.Parameter] | None = None,
    ) -> list[torch.Tensor | None]:
        """Compute and store gradients for a specific scope without modifying parameter states.

        Correctly handles active parameter policies (e.g. freeze_inactive) by filtering
        to parameters currently having ``requires_grad=True`` before invoking autograd,
        preventing RuntimeError on frozen parameters, and reconstructing a full
        manager-aligned gradient vector aligned with ``self._parameters``, with None
        for inactive or unincluded parameters.
        """
        param_to_idx = {id(p): idx for idx, p in enumerate(self._parameters)}
        if parameters is not None:
            target_params = list(parameters)
            for p in target_params:
                if id(p) not in param_to_idx:
                    raise ValueError(f"Supplied parameter is not managed by this GradientConflictManager: {p}")
        else:
            target_params = self._parameters

        active_params = [p for p in target_params if p.requires_grad]

        aligned_grads: list[torch.Tensor | None] = [None] * len(self._parameters)
        if active_params:
            computed_grads = torch.autograd.grad(
                loss,
                active_params,
                allow_unused=True,
                retain_graph=False,
            )
            for p, g in zip(active_params, computed_grads):
                if g is not None:
                    idx = param_to_idx[id(p)]
                    aligned_grads[idx] = g.detach().clone()

        self._recorded_gradients[scope_name] = aligned_grads
        return aligned_grads

    def resolve_gradients(
        self,
        method: str = "pcgrad",
    ) -> list[torch.Tensor | None]:
        """Resolve recorded multi-scope gradients into a single combined gradient per parameter."""
        scopes = list(self._recorded_gradients.keys())
        if not scopes:
            return []
        if len(scopes) == 1:
            return self._recorded_gradients[scopes[0]]

        if method == "pcgrad":
            if len(scopes) == 2:
                left_grads = self._recorded_gradients[scopes[0]]
                right_grads = self._recorded_gradients[scopes[1]]

                shared = [
                    (left, right)
                    for left, right in zip(left_grads, right_grads)
                    if left is not None and right is not None
                ]

                if shared:
                    dot = sum((left * right).sum() for left, right in shared)
                    left_norm = sum((left * left).sum() for left, _right in shared).clamp_min(1e-12)
                    right_norm = sum((right * right).sum() for _left, right in shared).clamp_min(1e-12)
                    conflict = bool(dot < 0)
                else:
                    dot = None
                    conflict = False

                resolved = []
                for left, right in zip(left_grads, right_grads):
                    if left is None:
                        combined = right
                    elif right is None:
                        combined = left
                    elif conflict:
                        combined = 0.5 * (
                            left - (dot / right_norm) * right
                            + right - (dot / left_norm) * left
                        )
                    else:
                        combined = 0.5 * (left + right)
                    resolved.append(combined)

                return resolved

            else:
                # Multi-scope PCGrad (> 2 scopes)
                num_params = len(self._parameters)
                projected_grads = {
                    s: [g.clone() if g is not None else None for g in self._recorded_gradients[s]]
                    for s in scopes
                }
                for i_idx, i_scope in enumerate(scopes):
                    for j_idx, j_scope in enumerate(scopes):
                        if i_idx == j_idx:
                            continue
                        shared = [
                            (projected_grads[i_scope][k], self._recorded_gradients[j_scope][k])
                            for k in range(num_params)
                            if projected_grads[i_scope][k] is not None
                            and self._recorded_gradients[j_scope][k] is not None
                        ]
                        if not shared:
                            continue
                        dot = sum((g_i * g_j).sum() for g_i, g_j in shared)
                        if dot < 0:
                            j_norm = sum((g_j * g_j).sum() for _g_i, g_j in shared).clamp_min(1e-12)
                            scale = dot / j_norm
                            for k in range(num_params):
                                g_i = projected_grads[i_scope][k]
                                g_j = self._recorded_gradients[j_scope][k]
                                if g_i is not None and g_j is not None:
                                    projected_grads[i_scope][k] = g_i - scale * g_j

                resolved = []
                for k in range(num_params):
                    valid = [projected_grads[s][k] for s in scopes if projected_grads[s][k] is not None]
                    if not valid:
                        resolved.append(None)
                    else:
                        resolved.append(sum(valid) / len(scopes))
                return resolved

        elif method == "average":
            param_count = len(self._parameters)
            resolved = []
            for i in range(param_count):
                active_grads = [
                    grads[i] for grads in self._recorded_gradients.values() if grads[i] is not None
                ]
                if not active_grads:
                    resolved.append(None)
                else:
                    resolved.append(sum(active_grads) / len(active_grads))
            return resolved

        else:
            raise ValueError(f"unknown gradient resolution method: {method}")

    def apply_resolved_gradients(
        self,
        resolved_gradients: Sequence[torch.Tensor | None] | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        clip_grad_norm: float | None = None,
        step_optimizer: bool = False,
    ) -> None:
        """Assign resolved gradients to parameters and optionally step the optimizer."""
        opt = optimizer or self.optimizer

        if resolved_gradients is None:
            resolved_gradients = self.resolve_gradients()

        for param, grad in zip(self._parameters, resolved_gradients):
            param.grad = None if grad is None else grad.detach().clone()

        # Ensure any parameters not in tracked target set have grad = None
        tracked_ids = {id(p) for p in self._parameters}
        for p in self.model.parameters():
            if id(p) not in tracked_ids:
                p.grad = None

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._parameters, clip_grad_norm)

        if step_optimizer and opt is not None:
            opt.step()
            opt.zero_grad(set_to_none=True)
            self.clear()

    def step(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        method: str = "pcgrad",
        clip_grad_norm: float | None = None,
    ) -> None:
        """Resolve all captured domain gradients and perform a single coordinated optimizer step.
        
        Strictly delays optimizer execution until all participating domain scopes have
        been captured, eliminating sequential parameter overwrites and cross-domain
        gradient interference. Releases the step transaction lock upon completion.
        """
        if not self._in_step:
            raise RuntimeError(
                "gradient_manager.step() called without an active step transaction. "
                "Call begin_step() before step()."
            )
        current_thread = threading.get_ident()
        if self._owner_thread != current_thread:
            raise RuntimeError(
                f"GradientConflictManager step transaction owned by thread {self._owner_thread}, "
                f"not current thread {current_thread}."
            )
        try:
            if not self._recorded_gradients:
                raise RuntimeError(
                    "gradient_manager.step() called with no captured domain gradients. "
                    "Call capture() in each active domain scope before stepping."
                )
            opt = optimizer or self.optimizer
            if opt is None:
                raise ValueError("optimizer must be provided either in __init__ or in step()")
            resolved = self.resolve_gradients(method=method)
            self.apply_resolved_gradients(
                resolved,
                optimizer=opt,
                clip_grad_norm=clip_grad_norm,
                step_optimizer=True,
            )
        finally:
            self._in_step = False
            self._owner_thread = None
            try:
                self._transaction_lock.release()
            except RuntimeError:
                pass

    def clear(self) -> None:
        """Clear recorded gradients."""
        self._recorded_gradients.clear()

    def _zero_model_grads(self) -> None:
        """Ensure all parameters in the model have .grad set to None."""
        for p in self.model.parameters():
            p.grad = None
