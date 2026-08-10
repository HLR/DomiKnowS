"""
Loss Calculator with unified T-Norm selection via TNormSelector.

Uses TNormSelector from adaptiveTNormLossCalculator for all t-norm decisions.
Supports three modes via the tnorm parameter:
  - Specific ("G","P","L","SP"): Use that t-norm for all constraints.
  - "default": Per-type optimal defaults.
  - "auto": Dynamic adaptation via AdaptiveTNormLossCalculator.
"""

from time import perf_counter_ns
import torch
from typing import Dict, Optional

from domiknows.graph import fixedL
from domiknows.graph.logicalConstrain import miotaL, queryL, sumL

from domiknows.solver.adaptiveTNormLossCalculator import (TNormSelector, get_constraint_type)

def _collect_tensors(obj):
    """Recursively collect tensors from nested lists/tuples."""
    tensors = []
    if torch.is_tensor(obj):
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_collect_tensors(item))
    return tensors


def multi_query_joint_nll(distribution, label, num_subclasses, *, epsilon=1e-6,
                          label_name="multi-answer queryL"):
    """Validate an aligned class-ID label and return joint per-position NLL."""
    target = torch.as_tensor(label, device=distribution.device).reshape(-1)
    if target.dtype.is_floating_point and not torch.all(target == target.round()):
        raise ValueError(f"{label_name} label must contain integer class IDs")
    target = target.long()
    if target.numel() != distribution.shape[0]:
        raise ValueError(
            f"{label_name} label has {target.numel()} values, but the constraint "
            f"grounded {distribution.shape[0]} candidates"
        )
    if torch.any((target < -1) | (target >= num_subclasses)):
        raise ValueError(
            f"{label_name} labels must be -1 or class IDs in "
            f"[0, {num_subclasses - 1}]"
        )
    membership = distribution.sum(dim=-1)
    chosen = torch.empty_like(membership)
    selected = target >= 0
    chosen[~selected] = 1.0 - membership[~selected]
    if selected.any():
        chosen[selected] = distribution[selected].gather(
            1, target[selected].unsqueeze(1)
        ).squeeze(1)
    clamped = chosen.clamp(epsilon, 1.0)
    stable = chosen + (clamped - chosen).detach()
    losses = -torch.log(stable)
    mean_loss = losses.mean() if losses.numel() else distribution.sum() * 0.0
    return target, chosen, losses, mean_loss


class LossCalculator:
    """
    Loss calculator with per-constraint t-norm selection via TNormSelector.
    """

    def __init__(self, solver, tnorm_selector: Optional[TNormSelector] = None):
        """
        Args:
            solver: Reference to gurobiILPOntSolver instance.
            tnorm_selector: Optional pre-configured TNormSelector. If None, one is created based tnorm and counting_tnorm parameters in calculateLoss.
        """
        self.solver = solver
        self._external_selector = tnorm_selector

    def _normalize_loss_list(self, lossTensor):
        """
        Normalize arbitrary nested lossTensor into a clean 1D tensor.

        Always returns:
            - 1D tensor (if any values exist)
            - None (if empty)
        """

        if lossTensor is None:
            return None

        # Collect every tensor recursively
        tensors = _collect_tensors(lossTensor)

        if not tensors:
            return None

        normalized = []

        for t in tensors:
            if t.dim() == 0:
                # scalar tensor
                normalized.append(t)
            else:
                # flatten vector or higher-dim tensor
                normalized.extend(t.reshape(-1))

        if not normalized:
            return None

        return torch.stack(normalized)

    def _normalize_query_distribution(self, query_output, num_subclasses):
        tensors = _collect_tensors(query_output)
        candidates = []
        for tensor in tensors:
            if tensor is None:
                continue
            if tensor.dim() == 0:
                continue
            flat = tensor.reshape(-1)
            if flat.numel() == num_subclasses:
                candidates.append(flat)

        if not candidates:
            return None

        distribution = torch.stack(candidates).mean(dim=0)
        total = distribution.sum()
        if (
            torch.is_tensor(total)
            and total.detach().abs().item() > 1e-12
            and abs(total.detach().item() - 1.0) > 1e-5
        ):
            distribution = distribution / total
        return distribution

    def _normalize_multi_query_distribution(self, query_output, num_subclasses):
        """Preserve the candidate axis of a multi-answer query result."""
        tensors = _collect_tensors(query_output)
        candidates = []
        for tensor in tensors:
            if tensor is None or tensor.dim() < 2:
                continue
            if tensor.shape[-1] == num_subclasses:
                candidates.append(tensor.reshape(-1, num_subclasses))
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        row_counts = {candidate.shape[0] for candidate in candidates}
        if len(row_counts) != 1:
            raise ValueError("multi-answer queryL produced inconsistent candidate counts")
        return torch.stack(candidates).mean(dim=0)

    @staticmethod
    def _decode_multi_query(distribution, threshold):
        if distribution is None:
            return None
        if distribution.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=distribution.device)
        membership = distribution.sum(dim=-1)
        answers = torch.argmax(distribution, dim=-1).to(torch.long)
        return torch.where(
            membership >= threshold,
            answers,
            torch.full_like(answers, -1),
        )

    def _normalize_selection_distribution(self, selection_output):
        """Flatten a vector-valued selector without normalizing its mass."""
        tensors = _collect_tensors(selection_output)
        if not tensors:
            return None
        return torch.cat([tensor.reshape(-1) for tensor in tensors])

    def calculate_single_lc_loss(self, lc, dn, key, tnorm='L', counting_tnorm=None, label=None):
        """
        Calculate loss for a single logical constraint.

        Args:
            lc: The logical constraint.
            dn: Data node.
            key: Softmax key.
            tnorm: A t-norm string ("L","P","SP","G","default", "auto").
            counting_tnorm: Deprecated —  but stil accepted for backward compatibility. Ignored if tnorm is a auto.
            label: Optional label for sumL constraints.
        """
        if not lc.headLC or not lc.active:
            return None
        if type(lc) is fixedL:
            return None

        selector =  self._external_selector if self._external_selector is not None else TNormSelector(tnorm, counting_tnorm)

        start = perf_counter_ns()
        myBooleanMethods = self.solver.myLcLossBooleanMethods
        result = {'lc': lc}

        selected_tnorm = selector.select(lc=lc)
        constraint_type = get_constraint_type(lc)
        result['tnorm_used'] = selected_tnorm
        result['constraint_type'] = constraint_type

        myBooleanMethods.setTNorm(selected_tnorm)

        if isinstance(lc, sumL) and label is None:
            _rawLabel = dn.getExecutableConstraintLabel(lc.lcName)
            if _rawLabel is None:
                return None
            label = _rawLabel.float()
        elif isinstance(lc, queryL) and lc.is_multi_answer and label is None:
            label = dn.getExecutableConstraintLabel(lc.lcName)

        self.solver.myLogger.info(f'Processing {lc} with t-norm {selected_tnorm}')

        self.solver.constraintConstructor.current_device = self.solver.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph

        if isinstance(lc, miotaL):
            selection_output, _lc_variables = self.solver.constraintConstructor.constructLogicalConstrains(
                lc, myBooleanMethods, None, dn, 0,
                key=key, headLC=False, loss=True, sample=False)
            selection_distribution = self._normalize_selection_distribution(selection_output)
            result['selectionDistribution'] = selection_distribution
            result['lossTensor'] = None
            result['conversionSigmoid'] = selection_distribution
            result['loss'] = None
            result['elapsedInMsLC'] = (perf_counter_ns() - start) / 1_000_000
            return result

        if isinstance(lc, queryL):
            query_output, _lc_variables = self.solver.constraintConstructor.constructLogicalConstrains(
                lc, myBooleanMethods, None, dn, 0,
                key=key, headLC=False, loss=True, sample=False)
            if lc.is_multi_answer:
                query_distribution = self._normalize_multi_query_distribution(
                    query_output, lc.num_subclasses)
            else:
                query_distribution = self._normalize_query_distribution(
                    query_output, lc.num_subclasses)
            result['queryDistribution'] = query_distribution
            if lc.is_multi_answer:
                result['queryAnswer'] = self._decode_multi_query(
                    query_distribution, lc.threshold)
            result['lossTensor'] = None
            if query_distribution is not None:
                result['conversionSigmoid'] = query_distribution
                if lc.is_multi_answer and label is not None:
                    _target, probability, losses, mean_loss = multi_query_joint_nll(
                        query_distribution,
                        label,
                        lc.num_subclasses,
                        label_name=f"multi-answer queryL {lc.lcName}",
                    )
                    result['probability'] = probability
                    result['lossTensor'] = losses
                    result['loss'] = mean_loss
                else:
                    result['loss'] = (
                        None if lc.is_multi_answer
                        else 1.0 - query_distribution.max()
                    )
            else:
                result['conversionSigmoid'] = None
                result['loss'] = None
            result['elapsedInMsLC'] = (perf_counter_ns() - start) / 1_000_000
            return result

        if isinstance(lc, sumL) and label is None:
            result = None
        else:
            lossTensor = self.solver.constraintConstructor.constructLogicalConstrains(
                lc, myBooleanMethods, None, dn, 0,
                key=key, headLC=True, loss=True, sample=False,
                **({'label': int(label)} if label is not None else {}))

        normalized = self._normalize_loss_list(lossTensor)
        result['lossTensor'] = normalized

        if normalized is None:
            lossTensor = None
        else:
            # define "loss" as mean of per-element losses
            lossTensor = normalized.mean()

        result['loss'] = lossTensor

        if lossTensor is not None:
            result['conversionSigmoid'] = 1.0 - lossTensor
            if isinstance(lc, sumL) or (hasattr(lc, 'innerLC') and isinstance(lc.innerLC, sumL)):
                result['expectedCount'] = lossTensor
        else:
            result['conversionSigmoid'] = None

        result['elapsedInMsLC'] = (perf_counter_ns() - start) / 1_000_000
        return result

    def calculateLoss(self, dn, tnorm='L', counting_tnorm=None, include_executable=False):
        """
        Calculate loss with per-constraint t-norm selection.

        Args:
            dn: Data node.
            tnorm: T-norm mode — "L","P","SP","G","default", or "auto".
            counting_tnorm: Deprecated — depricated but still accepted for backward compatibility. Ignored if tnorm is "auto".
            include_executable: Also evaluate ``execute()``-wrapped constraints.

                ``execute()`` *moves* a constraint out of ``graph.logicalConstrains``
                into ``graph.executableLCs``, so by default this method never sees
                it. That default is deliberate and load-bearing: ``InferenceProgram``
                scores executable constraints itself and adds a separately weighted
                graph-global term, relying on this exclusion to avoid double-counting
                (see domiknows/program/README_inference.md).

                Turn it on only when you want one number covering *every* constraint
                — e.g. comparing this path against the exact-circuit path, which does
                iterate both populations, so the two otherwise score different
                constraint sets on the same graph.
        """
        myBooleanMethods = self.solver.myLcLossBooleanMethods
        myBooleanMethods.current_device = dn.current_device
        myBooleanMethods.current_dtype = self.solver.constraintConstructor.current_dtype

        self.solver.myLogger.info('Calculating loss with per-constraint t-norms')
        self.solver.myLoggerTime.info('Calculating loss with per-constraint t-norms')

        key = "/local/softmax"
        lcLosses = {}

        dn.setActiveExecutableLCs()

        for graph in self.solver.myGraph:
            for _, lc in graph.logicalConstrains.items():
                result = self.calculate_single_lc_loss(lc, dn, key, tnorm=tnorm, counting_tnorm=counting_tnorm, label=None)
                if result is not None:
                    lcLosses[lc.lcName] = result

            if not include_executable:
                continue

            # Executable wrappers hold their real expression in `innerLC`, which
            # is intentionally marked non-head; a runtime label makes it a root.
            for executable_name, executable in graph.executableLCs.items():
                label = dn.getExecutableConstraintLabel(executable_name)
                inner = getattr(executable, 'innerLC', None)
                if label is None or inner is None:
                    continue

                old_head = inner.headLC
                inner.headLC = True
                try:
                    result = self.calculate_single_lc_loss(
                        inner, dn, key, tnorm=tnorm, counting_tnorm=counting_tnorm,
                        label=label)
                finally:
                    inner.headLC = old_head

                if result is not None:
                    result['executableName'] = executable_name
                    lcLosses[executable_name] = result

        return lcLosses
