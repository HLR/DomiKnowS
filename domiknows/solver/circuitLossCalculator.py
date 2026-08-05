"""Exact semantic-loss calculation by circuit compilation and torch WMC."""

from __future__ import annotations

from time import perf_counter_ns
import warnings

import torch

from domiknows.graph import fixedL
from domiknows.graph.logicalConstrain import miotaL, queryL, sumL

from .bdd import BDDNode, CircuitSizeLimitExceeded
from .circuitBooleanMethods import circuitBooleanMethods
from .lossCalculator import LossCalculator


def _collect_circuit_nodes(value):
    nodes = []
    if isinstance(value, BDDNode) or (
        hasattr(value, "is_literal") and hasattr(value, "elements")
    ):
        nodes.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            nodes.extend(_collect_circuit_nodes(item))
    return nodes


#: How a head constraint's groundings are combined into its loss.
#:
#: ``"joint"`` (default) compiles the conjunction of every grounding into one
#: circuit and reports ``-log P(all groundings hold)``. This is the exact joint
#: semantics: a concept variable appearing in several groundings is *one*
#: logical variable, so cross-grounding dependence is preserved (which
#: per-grounding factorisation — and the t-norm path — silently discards).
#:
#: ``"per_grounding"`` reports ``-log P(g)`` separately per grounding, giving a
#: ``[G]`` loss vector. It loses the cross-grounding dependence but (a) keeps the
#: loss scale independent of how many groundings a data item happens to have
#: (the joint ``-log`` grows roughly linearly with the grounding count, so
#: ``beta`` otherwise has to be retuned per task size), and (b) is required by
#: per-grounding dual mechanisms such as the amortized DualCritic (R5B), which
#: need one violation entry per grounding to attribute against.
AGGREGATIONS = ("joint", "per_grounding")


class CircuitLossCalculator:
    """Drive the shared DSL traversal with the exact circuit processor."""

    def __init__(self, solver, epsilon=1e-12, aggregation="joint"):
        self.solver = solver
        self.epsilon = float(epsilon)
        self.aggregation = self._check_aggregation(aggregation)
        self._compile_cache = {}

    @staticmethod
    def _check_aggregation(aggregation):
        if aggregation is None:
            return "joint"
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        return aggregation

    @staticmethod
    def _label_index(label):
        if label is None:
            return None
        if torch.is_tensor(label):
            values = label.detach().reshape(-1)
            if values.numel() == 0:
                return None
            if values.numel() > 1:
                return int(values.argmax().item())
            return int(values.item())
        return int(label)

    def calculate_single_lc_loss(
        self,
        lc,
        dn,
        key="/local/softmax",
        label=None,
        *,
        force_root=False,
        label_name=None,
        aggregation=None,
    ):
        if (not lc.headLC and not force_root) or not lc.active or type(lc) is fixedL:
            return None

        aggregation = self._check_aggregation(aggregation or self.aggregation)

        start = perf_counter_ns()
        processor = self.solver.myCircuitBooleanMethods
        processor.begin_evaluation()
        constructor = self.solver.constraintConstructor
        constructor.current_device = self.solver.current_device
        constructor.myGraph = self.solver.myGraph

        if isinstance(lc, (sumL, queryL, miotaL)) and label is None:
            label = dn.getExecutableConstraintLabel(label_name or lc.lcName)
        if isinstance(lc, sumL) and label is None:
            return None

        output, _ = constructor.constructLogicalConstrains(
            lc,
            processor,
            None,
            dn,
            0,
            key=key,
            headLC=not isinstance(lc, (queryL, miotaL)),
            loss=False,
            sample=False,
            label=self._label_index(label) if isinstance(lc, sumL) else None,
            circuit=True,
        )

        result = {
            "lc": lc,
            "backend": processor.backend_name,
            "nodeCount": processor.node_count,
            "groundingSignature": processor.grounding_signature(),
        }

        if isinstance(lc, miotaL):
            selection_nodes = _collect_circuit_nodes(output)
            probabilities = [processor.wmc(node).reshape(()) for node in selection_nodes]
            distribution = torch.stack(probabilities) if probabilities else None
            if distribution is None and label is not None:
                target_probe = torch.as_tensor(label).reshape(-1)
                if target_probe.numel() != 0:
                    raise ValueError(
                        f"miotaL label has {target_probe.numel()} values, but the "
                        "constraint grounded 0 candidates"
                    )
                distribution = torch.empty(0, device=self.solver.current_device)
            result['selectionDistribution'] = distribution
            result['conversionSigmoid'] = distribution
            if label is None:
                result['probability'] = None
                result['lossTensor'] = None
                result['loss'] = None
            else:
                target = torch.as_tensor(label, device=distribution.device).float().reshape(-1)
                if target.numel() != distribution.numel():
                    raise ValueError(
                        f"miotaL label has {target.numel()} values, but the constraint "
                        f"grounded {distribution.numel()} candidates"
                    )
                if not torch.all((target == 0) | (target == 1)):
                    raise ValueError("miotaL label must be a binary multi-hot vector")
                selected_probability = torch.where(target.bool(), distribution, 1.0 - distribution)
                losses = -torch.log(selected_probability.clamp_min(self.epsilon))
                result['probability'] = selected_probability
                result['lossTensor'] = losses
                result['loss'] = losses.mean() if losses.numel() else distribution.sum() * 0.0
        elif isinstance(lc, queryL):
            class_nodes = _collect_circuit_nodes(output)
            probabilities = [processor.wmc(node) for node in class_nodes]
            result["queryProbabilities"] = (
                torch.stack(probabilities) if probabilities else None
            )
            label_index = self._label_index(label)
            if label_index is None:
                result["probability"] = None
                result["lossTensor"] = None
                result["loss"] = None
            elif not 0 <= label_index < len(class_nodes):
                raise IndexError(
                    f"queryL label {label_index} is outside its {len(class_nodes)} subclasses"
                )
            else:
                probability = probabilities[label_index]
                loss = -torch.log(probability.clamp_min(self.epsilon))
                result["probability"] = probability
                result["lossTensor"] = loss.reshape(1)
                result["loss"] = loss
        else:
            grounded_nodes = _collect_circuit_nodes(output)
            if not grounded_nodes:
                return None

            result["aggregation"] = aggregation

            if aggregation == "per_grounding":
                # One -log P per grounding. Keeps the loss scale independent of
                # the grounding count and gives per-grounding dual mechanisms
                # (R5B) something to attribute against; the trade-off is that
                # dependence between groundings that share a variable is lost.
                probabilities = torch.stack(
                    [processor.wmc(node).reshape(()) for node in grounded_nodes])
                losses = -torch.log(probabilities.clamp_min(self.epsilon))
                result.update(
                    probability=probabilities,
                    lossTensor=losses,
                    loss=losses.mean(),
                    cacheHit=False,
                    groundingCount=len(grounded_nodes),
                )
            else:
                # A head LC means every grounding must hold.  Compiling their
                # conjunction in one circuit is what preserves shared-leaf identity.
                root = processor.manager.and_all(grounded_nodes)
                cache_key = (
                    id(lc),
                    processor.grounding_signature(),
                    processor.leaf_key_signature,
                    processor.backend_name,
                )
                cached = self._compile_cache.get(cache_key)
                root_identity = getattr(root, "node_id", getattr(root, "id", None))
                cached_identity = getattr(cached, "node_id", getattr(cached, "id", None))
                cache_hit = cached is not None and cached_identity == root_identity
                if cache_hit:
                    root = cached
                else:
                    self._compile_cache[cache_key] = root
                probability = processor.wmc(root)
                loss = -torch.log(probability.clamp_min(self.epsilon))
                result.update(
                    probability=probability,
                    lossTensor=loss.reshape(1),
                    loss=loss,
                    cacheHit=cache_hit,
                    groundingCount=len(grounded_nodes),
                )

        if isinstance(lc, miotaL):
            result["conversionSigmoid"] = result.get("selectionDistribution")
        else:
            result["conversionSigmoid"] = result.get("probability")
        result["elapsedInMsLC"] = (perf_counter_ns() - start) / 1_000_000
        return result

    def calculateCircuitLoss(self, dn, aggregation=None):
        processor = self.solver.myCircuitBooleanMethods
        processor.current_device = dn.current_device
        processor.current_dtype = getattr(dn, "current_dtype", None)
        dn.setActiveExecutableLCs()
        aggregation = self._check_aggregation(aggregation or self.aggregation)

        losses = {}

        def calculate(lc, *, label=None, force_root=False, label_name=None):
            try:
                return self.calculate_single_lc_loss(
                    lc,
                    dn,
                    label=label,
                    force_root=force_root,
                    label_name=label_name,
                    aggregation=aggregation,
                )
            except CircuitSizeLimitExceeded as error:
                warnings.warn(
                    f"{error} Falling back to Product t-norm for {label_name or lc.lcName!r}.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                old_processor = self.solver.myCircuitBooleanMethods
                # Discard the over-budget partial manager so later constraints
                # and batches can compile normally.
                self.solver.myCircuitBooleanMethods = circuitBooleanMethods(
                    backend=old_processor.requested_backend,
                    max_nodes=old_processor.max_nodes,
                    size_limit_action=old_processor.size_limit_action,
                )
                self._compile_cache.clear()
                loss_processor = self.solver.myLcLossBooleanMethods
                loss_processor.current_device = dn.current_device
                loss_processor.current_dtype = getattr(
                    self.solver.constraintConstructor,
                    "current_dtype",
                    None,
                )
                old_head = lc.headLC
                if force_root:
                    lc.headLC = True
                try:
                    fallback = LossCalculator(self.solver).calculate_single_lc_loss(
                        lc,
                        dn,
                        "/local/softmax",
                        tnorm="P",
                        label=label,
                    )
                finally:
                    lc.headLC = old_head
                if fallback is None:
                    return None
                fallback["backend"] = "tnorm"
                fallback["fallback"] = "circuit-size-limit"
                fallback["exact"] = False
                fallback["sizeLimitError"] = str(error)
                if isinstance(lc, queryL) and fallback.get("queryDistribution") is not None:
                    label_index = self._label_index(label)
                    if label_index is not None:
                        probability = fallback["queryDistribution"][label_index]
                        loss = -torch.log(probability.clamp_min(self.epsilon))
                        fallback["probability"] = probability
                        fallback["loss"] = loss
                        fallback["lossTensor"] = loss.reshape(1)
                return fallback

        for graph in self.solver.myGraph:
            for _, lc in graph.logicalConstrains.items():
                result = calculate(lc)
                if result is not None:
                    losses[lc.lcName] = result
            # Executable constraints (notably queryL and sumL) are wrappers
            # whose inner LC is intentionally marked non-head.  A present
            # runtime label makes that inner expression an exact circuit root.
            for executable_name, executable in graph.executableLCs.items():
                label = dn.getExecutableConstraintLabel(executable_name)
                inner = getattr(executable, "innerLC", None)
                if label is None or inner is None:
                    continue
                result = calculate(
                    inner,
                    label=label,
                    force_root=True,
                    label_name=executable_name,
                )
                if result is not None:
                    result["executableName"] = executable_name
                    losses[executable_name] = result
        return losses
