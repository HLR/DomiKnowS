"""Ahead-of-time plans for logical-constraint evaluators.

The graph formula is static while DataNodes and prediction tensors change for
each batch.  This module separates those two lifetimes:

* :class:`CompiledPlanCache` parses a logical-constraint tree once and keeps an
  immutable execution plan on the solver across batches;
* :class:`TensorizedCandidateResolver` binds that plan to one DataNode tree and
  executes the common identity/relation-path joins with padded index tensors.

Unusual candidate forms still delegate to :func:`getCandidates`.  That is a
correctness fallback, not a second implementation of their semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Optional

import torch

from domiknows.graph import LcElement, LogicalConstrain, V, CandidateSelection
from domiknows.graph.concept import Concept
from domiknows.graph.candidates import getCandidates


def _path_signature(value):
    """Stable-enough structural signature for invalidating a cached plan."""
    if isinstance(value, V):
        rel_info = value.relVarInfo
        if isinstance(rel_info, dict):
            rel_info = tuple(
                (str(k), _path_signature(v)) for k, v in rel_info.items())
        return ("V", value.name, _path_signature(value.v), rel_info)
    if isinstance(value, LogicalConstrain):
        return (
            "LC", id(value), type(value),
            tuple(_path_signature(e) for e in value.e),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_path_signature(v) for v in value))
    if isinstance(value, list):
        return ("list", tuple(_path_signature(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple((str(k), _path_signature(v)) for k, v in value.items()))
    if isinstance(value, (Concept, LcElement)):
        return (type(value), id(value), getattr(value, "name", None))
    try:
        hash(value)
        return value
    except TypeError:
        return (type(value), id(value), repr(value))


def constraint_signature(lc):
    return (type(lc), id(lc), tuple(_path_signature(e) for e in lc.e))


def _relation_name(value):
    if isinstance(value, str):
        return value
    return getattr(value, "name", None) or getattr(value, "fullname", None)


@dataclass(frozen=True)
class CandidatePlan:
    """Static candidate/path information for one formula element."""

    concept_name: str
    variable_name: Optional[str]
    paths: tuple[tuple[str, tuple[str, ...]], ...] = ()
    tensorizable: bool = False
    fallback_reason: Optional[str] = None

    @classmethod
    def compile(cls, element, variable, variable_name=None):
        concept_name = element[0].name
        variable_name = variable.name if variable_name is None else variable_name

        if variable.relVarInfo is not None:
            return cls(concept_name, variable_name, fallback_reason="relation-variable")

        path = variable.v
        if path is None:
            return cls(concept_name, variable_name, tensorizable=True)

        # eqL is intentionally handled by the established candidate resolver.
        from domiknows.graph.logicalConstrain import eqL
        if isinstance(path, eqL):
            return cls(concept_name, variable_name, fallback_reason="direct-eqL")
        if isinstance(path, str):
            path = (path,)
        if not isinstance(path, tuple) or not path:
            return cls(concept_name, variable_name, fallback_reason="invalid-path")

        if isinstance(path[0], str):
            raw_paths = (path,)
        else:
            raw_paths = tuple(path)

        parsed = []
        for raw in raw_paths:
            if not isinstance(raw, tuple) or not raw or not isinstance(raw[0], str):
                return cls(concept_name, variable_name, fallback_reason="complex-path")
            hops = []
            for hop in raw[1:]:
                if isinstance(hop, eqL):
                    return cls(concept_name, variable_name, fallback_reason="eqL-filter")
                name = _relation_name(hop)
                if name is None:
                    return cls(concept_name, variable_name, fallback_reason="non-relation-hop")
                hops.append(name)
            parsed.append((raw[0], tuple(hops)))

        return cls(
            concept_name, variable_name, paths=tuple(parsed), tensorizable=True)


@dataclass(frozen=True)
class CompiledElementPlan:
    index: int
    element: Any
    variable: V
    variable_name: str
    candidate: Optional[CandidatePlan] = None
    child: Optional["CompiledFormulaPlan"] = None


@dataclass(frozen=True)
class BatchedUnaryImplicationPlan:
    """A binary ``ifL`` whose two literals share one grounding domain.

    The plan is deliberately narrow: it recognizes the common KB-rule shape
    ``ifL(source("x"), target(path="x"))`` (and the equivalent explicit
    re-binding form).  Broader formulas remain on ``CompiledConstraintEvaluator``.
    """

    lc: LogicalConstrain
    source_element: tuple
    target_element: tuple
    source_concept: Concept
    target_concept: Concept


@dataclass(frozen=True)
class CompiledFormulaPlan:
    lc: LogicalConstrain
    steps: tuple[CompiledElementPlan, ...]
    signature: tuple
    dependency_revisions: tuple[tuple[LogicalConstrain, int], ...]
    batched_unary_implication: Optional[BatchedUnaryImplicationPlan] = None

    def is_current(self):
        return all(
            getattr(constraint, '_compile_revision', 0) == revision
            for constraint, revision in self.dependency_revisions)


class _PlanCompiler:
    def __init__(self):
        self.v_no = [1, 1]

    def compile(self, lc):
        first_v = None
        steps = []

        for index, element in enumerate(lc.e):
            if isinstance(element, V):
                continue

            if index + 1 < len(lc.e) and isinstance(lc.e[index + 1], V):
                variable = lc.e[index + 1]
            elif isinstance(element, LogicalConstrain):
                variable = V(name="_lc" + str(self.v_no[1]))
                self.v_no[1] += 1
            elif (
                isinstance(element, tuple) and element
                and isinstance(element[0], CandidateSelection)
            ):
                element[0].CandidateSelectionVariable = element[1]
                variable = V(name="_cs" + str(self.v_no[1]))
                self.v_no[1] += 1
            elif isinstance(element, (Concept, tuple, LcElement)):
                if first_v is None:
                    variable = V(name="_x" + str(self.v_no[0]))
                    if not isinstance(lc, CandidateSelection):
                        first_v = variable.name
                    self.v_no[0] += 1
                else:
                    variable = V(name="_x" + str(self.v_no[0]), v=(first_v,))
                    self.v_no[0] += 1
            else:
                # Integer/string parameters are retained as steps so execution
                # preserves the original operator's element order.
                variable = V(name="_arg" + str(index))

            if variable.name:
                variable_name = variable.name
            else:
                variable_name = "V" + str(self.v_no[0])
                self.v_no[0] += 1

            child = self.compile(element) if isinstance(element, LogicalConstrain) else None
            candidate = None
            if isinstance(element, (Concept, tuple)):
                try:
                    candidate = CandidatePlan.compile(element, variable, variable_name)
                except Exception:
                    candidate = None
            steps.append(CompiledElementPlan(
                index, element, variable, variable_name, candidate, child))

        dependencies = [(lc, getattr(lc, '_compile_revision', 0))]
        seen = {id(lc)}
        for step in steps:
            if step.child is None:
                continue
            for dependency in step.child.dependency_revisions:
                if id(dependency[0]) in seen:
                    continue
                dependencies.append(dependency)
                seen.add(id(dependency[0]))

        steps = tuple(steps)
        return CompiledFormulaPlan(
            lc, steps, constraint_signature(lc), tuple(dependencies),
            self._compile_batched_unary_implication(lc, steps))

    @staticmethod
    def _compile_batched_unary_implication(lc, steps):
        """Recognize identity-grounded binary implications at plan time."""
        from domiknows.graph.concept import EnumConcept
        from domiknows.graph.logicalConstrain import ifL

        if type(lc) is not ifL or len(steps) != 2:
            return None

        source, target = steps
        if source.index != 0 or target.index != 2:
            return None
        if not (
            isinstance(source.element, tuple)
            and len(source.element) == 4
            and isinstance(source.element[0], Concept)
            and isinstance(target.element, tuple)
            and len(target.element) == 4
            and isinstance(target.element[0], Concept)
        ):
            return None

        # Bare EnumConcepts contribute K columns, not one truth vector.  A
        # selected enum class remains a single-column literal and is eligible.
        if (
            isinstance(source.element[0], EnumConcept) and source.element[2] is None
        ) or (
            isinstance(target.element[0], EnumConcept) and target.element[2] is None
        ):
            return None

        source_candidate = source.candidate
        target_candidate = target.candidate
        if source_candidate is None or target_candidate is None:
            return None
        if not source_candidate.tensorizable or source_candidate.paths:
            return None
        if not target_candidate.tensorizable:
            return None

        # Either the target explicitly reuses the source variable name, or its
        # path is an identity reference to that source variable.
        explicit_rebind = (
            not target_candidate.paths
            and target.variable_name == source.variable_name
        )
        identity_path = target_candidate.paths == ((source.variable_name, ()),)
        if not (explicit_rebind or identity_path):
            return None

        return BatchedUnaryImplicationPlan(
            lc=lc,
            source_element=source.element,
            target_element=target.element,
            source_concept=source.element[0],
            target_concept=target.element[0],
        )


class CompiledPlanCache:
    """Constraint-plan cache with structural invalidation and diagnostics."""

    def __init__(self):
        self._plans = {}
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self._lock = RLock()

    def get(self, lc):
        with self._lock:
            key = id(lc)
            cached = self._plans.get(key)
            if cached is not None and cached.lc is lc and cached.is_current():
                self.hits += 1
                return cached
            if cached is not None:
                self.invalidations += 1
            plan = _PlanCompiler().compile(lc)
            self._plans[key] = plan
            self.misses += 1
            return plan

    def clear(self):
        with self._lock:
            self._plans.clear()

    def retain(self, constraints):
        """Drop plans for constraints no longer owned by the solver graphs."""
        with self._lock:
            live = {id(lc) for lc in constraints}
            for key in tuple(self._plans):
                if key not in live:
                    del self._plans[key]

    def info(self):
        with self._lock:
            return {
                "size": len(self._plans),
                "hits": self.hits,
                "misses": self.misses,
                "invalidations": self.invalidations,
            }


class TensorizedCandidateResolver:
    """Bind static candidate plans to one DataNode tree.

    Relation topology is converted lazily to padded destination-index tensors.
    Repeated paths in the same item then use tensor gather/masking rather than
    nested Python loops over every grounding and relation destination.
    """

    def __init__(self, probability_store):
        self.store = probability_store
        self._adjacency = {}
        self.tensorized_calls = 0
        self.fallback_calls = 0

    def rebind(self, probability_store):
        self.store = probability_store
        self._adjacency.clear()

    @staticmethod
    def _flatten_groups(groups):
        nodes = []
        group_ids = []
        item_ids = []
        for group_index, group in enumerate(groups):
            for item_index, node in enumerate(group):
                if node is None:
                    continue
                nodes.append(node)
                group_ids.append(group_index)
                item_ids.append(item_index)
        return nodes, group_ids, item_ids

    def _follow_once(self, nodes, group_ids, relation_name):
        """Follow one relation using a padded tensor gather."""
        if not nodes:
            return [], [], []

        key = (relation_name, tuple(id(node) for node in nodes))
        cached = self._adjacency.get(key)
        if cached is None:
            rows = []
            width = 1
            for node in nodes:
                destinations = node.getDnsForRelation(relation_name) or []
                destinations = [dest for dest in destinations if dest is not None]
                rows.append(destinations)
                width = max(width, len(destinations))

            unique = []
            node_index = {}
            encoded = torch.full((len(rows), width), -1, dtype=torch.long)
            for row_index, destinations in enumerate(rows):
                for column_index, destination in enumerate(destinations):
                    destination_index = node_index.get(id(destination))
                    if destination_index is None:
                        destination_index = len(unique)
                        node_index[id(destination)] = destination_index
                        unique.append(destination)
                    encoded[row_index, column_index] = destination_index
            cached = (encoded, unique)
            self._adjacency[key] = cached

        encoded, destinations = cached
        valid = encoded >= 0
        flat_destination_indices = encoded[valid]
        source_rows = torch.arange(encoded.shape[0]).unsqueeze(1).expand_as(encoded)[valid]
        group_tensor = torch.as_tensor(group_ids, dtype=torch.long)
        output_groups = group_tensor[source_rows]
        output_nodes = [destinations[index] for index in flat_destination_indices.tolist()]
        return output_nodes, output_groups.tolist()

    def _execute_path(self, groups, hops, *, expand):
        nodes, group_ids, item_ids = self._flatten_groups(groups)
        if expand:
            expansion_mapping = list(zip(group_ids, item_ids))
            group_ids = list(range(len(nodes)))
        else:
            expansion_mapping = None

        if not hops:
            output_nodes, output_groups = nodes, group_ids
        else:
            output_nodes, output_groups = nodes, group_ids
            for hop in hops:
                output_nodes, output_groups = self._follow_once(
                    output_nodes, output_groups, hop)
                if not output_nodes:
                    break

        group_count = len(nodes) if expand else len(groups)
        result = [[] for _ in range(group_count)]
        for node, group_index in zip(output_nodes, output_groups):
            result[group_index].append(node)
        result = [group if group else [None] for group in result]
        return result, expansion_mapping

    @staticmethod
    def _intersect(path_results):
        if len(path_results) == 1:
            return path_results[0]
        group_count = min(len(result) for result in path_results)
        output = []
        for group_index in range(group_count):
            first = [node for node in path_results[0][group_index] if node is not None]
            if not first:
                output.append([None])
                continue
            ids = torch.as_tensor([id(node) for node in first], dtype=torch.int64)
            keep = torch.ones(ids.shape[0], dtype=torch.bool)
            for result in path_results[1:]:
                other = [id(node) for node in result[group_index] if node is not None]
                if not other:
                    keep.zero_()
                    break
                keep &= torch.isin(ids, torch.as_tensor(other, dtype=torch.int64))
            selected = [node for node, include in zip(first, keep.tolist()) if include]
            output.append(selected or [None])
        return output

    @staticmethod
    def _apply_expansion(lc_variables_dns, mapping):
        expanded_names = list(lc_variables_dns.keys())
        for name in expanded_names:
            old = lc_variables_dns[name]
            new = []
            for group_index, item_index in mapping:
                if group_index >= len(old) or not old[group_index]:
                    new.append([None])
                elif len(old[group_index]) == 1:
                    new.append([old[group_index][0]])
                elif item_index < len(old[group_index]):
                    new.append([old[group_index][item_index]])
                else:
                    new.append([old[group_index][0]])
            lc_variables_dns[name] = new
        return expanded_names

    def get_candidates(self, plan, dn, element, variable, lc_variables_dns, lc, logger):
        if plan is None or not plan.tensorizable:
            self.fallback_calls += 1
            return getCandidates(dn, element, variable, lc_variables_dns, lc, logger)

        # Unqualified variables are either fresh concept domains or rebindings.
        if not plan.paths:
            if plan.variable_name in lc_variables_dns:
                self.tensorized_calls += 1
                return lc_variables_dns[plan.variable_name], [plan.variable_name], None
            # A parameterized executable plan retains the representative
            # concept name in ``plan`` but supplies the row's actual concept in
            # ``element``. Candidate topology must follow that runtime binding.
            concept_name = element[0].name
            dns = self.store.concept_datanodes(concept_name)
            self.tensorized_calls += 1
            return [[node] for node in dns], [], None

        # Forward references and other dynamic declaration forms keep the
        # established resolver until their declaration has been bound.
        if any(source not in lc_variables_dns for source, _ in plan.paths):
            self.fallback_calls += 1
            return getCandidates(dn, element, variable, lc_variables_dns, lc, logger)

        source_groups_by_path = [
            lc_variables_dns[source] for source, _ in plan.paths]
        expands = [
            any(len(group) > 1 for group in groups if group)
            for groups in source_groups_by_path]
        # The legacy resolver mutates prior bindings between paths during an
        # expansion. Preserve that uncommon ordering-sensitive case verbatim.
        if len(plan.paths) > 1 and any(expands):
            self.fallback_calls += 1
            return getCandidates(dn, element, variable, lc_variables_dns, lc, logger)
        if any(
            expand and any(node is None for group in groups for node in group)
            for expand, groups in zip(expands, source_groups_by_path)
        ):
            self.fallback_calls += 1
            return getCandidates(dn, element, variable, lc_variables_dns, lc, logger)

        path_results = []
        expansion_mapping = None
        for (source, hops), source_groups, expand in zip(
                plan.paths, source_groups_by_path, expands):
            result, mapping = self._execute_path(source_groups, hops, expand=expand)
            path_results.append(result)
            if mapping is not None:
                if expansion_mapping is not None and expansion_mapping != mapping:
                    self.fallback_calls += 1
                    return getCandidates(dn, element, variable, lc_variables_dns, lc, logger)
                expansion_mapping = mapping

        expansion_info = None
        if expansion_mapping is not None:
            expanded = self._apply_expansion(lc_variables_dns, expansion_mapping)
            expansion_info = {"mapping": expansion_mapping, "expanded_vars": expanded}

        self.tensorized_calls += 1
        return self._intersect(path_results), [source for source, _ in plan.paths], expansion_info
