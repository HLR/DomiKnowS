"""Adapters from DomiKnowS graph declarations to HMM masks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .constraints import (
    AllowedEmissionsSpec,
    AllowedTransitionsSpec,
    ConstraintApplicationReport,
    ConstraintDFAExportSpec,
    EmissionMaskSpec,
    ForbiddenEmissionsSpec,
    ForbiddenTransitionsSpec,
    StatePredicateTransitionSpec,
    TransitionMaskSpec,
    validate_mask,
)
from .dynamic import FactorizedStateSpace


def _name(obj: Any) -> str:
    return str(getattr(obj, "name", obj))


def _dedupe_names(items: Iterable[Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = _name(item)
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _domain_concept_names(items: Iterable[Any]) -> list[str]:
    return [name for name in _dedupe_names(items) if name != "constraint"]


class DomiKnowSGraphAdapter:
    """Extract graph-shaped HMM masks from a DomiKnowS graph.

    The adapter is intentionally conservative. It compiles explicit masks,
    tuple/dict constraint specs, and graph relation endpoints that can be
    mapped to HMM state names. Unsupported logical constraints are reported
    instead of approximated.
    """

    def __init__(
        self,
        graph,
        *,
        concepts: Iterable[Any] | None = None,
        relations: Iterable[Any] | None = None,
        constraints: Iterable[Any] | None = None,
        n_hidden_states: int | None = None,
        state_names: Iterable[str] | None = None,
        state_space: FactorizedStateSpace | None = None,
        symbols: Iterable[Any] | None = None,
        device=None,
        dtype: torch.dtype = torch.float64,
    ):
        self.graph = graph
        self._explicit_concepts = list(concepts) if concepts is not None else None
        self._explicit_relations = list(relations) if relations is not None else None
        self._explicit_constraints = list(constraints) if constraints is not None else None
        self.n_hidden_states = n_hidden_states
        self.state_space = state_space
        self.state_names = tuple(state_names) if state_names is not None else None
        self.symbols = tuple(symbols) if symbols is not None else None
        self.device = device
        self.dtype = dtype
        self.report = ConstraintApplicationReport()

    def concepts(self) -> list[str]:
        if self._explicit_concepts is not None:
            return _dedupe_names(self._explicit_concepts)
        graph_concepts = getattr(self.graph, "concepts", {}) if self.graph is not None else {}
        if isinstance(graph_concepts, dict):
            return _domain_concept_names(graph_concepts.keys())
        return _domain_concept_names(graph_concepts)

    def relations(self) -> list[str]:
        return [_relation_name(rel) for rel in self._relation_objects()]

    def constraints(self) -> list[Any]:
        if self._explicit_constraints is not None:
            return list(self._explicit_constraints)
        graph_constraints = getattr(self.graph, "logicalConstrains", {}) if self.graph is not None else {}
        constraints = list(graph_constraints.values()) if isinstance(graph_constraints, dict) else list(graph_constraints)
        return [
            constraint
            for constraint in constraints
            if getattr(constraint, "active", True) and getattr(constraint, "headLC", True)
        ]

    def set_symbols(self, symbols: Iterable[Any]) -> None:
        self.symbols = tuple(symbols)

    def allowed_transition_mask(self) -> torch.Tensor:
        state_names = self._state_names()
        shape = (len(state_names), len(state_names))
        mask = torch.ones(shape, dtype=self.dtype, device=self.device)

        relation_pairs = self._mapped_relation_pairs(state_names)
        if relation_pairs:
            mask = torch.zeros(shape, dtype=self.dtype, device=self.device)
            for src, dst in relation_pairs:
                mask[src, dst] = 1.0
            self.report.add_applied(f"compiled {len(relation_pairs)} graph relation transition(s)")

        for constraint in self.constraints():
            mask = self._apply_transition_constraint(mask, constraint, state_names)
        return mask

    def emission_type_mask(self) -> torch.Tensor:
        state_names = self._state_names()
        symbols = self.symbols or ()
        shape = (len(state_names), len(symbols))
        mask = torch.ones(shape, dtype=self.dtype, device=self.device)
        if not symbols:
            return mask
        for constraint in self.constraints():
            mask = self._apply_emission_constraint(mask, constraint, state_names, tuple(map(str, symbols)))
        return mask

    def _state_names(self) -> tuple[str, ...]:
        if self.state_names is not None:
            names = tuple(map(str, self.state_names))
        elif self.state_space is not None:
            names = tuple(map(str, self.state_space.state_names))
        else:
            concept_names = self.concepts()
            if self.n_hidden_states is not None and len(concept_names) == self.n_hidden_states:
                names = tuple(concept_names)
            elif self.n_hidden_states is not None:
                names = tuple(f"S{i}" for i in range(self.n_hidden_states))
            else:
                names = tuple(concept_names)
        if not names:
            raise ValueError("cannot infer HMM state names; pass n_hidden_states or state_names")
        if self.n_hidden_states is not None and len(names) != self.n_hidden_states:
            raise ValueError(f"state_names length {len(names)} does not match n_hidden_states={self.n_hidden_states}")
        if len(set(names)) != len(names):
            raise ValueError("state names must be unique")
        return names

    def _relation_objects(self) -> list[Any]:
        if self._explicit_relations is not None:
            return list(self._explicit_relations)
        graph_concepts = getattr(self.graph, "concepts", {}) if self.graph is not None else {}
        concept_values = graph_concepts.values() if isinstance(graph_concepts, dict) else graph_concepts
        relations: list[Any] = []
        seen: set[int] = set()
        for concept in concept_values:
            for rels in getattr(concept, "_out", {}).values():
                for rel in rels:
                    if getattr(rel, "is_reversed", False):
                        continue
                    if id(rel) not in seen:
                        relations.append(rel)
                        seen.add(id(rel))
        return relations

    def _mapped_relation_pairs(self, state_names: tuple[str, ...]) -> list[tuple[int, int]]:
        state_index = {name: idx for idx, name in enumerate(state_names)}
        pairs: list[tuple[int, int]] = []
        for rel in self._relation_objects():
            pair = _relation_pair(rel)
            if pair is None:
                self.report.add_unsupported(f"relation {rel!r} is not a state transition pair")
                continue
            src, dst = pair
            if src in state_index and dst in state_index:
                pairs.append((state_index[src], state_index[dst]))
            else:
                self.report.add_unsupported(f"relation {_relation_name(rel)!r} endpoints {src!r}->{dst!r} do not map to HMM states")
        return pairs

    def _apply_transition_constraint(self, mask: torch.Tensor, constraint: Any, state_names: tuple[str, ...]) -> torch.Tensor:
        specs = _normalize_constraint_specs(constraint)
        compiled = _compile_lc_to_transition_specs(constraint, state_names, device=mask.device, dtype=mask.dtype)
        if compiled is not None:
            specs.extend(compiled)
        if not specs:
            if _looks_like_logical_constraint(constraint):
                symbols = tuple(map(str, self.symbols or ()))
                if _compile_lc_to_emission_specs(constraint, state_names, symbols, device=mask.device, dtype=mask.dtype) is not None:
                    return mask
                self.report.add_unsupported(f"unsupported static HMM transition logical constraint {_lc_name(constraint)}")
            else:
                self.report.add_unsupported(f"unsupported HMM transition constraint {constraint!r}")
            return mask

        state_index = {name: idx for idx, name in enumerate(state_names)}
        for spec in specs:
            if isinstance(spec, TransitionMaskSpec):
                self.report.add_applied(_spec_message("applied transition mask", spec))
                mask = mask * validate_mask(spec.mask, tuple(mask.shape), name="transition_mask", device=mask.device, dtype=mask.dtype)
            elif isinstance(spec, AllowedTransitionsSpec):
                new_mask = torch.zeros_like(mask)
                for src, dst in _as_pairs(spec.transitions):
                    src_name = str(src)
                    dst_name = str(dst)
                    if src_name in state_index and dst_name in state_index:
                        new_mask[state_index[src_name], state_index[dst_name]] = 1.0
                    else:
                        self.report.add_unsupported(f"allowed transition {src!r}->{dst!r} does not map to HMM states")
                self.report.add_applied(_spec_message("applied allowed transition set", spec))
                mask = mask * new_mask
            elif isinstance(spec, ForbiddenTransitionsSpec):
                for src, dst in _as_pairs(spec.transitions):
                    src_name = str(src)
                    dst_name = str(dst)
                    if src_name in state_index and dst_name in state_index:
                        mask[state_index[src_name], state_index[dst_name]] = 0.0
                    else:
                        self.report.add_unsupported(f"forbidden transition {src!r}->{dst!r} does not map to HMM states")
                self.report.add_applied(_spec_message("applied forbidden transition set", spec))
            elif isinstance(spec, StatePredicateTransitionSpec):
                if self.state_space is None:
                    self.report.add_unsupported(_spec_message("StatePredicateTransitionSpec requires a FactorizedStateSpace", spec))
                    continue
                predicate_mask = self.state_space.transition_mask(spec.predicate, dtype=mask.dtype, device=mask.device)
                self.report.add_applied(_spec_message("applied factorized-state transition predicate", spec))
                mask = mask * predicate_mask
            elif isinstance(spec, ConstraintDFAExportSpec):
                self.report.add_applied(_spec_message("registered DFA-export constraint spec", spec))
            elif isinstance(spec, (EmissionMaskSpec, AllowedEmissionsSpec, ForbiddenEmissionsSpec)):
                continue
            else:
                self.report.add_unsupported(f"unsupported HMM transition spec {spec!r}")
        return mask

    def _apply_emission_constraint(
        self,
        mask: torch.Tensor,
        constraint: Any,
        state_names: tuple[str, ...],
        symbols: tuple[str, ...],
    ) -> torch.Tensor:
        specs = _normalize_constraint_specs(constraint)
        compiled = _compile_lc_to_emission_specs(constraint, state_names, symbols, device=mask.device, dtype=mask.dtype)
        if compiled is not None:
            specs.extend(compiled)
        if not specs:
            return mask

        state_index = {name: idx for idx, name in enumerate(state_names)}
        symbol_index = {name: idx for idx, name in enumerate(symbols)}
        for spec in specs:
            if isinstance(spec, EmissionMaskSpec):
                self.report.add_applied(_spec_message("applied emission mask", spec))
                mask = mask * validate_mask(spec.mask, tuple(mask.shape), name="emission_mask", device=mask.device, dtype=mask.dtype)
            elif isinstance(spec, AllowedEmissionsSpec):
                new_mask = torch.zeros_like(mask)
                for state, symbol in _as_pairs(spec.emissions):
                    state_name = str(state)
                    symbol_name = str(symbol)
                    if state_name in state_index and symbol_name in symbol_index:
                        new_mask[state_index[state_name], symbol_index[symbol_name]] = 1.0
                    else:
                        self.report.add_unsupported(f"allowed emission {state!r}->{symbol!r} does not map to HMM states/symbols")
                self.report.add_applied(_spec_message("applied allowed emission set", spec))
                mask = mask * new_mask
            elif isinstance(spec, ForbiddenEmissionsSpec):
                for state, symbol in _as_pairs(spec.emissions):
                    state_name = str(state)
                    symbol_name = str(symbol)
                    if state_name in state_index and symbol_name in symbol_index:
                        mask[state_index[state_name], symbol_index[symbol_name]] = 0.0
                    else:
                        self.report.add_unsupported(f"forbidden emission {state!r}->{symbol!r} does not map to HMM states/symbols")
                self.report.add_applied(_spec_message("applied forbidden emission set", spec))
            elif isinstance(spec, (TransitionMaskSpec, AllowedTransitionsSpec, ForbiddenTransitionsSpec, StatePredicateTransitionSpec, ConstraintDFAExportSpec)):
                continue
            else:
                self.report.add_unsupported(f"unsupported HMM emission spec {spec!r}")
        return mask

def _relation_name(rel: Any) -> str:
    pair = _relation_pair(rel)
    if pair is not None:
        return f"{pair[0]}->{pair[1]}"
    return _name(rel)


def _relation_pair(rel: Any) -> tuple[str, str] | None:
    if isinstance(rel, (tuple, list)) and len(rel) >= 2:
        return _name(rel[0]), _name(rel[1])
    src = getattr(rel, "src", None)
    dst = getattr(rel, "dst", None)
    if src is not None and dst is not None:
        return _name(src), _name(dst)
    return None


def _normalize_constraint_specs(constraint: Any) -> list[Any]:
    typed = (
        TransitionMaskSpec,
        EmissionMaskSpec,
        AllowedTransitionsSpec,
        ForbiddenTransitionsSpec,
        AllowedEmissionsSpec,
        ForbiddenEmissionsSpec,
        StatePredicateTransitionSpec,
        ConstraintDFAExportSpec,
    )
    if isinstance(constraint, typed):
        return [constraint]
    mapping = _constraint_mapping(constraint)
    if mapping is None:
        return []
    specs: list[Any] = []
    if "transition_mask" in mapping:
        specs.append(TransitionMaskSpec(mapping["transition_mask"]))
    if "emission_mask" in mapping:
        specs.append(EmissionMaskSpec(mapping["emission_mask"]))
    if "allowed_transitions" in mapping:
        specs.append(AllowedTransitionsSpec(tuple(_as_pairs(mapping["allowed_transitions"]))))
    transitions: list[tuple[Any, Any]] = []
    for key in ("forbidden_transitions", "forbid_transitions"):
        transitions.extend(_as_pairs(mapping.get(key)))
    if transitions:
        specs.append(ForbiddenTransitionsSpec(tuple(transitions)))
    if "allowed_emissions" in mapping:
        specs.append(AllowedEmissionsSpec(tuple(_as_pairs(mapping["allowed_emissions"]))))
    emissions: list[tuple[Any, Any]] = []
    for key in ("forbidden_emissions", "forbid_emissions"):
        emissions.extend(_as_pairs(mapping.get(key)))
    if emissions:
        specs.append(ForbiddenEmissionsSpec(tuple(emissions)))
    if "constraint_dfa" in mapping or "dfa" in mapping:
        specs.append(ConstraintDFAExportSpec(mapping.get("constraint_dfa", mapping.get("dfa"))))
    return specs


def _compile_lc_to_transition_specs(
    constraint: Any,
    state_names: tuple[str, ...],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> list[Any] | None:
    if not _looks_like_logical_constraint(constraint):
        return None
    mask = _compile_lc_to_transition_mask(constraint, set(state_names))
    if mask is None:
        return None
    return [TransitionMaskSpec(_mask_from_allowed(mask, state_names, state_names, device=device, dtype=dtype), name=f"lc:{_lc_name(constraint)}")]


def _compile_lc_to_emission_specs(
    constraint: Any,
    state_names: tuple[str, ...],
    symbols: tuple[str, ...],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> list[Any] | None:
    if not symbols or not _looks_like_logical_constraint(constraint):
        return None
    mask = _compile_lc_to_emission_mask(constraint, set(state_names), set(symbols))
    if mask is None:
        return None
    return [EmissionMaskSpec(_mask_from_allowed(mask, state_names, symbols, device=device, dtype=dtype), name=f"lc:{_lc_name(constraint)}")]


def _compile_lc_to_transition_mask(constraint: Any, state_universe: set[str]) -> set[tuple[str, str]] | None:
    name = _lc_name(constraint)
    children = _logical_children(constraint)
    if name == "ifL" and len(children) >= 2:
        return _compile_implication_mask(constraint, state_universe, state_universe)
    if name == "andL":
        child_masks = [_compile_lc_to_transition_mask(child, state_universe) for child in children]
        if any(child_mask is None for child_mask in child_masks):
            return None
        iterator = iter(child_masks)
        first = next(iterator, None)
        if first is None:
            return None
        allowed = set(first)
        for child_mask in iterator:
            allowed &= set(child_mask)
        return allowed
    if name == "orL":
        child_masks = [_compile_lc_to_transition_mask(child, state_universe) for child in children]
        if any(child_mask is None for child_mask in child_masks):
            return None
        allowed: set[tuple[str, str]] = set()
        for child_mask in child_masks:
            allowed |= set(child_mask)
        return allowed
    return None


def _compile_lc_to_emission_mask(constraint: Any, state_universe: set[str], symbol_universe: set[str]) -> set[tuple[str, str]] | None:
    name = _lc_name(constraint)
    children = _logical_children(constraint)
    if name == "ifL" and len(children) >= 2:
        return _compile_implication_mask(constraint, state_universe, symbol_universe)
    if name == "andL":
        child_masks = [_compile_lc_to_emission_mask(child, state_universe, symbol_universe) for child in children]
        if any(child_mask is None for child_mask in child_masks):
            return None
        iterator = iter(child_masks)
        first = next(iterator, None)
        if first is None:
            return None
        allowed = set(first)
        for child_mask in iterator:
            allowed &= set(child_mask)
        return allowed
    if name == "orL":
        child_masks = [_compile_lc_to_emission_mask(child, state_universe, symbol_universe) for child in children]
        if any(child_mask is None for child_mask in child_masks):
            return None
        allowed: set[tuple[str, str]] = set()
        for child_mask in child_masks:
            allowed |= set(child_mask)
        return allowed
    return None


def _compile_implication_mask(constraint: Any, source_universe: set[str], dest_universe: set[str]) -> set[tuple[str, str]] | None:
    if _lc_name(constraint) != "ifL":
        return None
    children = _logical_children(constraint)
    if len(children) < 2:
        return None
    source_set = _name_set_from_formula(children[0], source_universe)
    dest_set = _name_set_from_formula(_formula_from_children(children[1:]), dest_universe)
    if source_set is None or dest_set is None:
        return None
    allowed: set[tuple[str, str]] = set()
    for src in source_universe:
        destinations = dest_set if src in source_set else dest_universe
        for dst in destinations:
            allowed.add((src, dst))
    return allowed


def _formula_from_children(children: list[Any]) -> Any:
    if len(children) == 1:
        return children[0]
    return ("__or__", tuple(children))


def _name_set_from_formula(expr: Any, universe: set[str]) -> set[str] | None:
    if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == "__or__":
        result: set[str] = set()
        for child in expr[1]:
            child_set = _name_set_from_formula(child, universe)
            if child_set is None:
                return None
            result |= child_set
        return result
    atom_name = _atom_name(expr)
    if atom_name is not None:
        return {atom_name} if atom_name in universe else None
    if not _looks_like_logical_constraint(expr):
        return None
    name = _lc_name(expr)
    children = _logical_children(expr)
    if name == "notL" and len(children) == 1:
        child_set = _name_set_from_formula(children[0], universe)
        return None if child_set is None else set(universe) - child_set
    if name == "andL":
        child_sets = [_name_set_from_formula(child, universe) for child in children]
        if any(child_set is None for child_set in child_sets):
            return None
        iterator = iter(child_sets)
        first = next(iterator, None)
        if first is None:
            return None
        result = set(first)
        for child_set in iterator:
            result &= set(child_set)
        return result
    if name == "orL":
        result: set[str] = set()
        for child in children:
            child_set = _name_set_from_formula(child, universe)
            if child_set is None:
                return None
            result |= child_set
        return result
    return None


def _mask_from_allowed(
    allowed: set[tuple[str, str]],
    rows: tuple[str, ...],
    columns: tuple[str, ...],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    row_index = {name: idx for idx, name in enumerate(rows)}
    column_index = {name: idx for idx, name in enumerate(columns)}
    mask = torch.zeros((len(rows), len(columns)), dtype=dtype, device=device)
    for row, column in allowed:
        if row in row_index and column in column_index:
            mask[row_index[row], column_index[column]] = 1.0
    return mask


def _logical_children(constraint: Any) -> list[Any]:
    children: list[Any] = []
    for item in getattr(constraint, "e", ()):
        if _is_variable_ref(item):
            continue
        children.append(item)
    return children


def _is_variable_ref(item: Any) -> bool:
    return item.__class__.__name__ == "V" or hasattr(item, "relVarInfo")


def _looks_like_logical_constraint(value: Any) -> bool:
    return hasattr(value, "e") and value.__class__.__name__.endswith("L")


def _lc_name(value: Any) -> str:
    return value.__class__.__name__


def _atom_name(value: Any) -> str | None:
    if isinstance(value, tuple) and len(value) >= 2:
        return str(value[1])
    if isinstance(value, list) and value:
        names = [_atom_name(item) for item in value if not _is_variable_ref(item)]
        names = [name for name in names if name is not None]
        if len(names) == 1:
            return names[0]
    return None


def _spec_message(prefix: str, spec: Any) -> str:
    name = getattr(spec, "name", None)
    return f"{prefix}: {name}" if name else prefix


def _constraint_mapping(constraint: Any) -> dict[str, Any] | None:
    if isinstance(constraint, dict):
        result = dict(constraint)
    else:
        keys = (
            "transition_mask",
            "emission_mask",
            "allowed_transitions",
            "forbidden_transitions",
            "forbid_transitions",
            "allowed_emissions",
            "forbidden_emissions",
            "forbid_emissions",
            "constraint_dfa",
            "dfa",
        )
        result = {key: getattr(constraint, key) for key in keys if hasattr(constraint, key)}
    _append_pair_value(result, "forbidden_transitions", "forbid_transition")
    _append_pair_value(result, "forbidden_transitions", "forbidden_transition")
    _append_pair_value(result, "allowed_transitions", "allow_transition")
    _append_pair_value(result, "allowed_transitions", "allowed_transition")
    _append_pair_value(result, "forbidden_emissions", "forbid_emission")
    _append_pair_value(result, "forbidden_emissions", "forbidden_emission")
    _append_pair_value(result, "allowed_emissions", "allow_emission")
    _append_pair_value(result, "allowed_emissions", "allowed_emission")
    return result or None


def _append_pair_value(mapping: dict[str, Any], target_key: str, source_key: str) -> None:
    if source_key not in mapping:
        return
    existing = list(_as_pairs(mapping.get(target_key, ())))
    existing.extend(_as_pairs(mapping[source_key]))
    mapping[target_key] = existing


def _as_pairs(value: Any) -> list[tuple[Any, Any]]:
    if value is None:
        return []
    if isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], (tuple, list)):
        return [value]
    if isinstance(value, list) and len(value) == 2 and not isinstance(value[0], (tuple, list)):
        return [tuple(value)]
    pairs: list[tuple[Any, Any]] = []
    for item in value:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            pairs.append((item[0], item[1]))
        else:
            raise ValueError(f"expected pair constraints, got {item!r}")
    return pairs
