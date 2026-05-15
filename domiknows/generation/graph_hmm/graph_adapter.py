"""Bridge DomiKnowS graph declarations to static HMM-compatible masks.

This module translates selected graph concepts, relation edges, explicit
constraint specs, and a conservative subset of logical constraints into
transition/emission masks used by graph-aware HMM components.

The adapter is intentionally conservative: if a constraint cannot be mapped
exactly to local mask structure, it is reported as unsupported rather than
silently approximated.
Logical constraints that can be compiled (conservative subset):

    ifL/forAllL implications
    andL/orL/notL/nandL/norL/xorL/iffL/equivalenceL combinations
    local existsL/atLeastL/atMostL/exactL over supported formula children
    atMostAL(..., 0) / exactAL(..., 0) as global forbiddance masks
    relation-path endpoint hints such as path=("x", "rel", "y")
Atom names must map to known state names (and symbol names for emission typing).
Accumulated multi-token constraints that are not static masks are registered as
``ConstraintDFAExportSpec`` diagnostics instead of being approximated.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Atom:
    """Normalized concept predicate with variable/path binding metadata."""

    name: str
    variable: str | None = None
    path: tuple[Any, ...] | None = None


def _name(obj: Any) -> str:
    """Return a stable readable name for graph/spec objects."""
    return str(getattr(obj, "name", obj))


def _dedupe_names(items: Iterable[Any]) -> list[str]:
    """Deduplicate objects by normalized name while preserving order."""
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = _name(item)
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _domain_concept_names(items: Iterable[Any]) -> list[str]:
    """Filter out synthetic/internal concept entries from concept names."""
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
        """Return candidate concept names usable as HMM state names."""
        if self._explicit_concepts is not None:
            return _dedupe_names(self._explicit_concepts)
        graph_concepts = getattr(self.graph, "concepts", {}) if self.graph is not None else {}
        if isinstance(graph_concepts, dict):
            return _domain_concept_names(graph_concepts.keys())
        return _domain_concept_names(graph_concepts)

    def relations(self) -> list[str]:
        """Return normalized relation names for diagnostics/reporting."""
        return [_relation_name(rel) for rel in self._relation_objects()]

    def constraints(self) -> list[Any]:
        """Return active top-level constraints from explicit input or graph."""
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
        """Update emission symbol universe after learner vocabulary is known."""
        self.symbols = tuple(symbols)

    def allowed_transition_mask(self) -> torch.Tensor:
        """Compile transition compatibility mask from relations and constraints."""
        state_names = self._state_names()
        shape = (len(state_names), len(state_names))
        mask = torch.ones(shape, dtype=self.dtype, device=self.device)

        relation_pairs = self._mapped_relation_pairs(state_names)
        if relation_pairs:
            # If relation endpoints map to states, treat them as explicit allowed
            # transitions instead of unconstrained all-to-all connectivity.
            mask = torch.zeros(shape, dtype=self.dtype, device=self.device)
            for src, dst in relation_pairs:
                mask[src, dst] = 1.0
            self.report.add_applied(f"compiled {len(relation_pairs)} graph relation transition(s)")

        for constraint in self.constraints():
            mask = self._apply_transition_constraint(mask, constraint, state_names)
        return mask

    def emission_type_mask(self) -> torch.Tensor:
        """Compile emission compatibility mask from explicit/logical specs."""
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
        """Resolve final state-name list from explicit names/space/graph concepts."""
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
        """Collect relation objects from explicit config or graph concept edges."""
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
        """Map relation endpoints to state-id pairs when names align."""
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
        """Apply one constraint object/spec to the transition mask."""
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
                # Multiplicative composition preserves earlier constraints.
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
                # Evaluate predicate in factorized space, then project to flat ids.
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
        """Apply one constraint object/spec to the emission mask."""
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
    """Human-readable relation name, preferring ``src->dst`` when available."""
    pair = _relation_pair(rel)
    if pair is not None:
        return f"{pair[0]}->{pair[1]}"
    return _name(rel)


def _relation_pair(rel: Any) -> tuple[str, str] | None:
    """Extract a ``(src, dst)`` relation pair from tuple/list/object forms."""
    if isinstance(rel, (tuple, list)) and len(rel) >= 2:
        return _name(rel[0]), _name(rel[1])
    src = getattr(rel, "src", None)
    dst = getattr(rel, "dst", None)
    if src is not None and dst is not None:
        return _name(src), _name(dst)
    return None


def _normalize_constraint_specs(constraint: Any) -> list[Any]:
    """Normalize typed/object/dict constraints into typed spec instances."""
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
    """Compile supported logical-constraint fragments into transition specs."""
    if not _looks_like_logical_constraint(constraint):
        return None
    mask = _compile_lc_to_transition_mask(constraint, set(state_names))
    if mask is None:
        dfa_specs = _compile_lc_to_dfa_export_specs(constraint)
        return dfa_specs or None
    return [TransitionMaskSpec(_mask_from_allowed(mask, state_names, state_names, device=device, dtype=dtype), name=f"lc:{_lc_name(constraint)}")]


def _compile_lc_to_emission_specs(
    constraint: Any,
    state_names: tuple[str, ...],
    symbols: tuple[str, ...],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> list[Any] | None:
    """Compile supported logical-constraint fragments into emission specs."""
    if not symbols or not _looks_like_logical_constraint(constraint):
        return None
    mask = _compile_lc_to_emission_mask(constraint, set(state_names), set(symbols))
    if mask is None:
        return None
    return [EmissionMaskSpec(_mask_from_allowed(mask, state_names, symbols, device=device, dtype=dtype), name=f"lc:{_lc_name(constraint)}")]


def _compile_lc_to_transition_mask(constraint: Any, state_universe: set[str]) -> set[tuple[str, str]] | None:
    """Compile a limited boolean LC fragment into allowed transition pairs."""
    if not _is_supported_mask_root(constraint):
        return None
    if _is_accumulated_zero_forbid(constraint):
        atoms = _atoms_in_expr(_first_formula_operand(constraint))
        if not atoms or any(atom.name not in state_universe for atom in atoms):
            return None
        forbidden = {atom.name for atom in atoms}
        return {
            (src, dst)
            for src in state_universe
            for dst in state_universe
            if src not in forbidden and dst not in forbidden
        }
    allowed: set[tuple[str, str]] = set()
    saw_supported = False
    for src in state_universe:
        for dst in state_universe:
            value = _eval_transition_formula(constraint, src, dst, state_universe)
            if value is None:
                return None
            saw_supported = True
            if value:
                allowed.add((src, dst))
    return allowed if saw_supported else None


def _compile_lc_to_emission_mask(constraint: Any, state_universe: set[str], symbol_universe: set[str]) -> set[tuple[str, str]] | None:
    """Compile a limited boolean LC fragment into allowed emission pairs."""
    if not _is_supported_mask_root(constraint):
        return None
    if _is_accumulated_zero_forbid(constraint):
        atoms = _atoms_in_expr(_first_formula_operand(constraint))
        if not atoms:
            return None
        forbidden_states = {atom.name for atom in atoms if atom.name in state_universe}
        forbidden_symbols = {atom.name for atom in atoms if atom.name in symbol_universe}
        if not forbidden_states and not forbidden_symbols:
            return None
        return {
            (state, symbol)
            for state in state_universe
            for symbol in symbol_universe
            if state not in forbidden_states and symbol not in forbidden_symbols
        }
    allowed: set[tuple[str, str]] = set()
    saw_supported = False
    for state in state_universe:
        for symbol in symbol_universe:
            value = _eval_emission_formula(constraint, state, symbol, state_universe, symbol_universe)
            if value is None:
                return None
            saw_supported = True
            if value:
                allowed.add((state, symbol))
    return allowed if saw_supported else None


def _is_supported_mask_root(expr: Any) -> bool:
    """Return true for LC roots that form complete local/static mask rules."""

    if not _looks_like_logical_constraint(expr):
        return False
    name = _lc_name(expr)
    return name in {
        "ifL",
        "forAllL",
        "andL",
        "orL",
        "nandL",
        "norL",
        "xorL",
        "equivalenceL",
        "iffL",
        "existsL",
        "atLeastL",
        "atMostL",
        "exactL",
        "atMostAL",
        "exactAL",
    }


def _eval_transition_formula(expr: Any, src: str, dst: str, universe: set[str]) -> bool | None:
    """Evaluate a supported LC formula against one transition pair."""

    atom = _atom_from_expr(expr)
    if atom is not None:
        if not _atom_has_supported_local_path(atom):
            return None
        if atom.name not in universe:
            return None
        role = _atom_transition_role(atom)
        if role == "src":
            return src == atom.name
        if role == "dst":
            return dst == atom.name
        return src == atom.name or dst == atom.name
    return _eval_boolean_formula(
        expr,
        lambda child: _eval_transition_formula(child, src, dst, universe),
    )


def _eval_emission_formula(
    expr: Any,
    state: str,
    symbol: str,
    state_universe: set[str],
    symbol_universe: set[str],
) -> bool | None:
    """Evaluate a supported LC formula against one state/symbol emission pair."""

    atom = _atom_from_expr(expr)
    if atom is not None:
        if not _atom_has_supported_local_path(atom):
            return None
        if atom.name in symbol_universe and atom.name not in state_universe:
            return symbol == atom.name
        if atom.name in state_universe and atom.name not in symbol_universe:
            return state == atom.name
        if atom.name in state_universe and atom.name in symbol_universe:
            role = _atom_transition_role(atom)
            return symbol == atom.name if role == "dst" else state == atom.name
        return None
    return _eval_boolean_formula(
        expr,
        lambda child: _eval_emission_formula(child, state, symbol, state_universe, symbol_universe),
    )


def _eval_boolean_formula(expr: Any, child_eval) -> bool | None:
    """Evaluate supported boolean/count logical operators using child_eval."""

    if not _looks_like_logical_constraint(expr):
        return None
    name = _lc_name(expr)
    operands = _formula_operands(expr)
    if name == "notL" and len(operands) == 1:
        value = child_eval(operands[0])
        return None if value is None else not value
    if name == "andL":
        values = [child_eval(child) for child in operands]
        return None if any(value is None for value in values) else all(values)
    if name == "orL":
        values = [child_eval(child) for child in operands]
        return None if any(value is None for value in values) else any(values)
    if name == "nandL":
        value = _eval_synthetic("andL", operands, child_eval)
        return None if value is None else not value
    if name == "norL":
        value = _eval_synthetic("orL", operands, child_eval)
        return None if value is None else not value
    if name == "xorL":
        values = [child_eval(child) for child in operands]
        return None if any(value is None for value in values) else sum(bool(value) for value in values) == 1
    if name in {"equivalenceL", "iffL"}:
        values = [child_eval(child) for child in operands]
        return None if any(value is None for value in values) or not values else all(value == values[0] for value in values)
    if name in {"ifL", "forAllL"} and len(operands) >= 2:
        premise = child_eval(operands[0])
        consequents = [child_eval(child) for child in operands[1:]]
        if premise is None or any(value is None for value in consequents):
            return None
        return (not premise) or all(consequents)
    if name in {"existsL", "atLeastL", "atMostL", "exactL"}:
        values = [child_eval(child) for child in operands]
        if any(value is None for value in values):
            return None
        count = sum(bool(value) for value in values)
        limit = _count_limit(expr)
        if limit is None:
            return None
        if name == "existsL":
            return count >= 1
        if name == "atLeastL":
            return count >= limit
        if name == "atMostL":
            return count <= limit
        return count == limit
    if name in {"existsAL", "atLeastAL", "atMostAL", "exactAL"}:
        # Accumulated/global counts are not local pair/emission formulas except
        # for the zero-forbiddance special case handled before evaluation.
        return None
    return None


def _compile_lc_to_dfa_export_specs(constraint: Any) -> list[Any] | None:
    """Register non-local regular/window-like LCs for external DFA handling."""

    if not _looks_like_logical_constraint(constraint):
        return None
    name = _lc_name(constraint)
    if name in {"existsAL", "atLeastAL", "atMostAL", "exactAL"}:
        return [
            ConstraintDFAExportSpec(
                name=f"lc:{name}",
                description=(
                    f"{name} is an accumulated/multi-token constraint; "
                    "graph_hmm registered it for DFA/export diagnostics rather "
                    "than approximating it as a local HMM matrix"
                ),
            )
        ]
    atoms = _atoms_in_expr(constraint)
    if any(atom.path and len(atom.path) > 3 for atom in atoms):
        return [
            ConstraintDFAExportSpec(
                name=f"lc:{name}:path",
                description=(
                    "multi-hop relation path/window constraint requires a "
                    "finite-state DFA or dynamic monitor outside static HMM masks"
                ),
            )
        ]
    return None


def _eval_synthetic(name: str, operands: list[Any], child_eval) -> bool | None:
    """Evaluate a simple synthetic boolean form for derived operators."""

    values = [child_eval(child) for child in operands]
    if any(value is None for value in values):
        return None
    if name == "andL":
        return all(values)
    if name == "orL":
        return any(values)
    return None


def _formula_operands(expr: Any) -> list[Any]:
    """Return LC operands with atom tuples bound to their following V node."""

    if not _looks_like_logical_constraint(expr):
        return []
    return _group_atom_operands(getattr(expr, "e", ()), skip_count_limits=True)


def _group_atom_operands(items: Iterable[Any], *, skip_count_limits: bool) -> list[Any]:
    """Group flattened ``(concept tuple, V)`` pairs into _Atom objects."""

    values = list(items)
    result: list[Any] = []
    index = 0
    while index < len(values):
        item = values[index]
        if skip_count_limits and isinstance(item, int):
            index += 1
            continue
        if isinstance(item, list):
            atom = _atom_from_sequence(item)
            if atom is not None:
                result.append(atom)
            else:
                result.extend(_group_atom_operands(item, skip_count_limits=skip_count_limits))
            index += 1
            continue
        if _is_atom_tuple(item):
            variable = values[index + 1] if index + 1 < len(values) and _is_variable_ref(values[index + 1]) else None
            result.append(_atom_from_parts(item, variable))
            index += 2 if variable is not None else 1
            continue
        if _is_variable_ref(item):
            index += 1
            continue
        result.append(item)
        index += 1
    return result


def _atom_from_expr(expr: Any) -> _Atom | None:
    """Return a normalized atom if expr is a concept predicate."""

    if isinstance(expr, _Atom):
        return expr
    if isinstance(expr, list):
        return _atom_from_sequence(expr)
    if _is_atom_tuple(expr):
        return _atom_from_parts(expr, None)
    return None


def _atom_from_sequence(values: list[Any]) -> _Atom | None:
    """Normalize list-style concept call output to an atom."""

    for index, item in enumerate(values):
        if _is_atom_tuple(item):
            variable = values[index + 1] if index + 1 < len(values) and _is_variable_ref(values[index + 1]) else None
            return _atom_from_parts(item, variable)
    return None


def _atom_from_parts(atom_tuple: Any, variable: Any | None) -> _Atom:
    """Build an _Atom from the flattened DomiKnowS concept tuple and V."""

    return _Atom(
        name=str(atom_tuple[1]),
        variable=None if variable is None else getattr(variable, "name", None),
        path=None if variable is None or getattr(variable, "v", None) is None else tuple(getattr(variable, "v")),
    )


def _atoms_in_expr(expr: Any) -> list[_Atom]:
    """Collect normalized atoms from an expression tree."""

    atom = _atom_from_expr(expr)
    if atom is not None:
        return [atom]
    if not _looks_like_logical_constraint(expr):
        return []
    atoms: list[_Atom] = []
    for child in _formula_operands(expr):
        atoms.extend(_atoms_in_expr(child))
    return atoms


def _atom_transition_role(atom: _Atom) -> str | None:
    """Infer whether an atom is bound to source/current or destination/next."""

    path_role = _path_endpoint_role(atom.path)
    if path_role is not None:
        return path_role
    variable = str(atom.variable) if atom.variable is not None else ""
    if variable in {"y", "dst", "dest", "to", "next", "second", "t1", "t_next"}:
        return "dst"
    if variable in {"x", "src", "source", "from", "current", "first", "t", "t0"}:
        return "src"
    return None


def _atom_has_supported_local_path(atom: _Atom) -> bool:
    """Keep path support to one relation/window hop for static masks."""

    return atom.path is None or len(atom.path) <= 3


def _path_endpoint_role(path: tuple[Any, ...] | None) -> str | None:
    """Infer source/destination role from a DomiKnowS path tuple."""

    if not path:
        return None
    path_values = [str(getattr(item, "name", item)) for item in path]
    if path_values[-1] in {"y", "dst", "dest", "to", "next", "second", "t1", "t_next"}:
        return "dst"
    if path_values[-1] in {"x", "src", "source", "from", "current", "first", "t", "t0"}:
        return "src"
    if path_values[0] in {"y", "dst", "dest", "to", "next", "second"}:
        return "dst"
    if path_values[0] in {"x", "src", "source", "from", "current", "first"}:
        return "src"
    return None


def _first_formula_operand(expr: Any) -> Any | None:
    operands = _formula_operands(expr)
    return operands[0] if operands else None


def _is_accumulated_zero_forbid(expr: Any) -> bool:
    """Return true for accumulated count forms that statically forbid atoms."""

    if not _looks_like_logical_constraint(expr):
        return False
    name = _lc_name(expr)
    limit = _count_limit(expr)
    return name in {"atMostAL", "exactAL"} and limit == 0


def _count_limit(expr: Any) -> int | None:
    """Extract explicit/trailing count limits from DomiKnowS count constraints."""

    fixed = getattr(expr, "fixedLimit", None)
    if fixed is not None:
        return int(fixed)
    explicit = getattr(expr, "_explicitLimit", None)
    if explicit is not None:
        return int(explicit)
    for item in reversed(getattr(expr, "e", ())):
        if isinstance(item, int):
            return int(item)
    return 1


def _mask_from_allowed(
    allowed: set[tuple[str, str]],
    rows: tuple[str, ...],
    columns: tuple[str, ...],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert allowed pair set to dense 0/1 mask with fixed row/column order."""
    row_index = {name: idx for idx, name in enumerate(rows)}
    column_index = {name: idx for idx, name in enumerate(columns)}
    mask = torch.zeros((len(rows), len(columns)), dtype=dtype, device=device)
    for row, column in allowed:
        if row in row_index and column in column_index:
            mask[row_index[row], column_index[column]] = 1.0
    return mask


def _logical_children(constraint: Any) -> list[Any]:
    """Return logical children while skipping variable reference placeholders."""
    children: list[Any] = []
    for item in getattr(constraint, "e", ()):
        if _is_variable_ref(item):
            continue
        children.append(item)
    return children


def _is_atom_tuple(item: Any) -> bool:
    """Return true for flattened DomiKnowS concept predicate tuples."""

    return isinstance(item, tuple) and len(item) >= 2 and not isinstance(item[0], str)


def _is_variable_ref(item: Any) -> bool:
    """Heuristic check for DomiKnowS logical variable reference nodes."""
    return item.__class__.__name__ == "V" or hasattr(item, "relVarInfo")


def _looks_like_logical_constraint(value: Any) -> bool:
    """Best-effort check for logical-constraint-like objects."""
    return hasattr(value, "e") and value.__class__.__name__.endswith("L")


def _lc_name(value: Any) -> str:
    """Return the logical constraint class name."""
    return value.__class__.__name__


def _atom_name(value: Any) -> str | None:
    """Extract atom/concept-like names from parsed logical expressions."""
    if isinstance(value, tuple) and len(value) >= 2:
        return str(value[1])
    if isinstance(value, list) and value:
        names = [_atom_name(item) for item in value if not _is_variable_ref(item)]
        names = [name for name in names if name is not None]
        if len(names) == 1:
            return names[0]
    return None


def _spec_message(prefix: str, spec: Any) -> str:
    """Attach optional spec name to report messages."""
    name = getattr(spec, "name", None)
    return f"{prefix}: {name}" if name else prefix


def _constraint_mapping(constraint: Any) -> dict[str, Any] | None:
    """Read legacy dict/object constraint fields into a mapping."""
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
    """Normalize singular pair aliases into plural pair-list fields."""
    if source_key not in mapping:
        return
    existing = list(_as_pairs(mapping.get(target_key, ())))
    existing.extend(_as_pairs(mapping[source_key]))
    mapping[target_key] = existing


def _as_pairs(value: Any) -> list[tuple[Any, Any]]:
    """Normalize one pair or many pairs to a list of 2-tuples."""
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
