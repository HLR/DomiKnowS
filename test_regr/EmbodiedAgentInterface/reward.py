from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Set, Tuple

import torch

try:
    from dataset import ACTION_VOCAB, EOS_TOKEN
except ImportError:
    from .dataset import ACTION_VOCAB, EOS_TOKEN
from domiknows.generation.dfa.vocabulary import TokenVocabulary
from domiknows.reinforcement.rewards import flatten_generator_output

try:
    from world_graph import (
        canonical_state_name,
        is_goal_action,
        is_known_action,
        is_state_predicate,
        materialize_world_trajectory,
        positive_state_name,
        verify_world_constraints,
    )
except ImportError:
    from .world_graph import (
        canonical_state_name,
        is_goal_action,
        is_known_action,
        is_state_predicate,
        materialize_world_trajectory,
        positive_state_name,
        verify_world_constraints,
    )


Fact = Tuple[str, ...]
_ACTIVE_WORLD_BUNDLE: ContextVar[Any] = ContextVar("eai_active_world_bundle", default=None)


@contextmanager
def _using_world_bundle(world_bundle: Any):
    token = _ACTIVE_WORLD_BUNDLE.set(world_bundle)
    try:
        yield
    finally:
        _ACTIVE_WORLD_BUNDLE.reset(token)

def _normalize_name(value: Any) -> str:
    value = str(value or "").strip().lower().replace(".", "_").replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", value).strip("_")


def _canonical_predicate(value: Any) -> str:
    name = _normalize_name(value)
    return canonical_state_name(name, _ACTIVE_WORLD_BUNDLE.get())


def _negative_predicate(predicate: str) -> str:
    predicate = _canonical_predicate(predicate)
    if predicate == "closed":
        return "not_closed"
    if predicate == "off":
        return "on"
    if predicate.startswith("not_"):
        return predicate[4:]
    return f"not_{predicate}"


def _entity_parts(entity: str) -> tuple[str, str | None]:
    entity = _normalize_name(entity)
    match = re.match(r"^(.*?)(?:_(\d+))$", entity)
    return (match.group(1), match.group(2)) if match else (entity, None)


def _entities_match(required: str, actual: str) -> bool:
    """Match exactly, allowing only an omitted instance id on one side."""
    required = _normalize_name(required)
    actual = _normalize_name(actual)
    if required == actual:
        return True
    required_base, required_id = _entity_parts(required)
    actual_base, actual_id = _entity_parts(actual)
    if {required_base, actual_base} == {"agent", "character"}:
        return required_id is None or actual_id is None or required_id == actual_id
    return required_base == actual_base and (required_id is None or actual_id is None)


def _fact_matches(required: Fact, actual: Fact) -> bool:
    return (
        len(required) == len(actual)
        and required[0] == actual[0]
        and all(_entities_match(r, a) for r, a in zip(required[1:], actual[1:]))
    )


def _fact_present(required: Fact, state: Set[Fact]) -> bool:
    return any(_fact_matches(required, actual) for actual in state)


def _side_for_action(action: str) -> str | None:
    if action.startswith("left_"):
        return "left"
    if action.startswith("right_"):
        return "right"
    return None


def _placement_relation(action: str) -> str | None:
    if "inside" in action or action in {"put", "putin", "put_inside"}:
        return "inside"
    # VirtualHome PUTBACK places an object ON the destination; PUTIN contains it.
    if "ontop" in action or "on_top" in action or action in {
        "putback", "puton", "putontop", "put_ontop",
    }:
        return "ontop"
    if "nextto" in action or "next_to" in action:
        return "nextto"
    if "under" in action:
        return "under"
    return None


def tokens_from_labels(labels: Sequence[int | str | torch.Tensor], vocabulary: Any = None) -> list[str]:
    """Convert label IDs or strings to normalized EAI tokens, stopping at EOS."""
    tokens: list[str] = []
    for label in labels:
        if isinstance(label, str):
            token = label
        elif torch.is_tensor(label):
            idx = int(label.detach().cpu().reshape(-1)[0].item())
            token = vocabulary.token_for_label(idx) if vocabulary is not None and 0 <= idx < vocabulary.label_count else (ACTION_VOCAB[idx] if 0 <= idx < len(ACTION_VOCAB) else "other")
        else:
            idx = int(label)
            token = vocabulary.token_for_label(idx) if vocabulary is not None and 0 <= idx < vocabulary.label_count else (ACTION_VOCAB[idx] if 0 <= idx < len(ACTION_VOCAB) else "other")
        token = _normalize_name(token) if token != EOS_TOKEN else EOS_TOKEN
        if token == getattr(vocabulary, "other_token", None):
            token = "other"
        tokens.append(token)
        eos_token = vocabulary.eos_token if vocabulary is not None else EOS_TOKEN
        if token == eos_token:
            break
    return tokens


@dataclass(frozen=True)
class _ActionEvent:
    name: str
    args: tuple[str, ...] = ()


def _action_events_from_tokens(tokens: Sequence[str]) -> list[_ActionEvent]:
    """Decode variable-arity actions without shifting after STANDUP/SLEEP."""
    events: list[_ActionEvent] = []
    index = 0
    while index < len(tokens):
        action = _normalize_name(tokens[index])
        index += 1
        if not action or action == _normalize_name(EOS_TOKEN):
            break
        if not is_known_action(action, _ACTIVE_WORLD_BUNDLE.get()):
            continue
        args: tuple[str, ...] = ()
        if index < len(tokens):
            following = _normalize_name(tokens[index])
            if following and following != _normalize_name(EOS_TOKEN) and not is_known_action(following, _ACTIVE_WORLD_BUNDLE.get()):
                args = (following,)
                index += 1
        events.append(_ActionEvent(action, args))
    return events


def _set_fact(state: Set[Fact], fact: Fact, *remove: Fact) -> None:
    for old in remove:
        state.discard(old)
    state.add(fact)


# The benchmark supplies flat action tokens rather than a scene graph. This
# deliberately small transition model captures only effects relevant to goals.
def _simulate_events(
    events: Sequence[_ActionEvent],
    initial_state: Iterable[Fact] = (),
    goal_facts: Iterable[Fact] = (),
) -> tuple[list[Set[Fact]], list[_ActionEvent]]:
    state: Set[Fact] = set(initial_state)
    # Snapshot zero precedes the first action; later snapshots hold each action's
    # effects for final-state recall and ordered temporal evaluation.
    states: list[Set[Fact]] = [set(state)]
    held: dict[str, str | None] = {"left": None, "right": None}
    negative_spatial_goals = {
        fact for fact in goal_facts
        if fact and fact[0] in {"not_inside", "not_nextto", "not_ontop", "not_under"}
    }

    for event in events:
        action = event.name
        obj = event.args[0] if event.args else None
        side = _side_for_action(action)

        if any(part in action for part in ("grasp", "grab", "pickup", "take")) and obj:
            hand = side or ("right" if held["right"] is None else "left")
            held[hand] = obj
            hold_pred = "holds_lh" if hand == "left" else "holds_rh"
            state.add((hold_pred, "character", obj))
            for fact in negative_spatial_goals:
                if len(fact) >= 2 and _entities_match(fact[1], obj):
                    state.add(fact)
                    state.discard((fact[0][4:], *fact[1:]))
        else:
            relation = _placement_relation(action)
            if relation and obj:
                held_obj = next((value for value in held.values() if value and action == "puton" and _entities_match(value, obj)), None)
                held_obj = held_obj or (held.get(side) if side else None)
                held_obj = held_obj or held.get("right") or held.get("left")
                if held_obj:
                    destination = "character" if action == "puton" and _entities_match(held_obj, obj) else obj
                    if relation != "nextto":
                        for old in tuple(state):
                            if len(old) >= 2 and old[0] in {"inside", "onfloor", "ontop", "under"} and _entities_match(old[1], held_obj):
                                state.discard(old)
                    effective_relation = "onfloor" if relation == "ontop" and destination.startswith("room_floor_") else relation
                    state.add((effective_relation, held_obj, destination))
                    if relation == "ontop":
                        state.add(("touching", held_obj, destination))
                    if "nextto_ontop" in action:
                        state.add(("nextto", held_obj, destination))
                    state.discard((f"not_{effective_relation}", held_obj, destination))
                    for hand_name in ("left", "right"):
                        if held.get(hand_name) == held_obj:
                            held[hand_name] = None
                            state.discard(("holds_lh" if hand_name == "left" else "holds_rh", "character", held_obj))
                else:
                    state.add((relation, "object", obj))
            elif any(part in action for part in ("release", "drop", "putobjback")):
                held_obj = held.get(side or "right") or held.get("right") or held.get("left")
                if held_obj:
                    state.add(("released", held_obj))
                    for fact in goal_facts:
                        if fact and fact[0] == "onfloor" and len(fact) == 3 and _entities_match(fact[1], held_obj):
                            state.add(fact)
                    for hand_name in ("left", "right"):
                        if held.get(hand_name) == held_obj:
                            held[hand_name] = None
                            state.discard(("holds_lh" if hand_name == "left" else "holds_rh", "character", held_obj))
            elif action in {"clean", "wipe", "scrub", "wash", "rinse"} and obj:
                state.update({("clean", obj), ("not_dusty", obj), ("not_stained", obj)})
                state.discard(("dusty", obj))
                state.discard(("stained", obj))
                if action in {"wash", "rinse"}:
                    state.update({("washed", obj), ("rinsed", obj)})
            elif action == "open" and obj:
                _set_fact(state, ("open", obj), ("closed", obj), ("not_open", obj))
                state.add(("not_closed", obj))
            elif action == "close" and obj:
                _set_fact(state, ("closed", obj), ("open", obj), ("not_closed", obj))
                state.add(("not_open", obj))
            elif action in {"toggle_on", "switchon", "switch_on", "turn_on", "turnon"} and obj:
                _set_fact(state, ("on", obj), ("off", obj), ("not_on", obj))
            elif action in {"toggle_off", "switchoff", "switch_off", "turn_off", "turnoff"} and obj:
                _set_fact(state, ("off", obj), ("on", obj))
                state.add(("not_on", obj))
            elif action == "slice" and obj:
                state.add(("sliced", obj))
            elif action == "soak" and obj:
                state.add(("soaked", obj))
            elif action == "freeze" and obj:
                _set_fact(state, ("frozen", obj), ("not_frozen", obj))
            elif action == "unfreeze" and obj:
                _set_fact(state, ("not_frozen", obj), ("frozen", obj))
            elif action == "cook" and obj:
                state.add(("cooked", obj))
            elif action == "plugin" and obj:
                _set_fact(state, ("plugged_in", obj), ("not_plugged_in", obj))
            elif action == "plugout" and obj:
                state.discard(("plugged_in", obj))
                state.add(("not_plugged_in", obj))
            elif action == "pour" and obj:
                goal_source = next(
                    (
                        fact[1]
                        for fact in goal_facts
                        if len(fact) == 3 and fact[0] == "inside" and _entities_match(fact[2], obj)
                    ),
                    None,
                )
                held_obj = next(
                    (value for value in held.values() if value and goal_source and _entities_match(goal_source, value)),
                    None,
                )
                held_obj = held_obj or held.get("right") or held.get("left")
                if held_obj:
                    state.add(("inside", held_obj, obj))
            elif action in {"walk", "run"} and obj:
                state.update({
                    ("near", obj), ("near", "character", obj),
                    ("inside", "character", obj), ("nextto", "character", obj),
                })
            elif action in {"turnto", "lookat", "watch"} and obj:
                state.add(("facing", "character", obj))
            elif action == "touch" and obj:
                state.add(("touch", obj))
            elif action == "sit" and obj:
                state.update({("sitting", "character"), ("ontop", "character", obj)})
            elif action == "lie" and obj:
                state.update({("lying", "character"), ("ontop", "character", obj)})
        states.append(set(state))
    return states, list(events)


def _tokens_for_sequence(labels_or_tokens: Sequence[int | str | torch.Tensor], vocabulary: Any) -> list[str]:
    if isinstance(labels_or_tokens, (list, tuple)) and all(isinstance(x, str) for x in labels_or_tokens):
        tokens = [_normalize_name(x) if x != EOS_TOKEN else EOS_TOKEN for x in labels_or_tokens]
        eos = _normalize_name(EOS_TOKEN)
        return tokens[: next((i + 1 for i, token in enumerate(tokens) if _normalize_name(token) == eos), len(tokens))]
    return tokens_from_labels(labels_or_tokens, vocabulary)


def abstract_state_from_tokens(
    labels_or_tokens: Sequence[int | str | torch.Tensor],
    vocabulary: Any = None,
    *,
    initial_state: Iterable[Fact] = (),
    goal_facts: Iterable[Fact] = (),
) -> Set[Fact]:
    """Simulate the symbolic final state for an EAI action-token sequence."""
    events = _action_events_from_tokens(_tokens_for_sequence(labels_or_tokens, vocabulary))
    states, _events = _simulate_events(events, initial_state=initial_state, goal_facts=goal_facts)
    return states[-1]


# Dependency-free parser for the benchmark's SimpleTL grammar. The recursive
# descent order gives AND tighter binding than OR, and OR tighter binding than THEN.
@dataclass(frozen=True)
class _Atom:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class _Not:
    arg: Any


@dataclass(frozen=True)
class _And:
    args: tuple[Any, ...]


@dataclass(frozen=True)
class _Or:
    args: tuple[Any, ...]


@dataclass(frozen=True)
class _Then:
    args: tuple[Any, ...]


@dataclass(frozen=True)
class _Quantifier:
    kind: str
    variable: str
    arg: Any
    count: int | None = None


_TL_TOKEN_RE = re.compile(r"[A-Za-z_\-][A-Za-z0-9_\-]*(?:\.[0-9]+)?|\d+|[(),.]")


class _TLParser:
    def __init__(self, text: str):
        self.tokens = _TL_TOKEN_RE.findall(text)
        self.index = 0

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def pop(self, expected: str | None = None) -> str:
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of temporal goal")
        token = self.tokens[self.index]
        self.index += 1
        if expected is not None and token.lower() != expected:
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        return token

    def parse(self) -> Any:
        expression = self.parse_then()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token {self.peek()!r}")
        return expression

    def parse_then(self) -> Any:
        args = [self.parse_or()]
        while (self.peek() or "").lower() == "then":
            self.pop()
            args.append(self.parse_or())
        return args[0] if len(args) == 1 else _Then(tuple(args))

    def parse_or(self) -> Any:
        args = [self.parse_and()]
        while (self.peek() or "").lower() == "or":
            self.pop()
            args.append(self.parse_and())
        return args[0] if len(args) == 1 else _Or(tuple(args))

    def parse_and(self) -> Any:
        args = [self.parse_primary()]
        while (self.peek() or "").lower() == "and":
            self.pop()
            args.append(self.parse_primary())
        return args[0] if len(args) == 1 else _And(tuple(args))

    def parse_primary(self) -> Any:
        token = self.peek()
        if token is None:
            raise ValueError("Missing temporal-goal expression")
        lowered = token.lower()
        if token == "(":
            self.pop("(")
            expression = self.parse_then()
            self.pop(")")
            return expression
        if lowered == "not":
            self.pop()
            return _Not(self.parse_primary())
        if lowered in {"forall", "exists"}:
            kind = self.pop().lower()
            variable = self.pop()
            self.pop(".")
            return _Quantifier(kind, variable, self.parse_primary())
        if lowered == "forn":
            self.pop()
            count = int(self.pop())
            self.pop(".")
            variable = self.pop()
            self.pop(".")
            return _Quantifier("forn", variable, self.parse_primary(), count=count)
        name = self.pop()
        self.pop("(")
        args: list[str] = []
        if self.peek() != ")":
            while True:
                args.append(self.pop())
                if self.peek() != ",":
                    break
                self.pop(",")
        self.pop(")")
        return _Atom(name, tuple(args))


def _parse_tl_goal(text: str) -> Any:
    return _TLParser(text).parse() if text.strip() else None


def _walk_atoms(node: Any, *, negated: bool = False) -> Iterable[tuple[_Atom, bool]]:
    if isinstance(node, _Atom):
        yield node, negated
    elif isinstance(node, _Not):
        yield from _walk_atoms(node.arg, negated=not negated)
    elif isinstance(node, (_And, _Or, _Then)):
        for arg in node.args:
            yield from _walk_atoms(arg, negated=negated)
    elif isinstance(node, _Quantifier):
        yield from _walk_atoms(node.arg, negated=negated)


def _walk_reference_goal_atoms(node: Any, reference: Set[Fact], *, negated: bool = False):
    """Walk the demonstrated branch of OR goals while preserving all AND goals."""
    if isinstance(node, _Or):
        ranked: list[tuple[tuple[bool, float, int], Any]] = []
        for branch in node.args:
            atoms = [
                (atom, branch_negated)
                for atom, branch_negated in _walk_atoms(branch, negated=negated)
                if not _is_type_atom(atom)
            ]
            matched = sum(bool(_match_pattern(atom, branch_negated, reference)) for atom, branch_negated in atoms)
            total = len(atoms)
            score = (bool(total) and matched == total, matched / total if total else 0.0, matched)
            ranked.append((score, branch))
        best_score = max((score for score, _branch in ranked), default=(False, 0.0, 0))
        chosen = next((branch for score, branch in ranked if score == best_score), node.args[0])
        yield from _walk_reference_goal_atoms(chosen, reference, negated=negated)
    elif isinstance(node, _Atom):
        yield node, negated
    elif isinstance(node, _Not):
        yield from _walk_reference_goal_atoms(node.arg, reference, negated=not negated)
    elif isinstance(node, (_And, _Then)):
        for arg in node.args:
            yield from _walk_reference_goal_atoms(arg, reference, negated=negated)
    elif isinstance(node, _Quantifier):
        yield from _walk_reference_goal_atoms(node.arg, reference, negated=negated)


def _is_variable(value: str) -> bool:
    return bool(re.fullmatch(r"x\d*|[xyzuvw]", value.lower()))


def _is_action_atom(atom: _Atom) -> bool:
    return is_goal_action(_normalize_name(atom.name), _ACTIVE_WORLD_BUNDLE.get())


def _is_type_atom(atom: _Atom) -> bool:
    return not _is_action_atom(atom) and not is_state_predicate(
        _canonical_predicate(atom.name), _ACTIVE_WORLD_BUNDLE.get()
    )


def _atom_fact(atom: _Atom, *, negated: bool = False) -> Fact:
    predicate = _negative_predicate(atom.name) if negated else _canonical_predicate(atom.name)
    values = tuple(_normalize_name(arg) for arg in atom.args)
    return ("action", _normalize_name(atom.name), *values) if _is_action_atom(atom) else (predicate, *values)


def _reference_facts(states: Sequence[Set[Fact]], events: Sequence[_ActionEvent]) -> Set[Fact]:
    # Goal facts describe the final condition; transient placements must not
    # become extra requirements merely because the demonstration passed through
    # them on the way to the goal.
    facts = set(states[-1]) if states else set()
    facts.update(("action", event.name, *event.args) for event in events)
    return facts


def _match_pattern(atom: _Atom, negated: bool, reference: Set[Fact]) -> list[Fact]:
    prefix = ("action", _normalize_name(atom.name)) if _is_action_atom(atom) else (
        _negative_predicate(atom.name) if negated else _canonical_predicate(atom.name),
    )
    matches: list[Fact] = []
    for fact in reference:
        if fact[: len(prefix)] != prefix:
            continue
        fact_args = fact[len(prefix):]
        if len(fact_args) == len(atom.args) and all(
            _is_variable(pattern) or _entities_match(pattern, actual)
            for pattern, actual in zip(atom.args, fact_args)
        ):
            matches.append(fact)
    return matches


def _goal_tokens(sample: dict[str, Any], vocabulary: Any) -> list[str]:
    # Prefer self-describing strings; labels without a vocabulary are ambiguous.
    tokens = sample.get("target_action_tokens")
    if tokens is not None:
        return _tokens_for_sequence(list(tokens), vocabulary)
    labels = sample.get("target_action_labels")
    if labels is None:
        return []
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().tolist()
    return _tokens_for_sequence(labels, vocabulary)


def _goal_facts_and_reference(
    sample: dict[str, Any], vocabulary: Any,
) -> tuple[Any, Set[Fact], list[Set[Fact]], list[_ActionEvent]]:
    ast = _parse_tl_goal(str(sample.get("tl_goal", "")))
    # The reference action sequence grounds variables in the symbolic goal and
    # provides only the facts required to judge a generated sequence.
    events = _action_events_from_tokens(_goal_tokens(sample, vocabulary))
    states, events = _simulate_events(events)
    reference = _reference_facts(states, events)
    goal_facts: Set[Fact] = set()
    if ast is not None:
        for atom, negated in _walk_reference_goal_atoms(ast, reference):
            if _is_type_atom(atom):
                continue
            matches = _match_pattern(atom, negated, reference)
            if matches:
                goal_facts.update(matches)
            elif not any(_is_variable(arg) for arg in atom.args):
                goal_facts.add(_atom_fact(atom, negated=negated))
            elif _canonical_predicate(atom.name) == "onfloor" and not negated:
                # BEHAVIOR encodes moving an object to a room floor as RELEASE;
                # ground the quantified object from the demonstrated release.
                for event in events:
                    if "release" not in event.name or not event.args:
                        continue
                    grounded = tuple(
                        event.args[0] if _is_variable(arg) else _normalize_name(arg)
                        for arg in atom.args
                    )
                    goal_facts.add(("onfloor", *grounded))
    # GRAB establishes concrete negative spatial goals by removing the object.
    states, events = _simulate_events(events, goal_facts=goal_facts)
    return ast, goal_facts, states, events


def goal_facts_from_sample(
    sample: dict[str, Any], vocabulary: Any = None, world_bundle: Any = None,
) -> Set[Fact]:
    """Ground the benchmark ``tl_goal`` against the item's object instances."""
    with _using_world_bundle(world_bundle):
        _ast, facts, _states, _events = _goal_facts_and_reference(sample, vocabulary)
        return facts


def _initial_goal_facts(goal_facts: Set[Fact], reference: Set[Fact]) -> Set[Fact]:
    initial: Set[Fact] = set()
    for fact in goal_facts:
        if not fact or fact[0] == "action":
            continue
        positive = positive_state_name(fact[0], _ACTIVE_WORLD_BUNDLE.get())
        if positive is not None:
            if _fact_present(fact, reference):
                initial.add((positive, *fact[1:]))
            else:
                # Preserve initially true signed facts for dense recall.
                initial.add(fact)
        elif not _fact_present(fact, reference):
            # Persistent goal fact never produced by the demonstration, e.g.
            # PLUGGED_IN: it is true in the initial state.
            initial.add(fact)
    return initial


def _entity_universe(sample: dict[str, Any], ast: Any, events: Sequence[_ActionEvent]) -> tuple[str, ...]:
    entities = {_normalize_name(obj) for obj in sample.get("object_tokens", ()) if obj}
    entities.update(arg for event in events for arg in event.args)
    if ast is not None:
        for atom, _negated in _walk_atoms(ast):
            entities.update(_normalize_name(arg) for arg in atom.args if not _is_variable(arg))
    entities.add("character")
    return tuple(sorted(entity for entity in entities if entity))


def _type_root(name: str) -> str:
    name = re.sub(r"_part_\d+$", "", _normalize_name(name))
    name = re.sub(r"_n_\d+$", "", name)
    return re.sub(r"_\d+$", "", name)


def _infer_types(ast: Any, reference: Set[Fact], universe: Sequence[str]) -> dict[str, Set[str]]:
    types: dict[str, Set[str]] = {entity: {_type_root(entity)} for entity in universe}
    if ast is None:
        return types

    def scope_atoms(node: Any, variable: str, negated: bool = False):
        if isinstance(node, _Atom):
            yield node, negated
        elif isinstance(node, _Not):
            yield from scope_atoms(node.arg, variable, not negated)
        elif isinstance(node, (_And, _Or, _Then)):
            for child in node.args:
                yield from scope_atoms(child, variable, negated)
        elif isinstance(node, _Quantifier):
            # A nested quantifier that reuses the name introduces a new scope.
            if node.variable != variable:
                yield from scope_atoms(node.arg, variable, negated)

    def visit(node: Any):
        if isinstance(node, _Quantifier):
            scoped = list(scope_atoms(node.arg, node.variable))
            type_names = {
                _type_root(atom.name)
                for atom, _negated in scoped
                if _is_type_atom(atom) and atom.args == (node.variable,)
            }
            if type_names:
                candidates: list[tuple[str, str]] = []
                for atom, negated in scoped:
                    if _is_type_atom(atom) or node.variable not in atom.args:
                        continue
                    for match in _match_pattern(atom, negated, reference):
                        offset = 2 if match and match[0] == "action" else 1
                        for pattern, actual in zip(atom.args, match[offset:]):
                            if pattern == node.variable:
                                candidates.append((actual, _type_root(actual)))
                # Prefer lexical matches (strawberry_61_part_0 -> strawberry).
                # If the benchmark uses a scene synonym (hardback -> book), the
                # demonstrated relation remains the grounding source.
                lexical = [item for item in candidates if item[1] in type_names]
                for actual, _root in (lexical or candidates):
                    types.setdefault(actual, {_type_root(actual)}).update(type_names)
            visit(node.arg)
        elif isinstance(node, (_And, _Or, _Then)):
            for child in node.args:
                visit(child)
        elif isinstance(node, _Not):
            visit(node.arg)

    visit(ast)
    return types


@dataclass
class _EvalContext:
    universe: tuple[str, ...]
    types: dict[str, Set[str]]


@dataclass(frozen=True)
class PreparedEAIGoal:
    """Parsed and grounded task data reused by every sampled RL rollout."""
    task_id: str
    ast: Any
    gold_state: frozenset[Fact]
    reference_states: tuple[frozenset[Fact], ...]
    reference_events: tuple[_ActionEvent, ...]
    reference_facts: frozenset[Fact]
    initial_state: frozenset[Fact]
    entity_universe: tuple[str, ...]
    types: dict[str, Set[str]]
    tracked_binary_pairs: frozenset[tuple[str, str]]


def prepare_eai_goal(
    sample: dict[str, Any],
    vocabulary: Any = None,
    world_bundle: Any = None,
) -> PreparedEAIGoal:
    """Parse, ground, and cache all task data that is independent of a rollout."""
    with _using_world_bundle(world_bundle):
        return _prepare_eai_goal(sample, vocabulary)


def _prepare_eai_goal(sample: dict[str, Any], vocabulary: Any = None) -> PreparedEAIGoal:
    ast, gold_state, reference_states, reference_events = _goal_facts_and_reference(sample, vocabulary)
    reference = _reference_facts(reference_states, reference_events)
    initial_state = _initial_goal_facts(gold_state, reference)
    universe = _entity_universe(sample, ast, reference_events)
    pairs = {
        (fact[1], fact[2])
        for fact in (
            *gold_state,
            *reference,
            *initial_state,
            *(fact for state in reference_states for fact in state),
        )
        if len(fact) == 3 and fact[0] != "action"
    }
    pairs.update(
        (event.args[0], event.args[1])
        for event in reference_events
        if len(event.args) >= 2
    )
    return PreparedEAIGoal(
        task_id=str(sample.get("task_id", "")),
        ast=ast,
        gold_state=frozenset(gold_state),
        reference_states=tuple(frozenset(state) for state in reference_states),
        reference_events=tuple(reference_events),
        reference_facts=frozenset(reference),
        initial_state=frozenset(initial_state),
        entity_universe=universe,
        types=_infer_types(ast, reference, universe),
        tracked_binary_pairs=frozenset(pairs),
    )


def _typed_quantifier_body(node: _Quantifier, context: _EvalContext) -> tuple[tuple[str, ...], Any]:
    """Recognize EAI's ``not TYPE(x) or body`` typed-quantifier encoding."""
    if isinstance(node.arg, _Or):
        type_names: list[str] = []
        body: list[Any] = []
        for branch in node.arg.args:
            if (
                isinstance(branch, _Not)
                and isinstance(branch.arg, _Atom)
                and _is_type_atom(branch.arg)
                and branch.arg.args == (node.variable,)
            ):
                type_names.append(_type_root(branch.arg.name))
            else:
                body.append(branch)
        if type_names and body:
            domain = tuple(
                entity for entity in context.universe
                if any(name in context.types.get(entity, {_type_root(entity)}) for name in type_names)
            )
            expression = body[0] if len(body) == 1 else _Or(tuple(body))
            return domain, expression
    return context.universe, node.arg


def _eval_atom_at(atom: _Atom, state: Set[Fact], action: _ActionEvent | None, variables: dict[str, str], context: _EvalContext) -> bool:
    args = tuple(variables.get(arg, _normalize_name(arg)) for arg in atom.args)
    if _is_action_atom(atom):
        if action is None or action.name != _normalize_name(atom.name) or len(action.args) > len(args):
            return False
        # The flat EAI token dataset keeps one object for multi-object actions
        # such as POUR. Missing arguments are safe only when the TL expression
        # quantified them; concrete required arguments must never be discarded.
        if any(not _is_variable(original) for original in atom.args[len(action.args):]):
            return False
        return all(_entities_match(required, actual) for required, actual in zip(args, action.args))
    if _is_type_atom(atom):
        return len(args) == 1 and _type_root(atom.name) in context.types.get(args[0], {_type_root(args[0])})
    return _fact_present((_canonical_predicate(atom.name), *args), state)


def _eval_at(node: Any, state: Set[Fact], action: _ActionEvent | None, variables: dict[str, str], context: _EvalContext) -> bool:
    if isinstance(node, _Atom):
        return _eval_atom_at(node, state, action, variables, context)
    if isinstance(node, _Not):
        return not _eval_at(node.arg, state, action, variables, context)
    if isinstance(node, _And):
        return all(_eval_at(arg, state, action, variables, context) for arg in node.args)
    if isinstance(node, _Or):
        return any(_eval_at(arg, state, action, variables, context) for arg in node.args)
    if isinstance(node, _Quantifier):
        domain, body = _typed_quantifier_body(node, context)
        values = []
        for entity in domain:
            nested = dict(variables)
            nested[node.variable] = entity
            values.append(_eval_at(body, state, action, nested, context))
        if node.kind == "forall":
            return all(values)
        if node.kind == "exists":
            return any(values)
        return sum(values) == int(node.count or 0)
    if isinstance(node, _Then):
        raise ValueError("Temporal expression cannot be evaluated at one state")
    return False


def _first_satisfaction(node: Any, states: Sequence[Set[Fact]], actions: Sequence[_ActionEvent], context: _EvalContext, start: int = 0, variables: dict[str, str] | None = None) -> int | None:
    variables = variables or {}
    if isinstance(node, _Then):
        # Each term must be found after the prior one, though terms can hold at
        # different snapshots rather than needing to be true simultaneously.
        cursor = start
        for arg in node.args:
            found = _first_satisfaction(arg, states, actions, context, cursor, variables)
            if found is None:
                return None
            cursor = found + 1
        return cursor - 1
    if isinstance(node, _Quantifier) and isinstance(node.arg, _Then):
        matches: list[int] = []
        for entity in context.universe:
            nested = dict(variables)
            nested[node.variable] = entity
            found = _first_satisfaction(node.arg, states, actions, context, start, nested)
            if found is not None:
                matches.append(found)
        if node.kind == "forall" and len(matches) == len(context.universe):
            return max(matches, default=start)
        if node.kind == "exists" and matches:
            return min(matches)
        if node.kind == "forn" and len(matches) == int(node.count or 0):
            return max(matches, default=start)
        return None
    for index in range(start, len(states)):
        action = actions[index] if index < len(actions) else None
        if _eval_at(node, states[index], action, variables, context):
            return index
    return None


def _contains_then(node: Any) -> bool:
    if isinstance(node, _Then):
        return True
    if isinstance(node, (_And, _Or)):
        return any(_contains_then(arg) for arg in node.args)
    if isinstance(node, (_Not, _Quantifier)):
        return _contains_then(node.arg)
    return False


def state_recall(predicted_state: Set[Fact], gold_state: Set[Fact]) -> float:
    """Fraction of required facts matched with strict entity correlation."""
    if not gold_state:
        return 1.0 if not predicted_state else 0.0
    return sum(_fact_present(fact, predicted_state) for fact in gold_state) / len(gold_state)


def evaluate_goal_satisfaction(
    pred_labels_or_tokens: Sequence[int | str | torch.Tensor],
    sample: dict[str, Any],
    vocabulary: Any = None,
    world_bundle: Any = None,
    prepared_goal: PreparedEAIGoal | None = None,
    reward_mode: str = "binary",
    constraint_weight: float = 0.25,
    constraint_aggregate: str = "mean",
) -> dict[str, Any]:
    """Evaluate a generated trajectory against the sample's actual ``tl_goal``."""
    with _using_world_bundle(world_bundle):
        return _evaluate_goal_satisfaction(
            pred_labels_or_tokens,
            sample,
            vocabulary=vocabulary,
            world_bundle=world_bundle,
            prepared_goal=prepared_goal,
            reward_mode=reward_mode,
            constraint_weight=constraint_weight,
            constraint_aggregate=constraint_aggregate,
        )


def _evaluate_goal_satisfaction(
    pred_labels_or_tokens: Sequence[int | str | torch.Tensor],
    sample: dict[str, Any],
    vocabulary: Any = None,
    world_bundle: Any = None,
    prepared_goal: PreparedEAIGoal | None = None,
    reward_mode: str = "binary",
    constraint_weight: float = 0.25,
    constraint_aggregate: str = "mean",
) -> dict[str, Any]:
    if reward_mode not in {"binary", "dense"}:
        raise ValueError(f"Unsupported EAI reward mode: {reward_mode!r}")
    if not 0.0 <= float(constraint_weight) <= 1.0:
        raise ValueError("constraint_weight must be between 0 and 1")
    prepared = prepared_goal or _prepare_eai_goal(sample, vocabulary)
    ast = prepared.ast
    gold_state = set(prepared.gold_state)
    initial_state = set(prepared.initial_state)
    events = _action_events_from_tokens(_tokens_for_sequence(pred_labels_or_tokens, vocabulary))
    states, events = _simulate_events(events, initial_state=initial_state, goal_facts=gold_state)
    predicted_state = set(states[-1])
    predicted_state.update(("action", event.name, *event.args) for event in events)
    recall = state_recall(predicted_state, gold_state)
    parse_error = None
    if ast is None:
        is_success = recall >= 1.0 and bool(gold_state)
    else:
        context = _EvalContext(prepared.entity_universe, prepared.types)
        try:
            ast_success = _first_satisfaction(ast, states, events, context) is not None
            # Grounded fact recall is the authoritative final-condition check.
            # Temporal goals additionally require the ordered SimpleTL result.
            is_success = ast_success if _contains_then(ast) else recall >= 1.0 and bool(gold_state)
        except (ValueError, TypeError) as exc:
            parse_error = str(exc)
            is_success = recall >= 1.0 and bool(gold_state)
    task_reward = recall if reward_mode == "dense" else (1.0 if is_success else 0.0)
    constraint_evaluation = None
    if world_bundle is not None and world_bundle.has_constraints:
        root = materialize_world_trajectory(prepared, states, events, world_bundle)
        constraint_evaluation = verify_world_constraints(
            root, world_bundle, aggregate=constraint_aggregate,
        )
    world_constraint_score = (
        constraint_evaluation.score if constraint_evaluation is not None else None
    )
    rl_reward_score = task_reward
    if world_constraint_score is not None:
        rl_reward_score = (
            (1.0 - float(constraint_weight)) * task_reward
            + float(constraint_weight) * world_constraint_score
        )
    return {
        "is_success": 1.0 if is_success else 0.0,
        "recall": recall,
        "predicted_state": predicted_state,
        "gold_state": gold_state,
        "initial_state": initial_state,
        "parse_error": parse_error,
        "world_constraint_score": world_constraint_score,
        "world_constraint_results": (
            constraint_evaluation.results if constraint_evaluation is not None else None
        ),
        "rl_reward_score": rl_reward_score,
    }


def eai_goal_reward_function(
    generator_output: Any,
    data_item: Any = None,
    vocabulary: Any = None,
    mode: str = "binary",
    world_bundle: Any = None,
    prepared_goal: PreparedEAIGoal | None = None,
    constraint_weight: float = 0.25,
    constraint_aggregate: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """DomiKnowS-compatible binary goal-success or dense-recall reward."""
    if data_item is None:
        return torch.tensor([0.0], dtype=torch.float32)
    result = evaluate_goal_satisfaction(
        flatten_generator_output(generator_output),
        data_item,
        vocabulary=vocabulary,
        world_bundle=world_bundle,
        prepared_goal=prepared_goal,
        reward_mode=mode,
        constraint_weight=constraint_weight,
        constraint_aggregate=constraint_aggregate,
    )
    return torch.tensor([result["rl_reward_score"]], dtype=torch.float32)


def make_eai_reward_function(
    sample: dict[str, Any],
    vocabulary: Any = None,
    mode: str = "binary",
    world_bundle: Any = None,
    constraint_weight: float = 0.25,
    constraint_aggregate: str = "mean",
):
    """Create a per-item reward closure with inspection metadata."""
    prepared = prepare_eai_goal(sample, vocabulary, world_bundle=world_bundle)

    def _reward(generator_output: Any, **context) -> torch.Tensor:
        item = context.get("data_item") or sample
        active_prepared = (
            prepared if item is sample
            else prepare_eai_goal(item, vocabulary, world_bundle=world_bundle)
        )
        return eai_goal_reward_function(
            generator_output,
            data_item=item,
            vocabulary=vocabulary,
            mode=mode,
            world_bundle=world_bundle,
            prepared_goal=active_prepared,
            constraint_weight=constraint_weight,
            constraint_aggregate=constraint_aggregate,
        )

    _reward.task_id = str(sample.get("task_id", ""))
    _reward.gold_state = set(prepared.gold_state)
    _reward.mode = mode
    _reward.prepared_goal = prepared
    _reward.world_bundle = world_bundle
    _reward.constraint_weight = float(constraint_weight)
    _reward.constraint_aggregate = constraint_aggregate
    return _reward


def eai_action_decoder(samples: dict[Any, torch.Tensor], targets: list[Any], datanode: Any, data_item: Any) -> list[int]:
    """Map DomiKnowS sampled concept assignments to flat token IDs."""
    del datanode, data_item
    for concept in targets:
        name = getattr(concept, "name", str(concept))
        if name == "generated_token" or "generated_token" in str(concept):
            idx = samples.get(concept)
            if idx is not None:
                return idx.reshape(-1).tolist()
    for _concept, idx in samples.items():
        if idx is not None:
            return idx.reshape(-1).tolist()
    return []


def test_all_dataset_goals(dataset_name: str = "all", limit: int | None = None, max_steps: int = 135) -> dict[str, float]:
    """Verify temporal-goal satisfaction across EAI reference trajectories."""
    if isinstance(dataset_name, int) and limit is None:
        limit = dataset_name
        dataset_name = "all"
    try:
        from dataset import load_eai_dataset
    except ImportError:
        from .dataset import load_eai_dataset
    examples = load_eai_dataset(dataset_name=dataset_name, limit=limit, max_steps=max_steps, device="cpu")
    vocabulary = TokenVocabulary(examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
    successes = 0
    total_facts = 0
    for sample in examples:
        result = evaluate_goal_satisfaction(sample["target_action_tokens"], sample, vocabulary)
        successes += int(result["is_success"] == 1.0)
        total_facts += len(result["gold_state"])
    total = len(examples)
    success_rate = successes / total if total else 0.0
    avg_facts = total_facts / total if total else 0.0
    print(f"Verified {total} examples: goal success rate = {success_rate:.4f}, avg facts = {avg_facts:.2f}")
    return {"total": total, "success_rate": success_rate, "avg_facts": avg_facts}
