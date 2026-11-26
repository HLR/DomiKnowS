"""Reference-independent SimpleTL evaluation for Joint Stage 2."""

from dataclasses import dataclass, replace
import re

from .reward import (
    _ActionEvent, _And, _Atom, _Not, _Or, _Quantifier, _Then,
    _canonical_predicate, _contains_then, _entity_parts, _is_action_atom,
    _is_type_atom, _is_variable, _normalize_name, _parse_tl_goal,
    _tokens_for_sequence, _using_world_bundle, _walk_atoms,
)
from .utils.constraint_compiler import EAIExecutor, observed_entity_allowed
from .utils.goal_constraints import evaluate_goal_operator
from .utils.world_domain import ACTION_SPECS


def _problem_sections(text):
    stack = [[]]
    for token in re.findall(r"\(|\)|[^()\s]+", re.sub(r";[^\n]*", "", text)):
        if token == "(":
            child = []
            stack[-1].append(child)
            stack.append(child)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("unexpected close parenthesis in EAI transition model")
            stack.pop()
        else:
            stack[-1].append(token.lower())
    if len(stack) != 1 or len(stack[0]) != 1:
        raise ValueError("invalid EAI transition model")
    problem = stack[0][0]
    if not isinstance(problem, list) or not problem or problem[0] != "define":
        raise ValueError("EAI transition model must be a PDDL problem")
    sections = {part[0]: part[1:] for part in problem[1:] if isinstance(part, list) and part}
    if any(key not in sections for key in (":domain", ":objects", ":init")):
        raise ValueError("EAI transition model requires :domain, :objects, and :init")
    return sections


def _object_types(declarations):
    types = {}
    pending = []
    tokens = iter(declarations)
    for token in tokens:
        if token == "-":
            kind = _normalize_name(next(tokens, ""))
            if not kind or not pending:
                raise ValueError("invalid typed object declaration")
            types.update(dict.fromkeys(pending, kind))
            pending.clear()
        else:
            pending.append(_normalize_name(token))
    types.update(dict.fromkeys(pending, "object"))
    return types


def _category(name):
    normalized = _normalize_name(name)
    wordnet_base = re.sub(r"_n_\d+(?:_\d+)?$", "", normalized)
    return _entity_parts(normalized)[0] if wordnet_base == normalized else wordnet_base


def _observed_facts(expressions, aliases):
    positive, negative = set(), set()
    for expression in expressions:
        if not isinstance(expression, list) or not expression:
            raise ValueError("invalid EAI :init fact")
        if expression[0] == "not":
            relation, *arguments = expression[1]
            destination = negative
        else:
            relation, *arguments = expression
            destination = positive
        if relation in {"=", "increase", "decrease"}:
            continue  # numeric fluents are outside the action transition model
        names = tuple(aliases.get(_normalize_name(name), _normalize_name(name)) for name in arguments)
        destination.add((_canonical_predicate(relation), *names))
    return frozenset(positive), frozenset(negative)


@dataclass(frozen=True)
class ObservedEAIGoal:
    domain: str
    ast: object
    types: dict
    aliases: dict
    indexed_categories: dict
    positive: frozenset
    negative: frozenset
    unresolved: tuple

    def policy_context(self):
        capabilities = {"grabbable": "graspable", "can_open": "openable",
                        "pourable": "pourable", "containers": "container"}
        properties = {}
        for facts, truth in ((self.positive, True), (self.negative, False)):
            for fact in facts:
                if len(fact) == 2 and fact[0] in capabilities:
                    properties.setdefault(fact[1], {})[capabilities[fact[0]]] = truth
        return {
            "entities": tuple(self.types),
            "planning_facts": {
                "initial_state": {"positive": tuple(sorted(self.positive)),
                                  "negative": tuple(sorted(self.negative))},
                "entity_properties": properties,
            },
            "observation_entity_aliases": self.aliases,
            "observation_indexed_categories": tuple(self.indexed_categories),
            "require_observed_entities": True,
            "allow_empty_observed_plan": True,
            "observed_placement_semantics": True,
        }


def prepare_observed_eai_goal(sample, *, world_bundle=None):
    """Bind scene identities using PDDL and the task's published SimpleTL."""
    with _using_world_bundle(world_bundle):
        sections = _problem_sections(sample["transition_model"])
        domain, = sections[":domain"]
        ast = _parse_tl_goal(sample["tl_goal"])
        if ast is None:
            raise ValueError("EAI task has no SimpleTL goal")
        objects = _object_types(sections[":objects"])
        indexed_categories = {name: kind for name, kind in objects.items() if _entity_parts(name)[1] is None
                              and kind not in {"agent", "character"}} if domain == "virtualhome" else {}
        aliases = {name: "character" for name, kind in objects.items()
                   if kind in {"agent", "character"}}
        aliases.update({"agent": "character", "character": "character"})
        references = {}
        for atom, _ in _walk_atoms(ast):
            for arg in atom.args:
                if not _is_variable(arg):
                    name = _normalize_name(arg)
                    references.setdefault(_entity_parts(name)[0], set()).add(name)

        unresolved = set()
        types = {}
        for root, names in sorted(references.items()):
            for name in sorted(names):
                if name in objects or name in aliases:
                    continue
                if domain == "virtualhome" and root in objects:
                    # VirtualHome PDDL uses class-only objects. Retain every
                    # explicitly named instance, and lift the class observation
                    # only when its instance is unambiguous in this task.
                    types[name] = {objects[root], root}
                    if len(names) == 1:
                        aliases[root] = name
                    continue
                candidates = [obj for obj in objects if _category(obj) == root]
                if domain == "igibson" and len(candidates) == len(names) == 1:
                    aliases[name] = candidates[0]
                else:
                    unresolved.add(name)
        for name, kind in objects.items():
            types.setdefault(aliases.get(name, name), set()).update({kind, _category(name)})
        types.setdefault("character", set()).update(("character", "agent"))
        positive, negative = _observed_facts(sections[":init"], aliases)
        for fact in positive | negative:
            for entity in fact[1:]:
                types.setdefault(entity, {_category(entity)})
        return ObservedEAIGoal(domain, ast, types, aliases, indexed_categories,
                              positive, negative, tuple(sorted(unresolved)))


def _witness(node, states, events, goal, start=0, bindings=None):
    """Return the earliest witness, with strict order for successive Then terms."""
    bindings = {} if bindings is None else bindings
    if start >= len(states):
        return None
    if isinstance(node, _Then):
        for child in node.args:
            found = _witness(child, states, events, goal, start, bindings)
            if found is None:
                return None
            start = found + 1
        return start - 1
    if _contains_then(node):
        if isinstance(node, (_And, _Or)):
            matches = [_witness(child, states, events, goal, start, bindings) for child in node.args]
            if isinstance(node, _And):
                return max(matches) if all(value is not None for value in matches) else None
            return min((value for value in matches if value is not None), default=None)
        if isinstance(node, _Not):
            return start if _witness(node.arg, states, events, goal, start, bindings) is None else None
        if isinstance(node, _Quantifier):
            domain, body = _quantifier_domain(node, goal)
            witnesses = [_witness(body, states, events, goal, start, {**bindings, node.variable: name})
                         for name in domain]
            matches = [value for value in witnesses if value is not None]
            if evaluate_goal_operator(node.kind, [value is not None for value in witnesses], node.count or 0):
                return min(matches) if node.kind == "exists" else max(matches, default=start)
            return None
    return next((index for index in range(start, len(states))
                 if _matches(node, states[index], events[index], goal, bindings)), None)


def _quantifier_domain(node, goal):
    # In the benchmark, the leading "not TYPE(x) or ..." is a type
    # restriction. Counting and existential quantifiers range over that type.
    if isinstance(node.arg, _Or):
        kinds, clauses = set(), []
        for child in node.arg.args:
            if (isinstance(child, _Not) and isinstance(child.arg, _Atom)
                    and _is_type_atom(child.arg) and child.arg.args == (node.variable,)):
                kinds.add(_normalize_name(child.arg.name))
            else:
                clauses.append(child)
        if kinds and clauses:
            body = clauses[0] if len(clauses) == 1 else _Or(tuple(clauses))
            return (name for name, types in goal.types.items() if kinds & types), body
    return goal.types, node.arg


def _matches(node, facts, event, goal, bindings):
    if isinstance(node, _Atom):
        names = tuple(bindings[arg] if arg in bindings else goal.aliases.get(_normalize_name(arg), _normalize_name(arg))
                      for arg in node.args)
        if _is_action_atom(node):
            return event == _ActionEvent(_normalize_name(node.name), names)
        if _is_type_atom(node):
            return len(names) == 1 and _normalize_name(node.name) in goal.types.get(names[0], ())
        return (_canonical_predicate(node.name), *names) in facts
    if isinstance(node, _Not):
        return not _matches(node.arg, facts, event, goal, bindings)
    if isinstance(node, (_And, _Or)):
        return evaluate_goal_operator("and" if isinstance(node, _And) else "or",
                                      [_matches(child, facts, event, goal, bindings) for child in node.args])
    domain, body = _quantifier_domain(node, goal)
    return evaluate_goal_operator(node.kind,
        [_matches(body, facts, event, goal, {**bindings, node.variable: name}) for name in domain],
        node.count or 0)


def _top_level_score(node, states, events, goal):
    if _contains_then(node):
        return _witness(node, states, events, goal) is not None
    if any(_is_action_atom(atom) for atom, _ in _walk_atoms(node)):
        return any(_matches(node, states[-1], event, goal, {}) for event in events)
    return _matches(node, states[-1], None, goal, {})


def _events(tokens, aliases):
    result = []
    name, args = None, []
    for value in (*tokens, ""):
        token = _normalize_name(value)
        if token in ACTION_SPECS or value in ("", "<eos>"):
            if name is not None:
                spec = ACTION_SPECS[name]
                if not spec.min_args <= len(args) <= spec.max_args:
                    return None
                result.append(_ActionEvent(name, tuple(args)))
            name, args = token, []
            if value in ("", "<eos>"):
                break
        elif name is None:
            return None
        else:
            args.append(aliases.get(token, token))
    return result


def score_observed_eai_goal(labels, goal, vocabulary=None, *, world_bundle):
    """Score generated effects against the goal using exact scene identities."""
    if goal.unresolved:
        return {"is_success": False, "recall": 0.0, "unresolved": goal.unresolved}
    with _using_world_bundle(world_bundle):
        tokens = _tokens_for_sequence(labels, vocabulary)
        eos = getattr(vocabulary, "eos_token", "<eos>")
        if tokens and tokens[-1] == eos:
            tokens = tokens[:-1]
        events = _events(tokens, goal.aliases)
        if events is None:
            return {"is_success": False, "recall": 0.0, "unresolved": ()}
        extra_types = {}
        for name in {name for event in events for name in event.args} - goal.types.keys():
            if not observed_entity_allowed(name, goal.types, goal.indexed_categories):
                return {"is_success": False, "recall": 0.0, "unresolved": ()}
            category, _ = _entity_parts(name)
            extra_types[name] = {category, goal.indexed_categories[category]}
        if extra_types:
            # VirtualHome exports some objects by class alone, while the
            # action labels carry numeric IDs. Keep IDs distinct in effects
            # and action predicates. Never copy class facts to the new IDs.
            goal = replace(goal, types={**goal.types, **extra_types})
        executor = EAIExecutor(world_bundle, goal.policy_context(), goal.types)
        state = executor.planning.initial
        states = [state.facts.positive]
        for event in events:
            state, _, _ = executor.advance(state, event.name, event.args)
            states.append(state.facts.positive)
        timeline = (None, *events)
        parts = goal.ast.args if isinstance(goal.ast, _And) else (goal.ast,)
        outcomes = [_top_level_score(part, states, timeline, goal) for part in parts]
        return {"is_success": all(outcomes), "recall": sum(outcomes) / len(outcomes), "unresolved": ()}
