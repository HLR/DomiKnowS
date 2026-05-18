"""Generic planning-domain adapters for DomiKnowS generation graphs.

The helpers here keep task graphs declarative.  A human-authored graph defines
planning concepts, action labels, phase transitions, reference plans, and
logical constraints.  This module reads those graph constructs and builds the
execution artifacts used by demos and tests: a planning bundle, a hard DFA, and
graph-HMM masks.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

import torch

from .automata import DFA


@dataclass(frozen=True)
class PlanningBundle:
    """Execution view derived from a declarative planning graph."""

    graph: Any
    plan: Any
    step: Any
    planned_action: Any
    planned_task: Any | None
    precedes: Any
    earlier: Any
    later: Any
    action_names: tuple[str, ...]
    task_names: tuple[str, ...]
    phase_names: tuple[str, ...]
    terminal_action: str
    selected_task: str
    required_actions: Mapping[str, tuple[str, ...]]
    reference_plans: Mapping[str, tuple[str, ...]]
    phase_transitions: Mapping[tuple[str, str], str]
    action_count_limits: Mapping[str, int]
    non_terminal_limit: int | None

    @property
    def planned_dish(self):
        """Backward-compatible alias for cooking's original selected-task enum."""

        return self.planned_task

    @property
    def selected_reference_plan(self) -> tuple[str, ...]:
        """Return the graph-declared reference plan for ``selected_task``."""

        try:
            return self.reference_plans[self.selected_task]
        except KeyError as exc:
            raise ValueError(f"no reference plan declared for task {self.selected_task!r}") from exc

    @property
    def selected_required_actions(self) -> tuple[str, ...]:
        """Return required actions for ``selected_task``."""

        return self.required_actions.get(self.selected_task, ())


def planning_bundle_from_graph(
    graph,
    *,
    selected_task: str = "cookie",
    plan_name: str = "plan",
    step_name: str = "step",
    planned_action_name: str = "planned_action",
    planned_task_name: str = "planned_dish",
    before_relation_name: str = "precedes",
    first_role_name: str = "earlier",
    second_role_name: str = "later",
    action_root_name: str = "action",
    task_root_name: str = "dish",
    phase_root_name: str = "plan_phase",
    terminal_action: str = "done",
    required_action_schema_name: str = "dish_requires_action",
    reference_plan_schema_name: str = "reference_plan_step",
    task_role_name: str = "dish",
    position_role_name: str = "position",
    action_role_name: str = "action",
    phase_transition_schema_name: str = "phase_transition",
    source_phase_role_name: str = "source_phase",
    target_phase_role_name: str = "target_phase",
    action_count_limit_schema_name: str = "action_count_limit",
    non_terminal_limit_schema_name: str = "non_terminal_action_count_limit",
    limit_role_name: str = "limit",
) -> PlanningBundle:
    """Derive a planning execution bundle from a declarative DomiKnowS graph."""

    plan = _required_concept(graph, plan_name)
    step = _required_concept(graph, step_name)
    planned_action = _required_concept(graph, planned_action_name)
    planned_task = _optional_concept(graph, planned_task_name)
    precedes = _required_concept(graph, before_relation_name)
    action_root = _required_concept(graph, action_root_name)
    task_root = _required_concept(graph, task_root_name)
    phase_root = _required_concept(graph, phase_root_name)

    action_names = tuple(getattr(planned_action, "enum", ()))
    if terminal_action not in action_names:
        raise ValueError(f"terminal action {terminal_action!r} is not in planned_action enum")
    task_names = tuple(getattr(planned_task, "enum", ())) if planned_task is not None else tuple(
        concept.name for concept in _children_of(graph, task_root)
    )
    if selected_task not in task_names:
        raise ValueError(f"unknown selected_task {selected_task!r}; expected one of {task_names!r}")
    phase_names = tuple(concept.name for concept in _children_of(graph, phase_root))
    if not phase_names:
        raise ValueError("planning graph must declare at least one plan_phase subconcept")

    contains = _find_contains_relation(plan, step)
    earlier = _find_has_a_role(graph, precedes, first_role_name)
    later = _find_has_a_role(graph, precedes, second_role_name)
    if contains is None:
        raise ValueError(f"{plan_name}.contains({step_name}) relation is missing")

    required_actions = _extract_required_actions(
        graph,
        task_root,
        action_root,
        schema_name=required_action_schema_name,
        task_role_name=task_role_name,
        action_role_name=action_role_name,
    )
    reference_plans = _extract_reference_plans(
        graph,
        task_root,
        action_root,
        schema_name=reference_plan_schema_name,
        task_role_name=task_role_name,
        position_role_name=position_role_name,
        action_role_name=action_role_name,
    )
    phase_transitions = _extract_phase_transitions(
        graph,
        phase_root,
        action_root,
        schema_name=phase_transition_schema_name,
        source_phase_role_name=source_phase_role_name,
        action_role_name=action_role_name,
        target_phase_role_name=target_phase_role_name,
    )
    action_count_limits = _extract_action_count_limits(
        graph,
        action_root,
        schema_name=action_count_limit_schema_name,
        action_role_name=action_role_name,
        limit_role_name=limit_role_name,
    )
    non_terminal_limit = _extract_non_terminal_limit(
        graph,
        action_root,
        schema_name=non_terminal_limit_schema_name,
        limit_role_name=limit_role_name,
    )

    return PlanningBundle(
        graph=graph,
        plan=plan,
        step=step,
        planned_action=planned_action,
        planned_task=planned_task,
        precedes=precedes,
        earlier=earlier,
        later=later,
        action_names=action_names,
        task_names=task_names,
        phase_names=phase_names,
        terminal_action=terminal_action,
        selected_task=selected_task,
        required_actions=required_actions,
        reference_plans=reference_plans,
        phase_transitions=phase_transitions,
        action_count_limits=action_count_limits,
        non_terminal_limit=non_terminal_limit,
    )


def planning_dfa_from_graph(bundle: PlanningBundle) -> DFA:
    """Compile graph-declared phase/count/requirement facts into a hard DFA."""

    alphabet = frozenset(bundle.action_names)
    count_actions = tuple(sorted(bundle.action_count_limits))
    count_index = {action: index for index, action in enumerate(count_actions)}
    required_actions = tuple(bundle.selected_required_actions)
    required_index = {action: index for index, action in enumerate(required_actions)}
    start_phase = _phase_or_default(bundle.phase_names, "start", index=0)
    accepting_phase = _phase_or_default(bundle.phase_names, "done_phase", index=-1)
    dead = ("dead",)
    start = (start_phase, tuple(0 for _ in count_actions), 0, 0)

    states = {start, dead}
    transitions: dict[tuple[Any, str], Any] = {}
    accepting = set()
    queue = deque([start])

    def is_accepting(state) -> bool:
        if state == dead:
            return False
        phase, _counts, required_mask, _non_terminal_count = state
        all_required = required_mask == ((1 << len(required_actions)) - 1)
        return phase == accepting_phase and all_required

    while queue:
        state = queue.popleft()
        if is_accepting(state):
            accepting.add(state)
        for action in bundle.action_names:
            nxt = _planning_dfa_step(
                bundle,
                state,
                action,
                count_actions=count_actions,
                count_index=count_index,
                required_index=required_index,
                dead=dead,
            )
            transitions[(state, action)] = nxt
            if nxt not in states:
                states.add(nxt)
                queue.append(nxt)

    for action in bundle.action_names:
        transitions[(dead, action)] = dead

    return DFA(
        states=frozenset(states),
        alphabet=alphabet,
        transitions=transitions,
        start_state=start,
        accepting_states=frozenset(accepting),
        dead_states=frozenset({dead}),
    )


def planning_hmm_masks_from_graph(bundle: PlanningBundle, *, dtype=torch.float64):
    """Return ``(transition_mask, emission_mask)`` tensors for graph-HMM learning."""

    phase_to_id = {phase: index for index, phase in enumerate(bundle.phase_names)}
    action_to_id = {action: index for index, action in enumerate(bundle.action_names)}
    transition_mask = torch.zeros((len(bundle.phase_names), len(bundle.phase_names)), dtype=dtype)
    emission_mask = torch.zeros((len(bundle.phase_names), len(bundle.action_names)), dtype=dtype)
    for (source_phase, action), target_phase in bundle.phase_transitions.items():
        if source_phase not in phase_to_id or target_phase not in phase_to_id:
            continue
        if action not in action_to_id:
            continue
        transition_mask[phase_to_id[source_phase], phase_to_id[target_phase]] = 1
        emission_mask[phase_to_id[source_phase], action_to_id[action]] = 1

    if (transition_mask.sum(dim=1) == 0).any():
        empty = [bundle.phase_names[i] for i, value in enumerate(transition_mask.sum(dim=1)) if float(value) == 0.0]
        raise ValueError(f"phase transition graph leaves phases with no outgoing transitions: {empty}")
    if (emission_mask.sum(dim=1) == 0).any():
        empty = [bundle.phase_names[i] for i, value in enumerate(emission_mask.sum(dim=1)) if float(value) == 0.0]
        raise ValueError(f"phase emission graph leaves phases with no allowed actions: {empty}")
    return transition_mask, emission_mask


def reference_plans_from_graph(bundle: PlanningBundle) -> dict[str, tuple[str, ...]]:
    """Return graph-declared reference plans keyed by task name."""

    return dict(bundle.reference_plans)


def encode_plan(bundle: PlanningBundle, plan: Sequence[str]) -> tuple[int, ...]:
    """Map action names to compact action ids using the bundle enum order."""

    action_to_id = {action: index for index, action in enumerate(bundle.action_names)}
    try:
        return tuple(action_to_id[action] for action in plan)
    except KeyError as exc:
        raise ValueError(f"unknown action {exc.args[0]!r}") from exc


def decode_plan(bundle: PlanningBundle, labels: Sequence[int]) -> tuple[str, ...]:
    """Map compact action ids back to action names."""

    return tuple(bundle.action_names[int(label)] for label in labels)


def _planning_dfa_step(
    bundle: PlanningBundle,
    state,
    action: str,
    *,
    count_actions: tuple[str, ...],
    count_index: Mapping[str, int],
    required_index: Mapping[str, int],
    dead,
):
    if state == dead:
        return dead
    phase, counts, required_mask, non_terminal_count = state
    target_phase = bundle.phase_transitions.get((phase, action))
    if target_phase is None:
        return dead

    counts_list = list(counts)
    if action in count_index:
        idx = count_index[action]
        counts_list[idx] += 1
        if counts_list[idx] > bundle.action_count_limits[action]:
            return dead

    if action != bundle.terminal_action:
        non_terminal_count += 1
        if bundle.non_terminal_limit is not None and non_terminal_count > bundle.non_terminal_limit:
            return dead

    if action in required_index:
        required_mask |= 1 << required_index[action]

    return (target_phase, tuple(counts_list), required_mask, non_terminal_count)


def _extract_required_actions(
    graph,
    task_root,
    action_root,
    *,
    schema_name: str,
    task_role_name: str,
    action_role_name: str,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = defaultdict(list)
    for fact in _facts_for_schema(graph, schema_name):
        task = _role_target(graph, fact, task_role_name, task_root)
        action = _role_target(graph, fact, action_role_name, action_root)
        if task is not None and action is not None:
            result[task.name].append(action.name)
    return {task: tuple(actions) for task, actions in result.items()}


def _extract_reference_plans(
    graph,
    task_root,
    action_root,
    *,
    schema_name: str,
    task_role_name: str,
    position_role_name: str,
    action_role_name: str,
) -> dict[str, tuple[str, ...]]:
    position_root = _optional_concept(graph, "reference_position")
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for fact in _facts_for_schema(graph, schema_name):
        task = _role_target(graph, fact, task_role_name, task_root)
        action = _role_target(graph, fact, action_role_name, action_root)
        position = _role_target(graph, fact, position_role_name, position_root) if position_root is not None else None
        if task is None or action is None or position is None:
            continue
        grouped[task.name].append((_position_index(position.name), action.name))
    return {
        task: tuple(action for _position, action in sorted(items, key=lambda item: item[0]))
        for task, items in grouped.items()
    }


def _extract_phase_transitions(
    graph,
    phase_root,
    action_root,
    *,
    schema_name: str,
    source_phase_role_name: str,
    action_role_name: str,
    target_phase_role_name: str,
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for fact in _facts_for_schema(graph, schema_name):
        source = _role_target(graph, fact, source_phase_role_name, phase_root)
        action = _role_target(graph, fact, action_role_name, action_root)
        target = _role_target(graph, fact, target_phase_role_name, phase_root)
        if source is not None and action is not None and target is not None:
            result[(source.name, action.name)] = target.name
    if not result:
        raise ValueError(f"planning graph declares no {schema_name} facts")
    return result


def _extract_action_count_limits(
    graph,
    action_root,
    *,
    schema_name: str,
    action_role_name: str,
    limit_role_name: str,
) -> dict[str, int]:
    count_root = _optional_concept(graph, "count_limit")
    result: dict[str, int] = {}
    for fact in _facts_for_schema(graph, schema_name):
        action = _role_target(graph, fact, action_role_name, action_root)
        limit = _role_target(graph, fact, limit_role_name, count_root) if count_root is not None else None
        if action is not None and limit is not None:
            result[action.name] = _limit_value(limit.name)
    return result


def _extract_non_terminal_limit(
    graph,
    action_root,
    *,
    schema_name: str,
    limit_role_name: str,
) -> int | None:
    count_root = _optional_concept(graph, "count_limit")
    for fact in _facts_for_schema(graph, schema_name):
        limit = _role_target(graph, fact, limit_role_name, count_root) if count_root is not None else None
        if limit is not None:
            return _limit_value(limit.name)
    return None


def _facts_for_schema(graph, schema_name: str) -> list[Any]:
    schema = _optional_concept(graph, schema_name)
    if schema is None:
        return []
    return [
        concept
        for concept in _all_concepts(graph)
        if concept is not schema and _is_subconcept_of(concept, schema)
    ]


def _children_of(graph, root) -> list[Any]:
    return [
        concept
        for concept in _all_concepts(graph)
        if concept is not root and _is_subconcept_of(concept, root)
    ]


def _role_target(graph, fact, role_name: str, expected_root=None):
    relation = _find_has_a_role(graph, fact, role_name)
    if relation is None:
        return None
    candidates = []
    for concept in _all_concepts(graph):
        if concept is relation.dst:
            continue
        if relation in concept._in.get("has_a", ()):
            if expected_root is None or concept is expected_root or _is_subconcept_of(concept, expected_root):
                candidates.append(concept)
    if candidates:
        candidates.sort(key=lambda concept: _inheritance_depth(concept), reverse=True)
        return candidates[0]
    return relation.dst


def _find_contains_relation(src, dst):
    for relation in src._out.get("contains", ()):
        if relation.dst is dst:
            return relation
    return None


def _find_has_a_role(graph, concept, role_name: str):
    for relation in concept._out.get("has_a", ()):
        if _relation_role_name(relation) == role_name:
            return relation
    return None


def _relation_role_name(relation) -> str:
    return relation.name.split("-", 1)[0]


def _required_concept(graph, name: str):
    concept = graph.findConcept(name) if hasattr(graph, "findConcept") else None
    if concept is None:
        raise ValueError(f"planning graph is missing required concept {name!r}")
    return concept


def _optional_concept(graph, name: str):
    return graph.findConcept(name) if hasattr(graph, "findConcept") else None


def _all_concepts(graph) -> list[Any]:
    if hasattr(graph, "collectAllConcepts"):
        return list(graph.collectAllConcepts(include_supergraph=False, include_siblings=False).values())
    return list(getattr(graph, "_concepts", {}).values())


def _is_subconcept_of(concept, root) -> bool:
    seen = set()
    stack = [concept]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for relation in current._out.get("is_a", ()):
            parent = relation.dst
            if parent is root:
                return True
            stack.append(parent)
    return False


def _inheritance_depth(concept) -> int:
    if not concept._out.get("is_a"):
        return 0
    return 1 + max((_inheritance_depth(relation.dst) for relation in concept._out.get("is_a", ())), default=0)


def _position_index(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if not match:
        raise ValueError(f"reference position {name!r} must end with a numeric index")
    return int(match.group(1))


def _limit_value(name: str) -> int:
    match = re.fullmatch(r"max_(\d+)", name)
    if not match:
        raise ValueError(f"count limit {name!r} must be named like 'max_2'")
    return int(match.group(1))


def _phase_or_default(phase_names: Sequence[str], preferred: str, *, index: int) -> str:
    if preferred in phase_names:
        return preferred
    return phase_names[index]
