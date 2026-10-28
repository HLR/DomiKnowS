"""Grammar DFA plus causal knowledge over object ids, for the official protocol.

The official protocol shows no entity names, so the world graph's scene and task
rules cannot apply. The causal rules below need only the ids the plan itself
writes. Admission is compiled from the declarations in ``causal_graph.py``.
A pick (or a press) starts a subtask, since every benchmark pattern begins
with one; the picked object is then the subject of that subtask's destinations.

* ``anti_self_placement``: a destination (place, insert, pour, push) is never the held object;
* ``distinct_subtask_destinations``: one subtask never sends its object to the same destination twice;
* ``destinations_stay_put``: an object that has served as a destination is never picked later.

None of the 4,500 reference plans breaks one of them (checked on 27 October 2026).
The world graph's ``no_redundant_repeated_picks`` is left out: without task facts
no object is reusable, yet every ``take_chemistry_experiment`` reference picks
the same object twice.
"""
from __future__ import annotations

from functools import lru_cache

from domiknows.generation.dfa.predicate_monitor import PredicateEvent

try:
    from ..native_planning import NativePlanDFA
    from .causal_graph import build_causal_graph
except ImportError:  # pragma: no cover - flat-module execution
    from native_planning import NativePlanDFA
    from causal_graph import build_causal_graph

CAUSAL_RULES = ("anti_self_placement", "distinct_subtask_destinations", "destinations_stay_put")
SUBTASK_STARTS = frozenset({"pick", "press"})
DESTINATION_SKILLS = frozenset({"place", "insert", "pour", "push"})
NO_CAUSAL_STATE = (None, (), frozenset())  # (subject, subtask destinations, former destinations)


class CausalPlanDFA(NativePlanDFA):
    """``NativePlanDFA`` whose states also carry the causal state of the plan prefix.

    A state is the grammar state followed by ``(subject, destinations, former)``. An
    operation prefix is legal only while it can still end in an action the causal
    rules admit, so the rules prune ids as they are written and never leave a dead end.
    """

    def __init__(self, *args, causal_graph=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_graph = causal_graph if causal_graph is not None else build_causal_graph()
        self.causal_program = self.causal_graph.compile()
        self.start_state = self.start_state + NO_CAUSAL_STATE

    def causal_event(self, action, causal):
        """Observe this candidate against prior ids without deciding its legality."""
        skill, pointer = self.actions[action]
        subject, destinations, former = causal
        values = {"skill_" + name: skill == name for name in self.causal_graph.skills}
        values.update(
            target_is_subject=pointer == subject,
            target_used_in_subtask=pointer in destinations,
            target_was_destination=pointer in former,
        )
        return PredicateEvent("causal", values)

    def admits(self, action, causal) -> bool:
        return self.causal_program.check(self.causal_event(action, causal))

    def advance(self, action, causal):
        """Accumulate prefix observations after an admitted operation.

        Pick/press delimit subtasks in the official language. Local destination
        history resets there; plan-wide history persists. This stores facts,
        while ``causal_program`` decides which combinations are allowed.
        """
        skill, pointer = self.actions[action]
        subject, destinations, former = causal
        if skill in SUBTASK_STARTS:
            return (pointer if skill == "pick" else None), (), former
        if skill in DESTINATION_SKILLS:
            return subject, destinations + (pointer,), former | {pointer}
        return causal

    def operation_admitted(self, action, count, extras) -> bool:
        """Every compiled program admits ``action`` as operation ``count`` after a prefix with state ``extras``."""
        return self.admits(action, extras)

    def advance_state(self, action, count, extras):
        return self.advance(action, extras)

    def plan_complete(self, extras) -> bool:
        """The compiled plan-level formulas admit closing the plan here."""
        return True

    @lru_cache(maxsize=None)
    def reachable(self, node, prefixes, count, causal) -> bool:
        """Some action below operation-trie ``node`` is both grammatical here and admitted by the compiled rules."""
        skills = self.allowed_skills(prefixes, count)
        return any(self.actions[action][0] in skills and self.operation_admitted(action, count, causal)
                   for action in self.node_actions[node])

    def byte_step(self, state, byte):
        base, causal = state[:5], state[5:]
        following = super().byte_step(base, byte)
        if following is None:
            return None
        count, prefixes, phase, node, _ = base
        if following[0] > count:  # this byte closes an operation
            action = self.end_action[self.operations[node][byte]]
            if not self.operation_admitted(action, count, causal):
                return None
            return following + self.advance_state(action, count, causal)
        if phase == "separator" and following[2] == "closed" and not self.plan_complete(causal):
            return None
        if phase == "separator" and following[2] == "operation" and not self.reachable(0, prefixes, count, causal):
            return None  # no admitted operation can follow this comma
        if following[2] == "operation" and following[3] != 0 and following[3] != node:
            if not self.reachable(following[3], prefixes, count, causal):
                return None
        if following[2] == "operation" and following[4] == "ws" and node != 0:
            # Whitespace after an id leaves only the closing bytes; refuse it when they would be refused.
            extended = following + causal
            if not any(self.byte_step(extended, key) is not None for key in self.operations[node]):
                return None
        return following + causal

    @lru_cache(maxsize=None)
    def transitions(self, state):
        # The base class ends a plan with a five-field state; keep the causal fields.
        return tuple((label, target + state[5:] if len(target) == 5 else target)
                     for label, target in super().transitions(state))

    @lru_cache(maxsize=None)
    def effect_groups(self, state):
        """``NativePlanDFA.effect_groups`` over extended states (the grammar fields come first)."""
        count, _, phase, node, _ = state[:5]
        groups = {}
        for label, target in self.transitions(state):
            if label == self.eos_label:
                continue
            target_count, _, target_phase, target_node, _ = target[:5]
            key = None
            if target_count == count:
                if target_phase == "operation" and target_node != 0 and not (phase == "operation" and target_node == node):
                    key = ("action", target_node)
                elif phase == "separator":
                    key = {"operation": ("continue",), "closed": ("stop",)}.get(target_phase)
            elif target_count == count + 1:
                finished = self.completed_action(state, label)
                if finished is not None:
                    action, rest = finished
                    mode = "continue" if b"," in rest else "stop" if (b"]" in rest or b"\n" in rest) else "any"
                    key = ("done", action, mode)
            if key is not None:
                groups.setdefault(key, []).append(label)
        return {key: tuple(labels) for key, labels in groups.items()}
