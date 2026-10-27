"""Grammar DFA plus causal knowledge over object ids, for the official protocol.

The official protocol shows no entity names, so the world graph's scene and task
rules cannot apply. The causal rules below need only the ids the plan itself
writes. A pick (or a press) starts a subtask, since every benchmark pattern begins
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

try:
    from ..native_planning import NativePlanDFA
except ImportError:  # pragma: no cover - flat-module execution
    from native_planning import NativePlanDFA

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_state = self.start_state + NO_CAUSAL_STATE

    def admits(self, action, causal) -> bool:
        skill, pointer = self.actions[action]
        subject, destinations, former = causal
        if skill == "pick":
            return pointer not in former
        if skill in DESTINATION_SKILLS:
            return pointer != subject and pointer not in destinations
        return True

    def advance(self, action, causal):
        skill, pointer = self.actions[action]
        subject, destinations, former = causal
        if skill in SUBTASK_STARTS:
            return (pointer if skill == "pick" else None), (), former
        if skill in DESTINATION_SKILLS:
            return subject, destinations + (pointer,), former | {pointer}
        return causal

    @lru_cache(maxsize=None)
    def reachable(self, node, prefixes, count, causal) -> bool:
        """Some action below operation-trie ``node`` is both grammatical here and causally admitted."""
        skills = self.allowed_skills(prefixes, count)
        return any(self.actions[action][0] in skills and self.admits(action, causal)
                   for action in self.node_actions[node])

    def byte_step(self, state, byte):
        base, causal = state[:5], state[5:]
        following = super().byte_step(base, byte)
        if following is None:
            return None
        count, prefixes, phase, node, _ = base
        if following[0] > count:  # this byte closes an operation
            action = self.end_action[self.operations[node][byte]]
            if not self.admits(action, causal):
                return None
            return following + self.advance(action, causal)
        if following[2] == "operation" and following[3] != 0 and following[3] != node:
            if not self.reachable(following[3], prefixes, count, causal):
                return None
        return following + causal

    @lru_cache(maxsize=None)
    def transitions(self, state):
        # The base class ends a plan with a five-field state; keep the causal fields.
        return tuple((label, target + state[5:] if len(target) == 5 else target)
                     for label, target in super().transitions(state))

    def effect_groups(self, state):
        raise NotImplementedError("the action-level HMM lookahead is not defined over causal states")
