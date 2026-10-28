"""Plan DFA whose episode knowledge comes only from compiled logical constraints.

``KnowledgePlanDFA`` grounds the observations of ``decoding_graph.py`` (the model's
answers, the operation budget, the instruction's candidates) and, unless
``causal=False``, those of ``causal_graph.py``. Legality is decided by the compiled
programs alone; the episode facts never restrict the grammar directly.
"""
from __future__ import annotations

from functools import lru_cache

from domiknows.generation.dfa.predicate_monitor import PredicateConstraintProgram, PredicateEvent

try:
    from ..object_grounding import ObjectConstraints
    from .causal_dfa import CausalPlanDFA
    from .decoding_graph import build_decoding_graph
except ImportError:  # pragma: no cover - flat-module execution
    from object_grounding import ObjectConstraints
    from causal_dfa import CausalPlanDFA
    from decoding_graph import build_decoding_graph

START_PATTERN_PREFIXES = ((),)


class KnowledgePlanDFA(CausalPlanDFA):
    """A state is the grammar state, the causal state, then the named-pattern prefixes.

    ``press_answer``: the model's Yes/No to the press question. ``scene_bound``: the
    largest tag number plus one, or ``None`` when it was not read. ``chosen_patterns``:
    the subtask patterns the model named (empty: none). ``object_candidates``: the
    instruction's grounded ``ObjectConstraints``.
    """

    def __init__(self, *args, causal=True, causal_graph=None, decoding_graph=None, press_answer=False,
                 scene_bound=None, chosen_patterns=(), object_candidates=None, **kwargs):
        super().__init__(*args, causal_graph=causal_graph, **kwargs)
        if not causal:
            self.causal_program = PredicateConstraintProgram()
        self.decoding_graph = decoding_graph if decoding_graph is not None else build_decoding_graph()
        self.decoding_program = self.decoding_graph.compile()
        self.press_answer = bool(press_answer)
        self.scene_bound = scene_bound
        self.chosen_patterns = tuple(tuple(pattern) for pattern in chosen_patterns)
        self.object_candidates = object_candidates or ObjectConstraints()
        self.start_state = self.start_state + (START_PATTERN_PREFIXES,)

    @lru_cache(maxsize=None)
    def next_chosen(self, prefixes, skill):
        """``next_semantics`` over the named patterns only."""
        following = set()
        for prefix in prefixes:
            candidate = prefix + (skill,)
            for pattern in self.chosen_patterns:
                if pattern[:len(candidate)] == candidate:
                    following.add(() if len(candidate) == len(pattern) else candidate)
        return tuple(sorted(following))

    @lru_cache(maxsize=None)
    def continues_chosen(self, prefixes, skill, count) -> bool:
        remaining = self.max_operations - count - 1
        return any(prefix == () or any(pattern[:len(prefix)] == prefix and len(pattern) - len(prefix) <= remaining
                                       for pattern in self.chosen_patterns)
                   for prefix in self.next_chosen(prefixes, skill))

    def operation_event(self, action, count, extras):
        """Observe this candidate operation; the compiled formulas decide whether it is allowed."""
        skill, pointer = self.actions[action]
        roles = self.skill_arguments[skill]
        candidates = next((ids for s, r, ids in self.object_candidates.restrictions
                           if roles and (s, r) == (skill, roles[0])), None)
        return PredicateEvent("operation", {
            "press": skill == "press",
            "first_operation": count == 0,
            "press_answer_yes": self.press_answer,
            "has_target": pointer is not None,
            "scene_bound_read": self.scene_bound is not None,
            "target_within_scene_bound": pointer is not None and self.scene_bound is not None and pointer < self.scene_bound,
            "patterns_chosen": bool(self.chosen_patterns),
            "continues_chosen_pattern": bool(self.chosen_patterns) and self.continues_chosen(extras[3], skill, count),
            "instruction_candidates_known": candidates is not None,
            "target_is_instruction_candidate": candidates is not None and pointer in candidates,
        })

    def operation_admitted(self, action, count, extras) -> bool:
        return (self.admits(action, extras[:3])
                and self.decoding_program.check(self.operation_event(action, count, extras)))

    def advance_state(self, action, count, extras):
        skill = self.actions[action][0]
        chosen = self.next_chosen(extras[3], skill) if self.chosen_patterns else extras[3]
        return self.advance(action, extras[:3]) + (chosen,)

    def plan_complete(self, extras) -> bool:
        return self.decoding_program.check(PredicateEvent("plan", {
            "plan_patterns_chosen": bool(self.chosen_patterns),
            "chosen_patterns_complete": () in extras[3],
        }))
