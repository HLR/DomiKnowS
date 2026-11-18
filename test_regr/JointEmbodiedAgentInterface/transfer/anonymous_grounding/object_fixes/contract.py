"""Observation bindings for the DomiKnowS-compiled executor policy.

Token acceptance comes from compiled declarations and PredicateMonitorDFA.
This module contains no handwritten token filtering or legality guards.
"""
from dataclasses import dataclass, replace
from collections import Counter
from pathlib import Path
import hashlib
from domiknows.generation.dfa.predicate_monitor import (
    PredicateEvent, PredicateMonitorDFA, PredicateConstraintProgram, SemanticTransition,
)
from domiknows.generation.dfa.productive import DFACompletionBounds
from .executor_graph import CAPABILITIES, executor_declarations


def executor_policy_metadata():
    """Pin the declarative policy and compiler for provenance/resume checks."""
    from domiknows.generation.dfa import graph_discovery, predicate_monitor
    from test_regr.VLABenchAgentInterface import graph as grammar, generation_runtime
    files = [Path(__file__), Path(__file__).with_name('executor_graph.py'),
             Path(graph_discovery.__file__), Path(predicate_monitor.__file__),
             Path(grammar.__file__), Path(generation_runtime.__file__)]
    return {'implementation': ExecutorContract.implementation,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


@dataclass(frozen=True)
class ObservedState:
    grammar: object
    rule_states: tuple
    action: str | None = None
    held: int | None = None


class ExecutorObservations:
    """Interpret tokens and observed properties, never decide legality."""
    def __init__(self, declarations, vocabulary, public):
        self.declarations = declarations
        self.vocabulary = vocabulary
        self.entities = tuple(public['entities'])
        self.movable = frozenset(self.entities.index(n) for n in public['movable'])
        self.start_state = ObservedState(declarations.language.start,
            tuple(dfa.start_state for _, dfa, _ in declarations.grammar_rules))

    def step(self, state, label):
        language = self.declarations.language
        grammar = language.dfa.step(state.grammar, label)
        # This rejection is the verdict of compiled DomiKnowS declarations.
        if grammar not in language.live:
            return None
        token = self.vocabulary.token_for_label(label)
        rule_states = tuple(dfa.step(current, label) for current, (_, dfa, _) in
                            zip(state.rule_states, self.declarations.grammar_rules))
        successor = replace(state, grammar=grammar, rule_states=rule_states)
        events = ()
        if token.startswith('skill:'):
            successor = replace(successor, action=token[6:])
        elif token.startswith('obj:'):
            pointer = int(token[4:])
            events = (PredicateEvent('executor_operation', {
                'operation': True, 'pick': state.action == 'pick', 'place': state.action == 'place',
                'observed_target': pointer in range(len(self.entities)),
                'movable_target': pointer in self.movable,
                'subject_equals_destination': pointer == state.held,
            }),)
            successor = replace(successor, action=None, held=pointer if state.action == 'pick' else None)
        return SemanticTransition(successor, events)

    def is_accepting(self, state):
        return self.declarations.language.dfa.is_accepting(state.grammar)

    def acceptance_events(self, state):
        return ()  # termination and holding protocol are compiled grammar LCs


class ExecutorContract(PredicateMonitorDFA):
    """Shared compiled rules intersected with the compiled executor policy."""
    implementation = 'domiknows-compiled-executor-v1'

    def __init__(self, base, vocabulary, public, capabilities, max_operations=12):
        if capabilities != CAPABILITIES:
            raise ValueError('the declared executor interface is pick/place only')
        if (not public['entities'] or len(set(public['entities'])) != len(public['entities'])
            or len(public['entities']) > vocabulary.max_entities
            or not set(public['movable']) <= set(public['entities']) or max_operations < 2):
            raise ValueError('invalid observed entity/capability inventory')
        self.vocabulary = vocabulary
        self.max_operations = max_operations
        self.declarations = executor_declarations(vocabulary, max_operations)
        adapter = ExecutorObservations(self.declarations, vocabulary, public)
        bound = DFACompletionBounds.from_dfa(self.declarations.language.dfa)
        super().__init__(base, self.declarations.program, adapter,
                         completion_bound=lambda state: bound(state.semantic.grammar))
        self._rule_programs = tuple(PredicateConstraintProgram((rule,)) for rule in self.program.rules)

    def rejection_rules(self, state, label, prefix=()):
        """Diagnostics only: attribute rejections to compiled formulas."""
        transition = self.adapter.step(state.semantic, label)
        if transition is not None:
            return tuple(p.rules[0].name for p in self._rule_programs if not p.check_events(transition.events))
        reasons = []
        for current, (name, dfa, live) in zip(state.semantic.rule_states, self.declarations.grammar_rules):
            if dfa.step(current, label) not in live:
                reasons.append(name)
        return tuple(dict.fromkeys(reasons)) or ('executor_compiled_grammar',)

    def reason(self, state, label):
        reasons = self.rejection_rules(state, label)
        return reasons[0] if reasons else None

    def trace(self, labels):
        from ....generalization.shared_constraints import rule_trace
        result = rule_trace(self.base_dfa, labels)
        counts = Counter(); events = []; state = self.start_state; prefix = []
        for index, label in enumerate(labels):
            for candidate in self.base_dfa.allowed_tokens(state.grammar):
                reasons = self.rejection_rules(state, candidate, prefix)
                if reasons:
                    counts.update(reasons)
                    events.append({'step': index, 'candidate_label': int(candidate), 'rejected_by': list(reasons)})
            state = self.step(state, label)
            if state is None:
                break
            prefix.append(label)
        result.update(executor_rejections=dict(counts), executor_events=events,
                      executor_implementation=self.implementation,
                      executor_grammar_constraints=[name for name, _, _ in self.declarations.grammar_rules],
                      executor_semantic_constraints=list(self.program.coverage.compiled))
        return result
