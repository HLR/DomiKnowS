"""Operation progress projected from the declared generation formulas.

Argument labels are existentially hidden, not validated a second time. An
operation advances a compiled state; EOS availability means subtask completion.
No skill-pair table or independent subtask interpreter defines this language.
"""
from collections import deque
from dataclasses import replace
from functools import lru_cache
from itertools import count
from types import SimpleNamespace

from domiknows.generation.dfa.core import DFA

_BUILD_IDS = count()


def project_skill_progress(grammar, vocabulary):
    """Hide argument/object symbols and retain skills and legal EOS boundaries."""
    skill_labels = {vocabulary.label_for_token(f"skill:{skill}"): skill for skill in vocabulary.skills}
    hidden = {vocabulary.label_for_token(token) for token in vocabulary.tokens
              if token.startswith(("arg:", "obj:"))}
    eos = vocabulary.eos_label
    # Discard prefixes with no possible completion. Merely appearing in a local
    # progress formula does not guarantee that a state can ever reach EOS.
    eos_predecessors = {}
    for state in grammar.states:
        target = grammar.step(state, eos)
        if target is not None:
            eos_predecessors.setdefault(target, set()).add(state)
    terminating = set(grammar.accepting_states) - set(grammar.dead_states)
    pending = list(terminating)
    while pending:
        for source in eos_predecessors.get(pending.pop(), ()):
            if source not in terminating and source not in grammar.dead_states:
                terminating.add(source)
                pending.append(source)
    complete = {state for state in grammar.states if grammar.step(state, eos) in terminating}
    reverse = {}
    for (source, label), target in grammar.transitions.items():
        if label in hidden or label in skill_labels:
            reverse.setdefault(target, set()).add(source)
    productive = set(complete)
    queue = deque(complete)
    while queue:
        for source in reverse.get(queue.popleft(), ()):
            if source not in productive:
                productive.add(source)
                queue.append(source)

    def closure(seeds):
        reached = set(seeds) & productive
        pending = list(reached)
        while pending:
            source = pending.pop()
            for label in hidden:
                target = grammar.step(source, label)
                if target in productive and target not in reached:
                    reached.add(target)
                    pending.append(target)
        return frozenset(reached)

    start = closure((grammar.start_state,))
    states, accepting, transitions = {start}, set(), {}
    queue = deque((start,))
    while queue:
        source = queue.popleft()
        if source & complete:
            accepting.add(source)
        for label, skill in skill_labels.items():
            target = closure(grammar.step(state, label) for state in source)
            if not target:
                continue
            transitions[source, skill] = target
            if target not in states:
                states.add(target)
                queue.append(target)
    return DFA(frozenset(states), frozenset(skill_labels.values()), transitions,
               start, frozenset(accepting))


def progress_from_declarations(graph, bundle, vocabulary):
    """Compile the actual formula; its name carries no grammar semantics."""
    from domiknows.generation.dfa.auxiliary import compile_auxiliary_state_family
    if not any(rule is bundle.planner_grammar and rule.headLC and rule.active
               for rule in graph.logicalConstrains.values()):
        from domiknows.generation.dfa._constraints import accept_all_dfa
        return project_skill_progress(accept_all_dfa(bundle.vocabulary), vocabulary)
    compiled = compile_auxiliary_state_family(bundle.planner_grammar, bundle)
    if compiled is None:
        raise ValueError("planner progress must be declared as an auxiliary-state formula")
    return project_skill_progress(compiled, vocabulary)


@lru_cache(maxsize=128)
def _cached_progress(patterns, arguments, optional, checksum, language, builder, declaration):
    try:
        from .plan_codec import PlanVocabulary
    except ImportError:
        from plan_codec import PlanVocabulary
    world = SimpleNamespace(subtask_patterns=patterns, skill_arguments=dict(arguments),
                            optional_skill_arguments=dict(optional), domain_checksum=checksum,
                            skills=dict(arguments))
    vocabulary = PlanVocabulary.from_world(world, max_entities=1)
    graph, bundle = builder(world, vocabulary, max_operations=0, skill_language=language,
                            graph_name=f"vlabench_progress_{next(_BUILD_IDS)}")
    return progress_from_declarations(graph, bundle, vocabulary)


def pattern_progress_for(world, requirements):
    """Reuse a structural compilation, independent of scene identities/counts."""
    import importlib
    declarations = importlib.import_module(f"{__package__}.graph" if __package__ else "graph")
    try:
        from .task_policy import request_sequences
    except ImportError:
        from task_policy import request_sequences
    language = requirements.allowed_sequences
    if language is None:
        language = request_sequences(world, 0, requirements)
    # Only the progress formula is projected here. Operation/token budgets are
    # separate generation constraints and do not redefine subtask membership.
    from domiknows.generation.dfa.pattern_language import BoundedPatternLanguage
    if not isinstance(language, BoundedPatternLanguage):
        language = BoundedPatternLanguage(tuple(language), 0, repeat=False, allow_empty=() in language)
    language = replace(language, maximum=0)
    return _cached_progress(tuple(world.subtask_patterns), tuple(sorted(world.skill_arguments.items())),
                            tuple(sorted(world.optional_skill_arguments.items())), world.domain_checksum,
                            language, declarations.create_planner_generation_graph,
                            declarations._declare_plan_grammar)


def trace_progress(progress, skills):
    """Return each prefix's reachability and the final completion observation."""
    state = progress.start_state
    viable = []
    for skill in skills:
        state = None if state is None else progress.step(state, skill)
        viable.append(state is not None)
    return tuple(viable), state is not None and progress.is_accepting(state)
