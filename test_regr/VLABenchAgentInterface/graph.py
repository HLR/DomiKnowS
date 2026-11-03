"""The VLABench plan language in DomiKnowS concepts and logical constraints.

Plans concatenate complete subtasks. Each skill has ordered argument/object
pairs and an optional suffix. Auxiliary progress concepts describe where each
token belongs; compilation projects them away to obtain the token language.
"""
from __future__ import annotations

from domiknows.generation import GenerationEncoder, mark_for_dfa
from domiknows.graph import EnumConcept
from domiknows.graph.logicalConstrain import andL, atLeastAL, atMostAL, exactL, ifL, notL, orL

try:
    from .plan_codec import PlanVocabulary
except ImportError:
    from plan_codec import PlanVocabulary


def _any(predicates):
    predicates = tuple(predicates)
    return predicates[0] if len(predicates) == 1 else orL(*predicates)


def _declare_plan_grammar(bundle, vocabulary, language):
    """Describe positions within each subtask, without enumerating plans.

    Four formulas share the ``plan_progress`` positions; the compiler projects them jointly:

    - ``plan_progress_positions``: one position per token, the token each position allows, the legal starts;
    - ``plan_operation_arguments``: inside an operation, the skill, then argument name and entity per role;
    - ``plan_pattern_steps``: an operation end continues its optional roles or its pattern, or leaves it;
    - ``plan_only_completed_subtasks_end``: only completed subtasks end the plan or start another subtask.
    """
    ctx = bundle.context
    patterns = tuple(dict.fromkeys((*language.patterns, *(language.roots or ()))))
    roots = set(language.patterns if language.roots is None else language.roots)
    objects = tuple(f"obj:{index}" for index in range(vocabulary.max_entities))
    skill_arguments = vocabulary.skill_argument_map
    optional_arguments = dict(vocabulary.optional_skill_arguments)
    positions = {"finished": (vocabulary.eos_token,)}
    inside, ends = {}, {}  # successors of positions that do not / do end an operation
    starts, complete, repeating_starts = set(), set(), set()

    # One skill position and two positions per role: argument name and entity.
    # Optional roles form a suffix that may stop after any completed pair.
    for pattern_index, pattern in enumerate(patterns):
        previous_operation = set()
        for operation_index, skill in enumerate(pattern):
            prefix = f"subtask_{pattern_index}__step_{operation_index}__{skill}"
            skill_position = prefix + "__skill"
            positions[skill_position] = (f"skill:{skill}",)
            if operation_index == 0:
                if pattern in roots:
                    starts.add(skill_position)
                if language.repeat and pattern in language.patterns:
                    repeating_starts.add(skill_position)
            for previous in previous_operation:
                ends[previous].add(skill_position)
            required = tuple(skill_arguments[skill])
            optional = tuple(optional_arguments.get(skill, ()))
            if not objects:  # no object can fill a role: a skill with required roles never completes
                optional = ()
                if required:
                    inside[skill_position] = set()
                    previous_operation = set()
                    continue
            chain, operation_ends = [skill_position], set()
            for index, key in enumerate((*required, *optional)):
                if index >= len(required):
                    operation_ends.add(chain[-1])
                argument = f"{prefix}__argument_{index}__{key}"
                entity = f"{prefix}__entity_{index}__{key}"
                positions[argument], positions[entity] = (f"arg:{key}",), objects
                chain += [argument, entity]
            operation_ends.add(chain[-1])
            for position in chain:
                (ends if position in operation_ends else inside)[position] = set()
            for source, target in zip(chain, chain[1:]):
                (ends if source in operation_ends else inside)[source].add(target)
            previous_operation = operation_ends
        if pattern:
            complete.update(previous_operation)

    if language.allow_empty:
        starts.add("finished")
    leaving = {*repeating_starts, "finished"}
    progress = bundle.token(name="plan_progress", ConceptClass=EnumConcept, values=list(positions))

    def state(name, variable="position", path=None):
        return getattr(progress, name)(variable, **({"path": path} if path else {}))

    # Exactly one progress interpretation belongs to each token. Ambiguous
    # subtask prefixes retain multiple possible interpretations during decoding.
    rules = [exactL(*(state(name) for name in positions), 1)]
    for name, tokens in positions.items():
        rules.append(ifL(state(name), _any(ctx.token_value(token, "position") for token in tokens)))

    start_path = ("start", bundle.sequence_start_token)
    allowed_starts = [state(name, "first", start_path) for name in sorted(starts)]
    if not allowed_starts:
        # An empty language differs from a language containing an empty plan.
        first = state("finished", "first", start_path)
        allowed_starts = [andL(first, notL(first))]
    rules.append(ifL(ctx.sequence_start("start"), _any(allowed_starts)))
    declared = [mark_for_dfa(andL(*rules, name="plan_progress_positions"))]

    # For these local generation rules the ordering relation binds immediate successors.
    current = lambda name: state(name, "current", ("edge", bundle.first_token))
    following = lambda names: _any(state(name, "next", ("edge", bundle.second_token)) for name in sorted(names))

    def successors(table, name):
        """Each listed position is followed only by its declared successors (none: a dead end)."""
        rules = [ifL(current(source), following(targets) if targets else notL(current(source)))
                 for source, targets in table.items()]
        if rules:
            declared.append(mark_for_dfa(ifL(ctx.is_before_rel("edge"), andL(*rules), name=name)))

    successors(inside, "plan_operation_arguments")
    # An operation end continues with its optional roles or the pattern's next step, or leaves the subtask.
    successors({source: targets | leaving for source, targets in ends.items()}, "plan_pattern_steps")
    # Only completed subtasks may end the plan or start another subtask; an ended plan stays ended.
    declared.append(mark_for_dfa(ifL(ctx.is_before_rel("edge"), andL(
        ifL(following(leaving), _any(current(name) for name in sorted({*complete, "finished"}))),
        ifL(current("finished"), following({"finished"})),
    ), name="plan_only_completed_subtasks_end")))
    return tuple(declared)


def create_planner_generation_graph(
    world, vocabulary: PlanVocabulary | None = None, *, max_operations: int = 12,
    graph_name: str = "vlabench_planner_generation", skill_language=None,
):
    """Declare sequence boundaries, operation limits, and the plan grammar.

    ``skill_language`` supplies task-specific roots and empty-plan permission.
    These facts change declarations, never bypass compilation.
    """
    vocabulary = vocabulary or PlanVocabulary.from_world(world)
    if vocabulary.domain_checksum != world.domain_checksum:
        raise ValueError("generation vocabulary does not match the world graph")
    return create_plan_grammar_graph(vocabulary, world.subtask_patterns, max_operations=max_operations,
                                     graph_name=graph_name, skill_language=skill_language)


def create_plan_grammar_graph(
    vocabulary: PlanVocabulary, patterns, *, max_operations: int = 12,
    graph_name: str = "vlabench_plan_grammar", skill_language=None,
):
    """The plan grammar of one skill-argument table (``vocabulary``) and subtask-pattern list.

    The world graph's planner and the native JSON decoder (``NativePlanDFA``, one vocabulary
    per native profile) both decode with this declaration.
    """
    from domiknows.generation.dfa.pattern_language import BoundedPatternLanguage
    if max_operations < 0:
        raise ValueError("max_operations must be non-negative")
    language = skill_language
    if language is None:
        language = BoundedPatternLanguage(tuple(tuple(pattern) for pattern in patterns), max_operations)
    elif not isinstance(language, BoundedPatternLanguage):
        language = BoundedPatternLanguage(tuple(language), max_operations, repeat=False, allow_empty=() in language)
    graph, bundle = GenerationEncoder(
        vocab=list(vocabulary.tokens), eos_token=vocabulary.eos_token, graph_name=graph_name,
    ).build_graph()
    ctx = bundle.context
    with graph:
        # EOS must occur; every position after the first EOS is also EOS.
        mark_for_dfa(atLeastAL(ctx.token_value(vocabulary.eos_token, "position"), 1))
        mark_for_dfa(ifL(ctx.is_before_rel("before"), ifL(
            ctx.token_value(vocabulary.eos_token, "first", path=("before", ctx.first_token)),
            ctx.token_value(vocabulary.eos_token, "next", path=("before", ctx.second_token)),
        )))
        mark_for_dfa(atMostAL(notL(ctx.token_value(vocabulary.eos_token, "position")), max_operations * 5))
        mark_for_dfa(atMostAL(_any(
            ctx.token_value(f"skill:{skill}", "position") for skill in vocabulary.skills
        ), min(max_operations, language.maximum)))
        # planner_grammar declares the positions; the other rules constrain how they follow each other.
        bundle.planner_grammar_rules = _declare_plan_grammar(bundle, vocabulary, language)
        bundle.planner_grammar = bundle.planner_grammar_rules[0]
    return graph, bundle


def __getattr__(name):
    """Keep earlier imports working without runtime machinery in this file."""
    import importlib
    if name in {"plan_to_tokens", "tokens_to_plan", "labels_to_plan", "_pointer_index"}:
        module = "plan_codec"
    elif name in {"compile_planner_dfa", "bind_vlabench_contextual_dfa", "dfa_accepts_plan",
                  "_domain_dfa", "_legal_skill_sequences"}:
        module = "generation_runtime"
    elif name in {"COMPLEX_TASK_SCHEMAS", "ComplexTaskSchema"}:
        module = "task_policy"
    elif name in {"EOS_TOKEN", "canonicalize_plan", "validate_plan", "VLABENCH_DOMAIN_CHECKSUM",
                  "VLABenchWorldGraphBundle", "build_vlabench_world_graph"}:
        module = "world_graph"
    else:
        raise AttributeError(name)
    return getattr(importlib.import_module(f"{__package__}.{module}" if __package__ else module), name)
