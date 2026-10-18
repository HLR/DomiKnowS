"""Generation vocabulary and DFA derived from the VLABench world graph."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .world_graph import (
        EOS_TOKEN,
        VLABENCH_DOMAIN_CHECKSUM,
        VLABenchWorldGraphBundle,
        build_vlabench_world_graph,
        canonicalize_plan,
        validate_plan,
    )
except ImportError:
    from world_graph import (
        EOS_TOKEN,
        VLABENCH_DOMAIN_CHECKSUM,
        VLABenchWorldGraphBundle,
        build_vlabench_world_graph,
        canonicalize_plan,
        validate_plan,
    )


try:
    from .task_policy import COMPLEX_TASK_SCHEMAS, ComplexTaskSchema
except ImportError:
    from task_policy import COMPLEX_TASK_SCHEMAS, ComplexTaskSchema


_DEFAULT_WORLD_IDS = count()


@dataclass(frozen=True)
class PlanVocabulary:
    """Compact planner labels derived from one world-domain definition."""

    skills: tuple[str, ...]
    argument_keys: tuple[str, ...]
    skill_arguments: tuple[tuple[str, tuple[str, ...]], ...]
    max_entities: int = 64
    eos_token: str = EOS_TOKEN
    domain_checksum: str = VLABENCH_DOMAIN_CHECKSUM
    version: int = 2

    @property
    def tokens(self) -> tuple[str, ...]:
        return (
            self.eos_token,
            *(f"skill:{name}" for name in self.skills),
            *(f"arg:{key}" for key in self.argument_keys),
            *(f"obj:{index}" for index in range(self.max_entities)),
        )

    @property
    def skill_argument_map(self) -> Mapping[str, tuple[str, ...]]:
        return {name: tuple(arguments) for name, arguments in self.skill_arguments}

    @property
    def eos_label(self) -> int:
        return 0

    @property
    def label_count(self) -> int:
        # DomiKnowS TokenVocabulary reserves one compact label for unknown
        # tokenizer output even though it is not part of the declared tokens.
        return len(self.tokens) + 1

    @property
    def other_label(self) -> int:
        return len(self.tokens)

    def label_for_token(self, token: str) -> int:
        try:
            return self.tokens.index(str(token))
        except ValueError as exc:
            raise KeyError(token) from exc

    def token_for_label(self, label: int) -> str:
        if int(label) == self.other_label:
            return "<other>"
        try:
            return self.tokens[int(label)]
        except (IndexError, TypeError, ValueError) as exc:
            raise KeyError(label) from exc

    @property
    def checksum(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_world(cls, world: VLABenchWorldGraphBundle, max_entities: int = 64) -> "PlanVocabulary":
        if max_entities <= 0:
            raise ValueError("max_entities must be positive")
        signatures = tuple((name, tuple(world.skill_arguments[name])) for name in sorted(world.skills))
        return cls(
            skills=tuple(name for name, _keys in signatures),
            argument_keys=tuple(sorted({key for _name, keys in signatures for key in keys})),
            skill_arguments=signatures,
            max_entities=int(max_entities),
            domain_checksum=world.domain_checksum,
        )

    @classmethod
    def default(cls, max_entities: int = 64) -> "PlanVocabulary":
        world = build_vlabench_world_graph(
            f"vlabench_default_vocabulary_world_{next(_DEFAULT_WORLD_IDS)}"
        )
        return cls.from_world(world, max_entities=max_entities)

    @classmethod
    def from_plans(
        cls,
        plans: Iterable[Any],
        world: VLABenchWorldGraphBundle,
        max_entities: int = 64,
    ) -> "PlanVocabulary":
        # Dataset plans validate the graph definition; they never redefine it.
        for value in plans:
            result = validate_plan(
                value,
                skill_arguments=world.skill_arguments,
                patterns=world.subtask_patterns,
                require_pattern=True,
            )
            if not result.valid:
                raise ValueError("plan violates the world domain: " + "; ".join(result.errors))
        return cls.from_world(world, max_entities=max_entities)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**asdict(self), "checksum": self.checksum}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PlanVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        checksum = payload.pop("checksum", None)
        payload["skills"] = tuple(payload["skills"])
        payload["argument_keys"] = tuple(payload["argument_keys"])
        payload["skill_arguments"] = tuple(
            (str(name), tuple(arguments)) for name, arguments in payload["skill_arguments"]
        )
        value = cls(**payload)
        if checksum is not None and checksum != value.checksum:
            raise ValueError(f"vocabulary checksum mismatch for {path}")
        if value.domain_checksum != VLABENCH_DOMAIN_CHECKSUM:
            raise ValueError("vocabulary domain checksum differs from the current world graph")
        return value


def _pointer_index(value: Any, entity_table: Mapping[int, str] | Sequence[str] | None) -> int:
    if isinstance(value, int) or str(value).isdigit():
        return int(value)
    if entity_table is None:
        raise ValueError(f"named entity {value!r} requires an entity table")
    items = entity_table.items() if isinstance(entity_table, Mapping) else enumerate(entity_table)
    for index, name in items:
        if str(name) == str(value):
            return int(index)
    raise ValueError(f"entity {value!r} is absent from the entity table")


def plan_to_tokens(value: Any, entity_table=None, *, world: VLABenchWorldGraphBundle) -> list[str]:
    skill_arguments = world.skill_arguments
    tokens: list[str] = []
    for operation in canonicalize_plan(value):
        tokens.append(f"skill:{operation['name']}")
        for key in skill_arguments.get(operation["name"], ()):
            tokens.extend((f"arg:{key}", f"obj:{_pointer_index(operation['params'][key], entity_table)}"))
    tokens.append(EOS_TOKEN)
    return tokens


def tokens_to_plan(tokens: Iterable[str], *, world: VLABenchWorldGraphBundle | None = None) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_key: str | None = None
    for raw in tokens:
        token = str(raw)
        if token == EOS_TOKEN:
            break
        if token.startswith("skill:"):
            current = {"name": token[6:], "params": {}}
            plan.append(current)
            pending_key = None
        elif token.startswith("arg:"):
            pending_key = token[4:]
        elif token.startswith("obj:") and current is not None and pending_key:
            pointer = token[4:]
            current["params"][pending_key] = int(pointer) if pointer.isdigit() else pointer
            pending_key = None
    return plan


def labels_to_plan(labels: Iterable[int], vocabulary: PlanVocabulary, *, world=None) -> list[dict[str, Any]]:
    return tokens_to_plan((vocabulary.token_for_label(label) for label in labels), world=world)


def create_planner_generation_graph(
    world: VLABenchWorldGraphBundle,
    vocabulary: PlanVocabulary | None = None,
    *,
    max_operations: int = 8,
    graph_name: str = "vlabench_planner_generation",
):
    """Build the token graph after and from the semantic world graph."""
    from domiknows.generation import GenerationEncoder, mark_for_dfa
    from domiknows.graph.logicalConstrain import atMostAL, ifL, notL

    vocabulary = vocabulary or PlanVocabulary.from_world(world)
    if vocabulary.domain_checksum != world.domain_checksum:
        raise ValueError("generation vocabulary does not match the world graph")
    encoder = GenerationEncoder(vocab=list(vocabulary.tokens), eos_token=vocabulary.eos_token, graph_name=graph_name)
    graph, bundle = encoder.build_graph()
    ctx = bundle.context
    with graph:
        mark_for_dfa(ifL(
            ctx.is_before_rel("before"),
            ifL(
                ctx.token_value(vocabulary.eos_token, "x", path=("before", ctx.first_token)),
                ctx.token_value(vocabulary.eos_token, "y", path=("before", ctx.second_token)),
            ),
        ))
        mark_for_dfa(atMostAL(notL(ctx.token_value(vocabulary.eos_token, "x")), max_operations * 5))
    return graph, bundle


def _legal_skill_sequences(world: VLABenchWorldGraphBundle, max_operations: int):
    complete: set[tuple[str, ...]] = set()
    queue = deque([()])
    visited = {()}
    while queue:
        prefix = queue.popleft()
        for pattern in world.subtask_patterns:
            candidate = prefix + tuple(pattern)
            if len(candidate) > max_operations:
                continue
            complete.add(candidate)
            if candidate not in visited:
                visited.add(candidate)
                queue.append(candidate)
    prefixes = {sequence[:index] for sequence in complete for index in range(len(sequence) + 1)}
    return complete, prefixes


def _domain_dfa(world: VLABenchWorldGraphBundle, vocabulary: PlanVocabulary, max_operations: int, complete_sequences=None):
    from domiknows.generation.dfa.core import DFA

    complete, skill_prefixes = _legal_skill_sequences(world, max_operations)
    if complete_sequences is not None:
        complete = set(complete_sequences)
        skill_prefixes = {seq[:i] for seq in complete for i in range(len(seq)+1)}
    alphabet = frozenset(range(vocabulary.label_count))
    start = ((), None, 0, "skill", False)

    def successors(state):
        skills, current, argument_index, phase, ended = state
        if ended:
            yield vocabulary.eos_label, state
        elif phase == "skill":
            if skills in complete:
                yield vocabulary.eos_label, (skills, None, 0, "skill", True)
            for skill in vocabulary.skills:
                next_skills = skills + (skill,)
                if next_skills not in skill_prefixes:
                    continue
                arguments = world.skill_arguments[skill]
                yield vocabulary.label_for_token(f"skill:{skill}"), (
                    next_skills,
                    skill if arguments else None,
                    0,
                    "arg" if arguments else "skill",
                    False,
                )
        elif phase == "arg":
            key = world.skill_arguments[current][argument_index]
            yield vocabulary.label_for_token(f"arg:{key}"), (skills, current, argument_index, "obj", False)
        elif phase == "obj":
            arguments = world.skill_arguments[current]
            last = argument_index + 1 >= len(arguments)
            target = (skills, None if last else current, 0 if last else argument_index + 1, "skill" if last else "arg", False)
            for pointer in range(vocabulary.max_entities):
                yield vocabulary.label_for_token(f"obj:{pointer}"), target

    states = {start}
    transitions = {}
    accepting = set()
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state[-1]:
            accepting.add(state)
        for symbol, target in successors(state):
            transitions[(state, symbol)] = target
            if target not in states:
                states.add(target)
                queue.append(target)
    return DFA(
        states=frozenset(states),
        alphabet=alphabet,
        transitions=transitions,
        start_state=start,
        accepting_states=frozenset(accepting),
    )


def compile_planner_dfa(graph, generation_bundle, world, vocabulary, *, max_operations: int = 8):
    """Intersect graph LCs with the complete language declared by the world."""
    from domiknows.generation import constraints_to_dfa_from_graph
    from domiknows.generation.dfa.core import product_dfa

    graph_dfa = constraints_to_dfa_from_graph(graph, generation_bundle, on_unsupported="error")
    try:
        from .constraint_compiler import CompiledPlannerDFA, compile_world_constraints
    except ImportError:
        from constraint_compiler import CompiledPlannerDFA, compile_world_constraints
    program = compile_world_constraints(world)
    operation_dfa = _domain_dfa(world, vocabulary, max_operations)
    grammar = product_dfa([graph_dfa, operation_dfa])
    return CompiledPlannerDFA.from_dfa(grammar, program, world_constraint_bundle=world, token_dfa=graph_dfa,
                                     max_operations=max_operations, operation_dfa=operation_dfa)


def dfa_accepts_plan(dfa, bundle, plan, entity_table=None, *, world=None) -> bool:
    try:
        labels = [bundle.vocabulary.label_for_token(token) for token in plan_to_tokens(plan, entity_table, world=world)]
    except (KeyError, ValueError):
        return False
    if entity_table is not None:
        dfa = bind_vlabench_contextual_dfa(dfa, bundle.vocabulary, {"entities": entity_table})
    return bool(dfa.accepts(labels))


def bind_vlabench_contextual_dfa(base_dfa, vocabulary, context):
    """Bind explicit task and scene facts to compiled world constraints."""
    try:
        from .constraint_compiler import WorldConstraintDFA
        from .task_policy import task_requirements
    except ImportError:
        from constraint_compiler import WorldConstraintDFA
        from task_policy import task_requirements
    from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor
    if find_predicate_monitor(base_dfa) is not None:
        # Retain the exact bound policy for sampling and replay.
        return base_dfa
    if not hasattr(base_dfa, "constraint_program"):
        raise TypeError("VLABench binding requires a compiled world constraint program")
    get = context.get if isinstance(context, Mapping) else lambda k, default=None: getattr(context, k, default)
    entities = get("entity_table") or get("entities") or ()
    from dataclasses import replace
    from domiknows.generation.dfa.core import product_dfa
    try:
        from .task_policy import request_sequences
    except ImportError:
        from task_policy import request_sequences
    world = base_dfa.world_constraint_bundle
    requirements = task_requirements(context, world, entities)
    sequences = request_sequences(world, base_dfa.max_operations, requirements)
    requirements = replace(requirements, allowed_sequences=sequences)
    grammar = base_dfa
    if requirements.planning.completed_skill_prefix or requirements.planning.supplied:
        operation_dfa = _domain_dfa(world, vocabulary, base_dfa.max_operations, sequences)
        grammar = type(base_dfa).from_dfa(
            product_dfa([base_dfa.token_dfa, operation_dfa]), base_dfa.constraint_program,
            world_constraint_bundle=world, token_dfa=base_dfa.token_dfa,
            max_operations=base_dfa.max_operations, operation_dfa=operation_dfa,
        )
    return WorldConstraintDFA(grammar, vocabulary, world, base_dfa.constraint_program, entities, requirements=requirements)
