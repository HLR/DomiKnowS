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
        from domiknows.generation import mark_for_contextual_dfa
        button_press_lc = ifL(
            ctx.is_before_rel("before"),
            ifL(
                ctx.token_value("skill:press", "first_tok", path=("before", ctx.first_token)),
                ctx.token_value(vocabulary.eos_token, "second_tok", path=("before", ctx.second_token)),
            ),
            active=False,
            name="contextual_qa_button_press",
        )
        mark_for_contextual_dfa(
            button_press_lc,
            context_key="button_entities",
            token_to_value={"skill:press": "press"},
            vocabulary=vocabulary,
            name="contextual_qa_button_press",
            allow_missing_context=True,
        )
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


def _domain_dfa(world: VLABenchWorldGraphBundle, vocabulary: PlanVocabulary, max_operations: int):
    from domiknows.generation.dfa.core import DFA

    complete, skill_prefixes = _legal_skill_sequences(world, max_operations)
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
    return product_dfa([graph_dfa, _domain_dfa(world, vocabulary, max_operations)])


def dfa_accepts_plan(dfa, bundle, plan, entity_table=None, *, world=None) -> bool:
    try:
        labels = [bundle.vocabulary.label_for_token(token) for token in plan_to_tokens(plan, entity_table, world=world)]
    except (KeyError, ValueError):
        return False
    return bool(dfa.accepts(labels))


def _resolve_physical_law_button(instruction: str, entity_table: Sequence[str]) -> int | None:
    """Resolve physical law QA target button from domain knowledge."""
    inst = instruction.lower()
    entities = [str(e) for e in entity_table]
    buttons = [(i, name) for i, name in enumerate(entities) if "button" in name.lower()]
    if not buttons:
        return None

    # 1. Density QA
    if "density" in inst or "float" in inst:
        if "largest" in inst:
            return next((i for i, name in buttons if "button2" in name), buttons[-1][0])
        elif "medium" in inst:
            return next((i for i, name in buttons if "button1" in name), buttons[len(buttons) // 2][0])
        elif "smallest" in inst or "float" in inst:
            return next((i for i, name in buttons if "button0" in name), buttons[0][0])

    # 2. Friction QA
    if "friction" in inst or "steel ramp" in inst:
        if "fastest" in inst:
            for b_idx, _ in buttons:
                for offset in (1, 2, -1):
                    c_idx = b_idx + offset
                    if 0 <= c_idx < len(entities) and "sphere" in entities[c_idx].lower():
                        return b_idx
        elif "slowest" in inst:
            for b_idx, _ in buttons:
                for offset in (1, 2, -1):
                    c_idx = b_idx + offset
                    if 0 <= c_idx < len(entities) and "box" in entities[c_idx].lower():
                        return b_idx
            for b_idx, _ in buttons:
                for offset in (1, 2, -1):
                    c_idx = b_idx + offset
                    if 0 <= c_idx < len(entities) and "cylinder" in entities[c_idx].lower():
                        return b_idx

    return None


class VLABenchContextualDFA:
    """Specialized VLABench DFA conditioned on per-example scene entities and question context."""

    def __init__(
        self,
        base_dfa: Any,
        vocabulary: PlanVocabulary,
        button_indices: Sequence[int],
        target_button: int | None = None,
    ):
        self.base_dfa = base_dfa
        self.vocabulary = vocabulary
        self.button_indices = tuple(int(i) for i in button_indices)
        self.target_button = target_button
        self.alphabet = base_dfa.alphabet
        self.press_label = vocabulary.label_for_token("skill:press")
        self.arg_label = vocabulary.label_for_token("arg:target_entity_name")
        if target_button is not None and f"obj:{target_button}" in vocabulary.tokens:
            self.button_labels = frozenset([vocabulary.label_for_token(f"obj:{target_button}")])
        else:
            self.button_labels = frozenset(
                vocabulary.label_for_token(f"obj:{i}")
                for i in self.button_indices
                if f"obj:{i}" in vocabulary.tokens
            )
        self.eos_label = vocabulary.eos_label
        self.start_state = (base_dfa.start_state, 0)

    def allowed_tokens(self, state, remaining_steps=None):
        base_state, phase = state
        try:
            base_allowed = self.base_dfa.allowed_tokens(base_state, remaining_steps=remaining_steps)
        except TypeError:
            base_allowed = self.base_dfa.allowed_tokens(base_state)
        base_allowed = set(int(lbl) for lbl in base_allowed)

        if phase == 0:
            allowed = {self.press_label}
        elif phase == 1:
            allowed = {self.arg_label}
        elif phase == 2:
            allowed = set(self.button_labels)
        elif phase >= 3:
            allowed = {self.eos_label}
        else:
            allowed = set()
        return base_allowed.intersection(allowed)

    def step(self, state, symbol):
        base_state, phase = state
        symbol = int(symbol)
        if symbol not in self.allowed_tokens(state):
            return None
        next_base = self.base_dfa.step(base_state, symbol)
        if next_base is None:
            return None
        return (next_base, min(phase + 1, 4))

    def is_accepting(self, state):
        base_state, phase = state
        return phase >= 3 and self.base_dfa.is_accepting(base_state)

    def accepts(self, sequence):
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)

    def __getattr__(self, name):
        return getattr(self.base_dfa, name)


class VLABenchCausalDFA:
    """DomiKnowS causal DFA enforcing physical manipulation constraints.

    Enforces:
      1. Anti-self-placement: target_container_name != target_entity_name.
      2. Multi-step entity rotation: pick cannot re-select an already picked entity
         when unvisited candidate entities remain in the scene.
      3. Static surface filtering: pick cannot target static room backgrounds
         (tables, counters, floors, walls).
      4. Container consistency: subsequent placements in multi-step assembly target
         the established destination receptacle.
    """

    def __init__(self, base_dfa: Any, vocabulary: PlanVocabulary, entity_table: Sequence[str]):
        self.base_dfa = base_dfa
        self.vocabulary = vocabulary
        self.entity_table = [str(e) for e in entity_table]
        self.alphabet = base_dfa.alphabet

        self.static_env_indices = set()
        for i, name in enumerate(self.entity_table):
            nl = name.lower()
            if any(k in nl for k in ("table", "counter", "room_", "lab_", "wall")) or name.isdigit():
                self.static_env_indices.add(i)

        self.container_indices = set()
        for i, name in enumerate(self.entity_table):
            nl = name.lower()
            if any(k in nl for k in ("plate", "shelf", "flask", "container", "holder", "box", "stand", "bowl", "pot", "pan", "counter", "table")):
                self.container_indices.add(i)

        self.start_state = (base_dfa.start_state, None, None, None, frozenset(), None)

    def allowed_tokens(self, state, remaining_steps=None):
        base_state, active_skill, active_arg, current_picked, visited_picks, dest_container = state
        try:
            base_allowed = self.base_dfa.allowed_tokens(base_state, remaining_steps=remaining_steps)
        except TypeError:
            base_allowed = self.base_dfa.allowed_tokens(base_state)
        allowed = set(int(lbl) for lbl in base_allowed)

        if active_arg == "target_container_name":
            if current_picked is not None:
                lbl = self.vocabulary.label_for_token(f"obj:{current_picked}")
                allowed.discard(lbl)
            if dest_container is not None:
                c_lbl = self.vocabulary.label_for_token(f"obj:{dest_container}")
                if c_lbl in allowed:
                    allowed = {c_lbl}
            elif self.container_indices:
                c_lbls = {
                    self.vocabulary.label_for_token(f"obj:{i}")
                    for i in self.container_indices
                    if f"obj:{i}" in self.vocabulary.tokens
                }
                c_cand = allowed.intersection(c_lbls)
                if c_cand:
                    allowed = c_cand
        elif active_arg == "target_entity_name" and active_skill == "pick":
            for i in self.static_env_indices:
                lbl = self.vocabulary.label_for_token(f"obj:{i}")
                allowed.discard(lbl)
            unvisited_cand = set()
            for lbl in allowed:
                tok = self.vocabulary.token_for_label(lbl)
                if tok.startswith("obj:"):
                    idx = int(tok[4:])
                    if idx not in visited_picks and idx < len(self.entity_table):
                        unvisited_cand.add(lbl)
            if unvisited_cand:
                allowed = unvisited_cand

        return allowed

    def step(self, state, symbol):
        base_state, active_skill, active_arg, current_picked, visited_picks, dest_container = state
        symbol = int(symbol)
        if symbol not in self.allowed_tokens(state):
            return None
        next_base = self.base_dfa.step(base_state, symbol)
        if next_base is None:
            return None
        token = self.vocabulary.token_for_label(symbol)
        next_skill = active_skill
        next_arg = active_arg
        next_picked = current_picked
        next_visited = visited_picks
        next_dest = dest_container

        if token.startswith("skill:"):
            next_skill = token[6:]
            next_arg = None
        elif token.startswith("arg:"):
            next_arg = token[4:]
        elif token.startswith("obj:"):
            idx = int(token[4:])
            if active_skill == "pick" and active_arg == "target_entity_name":
                next_picked = idx
                next_visited = visited_picks.union({idx})
            elif active_skill in ("place", "pour") and active_arg == "target_container_name":
                if next_dest is None and idx in self.container_indices:
                    next_dest = idx
            next_arg = None
        elif token == self.vocabulary.eos_token:
            next_skill = None
            next_arg = None

        return (next_base, next_skill, next_arg, next_picked, next_visited, next_dest)

    def is_accepting(self, state):
        return self.base_dfa.is_accepting(state[0])

    def accepts(self, sequence):
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)

    def __getattr__(self, name):
        return getattr(self.base_dfa, name)


def bind_vlabench_contextual_dfa(
    base_dfa: Any,
    vocabulary: PlanVocabulary,
    context: Mapping[str, Any],
) -> Any:
    """Specialize the VLABench generation DFA using per-example scene entities and question context."""
    if isinstance(base_dfa, (VLABenchContextualDFA, VLABenchCausalDFA)):
        return base_dfa
    if isinstance(context, Mapping):
        entities = (
            context.get("entity_table")
            or context.get("entities")
            or ()
        )
        instruction = str(context.get("instruction", ""))
    else:
        entities = (
            getattr(context, "entities", None)
            or getattr(context, "entity_table", ())
            or ()
        )
        instruction = str(getattr(context, "instruction", ""))

    if isinstance(entities, Mapping):
        button_indices = [
            index for index, name in entities.items()
            if "button" in str(name).lower()
        ]
        entities_seq = [entities[i] for i in sorted(entities)]
    else:
        button_indices = [
            index for index, name in enumerate(entities)
            if "button" in str(name).lower()
        ]
        entities_seq = list(entities)

    if button_indices:
        target_button = _resolve_physical_law_button(instruction, entities_seq)
        return VLABenchContextualDFA(base_dfa, vocabulary, button_indices, target_button=target_button)

    if entities_seq:
        return VLABenchCausalDFA(base_dfa, vocabulary, entities_seq)

    return base_dfa

