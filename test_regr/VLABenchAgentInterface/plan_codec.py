"""Stable vocabulary serialization and plan/token conversion for VLABench."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import count
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .semantics import operation_schema_errors
    from .world_graph import (
        EOS_TOKEN,
        VLABENCH_DOMAIN_CHECKSUM,
        VLABenchWorldGraphBundle,
        build_vlabench_world_graph,
        canonicalize_plan,
        validate_plan,
    )
except ImportError:
    from semantics import operation_schema_errors
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
    optional_skill_arguments: tuple = ()

    @cached_property
    def tokens(self) -> tuple[str, ...]:
        # Cached: the decoder maps labels to tokens at every step of every search.
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

    @cached_property
    def _label_index(self) -> Mapping[str, int]:
        index: dict[str, int] = {}
        for label, token in enumerate(self.tokens):
            index.setdefault(token, label)
        return index

    def label_for_token(self, token: str) -> int:
        try:
            return self._label_index[str(token)]
        except KeyError as exc:
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
            argument_keys=tuple(sorted({key for _name, keys in signatures for key in keys} |
                                       {key for keys in world.optional_skill_arguments.values() for key in keys})),
            skill_arguments=signatures,
            max_entities=int(max_entities),
            domain_checksum=world.domain_checksum,
            optional_skill_arguments=tuple(sorted(world.optional_skill_arguments.items())),
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
        payload["optional_skill_arguments"] = tuple((name, tuple(keys)) for name, keys in payload.get("optional_skill_arguments", ()))
        value = cls(**payload)
        if checksum is not None and checksum != value.checksum:
            raise ValueError(f"vocabulary checksum mismatch for {path}")
        if value.domain_checksum != VLABENCH_DOMAIN_CHECKSUM:
            raise ValueError("vocabulary domain checksum differs from the current eleven-skill world graph; "
                             "nine-skill checkpoints require retraining, not automatic weight remapping")
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
        errors = operation_schema_errors(operation["name"], operation["params"], skill_arguments)
        if errors:
            raise ValueError("; ".join(errors))
        tokens.append(f"skill:{operation['name']}")
        for key in skill_arguments.get(operation["name"], ()):
            tokens.extend((f"arg:{key}", f"obj:{_pointer_index(operation['params'][key], entity_table)}"))
        for key in world.optional_skill_arguments.get(operation["name"], ()):
            if key in operation['params']:
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

