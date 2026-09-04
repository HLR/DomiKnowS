"""One reusable DomiKnowS root for EAI and VLABench execution."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from domiknows.graph import Concept, Graph

from test_regr.EmbodiedAgentInterface.graph import create_generation_graph
from test_regr.EmbodiedAgentInterface.world_graph import (
    ACTION_SPECS,
    PREDICATE_ALIASES,
    STATE_SPECS,
    EAIWorldGraphBundle,
    build_eai_world_graph,
)
from test_regr.VLABenchAgentInterface.graph import (
    PlanVocabulary,
    compile_planner_dfa,
    create_planner_generation_graph,
)
from test_regr.VLABenchAgentInterface.world_graph import (
    VLABenchWorldGraphBundle,
    build_vlabench_world_graph,
)


ACTIVATION_PROFILE_VERSION = 1
DOMAINS = ("eai", "vlabench")


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _eai_domain_checksum() -> str:
    return _checksum({
        "version": 1,
        "actions": {
            name: {
                "min_args": spec.min_args,
                "max_args": spec.max_args,
                "is_goal_action": spec.is_goal_action,
                "requires_task_entity": spec.requires_task_entity,
            }
            for name, spec in ACTION_SPECS.items()
        },
        "states": {
            name: {
                "arity": spec.arity,
                "aliases": list(spec.aliases),
                "positive_counterpart": spec.positive_counterpart,
            }
            for name, spec in STATE_SPECS.items()
        },
        "aliases": dict(PREDICATE_ALIASES),
    })


def _subgraph_concepts(graph: Graph) -> tuple[Any, ...]:
    concepts = tuple(graph.collectAllConcepts(
        include_subgraphs=True,
        include_supergraph=False,
        include_siblings=False,
    ).values())
    return tuple(
        concept for concept in concepts
        if getattr(concept.getOntologyGraph(), "constraint", None) is not concept
    )


@dataclass(frozen=True)
class JointWorldGraphBundle:
    root: Graph
    episode: Any
    entity: Any
    operation: Any
    eai: EAIWorldGraphBundle
    vlabench: VLABenchWorldGraphBundle
    eai_domain_checksum: str
    vlabench_domain_checksum: str
    combined_checksum: str


@dataclass
class JointDomainRuntime:
    world: JointWorldGraphBundle
    eai_generation_graph: Graph
    eai_generation: Any
    eai_dfa: Any
    vlabench_generation_graph: Graph
    vlabench_generation: Any
    vlabench_dfa: Any
    eai_vocabulary: Any
    vlabench_vocabulary: PlanVocabulary
    max_eai_steps: int
    max_vlabench_operations: int
    activation_profiles: Mapping[str, tuple[Any, ...]]
    runtime_checksum: str
    activation_profile_version: int = ACTIVATION_PROFILE_VERSION
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _domain_stack: list[str | None] = field(default_factory=list, repr=False)
    _eai_policy_cache: dict[int, Any] = field(default_factory=dict, repr=False)
    _selected_domain: str | None = field(default=None, repr=False)

    @property
    def root(self) -> Graph:
        return self.world.root

    @property
    def active_domain(self) -> str | None:
        return self._selected_domain

    def activate_domain(
        self,
        domain: str | None,
        extra_concepts: Iterable[Any] = (),
    ) -> tuple[Any, ...]:
        """Activate one complete domain profile, or reset all concepts."""
        if domain is None:
            active = self.root.set_active_concepts(None)
            self._selected_domain = None
            return active
        if domain not in DOMAINS:
            raise ValueError(f"unknown joint domain {domain!r}; expected one of {DOMAINS}")
        active = self.root.set_active_concepts([
            *self.activation_profiles[domain],
            *tuple(extra_concepts),
        ])
        self._selected_domain = domain
        return active

    def dfa_for(self, domain: str, context: Mapping[str, Any] | None = None):
        """Return the immutable domain DFA, binding EAI contextual facts."""
        if domain == "vlabench":
            return self.vlabench_dfa
        if domain != "eai":
            raise ValueError(f"unknown joint domain {domain!r}; expected one of {DOMAINS}")
        from domiknows.generation import bind_contextual_dfa

        key = id(context)
        cached = self._eai_policy_cache.get(key)
        if cached is None:
            cached = bind_contextual_dfa(
                self.eai_dfa,
                self.eai_generation_graph,
                context or {},
            )
            self._eai_policy_cache[key] = cached
        return cached

    @contextmanager
    def domain_scope(
        self,
        domain: str,
        extra_concepts: Iterable[Any] = (),
    ):
        """Serialize mutable activation and restore the preceding selection."""
        with self._lock:
            previous = self.active_domain
            entered = False
            try:
                self.activate_domain(domain, extra_concepts)
                self._domain_stack.append(domain)
                entered = True
                yield self
            finally:
                if entered:
                    self._domain_stack.pop()
                self.activate_domain(previous)


def build_joint_world_graph(
    graph_name: str = "joint_embodied_world",
    *,
    eai_constraint_builders=(),
    vlabench_constraint_builders=(),
    include_eai_default_constraints: bool = True,
    include_vlabench_default_constraints: bool = True,
) -> JointWorldGraphBundle:
    """Build the semantic spine and both authoritative domain subgraphs once."""
    with Graph(graph_name) as root:
        episode = Concept(name="embodied_episode")
        entity = Concept(name="embodied_entity")
        operation = Concept(name="embodied_operation")
        parents = {"episode": episode, "entity": entity, "operation": operation}
        eai = build_eai_world_graph(
            f"{graph_name}_eai",
            constraint_builders=eai_constraint_builders,
            include_default_constraints=include_eai_default_constraints,
            semantic_parents=parents,
        )
        vlabench = build_vlabench_world_graph(
            f"{graph_name}_vlabench",
            constraint_builders=vlabench_constraint_builders,
            include_default_constraints=include_vlabench_default_constraints,
            semantic_parents=parents,
        )
    eai_checksum = _eai_domain_checksum()
    combined = _checksum({
        "version": ACTIVATION_PROFILE_VERSION,
        "eai": eai_checksum,
        "vlabench": vlabench.domain_checksum,
        "spine": [episode.name, entity.name, operation.name],
    })
    return JointWorldGraphBundle(
        root=root,
        episode=episode,
        entity=entity,
        operation=operation,
        eai=eai,
        vlabench=vlabench,
        eai_domain_checksum=eai_checksum,
        vlabench_domain_checksum=vlabench.domain_checksum,
        combined_checksum=combined,
    )


def build_joint_runtime(
    world_bundle: JointWorldGraphBundle,
    eai_vocabulary,
    vlabench_vocabulary: PlanVocabulary | None = None,
    *,
    max_eai_steps: int = 60,
    eai_object_tokens=(),
    eai_action_tokens=(),
    eai_action_sequence_tokens=(),
    eai_openable_object_tokens=(),
    eai_action_object_constraint_tokens=None,
    max_vlabench_entities: int = 64,
    max_vlabench_operations: int = 8,
) -> JointDomainRuntime:
    """Attach both generation schemas and compile both immutable DFAs."""
    from domiknows.generation import constraints_to_dfa_from_graph

    eai_graph, eai_generation = create_generation_graph(
        max_steps=max_eai_steps,
        vocab=tuple(getattr(eai_vocabulary, "tokens", eai_vocabulary)),
        object_tokens=eai_object_tokens,
        action_tokens=eai_action_tokens,
        action_sequence_tokens=eai_action_sequence_tokens,
        openable_object_tokens=eai_openable_object_tokens,
        action_object_constraint_tokens=eai_action_object_constraint_tokens or {},
    )
    if vlabench_vocabulary is None:
        vlabench_vocabulary = PlanVocabulary.from_world(
            world_bundle.vlabench,
            max_entities=max_vlabench_entities,
        )
    else:
        expected = PlanVocabulary.from_world(
            world_bundle.vlabench,
            max_entities=vlabench_vocabulary.max_entities,
        )
        if vlabench_vocabulary.checksum != expected.checksum:
            raise ValueError("VLABench vocabulary differs from the joint world definition")
    vlabench_graph, vlabench_generation = create_planner_generation_graph(
        world_bundle.vlabench,
        vlabench_vocabulary,
        max_operations=max_vlabench_operations,
        graph_name="joint_vlabench_generation",
    )
    # GenerationEncoder owns graph construction and may reset its context;
    # attach the completed schemas explicitly to the already-built joint root.
    world_bundle.root.attach(eai_graph)
    world_bundle.root.attach(vlabench_graph)
    eai_dfa = constraints_to_dfa_from_graph(
        eai_graph, eai_generation, on_unsupported="raise", minimize=False,
    )
    vlabench_dfa = compile_planner_dfa(
        vlabench_graph,
        vlabench_generation,
        world_bundle.vlabench,
        vlabench_vocabulary,
        max_operations=max_vlabench_operations,
    )
    profiles = {
        "eai": (*_subgraph_concepts(world_bundle.eai.graph), *_subgraph_concepts(eai_graph)),
        "vlabench": (
            *_subgraph_concepts(world_bundle.vlabench.graph),
            *_subgraph_concepts(vlabench_graph),
        ),
    }
    runtime_checksum = _checksum({
        "world": world_bundle.combined_checksum,
        "activation_profile_version": ACTIVATION_PROFILE_VERSION,
        "eai_vocabulary": list(getattr(eai_vocabulary, "labels", getattr(eai_vocabulary, "tokens", ()))),
        "vlabench_vocabulary": vlabench_vocabulary.checksum,
        "max_eai_steps": int(max_eai_steps),
        "max_vlabench_operations": int(max_vlabench_operations),
    })
    runtime = JointDomainRuntime(
        world=world_bundle,
        eai_generation_graph=eai_graph,
        eai_generation=eai_generation,
        eai_dfa=eai_dfa,
        vlabench_generation_graph=vlabench_graph,
        vlabench_generation=vlabench_generation,
        vlabench_dfa=vlabench_dfa,
        eai_vocabulary=eai_vocabulary,
        vlabench_vocabulary=vlabench_vocabulary,
        max_eai_steps=int(max_eai_steps),
        max_vlabench_operations=int(max_vlabench_operations),
        activation_profiles=profiles,
        runtime_checksum=runtime_checksum,
    )
    runtime.activate_domain(None)
    return runtime
