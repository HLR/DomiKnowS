"""Joint semantic concepts and composition of authoritative domain constraints.

Episode, entity and operation concepts are shared parents. EAI and VLABench
retain their own action vocabularies and logical rules: sharing a parent does
not assert that grasp and pick, for example, are equivalent. There are no
additional cross-domain behavioral rules. See docs/JointLogicalConstraints.md.
Compilation, fact binding and activation live in utils/generation_runtime.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from domiknows.graph import Concept, Graph
from test_regr.EmbodiedAgentInterface.world_graph import EAIWorldGraphBundle, build_eai_world_graph
from test_regr.VLABenchAgentInterface.world_graph import (
    VLABenchWorldGraphBundle,
    build_vlabench_world_graph,
)
from .utils.world_schema import ACTIVATION_PROFILE_VERSION, DOMAINS, _checksum, _eai_domain_checksum


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
    combined = _checksum(
        {
            "version": ACTIVATION_PROFILE_VERSION,
            "eai": eai_checksum,
            "vlabench": vlabench.domain_checksum,
            "spine": [episode.name, entity.name, operation.name],
        }
    )
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


def __getattr__(name):
    """Keep former imports available after separating declarations and runtime."""
    if name.startswith("__"):
        raise AttributeError(name)
    from .utils import generation_runtime, world_schema

    for module in (generation_runtime, world_schema):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)
