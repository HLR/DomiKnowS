"""Canonical, inspectable provenance of declared knowledge and compiled rules.

Schema hashes retain their established checkpoint meaning. This separate
manifest fingerprints actual formulas, relation paths, contextual bindings,
activation, compiled predicate programs, and auxiliary declaration sources.
Activation is recorded separately so intentional knowledge ablations remain
loadable without pretending their enforcement is identical.
"""

from __future__ import annotations
import ast
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
import re
from domiknows.graph import Concept
from domiknows.graph.relation import Relation
from domiknows.graph.logicalConstrain import V
from .world_schema import _checksum

MANIFEST_VERSION = 1


def _name(value):
    return re.sub(r"__(?:eai|vlabench)_world_[0-9]+__", "__world__", str(value))


def _source_definition(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]  # prose and formatting do not change the knowledge
    return _checksum(ast.dump(tree))


def knowledge_manifest(runtime):
    """Read current declarations, without process IDs or repr-based hashes."""
    roots = {
        "joint": runtime.world.root,
        "eai_world": runtime.world.eai.graph,
        "vlabench_world": runtime.world.vlabench.graph,
        "eai_generation": runtime.eai_generation_graph,
        "vlabench_generation": runtime.vlabench_generation_graph,
    }
    scopes = {id(graph): label for label, graph in roots.items()}

    def walk(graph, label):
        yield graph, label
        for index, child in enumerate(getattr(graph, "subgraphs", {}).values()):
            if id(child) in scopes:
                continue  # sibling components have their own stable labels
            child_label = label + "/subgraph_" + str(index)
            scopes[id(child)] = child_label
            yield from walk(child, child_label)

    components = {label: list(walk(graph, label)) for label, graph in roots.items()}

    def encode(value):
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return _name(value)
        if isinstance(value, V):
            return {
                "variable": value.name,
                "path": encode(value.v),
                "roles": encode(value.relVarInfo),
            }
        if isinstance(value, (Concept, Relation)):
            owner = (
                value.src.getOntologyGraph()
                if isinstance(value, Relation)
                else value.getOntologyGraph()
            )
            reference = {"concept": _name(value.name), "scope": scopes.get(id(owner), "external")}
            if isinstance(value, Relation):
                reference.update(source=encode(value.src), target=encode(value.dst))
            return reference
        if hasattr(value, "e"):
            return {
                "operator": type(value).__name__,
                "arguments": encode(value.e),
                "fixed_limit": getattr(value, "fixedLimit", None),
                "explicit_limit": getattr(value, "_explicitLimit", None),
            }
        if isinstance(value, Mapping):
            rows = [[encode(key), encode(item)] for key, item in value.items()]
            return sorted(rows, key=lambda row: _checksum(row[0]))
        if isinstance(value, (set, frozenset)):
            return sorted((encode(item) for item in value), key=_checksum)
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if is_dataclass(value):
            return {
                field.name: encode(getattr(value, field.name))
                for field in fields(value)
                if field.name != "name"
            }  # contextual rule names are labels
        raise TypeError(f"Cannot fingerprint knowledge value of type {type(value).__name__}")

    result = {}
    for label, graphs in components.items():
        rows = []
        for graph, scope in graphs:
            for constraint in graph.logicalConstrains.values():
                if not getattr(constraint, "headLC", True):
                    continue
                formula = {
                    "scope": scope,
                    "formula": encode(constraint),
                    "priority": getattr(constraint, "p", None),
                    "dfa": bool(getattr(constraint, "_generation_dfa_constraint", False)),
                    "context_bindings": encode(
                        getattr(constraint, "_generation_contextual_dfa_specs", ())
                    ),
                }
                rows.append({"name": constraint.name, "active": bool(constraint.active), **formula})
        definitions = [
            {k: v for k, v in row.items() if k not in {"name", "active"}} for row in rows
        ]
        activations = [{k: v for k, v in row.items() if k != "name"} for row in rows]
        ontology = []
        for graph, scope in graphs:
            for concept in graph.collectAllConcepts(
                include_subgraphs=False, include_supergraph=False, include_siblings=False
            ).values():
                if concept is getattr(graph, "constraint", None):
                    continue
                ontology.append(
                    {
                        "concept": encode(concept),
                        "parents": sorted(
                            (encode(role.dst) for role in concept.is_a()), key=_checksum
                        ),
                    }
                )
        result[label] = {
            "constraint_count": len(rows),
            "rules": rows,
            "ontology": ontology,
            "definition_sha256": _checksum(
                {
                    "rules": sorted(definitions, key=_checksum),
                    "ontology": sorted(ontology, key=_checksum),
                }
            ),
            "activation_sha256": _checksum(sorted(activations, key=_checksum)),
        }
    for label, dfa in (("eai_world", runtime.eai_dfa), ("vlabench_world", runtime.vlabench_dfa)):
        program = getattr(dfa, "constraint_program", None)
        if program is not None:
            result[label]["compiled_sha256"] = _checksum(
                [
                    encode(
                        (
                            rule.domain,
                            rule.expression,
                            rule.phase,
                            rule.variables,
                            rule.free_variables,
                        )
                    )
                    for rule in program.rules
                ]
            )
    # Parametric goal/effect/task declarations can produce different graph
    # sizes per request. Fingerprint their complete syntax, not a sample goal.
    repo = Path(__file__).resolve().parents[3]
    paths = (
        "test_regr/EmbodiedAgentInterface/utils/state_effects.py",
        "test_regr/EmbodiedAgentInterface/utils/effect_runtime.py",
        "test_regr/EmbodiedAgentInterface/reward.py",
        "test_regr/EmbodiedAgentInterface/utils/goal_constraints.py",
        "test_regr/EmbodiedAgentInterface/utils/task_knowledge.py",
        "test_regr/EmbodiedAgentInterface/utils/world_domain.py",
        "test_regr/EmbodiedAgentInterface/utils/world_facts.py",
        "domiknows/generation/grounded.py",
        "domiknows/generation/grounded_facts.py",
        "test_regr/JointEmbodiedAgentInterface/utils/planner_context.py",
    )
    paths = (
        *paths,
        "test_regr/VLABenchAgentInterface/world_domain.py",
        "test_regr/VLABenchAgentInterface/task_policy.py",
        "test_regr/VLABenchAgentInterface/pattern_runtime.py",
        *(
            path.relative_to(repo).as_posix()
            for path in sorted((repo / "test_regr/VLABenchAgentInterface/knowledge").glob("*.py"))
        ),
    )
    sources = {path: _source_definition(repo / path) for path in paths}
    state = runtime.world.vlabench.knowledge_state
    settings = {"vlabench_level": state.level, "vlabench_task_identity": state.task_identity}
    return {
        "version": MANIFEST_VERSION,
        "components": result,
        "declaration_sources": sources,
        "knowledge_settings": settings,
        "definition_sha256": _checksum(
            {
                "components": {label: data["definition_sha256"] for label, data in result.items()},
                "sources": sources,
            }
        ),
        "activation_sha256": _checksum(
            {
                "components": {label: data["activation_sha256"] for label, data in result.items()},
                "settings": settings,
            }
        ),
    }


def check_knowledge_compatibility(
    saved, current, *, planning_migration=False, knowledge_ablation=False
):
    """Check definitions, record activation changes, and identify old evidence gaps."""
    if saved is None:
        return {"status": "unrecorded_legacy", "activation_changed": None}
    if saved.get("version") != MANIFEST_VERSION:
        raise ValueError("unsupported joint knowledge manifest version")
    for label, component in current["components"].items():
        if (planning_migration or knowledge_ablation) and label.startswith("vlabench_"):
            continue
        old = saved.get("components", {}).get(label, {})
        if old.get("definition_sha256") != component["definition_sha256"]:
            raise ValueError(f"joint checkpoint knowledge definitions differ for {label}")
        if old.get("activation_sha256") == component["activation_sha256"] and old.get(
            "compiled_sha256"
        ) != component.get("compiled_sha256"):
            raise ValueError(f"joint checkpoint compiled knowledge differs for {label}")
    old_sources = saved.get("declaration_sources", {})
    for path in old_sources.keys() | current["declaration_sources"].keys():
        if planning_migration and path.startswith("test_regr/VLABenchAgentInterface/"):
            continue
        if old_sources.get(path) != current["declaration_sources"].get(path):
            raise ValueError(f"joint checkpoint auxiliary knowledge declarations differ: {path}")
    return {
        "status": (
            "planning_migration"
            if planning_migration
            else ("knowledge_ablation" if knowledge_ablation else "verified")
        ),
        "activation_changed": saved.get("activation_sha256") != current["activation_sha256"],
    }
