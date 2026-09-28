"""Compile portable, finite-state monitors for governed planning execution.

The artifact is data only.  KAoS can verify it without importing DomiKnowS.
Its state is the finite tuple of node statuses, current award epochs, and
dispatch reservations; transitions are evaluated lazily to avoid enumerating
every interleaving of independent mission parts.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


VERSION = 1
STEP_CONTRACT_FIELDS = ("id", "actionClass", "actor", "handler", "input", "properties",
                        "dependsOn", "children", "directive", "maxIterations", "k",
                        "node", "deadline", "dispatchId")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def artifact_digest(value):
    return hashlib.sha256(canonical({k: v for k, v in value.items() if k != "digest"}).encode()).hexdigest()


def step_contract_digest(step):
    return hashlib.sha256(canonical({key: step[key] for key in STEP_CONTRACT_FIELDS if key in step}).encode()).hexdigest()


def _artifact(*, scope, mission, revision, domain_fingerprint, plan_digest,
              nodes, actor=None, award_key=None, epoch=None, groups=(), directives=()):
    try:
        from domiknows_planning_monitor import compile_monitor
    except ImportError as exc:
        raise RuntimeError("install domiknows[planning-monitor] to compile planning monitors") from exc
    return compile_monitor({
        "scope": scope,
        "identity": {"mission": mission, "revision": revision,
                     "domainFingerprint": domain_fingerprint,
                     "planDigest": plan_digest, "actor": actor,
                     "awardKey": award_key, "epoch": epoch},
        "nodes": nodes, "groups": groups, "directives": directives,
    })


def compile_local_monitor(run: Mapping, *, domain_fingerprint: str,
                          award_key: str, epoch: int):
    """Compile an executive run, including its explicit dependency edges."""
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("award epoch must be a nonnegative integer")
    leaves = [step for step in run["steps"] if not step.get("directive")]
    leaf_ids = {step["id"] for step in leaves}
    by_id = {step["id"]: step for step in run["steps"]}
    directives = [step["id"] for step in run["steps"] if step.get("directive")]
    nodes = []
    repeat = {child: int(step["maxIterations"])
              for step in run["steps"] if step.get("directive") == "until"
              for child in step.get("children") or ()}
    for step in leaves:
        dependencies = set()
        directive_dependencies = set()
        stack = list(step.get("dependsOn") or ())
        while stack:
            previous = stack.pop()
            if previous in leaf_ids:
                dependencies.add(previous)
            elif previous in by_id:
                directive_dependencies.add(previous)
            else:
                raise ValueError(f"unknown local dependency {previous!r}")
        nodes.append(dict(id=step["id"], actionClass=step["actionClass"],
                          actor=step["actor"], dependsOn=sorted(dependencies),
                          dependsOnDirectives=sorted(directive_dependencies),
                          stepDigest=step_contract_digest(step),
                          maxAttempts=repeat.get(step["id"], 1)))
    groups = []
    for step in run["steps"]:
        if step.get("directive") in {"case_or", "k_of_n"}:
            children = list(step.get("children") or ())
            if not set(children) <= leaf_ids:
                raise ValueError("monitor directive has non-leaf children")
            groups.append(dict(id=step["id"], kind=step["directive"], members=children,
                               k=int(step.get("k", 1))))
    plan_digest = hashlib.sha256(canonical({k: v for k, v in run.items()
                                           if k not in {"requestDigest", "monitor"}}).encode()).hexdigest()
    return _artifact(scope="local", mission=run["mission"], revision=run["revision"],
                     domain_fingerprint=domain_fingerprint, plan_digest=plan_digest,
                     nodes=nodes, actor=run.get("actor"), award_key=award_key,
                     epoch=epoch, groups=groups, directives=directives)


def compile_mission_monitor(board: Mapping, *, domain_fingerprint: str):
    """Compile top-level plan parts; directed siblings wait for prior siblings.

    Split parts are replaced by their active children.  Undirected siblings
    have no ordering edge and may be awarded or performed concurrently.
    """
    active = {part["task"]: part for part in board["parts"] if part["status"] != "split"}
    def leaves(node):
        task = node.get("actionClass") or node.get("task")
        if task in active:
            return [task]
        return [name for child in node.get("children") or () for name in leaves(child)]
    root = board["tree"]
    while root.get("connector") == "OR" and root.get("children"):
        root = root["children"][0]
    siblings = root.get("children") or [root]
    ordered = root.get("precedence") == "DIRECTED"
    dependencies = {name: set() for name in active}
    previous = []
    for child in siblings:
        current = leaves(child)
        for name in current:
            if name not in dependencies:
                raise ValueError(f"plan part {name!r} has no active award entry")
            if ordered:
                dependencies[name].update(previous)
        if ordered:
            previous = current
    if set(name for child in siblings for name in leaves(child)) != set(active):
        raise ValueError("mission plan parts do not match the derivation tree")
    nodes = [dict(id=name, actionClass=name, actor=None,
                  dependsOn=sorted(dependencies[name])) for name in active]
    source = {"tree": board["tree"], "parts": [
        {"task": part["task"], "parent": part.get("parent")}
        for part in board["parts"] if part["status"] != "split"]}
    plan_digest = hashlib.sha256(canonical(source).encode()).hexdigest()
    return _artifact(scope="mission", mission=board["mission"],
                     revision=board["revision"], domain_fingerprint=domain_fingerprint,
                     plan_digest=plan_digest, nodes=nodes)


def validate_action_candidate(constraint_plan, actions, data_nodes, *, expected=None,
                              request_values=None, path_resolver=None):
    """Check finite proposed actions against request-bound DomiKnowS constraints.

    The caller supplies one DataNode per proposed position. Missing attributes,
    observed labels, scores, variable bindings and expected query answers are
    errors from ``bind_data_nodes``; no hard constraint is silently discarded.
    """
    actions, data_nodes = tuple(actions), tuple(data_nodes)
    if len(actions) != len(data_nodes):
        raise ValueError("each proposed action requires one ordered DataNode")
    from ..dfa.graph_discovery import analyze_generation_constraints
    deferred_names = {name for name, _ in constraint_plan.deferred}
    for analysis in analyze_generation_constraints(
            constraint_plan.graph, constraint_plan.bundle, on_unsupported="ignore",
            max_sequence_length=len(actions)):
        if (analysis.relevant and not analysis.supported and
                analysis.lc_name not in deferred_names and
                getattr(constraint_plan.graph.logicalConstrains[analysis.lc_name], "hard", True)):
            raise ValueError(f"unsupported hard planning constraint {analysis.lc_name}: {analysis.reason}")
    required_answers = {name for name, expression in constraint_plan.deferred
                        if type(expression).__name__ in {"queryL", "sumL"}}
    missing_answers = required_answers - set(expected or {})
    if missing_answers:
        raise ValueError(f"request-bound planning answer is missing for {sorted(missing_answers)}")
    vocabulary = constraint_plan.bundle.vocabulary
    try:
        labels = tuple(vocabulary.label_for_token(action) for action in actions)
    except (KeyError, ValueError) as exc:
        raise ValueError("proposed action is outside the planning vocabulary") from exc
    bound = constraint_plan.bind_data_nodes(
        data_nodes, max_sequence_length=len(actions), expected=expected,
        request_values=request_values, path_resolver=path_resolver,
    )
    if not bound.accepts(labels):
        raise ValueError("proposed actions violate a request-bound planning constraint")
    state = bound.start_state
    for label in labels:
        state = bound.step(state, label)
    return bound.values(state)
