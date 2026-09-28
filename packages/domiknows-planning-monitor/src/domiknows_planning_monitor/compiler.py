"""Compile normalized plan graphs to the existing version-1 artifact format."""
from __future__ import annotations

from collections.abc import Mapping

from .engine import MonitorRefused, digest, validate


VERSION = 1


def compile_monitor(spec: Mapping) -> dict:
    """Compile a normalized graph with scope, identity, nodes and optional groups.

    ``identity`` contains mission, revision, domainFingerprint, planDigest,
    actor, awardKey and epoch. Nodes contain id, actionClass, actor, dependsOn,
    dependsOnDirectives, maxAttempts and optional stepDigest. The digest and
    field defaults match DomiKnowS's original version-1 artifacts exactly.
    """
    scope = spec["scope"]
    if scope not in {"mission", "local"}:
        raise ValueError("scope must be mission or local")
    source = spec["identity"]
    if any(not source.get(key) for key in ("mission", "revision", "domainFingerprint", "planDigest")):
        raise ValueError("monitor identity requires mission, revision, domain fingerprint and plan digest")
    identity = {
        "mission": str(source["mission"]), "revision": str(source["revision"]),
        "domainFingerprint": str(source["domainFingerprint"]),
        "planDigest": str(source["planDigest"]), "actor": source.get("actor"),
        "awardKey": source.get("awardKey"), "epoch": source.get("epoch"),
    }
    if scope == "local" and (not identity["actor"] or not identity["awardKey"]
                             or type(identity["epoch"]) is not int or identity["epoch"] < 0):
        raise ValueError("local monitor requires actor, award key and nonnegative epoch")
    nodes = []
    for node in spec["nodes"]:
        attempt_count = node.get("maxAttempts", 1)
        if type(attempt_count) is not int:
            raise ValueError("monitor attempts must be integers")
        nodes.append({
            "id": str(node["id"]), "actionClass": str(node["actionClass"]),
            "actor": node.get("actor"),
            "dependsOn": sorted(set(node.get("dependsOn") or ())),
            "dependsOnDirectives": sorted(set(node.get("dependsOnDirectives") or ())),
            "maxAttempts": attempt_count,
            **({"stepDigest": node["stepDigest"]} if node.get("stepDigest") else {}),
        })
    artifact = {
        "version": VERSION, "scope": scope, "identity": identity,
        "nodes": sorted(nodes, key=lambda node: node["id"]),
        "groups": list(spec.get("groups") or ()),
        "directives": list(spec.get("directives") or ()),
    }
    artifact["digest"] = digest(artifact)
    try:
        validate(artifact)
    except MonitorRefused as exc:
        raise ValueError(str(exc)) from exc
    return artifact
