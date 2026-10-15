"""Conservative instruction grounding from scene components, never goal metadata.

Rules resolve explicit names and coarse object types. They deliberately abstain
on negation, spatial relations, ordinals, and unstated semantic knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SceneObject:
    object_id: int
    name: str
    kind: str = ""


def scene_objects(config, entities):
    """Read only component identities/classes; conditions and instructions excluded."""
    by_name = {}
    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            if isinstance(value.get("name"), str):
                by_name.setdefault(value["name"], str(value.get("class", "")))
            for key in ("components", "subentities", "entities"):
                visit(value.get(key))
    visit(config.get("task", config).get("components", []))
    return tuple(SceneObject(i, name, by_name[name]) for i, name in enumerate(entities) if name in by_name)


_CONTAINERS = frozenset({"Plate", "Vase", "FlatContainer", "CommonContainer", "Fridge",
                        "Shelf", "TubeStand", "CardHolder", "BilliardTable", "Container"})
_SUPPORTS = frozenset({"Table", "Counter", "Stove", "NameTag", "Scene"})
_TYPE_WORDS = {
    "Book": r"\bbooks?\b", "Fruit": r"\bfruits?\b", "Poker": r"\b(?:pokers?|cards?)\b",
    "ChemistryTube": r"\b(?:solutions?|tubes?)\b", "Drink": r"\bdrinks?\b",
    "Toy": r"\btoys?\b", "BilliardBall": r"\b(?:billiards?|balls?)\b",
    "CommonGraspedEntity": r"\b(?:flowers?|condiments?)\b",
}


def _normal(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _mentioned(objects, text):
    text = " " + _normal(text) + " "
    return tuple(obj.object_id for obj in objects
                 if " " + _normal(obj.name) + " " in text)


@dataclass(frozen=True)
class ObjectConstraints:
    """Hashable skill/role candidate sets, suitable for DFA and HMM cache identity."""
    restrictions: tuple[tuple[str, str, tuple[int, ...]], ...] = ()
    evidence: tuple[str, ...] = ()

    def ids(self, skill, role, entity_count):
        return next((ids for s, r, ids in self.restrictions if (s, r) == (skill, role)),
                    tuple(range(entity_count)))

    def as_dict(self):
        return {"restrictions": [{"skill": s, "role": r, "ids": list(ids)}
                                  for s, r, ids in self.restrictions], "evidence": list(self.evidence)}


def instruction_object_constraints(instruction, objects):
    """Only hard-mask candidates when an explicit rule supplies positive evidence.

    This is a candidate filter, not a solver for natural-language instructions.
    Ambiguous names retain their union; unsupported descriptions retain all IDs.
    """
    text = instruction.lower()
    if re.search(r"\b(?:not|except|without|instead|rather|besides)\b|\bother\s+than\b", text):
        return ObjectConstraints(evidence=("abstain: negation or exclusion",))
    if re.search(r"\b(?:left|right|top|bottom|nearest|farthest|row|column|first|second|third|\d+(?:st|nd|rd|th))\b", text):
        return ObjectConstraints(evidence=("abstain: spatial or ordinal description",))
    # Restrict explicit-source selection only in an action clause, not arbitrary prose.
    action = re.search(r"\b(?:pick|take|fetch|grab|select|put|insert|add|pour)\b\s+(.*)", text)
    if action is None:
        return ObjectConstraints(evidence=("abstain: unsupported action wording",))
    clause = action.group(1)
    destination = re.search(r"\b(?:into|to)\b\s+(.*)", clause) or re.search(r"\bin\b\s+(.*)", clause)
    source = clause[:destination.start()] if destination else clause
    # A source relation needs geometry/container membership; exact bare names can be misleading.
    if re.search(r"\b(?:on|in|inside|from)\b", source):
        return ObjectConstraints(evidence=("abstain: source relation",))
    manipulable = tuple(obj for obj in objects if obj.kind not in _CONTAINERS | _SUPPORTS
                        and obj.kind)  # Unknown class does not justify a hard restriction.
    candidates = _mentioned(manipulable, source)
    evidence, restrictions = [], []
    if candidates:
        restrictions.append(("pick", "target_entity_name", candidates))
        evidence.append("pick: explicit scene object name in source clause")
    else:
        kinds = {kind for kind, pattern in _TYPE_WORDS.items() if re.search(pattern, source)}
        candidates = tuple(obj.object_id for obj in manipulable if obj.kind in kinds)
        if candidates:
            restrictions.append(("pick", "target_entity_name", candidates))
            evidence.append("pick: instruction object type and scene class")
    if destination:
        containers = tuple(obj for obj in objects if obj.kind in _CONTAINERS)
        candidates = _mentioned(containers, destination.group(1))
        if candidates:
            for skill in ("place", "insert", "pour", "push"):
                restrictions.append((skill, "target_container_name", candidates))
            evidence.append("destination: explicit container name")
    return ObjectConstraints(tuple(restrictions), tuple(evidence) or ("abstain: no grounded match",))
