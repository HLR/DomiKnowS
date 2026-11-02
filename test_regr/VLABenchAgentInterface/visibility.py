"""Explicit initial-observation entity support, independent of scene metadata.

None means unavailable evidence; an empty collection means an observed empty
support set. Callers supply ids/names grounded from the initial observation.
Neither the entity inventory nor the largest tag id is visibility evidence.
"""
from collections.abc import Mapping


def visible_entity_ids(values, entities):
    if values is None:
        return None
    table = dict(entities) if isinstance(entities, Mapping) else dict(enumerate(entities))
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError("visible_entities must be a collection of entity ids or names")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError("visible_entities must be a collection of entity ids or names") from exc
    result = set()
    for value in values:
        if type(value) is int:
            matches = [value] if value in table else []
        elif isinstance(value, str):
            matches = [int(value)] if value.isdigit() and int(value) in table else [i for i, name in table.items() if name == value]
        else:
            matches = []
        if len(matches) != 1:
            raise ValueError(f"visible entity {value!r} is absent or ambiguous in the entity inventory")
        result.add(matches[0])
    return frozenset(result)


def reference_is_visible(value, entities, visible):
    if visible is None:
        return False
    try:
        ids = visible_entity_ids((value,), entities)
    except ValueError:
        return False
    return bool(ids <= visible)


def instruction_candidate_ids(values, entities):
    """``{(skill, role): frozenset(ids)}`` from instruction candidates, or None when none were observed.

    ``values`` is an ``ObjectConstraints``, its ``as_dict()`` form, or ``(skill, role, ids)`` triples.
    A skill and role without an entry is unrestricted; ids resolve against the inventory like visible ids.
    """
    if values is None:
        return None
    if hasattr(values, "restrictions"):
        values = values.restrictions
    elif isinstance(values, Mapping):
        values = [(row["skill"], row["role"], row["ids"]) for row in values.get("restrictions", ())]
    table = {}
    for skill, role, ids in values:
        table[(str(skill), str(role))] = visible_entity_ids(tuple(ids), entities)
    return table or None


def reference_is_candidate(value, skill, role, entities, candidates):
    """``(known, member)``: whether the instruction restricts this skill and role, and whether ``value`` is allowed."""
    allowed = (candidates or {}).get((skill, role))
    if allowed is None or value is None:
        return False, False
    try:
        ids = visible_entity_ids((value,), entities)
    except ValueError:
        return True, False
    return True, bool(ids <= allowed)


def visibility_from_segmentation(masks, segment_entities, entities):
    """Union supported ids across initial-view integer instance-label masks.

    segment_entities explicitly maps segmentation labels to canonical ids/names.
    Unbound labels (background, robot, etc.) supply no entity support. A bound
    entity is supported only when at least one of its label's pixels is present.
    """
    import numpy as np

    bindings = {}
    for label, entity in segment_entities.items():
        if type(label) is not int:
            raise ValueError("segmentation labels must be integers")
        bindings[label] = next(iter(visible_entity_ids((entity,), entities)))
    seen = set()
    for mask in masks:
        array = np.asarray(mask)
        if array.ndim != 2 or array.dtype.kind not in "iu":
            raise ValueError("segmentation masks must be 2D integer label arrays")
        seen.update(bindings[int(label)] for label in np.unique(array) if int(label) in bindings)
    return frozenset(seen)
