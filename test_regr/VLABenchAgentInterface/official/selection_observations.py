"""Observations for ``selection_graph.py`` from the instruction and image perception.

The instruction is parsed into a request: a group noun, an ordinal and an order (left/right/
bottom/top in the image, or a read value), an optional shelf layer, or a named property.
Perception supplies the group (tags whose predicted kind names the noun, or tags with a reading
of that noun), the order (normalized image boxes, compared strictly) and the readings. Which
words name a readable property is the business of a reading source: ``card_observations.py`` is
the one in use.

"Left" is the viewer's left in the image the model sees; this convention is declared, not
fitted. Every unsure observation abstains rather than restricting: a low-confidence or
unreadable object, a group larger than ``MAX_GROUP``, two members whose order the boxes do
not settle, equal read values, or a request no tag satisfies. Rows and columns of a grid are
not read: an empty grid slot has no tag, so the image cannot fix a column number.
"""
from __future__ import annotations

import re

try:
    from ..object_grounding import ObjectConstraints, _category_named
    from .card_observations import CARD_NOUNS, card_request
    from .selection_graph import MAX_GROUP
except ImportError:  # pragma: no cover - flat-module execution
    from object_grounding import ObjectConstraints, _category_named
    from card_observations import CARD_NOUNS, card_request
    from selection_graph import MAX_GROUP

ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}
_ORDINAL = r"(?:\d+(?:st|nd|rd|th)|" + "|".join(ORDINAL_WORDS) + ")"
_SPATIAL = re.compile(rf"\b(?P<ordinal>{_ORDINAL})\s+(?P<noun>[a-z]+)\s+from\s+(?:the\s+)?(?P<start>left|right|bottom|top)\b")
_LAYER = re.compile(r"\bon\s+the\s+(?P<layer>bottom|bottommost|lowest|top|topmost|highest|upper)\s+(?:layer|shelf)\b")
_NEGATION = re.compile(r"\b(?:not|except|without|instead|rather|besides)\b|\bother\s+than\b")
READING_SOURCES = (card_request,)  # each turns an instruction into a request about a readable property
READING_NOUNS = CARD_NOUNS  # the nouns whose group members are known by having a reading


def ordinal(text):
    return ORDINAL_WORDS.get(text) or int(re.match(r"\d+", text).group())


def selection_request(instruction):
    """The instruction's selection request as a dict, or None when it names no supported selection."""
    text = " ".join(str(instruction).lower().replace("_", " ").split())
    if _NEGATION.search(text):
        return None
    for source in READING_SOURCES:
        if (request := source(text, _ORDINAL, ordinal)) is not None:
            return request
    if match := _SPATIAL.search(text):
        layer = _LAYER.search(text)
        return {"kind": "spatial", "noun": match["noun"],
                "rank": ordinal(match["ordinal"]), "start": match["start"],
                "layer": None if layer is None else ("bottom" if layer["layer"] in ("bottom", "bottommost", "lowest") else "top")}
    return None


def _before(a, b, start):
    """``a`` strictly precedes ``b`` counting from ``start``; boxes are [left, top, right, bottom] in [0, 1]."""
    if start == "left":
        return a[2] < b[0]
    if start == "right":
        return b[2] < a[0]
    if start == "top":
        return a[3] < b[1]
    return b[3] < a[1]  # bottom: image y grows downwards


def _layer(group, boxes, which):
    """The members on the requested shelf layer: vertical-overlap clusters, lowest or highest."""
    clusters = []
    for tag in sorted(group, key=lambda t: boxes[t][1]):
        if clusters and boxes[tag][1] < max(boxes[t][3] for t in clusters[-1]):
            clusters[-1].append(tag)
        else:
            clusters.append([tag])
    pick = max if which == "bottom" else min
    return set(pick(clusters, key=lambda c: sum(boxes[t][3] for t in c) / len(c)))


def selection_observations(request, visible, boxes, categories, readings):
    """The selection graph's observations for one episode, or None to abstain.

    ``readings`` maps a tag to what could be read on it, a tuple whose first element orders the objects.
    """
    by_reading = request["kind"] in ("property", "value") or request["noun"] in READING_NOUNS
    noun = " ".join(READING_NOUNS) if by_reading else request["noun"]
    # A member is a tag whose predicted kind the noun names ("red book" for "book"), or a tag with a reading.
    group = {t for t in visible if (by_reading and t in readings)
             or (categories.get(t) and _category_named(categories[t], noun))}
    if not group:
        return None
    if request["kind"] == "property":
        matches = {t for t in group if readings.get(t) == request["value"]}
        return {"property_requested": True, "shows_property": matches} if matches else None
    if len(group) > MAX_GROUP:
        return None
    if request["kind"] == "value":
        if any(t not in readings for t in group) or len({readings[t][0] for t in group}) != len(group):
            return None
        larger = request["start"] == "largest"
        precedes = {(a, b) for a in group for b in group
                    if (readings[a][0] > readings[b][0] if larger else readings[a][0] < readings[b][0])}
        return {"rank": request["rank"], "members": group, "precedes": precedes}
    if any(t not in boxes for t in group):
        return None
    if request["layer"] is not None:
        group = _layer(group, boxes, request["layer"])
    precedes = {(a, b) for a in group for b in group if a != b and _before(boxes[a], boxes[b], request["start"])}
    if any((a, b) not in precedes and (b, a) not in precedes for a in group for b in group if a < b):
        return None  # two members overlap along the axis: their order is not observed
    if request["rank"] > len(group):
        return None
    return {"rank": request["rank"], "members": group, "precedes": precedes}


def selection_constraints(graph, instruction, visible, boxes, categories, readings, uncertain=False):
    """Pick candidates admitted by the selection formulas, or no restriction when any observation is unsure."""
    request = selection_request(instruction)
    if request is None:
        return ObjectConstraints(evidence=("selection: no ordinal or property request",))
    if uncertain or not visible:
        return ObjectConstraints(evidence=("selection: abstain, perception uncertain",))
    observed = selection_observations(request, tuple(visible), boxes, categories, readings)
    if observed is None:
        return ObjectConstraints(evidence=("selection: abstain, request not settled by the image",))
    admitted = graph.admitted(visible, **observed)
    if not admitted:
        return ObjectConstraints(evidence=("selection: abstain, no tag admitted",))
    return ObjectConstraints((("pick", "target_entity_name", admitted),),
                             (f"pick: {request['kind']} selection from image perception",))
