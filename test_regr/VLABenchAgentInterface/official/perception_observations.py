"""Image perception as observations of the official decoder, over tag ids only.

The frozen perception model reads the episode's unlabeled image, its number-labeled
image and the instruction (the official inputs) and reports the confidently visible
tags, the kind of each one (``"tomato"``, ``"flask"``: a category read from pixels, never a
scene name) and the instruction's role hypotheses. No entity table is passed: pointers stay
tag ids and completeness is the model's own claim about the image, never a comparison
with a scene inventory.

The facts reach the plan DFA only as observations of compiled logical constraints:

* the visible tags ground ``observation_supported_references`` (``decoding_graph.py``);
* the predicted kinds named by the instruction's source and destination clauses ground
  ``instruction_candidates_bound_targets`` (``object_grounding.perceived_object_constraints``);
* ordinal and named-card requests ("2nd poker from left to right", "3rd largest poker",
  "poker jack of spades") ground the selection formulas of ``selection_graph.py`` from the
  perceived boxes and card faces; the tags they admit join the same candidate LC;
* the hypotheses ground the research grounding formulas (``knowledge/research_lcs.py``),
  evaluated by the same ``ResearchMonitor`` as the internal runtime. The default joint
  policy needs the plan's ids to keep at least one hypothesis alive. ``--grounding-policy
  robust`` or ``abstain`` may be chosen explicitly; they refuse to act under ambiguity or
  incomplete grounding, which in a planning-only benchmark is a format error.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from domiknows.generation.grounded_facts import Operation

try:
    from ..research.config import ResearchConfig
    from ..research.monitor import ResearchMonitor, ResearchState
    from ..research.perception import PerceptionUnavailable, perceive
    from ..knowledge.research_lcs import build_research_knowledge
    from ..object_grounding import ObjectConstraints, perceived_object_constraints
    from .selection_graph import build_selection_graph
    from .selection_observations import selection_constraints
except ImportError:  # pragma: no cover - flat-module execution
    from research.config import ResearchConfig
    from research.monitor import ResearchMonitor, ResearchState
    from research.perception import PerceptionUnavailable, perceive
    from knowledge.research_lcs import build_research_knowledge
    from object_grounding import ObjectConstraints, perceived_object_constraints
    from selection_graph import build_selection_graph
    from selection_observations import selection_constraints

GROUNDING_POLICY = "joint"
DESTINATION_ROLE = "target_container_name"
_SELECTION = []  # the compiled selection graph, built on first use


def selection_graph():
    if not _SELECTION:
        _SELECTION.append(build_selection_graph())
    return _SELECTION[0]


def merged_constraints(*parts: ObjectConstraints) -> ObjectConstraints:
    """Every source's candidates for a skill and role must hold; sources that disagree entirely abstain."""
    merged, dropped = {}, set()
    for part in parts:
        for skill, role, ids in part.restrictions:
            key = (skill, role)
            merged[key] = tuple(sorted(set(merged[key]) & set(ids))) if key in merged else tuple(ids)
            if not merged[key]:
                dropped.add(key)
    restrictions = tuple((skill, role, ids) for (skill, role), ids in merged.items() if (skill, role) not in dropped)
    evidence = tuple(e for part in parts for e in part.evidence)
    evidence += tuple(f"abstain: {skill}.{role} sources disagree" for skill, role in sorted(dropped))
    return ObjectConstraints(restrictions, evidence)


def perceived_boxes(result) -> dict:
    prefix = "image_bbox:"
    return {int(r["id"][len(prefix):]): tuple(r["value"]) for r in result["evidence"]["records"]
            if r["id"].startswith(prefix) and r["id"][len(prefix):].isdigit()}


def episode_inputs(items: list[list[Any]]) -> tuple[Any, Any, str]:
    """The target episode's unlabeled image, labeled image and instruction from an official ti-list.

    Few-shot examples come first, so the episode's own inputs are the last two images and
    the text after the last ``Language instruction:`` marker.
    """
    images = [value for kind, value in items if kind == "image"]
    texts = [value for kind, value in items if kind == "text"]
    marker = max(i for i, text in enumerate(texts) if text == "Language instruction:")
    if len(images) < 2 or marker + 1 >= len(texts):
        raise ValueError("an official ti-list ends with two images and an instruction")
    return images[-2], images[-1], texts[marker + 1]


@dataclass(frozen=True)
class _View:
    """The part of a grounded state the research monitor reads."""
    monitor_state: tuple = ()


class PerceivedGrounding:
    """The research grounding formulas over one episode's hypotheses; states are alive-hypothesis tuples."""

    def __init__(self, grounding: dict, knowledge=None, disabled_formulas=(), policy=GROUNDING_POLICY):
        config = ResearchConfig("research_v1", policy, disabled_formulas=tuple(disabled_formulas))
        self.monitor = ResearchMonitor(knowledge or build_research_knowledge(), {"grounding": grounding}, config)
        self.start = self.monitor.state(_View()).alive

    @staticmethod
    def _view(alive):
        return _View(((ResearchMonitor.key, ResearchState(alive=alive)),))

    def admits(self, operation: Operation, alive: tuple) -> bool:
        checks = self.monitor.operation(self._view(alive), operation)
        return all(check.satisfied is not False for check in checks)

    def advance(self, operation: Operation, alive: tuple) -> tuple:
        after = self.monitor.transition(self._view(alive), self._view(alive), operation)
        return self.monitor.state(after).alive


def operation_for(skill: str, roles: tuple, pointer: int | None) -> Operation:
    """The research view of one official operation: its id is a subject or a destination."""
    if roles and roles[0] == DESTINATION_ROLE:
        return Operation(skill, None, pointer)
    return Operation(skill, pointer, None)


@dataclass(frozen=True)
class EpisodePerception:
    visible: tuple[int, ...] | None
    grounding: dict | None
    audit: dict
    categories: tuple[tuple[int, str], ...] = ()  # (tag, predicted kind) for confidently visible tags
    candidates: ObjectConstraints = ObjectConstraints()
    cards: tuple[tuple[int, tuple[int, str]], ...] = ()  # (tag, (rank 2..14, suit)) for read card faces

    @property
    def supplies_facts(self) -> bool:
        return self.visible is not None or self.grounding is not None or bool(self.candidates.restrictions)

    def note(self) -> str:
        hypotheses = len(self.grounding["hypotheses"]) if self.grounding else 0
        status = self.audit.get("status", "observed")
        visible = list(self.visible) if self.visible is not None else None
        complete = bool(self.grounding and self.grounding.get("complete"))
        categories = json.dumps({str(tag): kind for tag, kind in self.categories})
        cards = json.dumps({str(tag): list(face) for tag, face in self.cards})
        candidates = json.dumps({f"{skill}.{role}": list(ids) for skill, role, ids in self.candidates.restrictions})
        return (f"<!-- perception: status={status} visible={visible} categories={categories} cards={cards} "
                f"candidates={candidates} hypotheses={hypotheses} complete={str(complete).lower()} -->")


def observe_episode(model, items: list[list[Any]], episode: str = "official") -> EpisodePerception:
    """Run perception on the episode's official inputs; any failure supplies no facts."""
    raw, numbered, instruction = episode_inputs(items)
    try:
        result = perceive(model, raw, numbered, instruction, episode=episode, step=0, entity_table=None)
    except PerceptionUnavailable as error:
        return EpisodePerception(None, None, {"status": "unavailable", "reason": str(error)})
    visible = tuple(result["perception"].get("visible_entities") or ()) or None
    grounding = result["grounding"]
    if not grounding.get("requests") or not grounding.get("hypotheses"):
        grounding = None  # no role was bound: the grounding formulas have nothing to observe
    categories = tuple((int(tag), kind) for tag, kind in result["perception"].get("visible_categories", {}).items())
    known = dict(categories)
    cards = tuple((int(tag), (face[0], face[1])) for tag, face in result["perception"].get("visible_cards", {}).items())
    # Any reported object below the confidence threshold may belong to a counted group: ordinals abstain.
    reported = {int(tag) for tag in result["perception"].get("object_confidences", {})}
    uncertain = bool(reported - set(visible or ()))
    candidates = merged_constraints(
        perceived_object_constraints(instruction, known, tuple(t for t in visible or () if t not in known)),
        selection_constraints(selection_graph(), instruction, visible or (), perceived_boxes(result), known,
                              dict(cards), uncertain))
    return EpisodePerception(visible, grounding, result["perception"], categories, candidates, cards)
