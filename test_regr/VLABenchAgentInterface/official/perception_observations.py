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
* the hypotheses ground the research grounding formulas (``knowledge/research_lcs.py``),
  evaluated by the same ``ResearchMonitor`` as the internal runtime, with the joint
  policy: the plan's ids must keep at least one hypothesis alive. The robust and abstain
  policies refuse to act under ambiguity, which in a planning-only benchmark would turn
  every ambiguous episode into a format error.
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
except ImportError:  # pragma: no cover - flat-module execution
    from research.config import ResearchConfig
    from research.monitor import ResearchMonitor, ResearchState
    from research.perception import PerceptionUnavailable, perceive
    from knowledge.research_lcs import build_research_knowledge
    from object_grounding import ObjectConstraints, perceived_object_constraints

GROUNDING_POLICY = "joint"
DESTINATION_ROLE = "target_container_name"


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

    def __init__(self, grounding: dict, knowledge=None, disabled_formulas=()):
        config = ResearchConfig("research_v1", GROUNDING_POLICY, disabled_formulas=tuple(disabled_formulas))
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

    @property
    def supplies_facts(self) -> bool:
        return self.visible is not None or self.grounding is not None or bool(self.candidates.restrictions)

    def note(self) -> str:
        hypotheses = len(self.grounding["hypotheses"]) if self.grounding else 0
        status = self.audit.get("status", "observed")
        visible = list(self.visible) if self.visible is not None else None
        complete = bool(self.grounding and self.grounding.get("complete"))
        categories = json.dumps({str(tag): kind for tag, kind in self.categories})
        candidates = json.dumps({f"{skill}.{role}": list(ids) for skill, role, ids in self.candidates.restrictions})
        return (f"<!-- perception: status={status} visible={visible} categories={categories} "
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
    candidates = perceived_object_constraints(instruction, known, tuple(t for t in visible or () if t not in known))
    return EpisodePerception(visible, grounding, result["perception"], categories, candidates)
