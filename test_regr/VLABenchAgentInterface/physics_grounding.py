"""Input-only adapters for VLABench's declared qualitative physics models.

These ordinal scores describe benchmark models, not physical measurements.
Callers supply scene components only: conditions and reference plans are never
arguments to this adapter. Reflection requires explicitly bound observations.
"""

from copy import deepcopy
from math import dist, isfinite
import re
from domiknows.generation.property_observations import PropertyObservations

try:
    from .world_graph import QA_TASK_PATTERNS
except ImportError:
    from world_graph import QA_TASK_PATTERNS


PHYSICS_PROFILE = "scene_metadata_physics_v1"
VISUAL_QA_PROFILE = "visual_qa_v1"
# Public models in VLABench/tasks/hierarchical_tasks/primitive/physical_series:
# density_qa.py, sound_speed_qa.py, thermal_expansion_qa.py, friction_qa.py,
# and magnetism_qa.py. Version these separately from physical measurements.
MATERIAL_ORDERS = {
    "density_qa": ("wood", "rubber", "water", "glass", "stone", "metal"),
    "speed_of_sound_qa": ("rubber", "water", "wood", "stone", "glass", "metal"),
    "thermal_expansion_qa": ("glass", "wood", "stone", "metal", "rubber"),
}
SLIDE_ORDER = ("sphere", "cylinder", "box")
KNOWN_MATERIALS = frozenset(("wood", "rubber", "water", "glass", "stone", "metal"))


def visual_qa_planning_facts(task, instruction, components):
    """Declare action shape and unknown properties, leaving the choice to vision.

    Only exposed option identities and instruction comparisons are consumed.
    Material values, saved reflectance, sidecars and target annotations do not
    supply answer-selection facts in this profile.
    """
    candidates = [c["name"] for c in components if c.get("class") == "Button"]
    property = "reflectance_fraction" if task == "reflection_qa" else task + ":visual_property"
    facts = {"observation_report": {
        "property": property, "sources": [], "status": "unavailable",
        "missing_candidates": candidates,
        "reasons": ["visual profile delegates property selection to the VLM; no symbolic property observations consumed"],
        "prediction_source": "vlm_visual",
    }}
    if candidates:
        facts["selections"] = [{"action": "press", "argument": "subject", "property": property,
                                 "candidates": candidates, "scores": dict.fromkeys(candidates),
                                 **comparison(task, instruction, len(candidates))}]
    return facts, VISUAL_QA_PROFILE


def comparison(task, instruction, candidate_count):
    text = " ".join(instruction.lower().split())
    if task == "magnetism_qa":
        if "non-magnetic" in text or "non magnetic" in text:
            return {"criterion": "min"}
        if "magnetic" in text:
            return {"criterion": "max"}
    if task == "density_qa" and "float in the water" in text:
        return {"criterion": "below", "threshold": MATERIAL_ORDERS[task].index("water")}
    if ("second fastest" in text or "second-fastest" in text) and candidate_count >= 2:
        # Slide cost is ascending from fastest, sound speed from slowest.
        rank = 1 if task == "friction_qa" else candidate_count - 2
        return {"criterion": "rank", "rank": rank}
    if any(word in text for word in ("medium", "moderately")):
        return {"criterion": "rank", "rank": candidate_count // 2} if candidate_count % 2 else {}
    small = any(word in text for word in ("smallest", "least", "slowest"))
    large = any(word in text for word in ("largest", "most", "fastest"))
    if task == "friction_qa":
        small, large = large, small
    return {"criterion": "min" if small else "max"} if small != large else {}


def _xy(component):
    position = component.get("position", ())
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    values = position[:2]
    if any(type(v) not in (float, int) or not isfinite(v) for v in values):
        return None
    return values


def option_objects(components):
    """Bind explicitly nested objects or unambiguous nearest planar neighbors.

    Unresolved/ambiguous associations remain unknown. No option number, list
    index, success condition, or desired answer determines an association.
    """
    buttons = [c for c in components if c.get("class") == "Button"]
    objects = [c for c in components if c.get("class") == "RandomGeom"]
    result = {}
    for button in buttons:
        name = button["name"]
        nested = [c for c in button.get("subentities", ()) if c.get("class") == "RandomGeom"]
        result[name] = nested[0] if len(nested) == 1 else None
        if nested or _xy(button) is None:
            continue
        distances = sorted((dist(_xy(button), _xy(obj)), i) for i, obj in enumerate(objects) if _xy(obj) is not None)
        if distances and distances[0][0] <= .25 and (len(distances) == 1 or distances[1][0] > 2 * distances[0][0]):
            result[name] = objects[distances[0][1]]
    # A flat object cannot stand for two different answer options.
    used = [id(obj) for obj in result.values() if obj is not None]
    return {name: obj if obj is None or used.count(id(obj)) == 1 else None for name, obj in result.items()}


def reflection_observations(components, candidates, supplied=None):
    """Consume explicit button/reflectance bindings, never infer vector order."""
    observations, missing_reasons = [], []
    def visit(items):
        for component in items:
            yield component
            yield from visit(component.get("subentities", ()))
    for component in visit(components):
        if component.get("class") != "Mirrors":
            continue
        values, buttons = component.get("reflectance"), component.get("option_buttons")
        if values is None or buttons is None:
            if values is None:
                missing_reasons.append("saved Mirrors component omits reflectance")
            if buttons is None:
                missing_reasons.append("saved Mirrors component omits explicit option_buttons binding")
            continue
        if not isinstance(values, list) or not isinstance(buttons, list) or len(values) != len(buttons):
            raise ValueError("reflectance and option_buttons must be parallel arrays")
        observations.append(PropertyObservations.parse({
            "version": 1, "source": "simulator_material_metadata",
            "observations": [{"entity": button, "property": "reflectance", "unit": "fraction", "value": value}
                             for button, value in zip(buttons, values)],
        }, entities=candidates))
    if supplied is not None:
        observations.append(PropertyObservations.parse(supplied, entities=candidates))
    scores = dict.fromkeys(candidates)
    sources = []
    for observation in observations:
        values = observation.scores("reflectance", "fraction")
        if values:
            sources.append(observation.source)
        for button, value in values.items():
            if value is not None and not 0 <= value <= 1:
                raise ValueError("reflectance fractions must be within [0, 1]")
            if scores[button] is not None and value is not None and scores[button] != value:
                raise ValueError(f"conflicting reflectance observations for {button}")
            if value is not None:
                scores[button] = value
    missing = [button for button, value in scores.items() if value is None]
    report = {"property": "reflectance", "unit": "fraction", "sources": sorted(set(sources)),
              "status": "unavailable" if len(missing) == len(scores) else "partial" if missing else "observed",
              "missing_candidates": missing,
              "reasons": sorted(set(missing_reasons or (["no explicitly bound reflectance observations"] if missing else [])))}
    if not missing:
        report["reasons"] = []
    return scores, report


def physics_planning_facts(task, instruction, components, existing=None, observations=None):
    """Return normalized input requirements and an explicit observation profile.

    Explicit press-selection annotations take precedence over inferred metadata
    models. Missing property observations stay missing, including reflectance.
    """
    if task not in QA_TASK_PATTERNS:
        return existing, None
    facts = deepcopy(existing) if existing is not None else {}
    if any(s.get("action") == "press" for s in facts.get("selections", ())):
        return facts, "declared_planning_facts"
    options = option_objects(components)
    if not options:
        if task == "reflection_qa":
            if observations is not None:
                PropertyObservations.parse(observations, entities=())
            facts["observation_report"] = {"property": "reflectance", "unit": "fraction", "sources": [],
                                           "status": "unavailable", "missing_candidates": [],
                                           "reasons": ["reflection candidate universe unavailable"]}
            return facts, "reflection_unavailable_v1"
        return facts or None, PHYSICS_PROFILE
    if task == "reflection_qa":
        scores, report = reflection_observations(components, tuple(options), observations)
        facts["observation_report"] = report
        facts["selections"] = [*facts.get("selections", ()), {
            "action": "press", "argument": "subject", "candidates": list(options),
            "scores": scores, "property": "reflectance_fraction",
            **comparison(task, instruction, len(options)),
        }]
        return facts, "reflection_observations_v1" if report["sources"] else "reflection_unavailable_v1"
    scores = {}
    for button, obj in options.items():
        value = None
        if obj is not None:
            material = obj.get("material")
            if task in MATERIAL_ORDERS and material in MATERIAL_ORDERS[task]:
                value = MATERIAL_ORDERS[task].index(material)
            elif task == "magnetism_qa" and material in KNOWN_MATERIALS:
                value = int(material == "metal")
            elif task == "friction_qa":
                shape = obj.get("geom_type")
                if shape is None:
                    match = re.search(r"_(sphere|cylinder|box|capsule)(?:_\d+)?$", obj.get("name", ""))
                    shape = match.group(1) if match else None
                if shape in SLIDE_ORDER:
                    value = SLIDE_ORDER.index(shape)
        scores[button] = value
    requirement = {"action": "press", "argument": "subject", "candidates": list(options),
                   "scores": scores, "property": f"{task}:qualitative_model_v1",
                   **comparison(task, instruction, len(options))}
    facts["selections"] = [*facts.get("selections", ()), requirement]
    return facts, PHYSICS_PROFILE
