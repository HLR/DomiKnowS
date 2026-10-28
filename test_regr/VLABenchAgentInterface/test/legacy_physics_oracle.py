"""Frozen pre-LC physics knowledge as of commit e358e881, the reference for the equivalence tests.

``legacy_selection`` rebuilds the old scored press selection; ``SelectionRequirement.check`` then judges it as before.
"""
import re

MATERIAL_ORDERS = {
    "density_qa": ("wood", "rubber", "water", "glass", "stone", "metal"),
    "speed_of_sound_qa": ("rubber", "water", "wood", "stone", "glass", "metal"),
    "thermal_expansion_qa": ("glass", "wood", "stone", "metal", "rubber"),
}
SLIDE_ORDER = ("sphere", "cylinder", "box")
KNOWN_MATERIALS = frozenset(("wood", "rubber", "water", "glass", "stone", "metal"))


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



def legacy_scores(task, options):
    """Ordinal scores of the old adapter from ``{button: object component or None}``."""
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
    return scores


def legacy_selection(task, instruction, options, scores=None):
    scores = legacy_scores(task, options) if scores is None else scores
    return {"action": "press", "argument": "subject", "candidates": list(options), "scores": scores,
            "property": f"{task}:qualitative_model_v1", **comparison(task, instruction, len(options))}
