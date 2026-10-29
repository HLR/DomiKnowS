"""Complex task observations from exposed language and scene names only.

No reference plans, conditions, or prepared simulator goal state are inputs. This module binds scene entities
to roles and reads the instruction; the recipes, step orders, destinations and goals are logical constraints
in knowledge/complex_lcs.py, generated from knowledge/tables.py.
"""
from copy import deepcopy
import re

try:
    from .knowledge.tables import (
        COMPLEX_EFFECTS, COMPLEX_ROLE_FRAGMENTS, COMPLEX_SKILL_TEMPLATES, COMPLEX_TASKS, RECIPES,
    )
except ImportError:
    from knowledge.tables import COMPLEX_EFFECTS, COMPLEX_ROLE_FRAGMENTS, COMPLEX_SKILL_TEMPLATES, COMPLEX_TASKS, RECIPES

REVERSE_PHRASES = ("latest to earliest", "newest to oldest", "descending", "reverse chronological")


def normalized(text):
    return " ".join(str(text).lower().replace("_", " ").split())


def complex_observations(task, instruction, entities):
    """Scene roles, the named recipe(s), book years, direction and reagents: observations, no rule."""
    names = tuple(entities)
    text = normalized(instruction)

    def unique(fragment):
        matches = [name for name in names if fragment in name.lower()]
        return [matches[0]] if len(matches) == 1 else []
    roles = {role: unique(fragment) for role, fragment in COMPLEX_ROLE_FRAGMENTS.items()}
    paintings = [name for name in names if name.endswith("_painting") and normalized(name[:-9]) in text]
    roles["painting"] = paintings if len(paintings) == 1 else []
    years = {name: int(match.group(1)) for name in names if (match := re.match(r"century(\d+)_.*_book$", name))}
    roles["book"] = sorted(years)
    # Product names may also be visible distractors: a reagent is a formula the instruction names as a solution.
    formulae = [name for name in names if re.fullmatch(r"[A-Z][A-Za-z0-9()]*", name)]
    by_lower = {name.lower(): name for name in formulae}
    reagents = {name for name in formulae if re.search(r"(?<!\w)" + re.escape(name.lower()) + r"\s+solution\b", text)}
    if formulae:
        choice = "(?:" + "|".join(re.escape(name.lower()) for name in sorted(formulae, key=len, reverse=True)) + ")"
        grouped = (rf"(?<!\w)({choice})\s+(?:solution\s+)?(?:and|with)\s+({choice})\s+solutions?\b"
                   rf"|solutions?\s+(?:of|using)\s+({choice})\s+(?:and|with)\s+({choice})(?!\w)")
        for match in re.finditer(grouped, text):
            reagents.update(by_lower[value] for value in match.groups() if value)
    roles["reagent"] = sorted(reagents, key=lambda name: text.find(name.lower()))
    return {"task": task, "entities": list(names), "roles": roles,
            "recipes": [recipe for recipe in RECIPES if normalized(recipe) in text],
            "years": years, "books_keyed": bool(years) and len(set(years.values())) == len(years),
            "asks_reverse": any(phrase in text for phrase in REVERSE_PHRASES), "formulae": formulae}


def bound_subjects(task, block):
    """Subjects the task template is bound to, or None when the episode does not bind it."""
    roles, names = block["roles"], set(block["entities"])
    if task == "cook_dishes":
        if len(block["recipes"]) == 1 and roles["plate"] and all(food in names for food in RECIPES[block["recipes"][0]]):
            return list(RECIPES[block["recipes"][0]])
    elif task == "book_rearrange":
        if roles["book"] and roles["shelf"] and block["books_keyed"]:
            return roles["book"]
    elif task == "take_chemistry_experiment":
        if roles["reagent"] and roles["flask"] and roles["table"]:
            return roles["reagent"]
    elif task == "hammer_nail_and_hang_picture":
        if all(roles[role] for role in ("hammer", "nail", "counter", "painting")):
            return [*roles["hammer"], *roles["painting"]]
    return None


REASONS = {
    "cook_dishes": "unique named recipe, visible ingredients, and plate required",
    "book_rearrange": "unique chronological keys for visible books and a shelf required",
    "take_chemistry_experiment": "explicitly named reagents, flask, and table required; no amounts inferred",
    "hammer_nail_and_hang_picture": "unique hammer, nail, counter, and instruction-selected painting required",
    "texas_holdem": "winning card combination must be inferred from observed cards; no hidden winning-hand target imported",
}


def complex_planning_facts(task, instruction, entities, existing=None):
    """Return facts and coverage. Explicit declarations remain authoritative."""
    if task not in COMPLEX_TASKS:
        return existing
    facts = deepcopy(existing or {})
    if task == "texas_holdem" and "poker_selection" in facts:
        facts["complex_grounding_report"] = {"profile": "observed_poker_v1", "status": "supplied",
            "reasons": ["card faces and ordering are evaluated by poker logical constraints; no winning targets imported"]}
        return facts
    if any(key in facts for key in ("goals", "milestones", "completed_skill_prefix")):
        return facts
    block = complex_observations(task, instruction, entities)
    subjects = bound_subjects(task, block)
    kind, skills = COMPLEX_SKILL_TEMPLATES.get(task, (None, ()))
    if kind == "cycle":
        facts["skill_cycle"] = list(skills)
        # Chemistry formulae may be picked again for a further transfer.
        facts["reusable_entities"] = sorted(set(facts.get("reusable_entities", ())) | set(block["formulae"]))
    if subjects is not None:
        facts["complex_request"] = block
        if kind == "per_subject":
            facts["required_skill_sequence"] = list(skills) * len(subjects)
        elif kind == "fixed":
            facts["required_skill_sequence"] = list(skills)
        for action, effect in COMPLEX_EFFECTS.get(task, {}).items():
            facts.setdefault("effects", {}).setdefault(action, deepcopy(effect))
    facts["complex_grounding_report"] = {"profile": "input_complex_v1",
                                         "status": "checked" if subjects is not None else "unavailable",
                                         "reasons": [] if subjects is not None else [REASONS[task]]}
    return facts
