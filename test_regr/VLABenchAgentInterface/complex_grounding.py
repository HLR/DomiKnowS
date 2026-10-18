"""Complex task requirements from exposed language and scene names only.

No reference plans, conditions, or prepared simulator goal state are inputs.
Recipes are the public VLABench recipe.json task vocabulary.
"""
from copy import deepcopy
import re

RECIPES = {
    "tomato_and_egg_stir_fry": ("tomato", "egg"),
    "garlic_steak_with_mushroom": ("steak", "garlic", "mushroom"),
    "broccoli_and_cheese_bake": ("broccoli", "cheese"),
    "carrot_and_corn_salad": ("carrot", "corn"),
    "bell_pepper_and_eggplant_stir_fry": ("bell_pepper", "eggplant", "garlic"),
    "fish_with_garlic_and_onion": ("fish", "garlic", "onion"),
    "sweet_potato_and_cheese_bake": ("sweet_potato", "cheese"),
    "steak_and_potato_skillet": ("steak", "potato", "onion"),
    "eggplant_and_tomato_stew": ("eggplant", "tomato", "garlic"),
    "mushroom_and_broccoli_stir_fry": ("mushroom", "broccoli", "garlic"),
    "carrot_and_sweet_potato_roast": ("carrot", "sweet_potato", "garlic"),
    "fish_and_bell_pepper_sauté": ("fish", "bell_pepper", "onion"),
    "corn_and_cheese_omelette": ("corn", "egg", "cheese"),
    "tomato_and_onion_salad": ("tomato", "onion"),
    "eggplant_and_mushroom_gratin": ("eggplant", "mushroom", "cheese"),
}


def normalized(text):
    return " ".join(str(text).lower().replace("_", " ").split())


def complex_planning_facts(task, instruction, entities, existing=None):
    """Return facts and coverage. Explicit declarations remain authoritative."""
    if task not in {"cook_dishes", "book_rearrange", "take_chemistry_experiment", "hammer_nail_and_hang_picture", "texas_holdem"}:
        return existing
    facts = deepcopy(existing or {})
    if any(key in facts for key in ("goals", "milestones", "completed_skill_prefix")):
        return facts
    names = tuple(entities)
    text = normalized(instruction)
    goals, milestones, missing = [], [], []
    sequence = None
    selectable = []

    def unique(fragment):
        matches = [name for name in names if fragment in name.lower()]
        return matches[0] if len(matches) == 1 else None

    def milestone(name, action, subject, destination=None, parents=()):
        milestones.append(dict(name=name, action=action, subject=subject, destination=destination, requires=list(parents)))

    if task == "cook_dishes":
        recipes = [ingredients for recipe, ingredients in RECIPES.items() if normalized(recipe) in text]
        plate = unique("plate")
        if len(recipes) == 1 and plate and all(name in names for name in recipes[0]):
            selectable = list(recipes[0])
            sequence = ["pick", "place"] * len(selectable)
            for obj in recipes[0]:
                goals.append(["at", obj, plate])
                milestone("ingredient:" + obj, "place", obj, plate)
            facts["destinations"] = {**{obj: plate for obj in recipes[0]}, **facts.get("destinations", {})}
        else:
            missing.append("unique named recipe, visible ingredients, and plate required")
    elif task == "book_rearrange":
        books = [(int(match.group(1)), name) for name in names if (match := re.match(r"century(\d+)_.*_book$", name))]
        shelf = unique("shelf")
        if books and shelf and len({year for year, _ in books}) == len(books):
            reverse = any(phrase in text for phrase in ("latest to earliest", "newest to oldest", "descending", "reverse chronological"))
            ordered_books = sorted(books, reverse=reverse)
            selectable = [obj for _, obj in ordered_books]
            sequence = ["pick", "push", "pull"] * len(books)
            previous = ()
            for _, obj in ordered_books:
                name = "shelved:" + obj
                milestone("picked:" + obj, "pick", obj, parents=previous)
                milestone(name, "push", obj, shelf, ("picked:" + obj,))
                goals.append(["pushed_to", obj, shelf])
                previous = (name,)
            # The public book task releases its gripper during pull.
            facts.setdefault("effects", {}).setdefault("pull", {"release_hands": ["right"]})
            facts["destinations"] = {**{obj: shelf for _, obj in books}, **facts.get("destinations", {})}
        else:
            missing.append("unique chronological keys for visible books and a shelf required")
    elif task == "take_chemistry_experiment":
        # Product names may also be visible distractors. Require an explicit
        # solution mention rather than treating every formula as a reactant.
        formulae = [name for name in names if re.fullmatch(r"[A-Z][A-Za-z0-9()]*", name)]
        by_lower = {name.lower(): name for name in formulae}
        reagents = {name for name in formulae if re.search(r"(?<!\w)" + re.escape(name.lower()) + r"\s+solution\b", text)}
        if formulae:
            choice = "(?:" + "|".join(re.escape(name.lower()) for name in sorted(formulae, key=len, reverse=True)) + ")"
            grouped = rf"(?<!\w)({choice})\s+(?:solution\s+)?(?:and|with)\s+({choice})\s+solutions?\b|solutions?\s+(?:of|using)\s+({choice})\s+(?:and|with)\s+({choice})(?!\w)"
            for match in re.finditer(grouped, text):
                reagents.update(by_lower[value] for value in match.groups() if value)
        reagents = sorted(reagents, key=lambda name: text.find(name.lower()))
        facts["skill_cycle"] = ["pick", "pour", "place"]
        facts["reusable_entities"] = sorted(set(facts.get("reusable_entities", ())) | set(formulae))
        flask, table = unique("flask"), unique("table")
        if reagents and flask and table:
            selectable = reagents
            for obj in reagents:
                goals.extend((["poured_into", obj, flask], ["at", obj, table]))
                milestone("transfer:" + obj, "pour", obj, flask)
                milestone("returned:" + obj, "place", obj, table, ("transfer:" + obj,))
            facts.setdefault("action_destinations", []).extend({"action": action, "subject": obj, "destination": dest}
                for obj in reagents for action, dest in (("pour", flask), ("place", table)))
            facts["reusable_entities"] = sorted(set(facts.get("reusable_entities", ())) | set(reagents))
        else:
            missing.append("explicitly named reagents, flask, and table required; no amounts inferred")
    elif task == "hammer_nail_and_hang_picture":
        hammer, nail, counter = unique("hammer"), unique("nail"), unique("counter")
        paintings = [name for name in names if name.endswith("_painting") and normalized(name[:-9]) in text]
        if hammer and nail and counter and len(paintings) == 1:
            painting = paintings[0]
            selectable = [hammer, painting]
            sequence = ["pick", "push", "place", "pick", "push"]
            milestone("tool_grasped", "pick", hammer)
            milestone("nail_secured", "push", hammer, nail, ("tool_grasped",))
            milestone("hammer_returned", "place", hammer, counter, ("nail_secured",))
            milestone("painting_grasped", "pick", painting, parents=("hammer_returned",))
            milestone("painting_hung", "push", painting, nail, ("painting_grasped",))
            goals.extend((["at", hammer, counter], ["pushed_to", painting, nail]))
            facts.setdefault("action_destinations", []).extend({"action": action, "subject": obj, "destination": dest}
                for action, obj, dest in (("push", hammer, nail), ("place", hammer, counter), ("push", painting, nail)))
        else:
            missing.append("unique hammer, nail, counter, and instruction-selected painting required")
    else:
        missing.append("winning card combination must be inferred from observed cards; no hidden winning-hand target imported")
    facts["complex_grounding_report"] = {"profile": "input_complex_v1", "status": "checked" if goals else "unavailable", "reasons": missing}
    if goals:
        facts["goals"] = {"positive": goals}
        facts["milestones"] = milestones
        facts.setdefault("selections", []).append({"action": "pick", "candidates": selectable})
        if sequence is not None:
            facts["required_skill_sequence"] = sequence
    return facts
