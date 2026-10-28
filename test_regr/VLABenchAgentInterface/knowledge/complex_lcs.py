"""Complex VLABench tasks as DomiKnowS logical constraints.

``knowledge.tables`` holds the recipes, the scene roles, the hammer chain and the skill templates; this module
turns them into rules over one operation (variables ``q`` the request, ``x`` its subject, ``y`` its destination,
``e``/``c``/``v``/``w`` the role-bearing entities of the scene) and over the finished plan:

* which objects a pick may take (``affordance``) and where they go (``destination``);
* the order of the steps, read from the operations already performed (``milestone_order``), so a decoder is
  stopped at the first operation that breaks it;
* at the end, every required step done (hard) and the goal facts true when they are known (``goal_completion``).

Observations only describe the episode: the task, the recipe named in the instruction, which scene entities
bind each role, which foods are present, the books' years and the requested direction, the operation, the
operations performed so far and the final facts. ``ComplexMonitor`` relays the rules through
``domiknows.generation.lc_relay`` into the grounded requirement families.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from domiknows.generation.dfa.predicate_monitor import PredicateEvent
from domiknows.generation.grounded_facts import Check
from domiknows.generation.lc_relay import RelayedKnowledge

from .tables import COMPLEX_ROLES, COMPLEX_TASK_ROLES, FOODS, HAMMER_CHAIN, RECIPES

DOMAIN = "complex_task"
SKILLS = ("pick", "place", "push", "pull", "pour")
RELATIONS = ("at", "pushed_to", "poured_into")
DONE = ("push", "place", "pour")  # performed (skill, subject, destination) pairs; picks are unary
_IDS = count()


def recipe_key(recipe):
    return "".join(ch if ch.isalnum() else "_" for ch in recipe)


def declare_complex(rules):
    from domiknows.graph import Concept
    from domiknows.graph.logicalConstrain import andL, existsL, notL, orL

    build = next(_IDS)
    request = Concept(name="complex_request")
    entity = Concept(name="complex_entity")
    pairs = {}

    def relation(name):
        concept = Concept(name=f"complex_{name}")
        concept.has_a(**{f"complex_{name}_a_{build}": entity, f"complex_{name}_b_{build}": entity})
        pairs[name] = concept
        return concept

    tasks = {task: request(name=f"task__{task}") for task in COMPLEX_TASK_ROLES}
    recipes = {recipe: request(name=f"recipe__{recipe_key(recipe)}") for recipe in RECIPES}
    bound = {role: request(name=f"role_bound__{role}") for role in COMPLEX_ROLES}
    flags = {name: request(name=name) for name in ("books_keyed", "asks_reverse")}
    ops = {skill: request(name=f"op__{skill}") for skill in SKILLS}
    ending = request(name="at_completion")
    roles = {role: entity(name=f"role__{role}") for role in COMPLEX_ROLES}
    foods = {food: entity(name=f"food__{food}") for food in FOODS}
    done_pick = entity(name="done__pick")
    earlier = relation("earlier_year")
    done = {skill: relation(f"done__{skill}") for skill in DONE}
    final = {name: relation(f"final__{name}") for name in RELATIONS}
    known = {name: relation(f"known__{name}") for name in RELATIONS}

    def guard(task):
        return [tasks[task]("q"), *(bound[role]("q") for role in COMPLEX_TASK_ROLES[task])]

    def placed(e, skill, role, local):
        """``e`` was taken to the entity bound to ``role`` by ``skill``."""
        return existsL(andL(roles[role](local), done[skill](e, local)))

    def open_goal(name, rel, subject_role, destination_role, condition):
        rules.rule(name, "goal_completion",
                   lambda: andL(*condition(), ending("q"), subject_role(), roles[destination_role]("c")),
                   lambda: orL(notL(known[rel]("e", "c")), final[rel]("e", "c")),
                   lambda: andL(known[rel]("e", "c"), final[rel]("e", "c")))

    # Cooking: the named recipe's ingredients go to the plate.
    for recipe, ingredients in RECIPES.items():
        key = recipe_key(recipe)

        def cook(recipe=recipe, ingredients=ingredients):
            return [*guard("cook_dishes"), recipes[recipe]("q"),
                    *(notL(recipes[other]("q")) for other in RECIPES if other != recipe),
                    *(existsL(foods[food]("v")) for food in ingredients)]

        def ingredient(z, ingredients=ingredients):
            atoms = [foods[food](z) for food in ingredients]
            return atoms[0] if len(atoms) == 1 else orL(*atoms)
        rules.rule(f"complex__cook__{key}__pick", "affordance",
                   lambda c=cook: andL(*c(), ops["pick"]("q")), lambda i=ingredient: i("x"))
        rules.rule(f"complex__cook__{key}__destination", "destination",
                   lambda c=cook, i=ingredient: andL(*c(), i("x")), lambda: roles["plate"]("y"))
        rules.rule(f"complex__cook__{key}__placed", "goal_completion",
                   lambda c=cook, i=ingredient: andL(*c(), ending("q"), i("e")),
                   lambda: placed("e", "place", "plate", "c"))
        open_goal(f"complex__cook__{key}__at", "at", lambda i=ingredient: i("e"), "plate", cook)

    # Books: each is shelved in the requested chronological order.
    def books():
        return [*guard("book_rearrange"), flags["books_keyed"]("q")]

    def precedes(a, b):
        return orL(andL(notL(flags["asks_reverse"]("q")), earlier(a, b)), andL(flags["asks_reverse"]("q"), earlier(b, a)))
    rules.rule("complex__book__pick", "affordance", lambda: andL(*books(), ops["pick"]("q")), lambda: roles["book"]("x"))
    rules.rule("complex__book__destination", "destination",
               lambda: andL(*books(), roles["book"]("x")), lambda: roles["shelf"]("y"))
    rules.rule("complex__book__order", "milestone_order",
               lambda: andL(*books(), ops["pick"]("q"), roles["book"]("x"), notL(done_pick("x")),
                            roles["book"]("c"), precedes("c", "x")),
               lambda: placed("c", "push", "shelf", "v"))
    rules.rule("complex__book__shelve_after_pick", "milestone_order",
               lambda: andL(*books(), ops["push"]("q"), roles["book"]("x"), roles["shelf"]("y"),
                            notL(done["push"]("x", "y"))),
               lambda: done_pick("x"))
    rules.rule("complex__book__shelved", "goal_completion",
               lambda: andL(*books(), ending("q"), roles["book"]("e")),
               lambda: andL(done_pick("e"), placed("e", "push", "shelf", "c")))
    open_goal("complex__book__pushed_to", "pushed_to", lambda: roles["book"]("e"), "shelf", books)

    # Chemistry: every named reagent is poured into the flask, then returned to the table.
    def chemistry():
        return [*guard("take_chemistry_experiment"), existsL(roles["reagent"]("v"))]
    rules.rule("complex__chemistry__pick", "affordance",
               lambda: andL(*chemistry(), ops["pick"]("q")), lambda: roles["reagent"]("x"))
    for skill, role in (("pour", "flask"), ("place", "table")):
        rules.rule(f"complex__chemistry__{skill}_destination", "destination",
                   lambda s=skill: andL(*chemistry(), ops[s]("q"), roles["reagent"]("x")),
                   lambda r=role: roles[r]("y"))
        rules.rule(f"complex__chemistry__{skill}ed", "goal_completion",
                   lambda: andL(*chemistry(), ending("q"), roles["reagent"]("e")),
                   lambda s=skill, r=role: placed("e", s, r, "c"))
    rules.rule("complex__chemistry__return_after_pour", "milestone_order",
               lambda: andL(*chemistry(), ops["place"]("q"), roles["reagent"]("x"), roles["table"]("y"),
                            notL(done["place"]("x", "y"))),
               lambda: placed("x", "pour", "flask", "v"))
    open_goal("complex__chemistry__poured_into", "poured_into", lambda: roles["reagent"]("e"), "flask", chemistry)
    open_goal("complex__chemistry__at", "at", lambda: roles["reagent"]("e"), "table", chemistry)

    # Hammer and picture: the five steps of HAMMER_CHAIN, in order.
    def hammer():
        return guard("hammer_nail_and_hang_picture")

    def step_done(step, v, w):
        skill, subject, destination = HAMMER_CHAIN[step]
        if destination is None:
            return existsL(andL(roles[subject](v), done_pick(v)))
        return existsL(andL(roles[subject](v), placed(v, skill, destination, w)))
    rules.rule("complex__hammer__pick", "affordance", lambda: andL(*hammer(), ops["pick"]("q")),
               lambda: orL(roles["hammer"]("x"), roles["painting"]("x")))
    for skill, subject, destination in dict.fromkeys(step for step in HAMMER_CHAIN if step[2] is not None):
        rules.rule(f"complex__hammer__{skill}_{subject}_destination", "destination",
                   lambda s=skill, r=subject: andL(*hammer(), ops[s]("q"), roles[r]("x")),
                   lambda d=destination: roles[d]("y"))
    for index in range(1, len(HAMMER_CHAIN)):
        skill, subject, destination = HAMMER_CHAIN[index]

        def fresh(skill=skill, destination=destination):
            if destination is None:
                return [roles[HAMMER_CHAIN[index][1]]("x"), notL(done_pick("x"))]
            return [roles[destination]("y"), notL(done[skill]("x", "y"))]
        rules.rule(f"complex__hammer__step{index + 1}_after_step{index}", "milestone_order",
                   lambda s=skill, r=subject, f=fresh: andL(*hammer(), ops[s]("q"), roles[r]("x"), *f()),
                   lambda i=index: step_done(i - 1, "v", "w"))
    rules.rule("complex__hammer__steps_done", "goal_completion",
               lambda: andL(*hammer(), ending("q")),
               lambda: andL(*(step_done(i, "v", "w") for i in range(len(HAMMER_CHAIN)))))
    open_goal("complex__hammer__at_counter", "at", lambda: roles["hammer"]("e"), "counter", hammer)
    open_goal("complex__hammer__pushed_to_nail", "pushed_to", lambda: roles["painting"]("e"), "nail", hammer)

    rows = [(tasks[t], f"task__{t}", "immediate") for t in tasks]
    rows += [(recipes[r], f"recipe__{recipe_key(r)}", "immediate") for r in recipes]
    rows += [(bound[r], f"role_bound__{r}", "immediate") for r in bound]
    rows += [(flags[n], n, "immediate") for n in flags] + [(ops[s], f"op__{s}", "immediate") for s in ops]
    rows += [(ending, "at_completion", "completion")]
    rows += [(roles[r], f"role__{r}", "immediate") for r in roles] + [(foods[f], f"food__{f}", "immediate") for f in foods]
    rows += [(done_pick, "done__pick", "immediate"), (earlier, "earlier_year", "immediate")]
    rows += [(done[s], f"done__{s}", "immediate") for s in done]
    rows += [(final[n], f"final__{n}", "completion") for n in final] + [(known[n], f"known__{n}", "completion") for n in known]
    return [(concept, name, DOMAIN, phase) for concept, name, phase in rows]


def build_complex_knowledge(active=True):
    return RelayedKnowledge("vlabench_complex_knowledge", declare_complex, active=active)


def complex_event(task, block, state, op=None):
    """One complete valuation of an operation (``op``) or of the finished plan (``op is None``)."""
    roles = {role: set(block.get("roles", {}).get(role, ())) for role in COMPLEX_ROLES}
    present_foods = {name for name in block.get("entities", ()) if name in FOODS}
    universe = set().union(*roles.values(), present_foods)
    subject = destination = None
    if op is not None:
        subject, destination = op.subject, op.destination
    entities = tuple(sorted(universe | {e for e in (subject, destination) if e is not None}, key=str))
    named = set(block.get("recipes", ()))
    years = block.get("years", {})
    performed = state.performed if state is not None else frozenset()
    values = {(f"task__{t}", "request"): t == task for t in COMPLEX_TASK_ROLES}
    values.update({(f"recipe__{recipe_key(r)}", "request"): r in named for r in RECIPES})
    values.update({(f"role_bound__{r}", "request"): bool(roles[r]) for r in COMPLEX_ROLES})
    values[("books_keyed", "request")] = bool(block.get("books_keyed"))
    values[("asks_reverse", "request")] = bool(block.get("asks_reverse"))
    values.update({(f"op__{s}", "request"): op is not None and op.action == s for s in SKILLS})
    values[("at_completion", "request")] = op is None
    picked = {s for a, s, _ in performed if a == "pick"}
    for e in entities:
        values.update({(f"role__{r}", e): e in roles[r] for r in COMPLEX_ROLES})
        values.update({(f"food__{f}", e): e == f and e in present_foods for f in FOODS})
        values[("done__pick", e)] = e in picked
    facts = state.facts if state is not None else None
    for a in entities:
        for b in entities:
            values[("earlier_year", a, b)] = a in years and b in years and years[a] < years[b]
            for skill in DONE:
                values[(f"done__{skill}", a, b)] = (skill, a, b) in performed
            for rel in RELATIONS:
                truth = None if facts is None else facts.truth((rel, a, b))
                values[(f"final__{rel}", a, b)] = truth is True
                values[(f"known__{rel}", a, b)] = truth is not None
    domains = {"q": ("request",), "x": () if subject is None else (subject,),
               "y": () if destination is None else (destination,),
               "e": entities, "c": entities, "v": entities, "w": entities}
    return PredicateEvent(DOMAIN, values, "immediate" if op is not None else "completion", variable_domains=domains)


_TASK_PREFIX = {"cook_dishes": "complex__cook__", "book_rearrange": "complex__book__",
                "take_chemistry_experiment": "complex__chemistry__", "hammer_nail_and_hang_picture": "complex__hammer__"}


@dataclass
class ComplexMonitor:
    """Requirement monitor for one complex-task episode; it reads the operations already performed."""

    knowledge: RelayedKnowledge
    task: str
    block: dict
    tracks_history = True
    _cache: dict = field(default_factory=dict, repr=False)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def _names(self):
        prefix = _TASK_PREFIX.get(self.task)
        return {name for name in self.knowledge.enforce_constraints if prefix and name.startswith(prefix)}

    def _checks(self, event):
        by_family = {}
        for family, observation in self.knowledge.relay(event, self._names()):
            by_family.setdefault(family, []).append(observation)
        checks = []
        for family, observations in by_family.items():
            truths = [o.truth for o in observations]
            satisfied = False if False in truths else True if True in truths else None
            checks.append(Check(family, True, satisfied, (), tuple(observations)))
        return tuple(checks)

    def operation(self, state, op):
        if self.task not in _TASK_PREFIX:
            return ()
        key = (op.action, op.subject, op.destination, state.performed if state is not None else frozenset())
        if key not in self._cache:
            self._cache[key] = self._checks(complex_event(self.task, self.block, state, op))
        return self._cache[key]

    def completion(self, state):
        if self.task not in _TASK_PREFIX:
            return ()
        return self._checks(complex_event(self.task, self.block, state))
