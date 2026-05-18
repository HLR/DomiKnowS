"""Declarative DomiKnowS graph for the cooking planner domain.

This file is the human-authored source of truth for the cooking domain.  It
declares concepts, subconcepts, relations, domain facts, and logical
constraints.  Execution layers such as DFA construction and graph-HMM masks are
derived elsewhere from this graph.
"""
from __future__ import annotations

from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, atMostAL, existsAL, ifL, notL, orL


def build_graph(dish: str = "cookie", *, max_steps: int = 8):
    """Build the declarative cooking planner graph.

    Args:
        dish: Name of the dish that a later execution adapter may select.
            The graph still declares all supported dishes and all conditional
            dish rules; this argument is validated for early feedback only.
        max_steps: Maximum number of non-terminal actions allowed in a plan.

    Returns:
        ``(graph, parts)`` where ``parts`` contains the primary graph objects
        useful for review or later adapter construction.
    """
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("cooking_planner") as graph:
        # Core planning concepts.
        plan = Concept(name="plan")
        step = Concept(name="step")
        dish_concept = Concept(name="dish")
        action = Concept(name="action")
        plan_phase = Concept(name="plan_phase")
        reference_position = Concept(name="reference_position")
        count_limit = Concept(name="count_limit")

        contains, = plan.contains(step)

        # A pairwise temporal relation over plan steps.
        precedes = Concept(name="precedes")
        earlier, later = precedes.has_a(earlier=step, later=step)

        # Action taxonomy.
        fridge_action = action(name="fridge_action")
        take_action = action(name="take_action")
        table_action = action(name="table_action")
        prep_action = action(name="prep_action")
        cook_action = action(name="cook_action")
        serve_action = action(name="serve_action")
        terminal_action = action(name="terminal_action")

        done = terminal_action(name="done")
        open_fridge = fridge_action(name="open_fridge")
        close_fridge = fridge_action(name="close_fridge")
        put_on_table = table_action(name="put_on_table")
        take_eggs = take_action(name="take_eggs")
        take_butter = take_action(name="take_butter")
        take_lettuce = take_action(name="take_lettuce")
        take_cheese = take_action(name="take_cheese")
        mix_dough = prep_action(name="mix_dough")
        chop_lettuce = prep_action(name="chop_lettuce")
        bake_cookies = cook_action(name="bake_cookies")
        cook_omelette = cook_action(name="cook_omelette")
        serve = serve_action(name="serve")

        actions = (
            done,
            open_fridge,
            take_eggs,
            take_butter,
            take_lettuce,
            take_cheese,
            close_fridge,
            put_on_table,
            mix_dough,
            bake_cookies,
            cook_omelette,
            chop_lettuce,
            serve,
        )

        # The multiclass action a model/planner assigns to each step.
        planned_action = step(
            name="planned_action",
            ConceptClass=EnumConcept,
            values=[action_concept.name for action_concept in actions],
        )

        # Dish taxonomy plus a plan-level selected dish property.
        cookie = dish_concept(name="cookie")
        omelette = dish_concept(name="omelette")
        salad = dish_concept(name="salad")
        dishes = (cookie, omelette, salad)

        planned_dish = plan(
            name="planned_dish",
            ConceptClass=EnumConcept,
            values=[dish_item.name for dish_item in dishes],
        )

        if dish not in planned_dish.enum:
            raise ValueError(
                f"unknown dish {dish!r}; expected one of {tuple(planned_dish.enum)!r}"
            )

        # Plan-phase taxonomy for later HMM masks.
        start = plan_phase(name="start")
        fridge_open = plan_phase(name="fridge_open")
        after_fridge = plan_phase(name="after_fridge")
        table_ready = plan_phase(name="table_ready")
        prep = plan_phase(name="prep")
        cook = plan_phase(name="cook")
        served = plan_phase(name="served")
        done_phase = plan_phase(name="done_phase")

        # Finite reference positions.  Later adapters may interpret declaration
        # order as the reference-plan order; the graph itself contains only
        # concepts and relations.
        pos_0 = reference_position(name="pos_0")
        pos_1 = reference_position(name="pos_1")
        pos_2 = reference_position(name="pos_2")
        pos_3 = reference_position(name="pos_3")
        pos_4 = reference_position(name="pos_4")
        pos_5 = reference_position(name="pos_5")
        pos_6 = reference_position(name="pos_6")
        pos_7 = reference_position(name="pos_7")
        pos_8 = reference_position(name="pos_8")

        max_2 = count_limit(name="max_2")
        max_plan_steps = count_limit(name=f"max_{max_steps}")

        # Declarative relation schemas for domain facts.
        dish_requires_action = Concept(name="dish_requires_action")
        dish_requires_action.has_a(dish=dish_concept, action=action)

        reference_plan_step = Concept(name="reference_plan_step")
        reference_plan_step.has_a(
            dish=dish_concept,
            position=reference_position,
            action=action,
        )

        phase_allows_action = Concept(name="phase_allows_action")
        phase_allows_action.has_a(phase=plan_phase, action=action)

        phase_transition = Concept(name="phase_transition")
        phase_transition.has_a(
            source_phase=plan_phase,
            action=action,
            target_phase=plan_phase,
        )

        action_count_limit = Concept(name="action_count_limit")
        action_count_limit.has_a(action=action, limit=count_limit)

        non_terminal_action_count_limit = Concept(name="non_terminal_action_count_limit")
        non_terminal_action_count_limit.has_a(terminal_action=action, limit=count_limit)

        # Declared domain facts: count limits.
        open_fridge_at_most_twice = action_count_limit(name="open_fridge_at_most_twice")
        open_fridge_at_most_twice.has_a(action=open_fridge, limit=max_2)

        non_terminal_steps_at_most_max = non_terminal_action_count_limit(
            name="non_terminal_steps_at_most_max"
        )
        non_terminal_steps_at_most_max.has_a(
            terminal_action=done,
            limit=max_plan_steps,
        )

        # Declared domain facts: required actions per dish.
        cookie_requires_eggs = dish_requires_action(name="cookie_requires_eggs")
        cookie_requires_eggs.has_a(dish=cookie, action=take_eggs)
        cookie_requires_butter = dish_requires_action(name="cookie_requires_butter")
        cookie_requires_butter.has_a(dish=cookie, action=take_butter)
        cookie_requires_mix = dish_requires_action(name="cookie_requires_mix_dough")
        cookie_requires_mix.has_a(dish=cookie, action=mix_dough)
        cookie_requires_bake = dish_requires_action(name="cookie_requires_bake_cookies")
        cookie_requires_bake.has_a(dish=cookie, action=bake_cookies)
        cookie_requires_serve = dish_requires_action(name="cookie_requires_serve")
        cookie_requires_serve.has_a(dish=cookie, action=serve)

        omelette_requires_eggs = dish_requires_action(name="omelette_requires_eggs")
        omelette_requires_eggs.has_a(dish=omelette, action=take_eggs)
        omelette_requires_cheese = dish_requires_action(name="omelette_requires_cheese")
        omelette_requires_cheese.has_a(dish=omelette, action=take_cheese)
        omelette_requires_cook = dish_requires_action(name="omelette_requires_cook")
        omelette_requires_cook.has_a(dish=omelette, action=cook_omelette)
        omelette_requires_serve = dish_requires_action(name="omelette_requires_serve")
        omelette_requires_serve.has_a(dish=omelette, action=serve)

        salad_requires_lettuce = dish_requires_action(name="salad_requires_lettuce")
        salad_requires_lettuce.has_a(dish=salad, action=take_lettuce)
        salad_requires_chop = dish_requires_action(name="salad_requires_chop")
        salad_requires_chop.has_a(dish=salad, action=chop_lettuce)
        salad_requires_serve = dish_requires_action(name="salad_requires_serve")
        salad_requires_serve.has_a(dish=salad, action=serve)

        def reference_fact(name, dish_value, position, action_value):
            fact = reference_plan_step(name=name)
            fact.has_a(dish=dish_value, position=position, action=action_value)
            return fact

        # Declared domain facts: valid reference plans.
        reference_fact("cookie_step_0_open_fridge", cookie, pos_0, open_fridge)
        reference_fact("cookie_step_1_take_eggs", cookie, pos_1, take_eggs)
        reference_fact("cookie_step_2_take_butter", cookie, pos_2, take_butter)
        reference_fact("cookie_step_3_close_fridge", cookie, pos_3, close_fridge)
        reference_fact("cookie_step_4_put_on_table", cookie, pos_4, put_on_table)
        reference_fact("cookie_step_5_mix_dough", cookie, pos_5, mix_dough)
        reference_fact("cookie_step_6_bake_cookies", cookie, pos_6, bake_cookies)
        reference_fact("cookie_step_7_serve", cookie, pos_7, serve)
        reference_fact("cookie_step_8_done", cookie, pos_8, done)

        reference_fact("omelette_step_0_open_fridge", omelette, pos_0, open_fridge)
        reference_fact("omelette_step_1_take_eggs", omelette, pos_1, take_eggs)
        reference_fact("omelette_step_2_take_cheese", omelette, pos_2, take_cheese)
        reference_fact("omelette_step_3_close_fridge", omelette, pos_3, close_fridge)
        reference_fact("omelette_step_4_put_on_table", omelette, pos_4, put_on_table)
        reference_fact("omelette_step_5_cook_omelette", omelette, pos_5, cook_omelette)
        reference_fact("omelette_step_6_serve", omelette, pos_6, serve)
        reference_fact("omelette_step_7_done", omelette, pos_7, done)

        reference_fact("salad_step_0_open_fridge", salad, pos_0, open_fridge)
        reference_fact("salad_step_1_take_lettuce", salad, pos_1, take_lettuce)
        reference_fact("salad_step_2_close_fridge", salad, pos_2, close_fridge)
        reference_fact("salad_step_3_put_on_table", salad, pos_3, put_on_table)
        reference_fact("salad_step_4_chop_lettuce", salad, pos_4, chop_lettuce)
        reference_fact("salad_step_5_serve", salad, pos_5, serve)
        reference_fact("salad_step_6_done", salad, pos_6, done)

        def phase_action(name, source_phase, action_value, target_phase):
            fact = phase_transition(name=name)
            fact.has_a(
                source_phase=source_phase,
                action=action_value,
                target_phase=target_phase,
            )
            allowed = phase_allows_action(name=f"{name}_allowed")
            allowed.has_a(phase=source_phase, action=action_value)
            return fact

        # Declared domain facts: phase/action transitions for later HMM masks.
        phase_action("start_open_fridge", start, open_fridge, fridge_open)
        phase_action("fridge_open_take_eggs", fridge_open, take_eggs, fridge_open)
        phase_action("fridge_open_take_butter", fridge_open, take_butter, fridge_open)
        phase_action("fridge_open_take_lettuce", fridge_open, take_lettuce, fridge_open)
        phase_action("fridge_open_take_cheese", fridge_open, take_cheese, fridge_open)
        phase_action("fridge_open_close_fridge", fridge_open, close_fridge, after_fridge)
        phase_action("after_fridge_put_on_table", after_fridge, put_on_table, table_ready)
        phase_action("table_ready_mix_dough", table_ready, mix_dough, prep)
        phase_action("table_ready_chop_lettuce", table_ready, chop_lettuce, prep)
        phase_action("table_ready_cook_omelette", table_ready, cook_omelette, cook)
        phase_action("prep_bake_cookies", prep, bake_cookies, cook)
        phase_action("prep_serve", prep, serve, served)
        phase_action("cook_serve", cook, serve, served)
        phase_action("served_done", served, done, done_phase)
        phase_action("done_stays_done", done_phase, done, done_phase)

        def action_value(action_concept):
            return getattr(planned_action, action_concept.name)

        def exists_action(action_concept, variable):
            return existsAL(action_value(action_concept)(variable))

        def any_action(action_concepts, variable):
            return orL(
                *[action_value(action_concept)(f"{variable}_{i}") for i, action_concept in enumerate(action_concepts)]
            )

        def require_actions_for_dish(dish_value, required_actions):
            ifL(
                getattr(planned_dish, dish_value.name)("p"),
                andL(
                    *[
                        exists_action(action_concept, f"{dish_value.name}_{action_concept.name}")
                        for action_concept in required_actions
                    ]
                ),
            )

        # If a plan is for a dish, it must contain that dish's declared actions.
        require_actions_for_dish(
            cookie,
            (take_eggs, take_butter, mix_dough, bake_cookies, serve),
        )
        require_actions_for_dish(
            omelette,
            (take_eggs, take_cheese, cook_omelette, serve),
        )
        require_actions_for_dish(
            salad,
            (take_lettuce, chop_lettuce, serve),
        )

        # After terminal action `done`, every later step must also be `done`.
        ifL(
            precedes("order"),
            ifL(
                action_value(done)("earlier_step", path=("order", earlier)),
                action_value(done)("later_step", path=("order", later)),
            ),
        )

        # Bound the number of non-terminal steps and fridge openings.
        atMostAL(notL(action_value(done)("non_terminal_step")), max_steps)
        atMostAL(action_value(open_fridge)("fridge_open_step"), 2)

        # Any food-taking action requires some fridge opening in the plan.
        ifL(
            existsAL(any_action((take_eggs, take_butter, take_lettuce, take_cheese), "take")),
            exists_action(open_fridge, "required_open_fridge"),
        )

        # Any prep/cook action requires the food to be placed on the table.
        ifL(
            existsAL(
                any_action(
                    (mix_dough, chop_lettuce, bake_cookies, cook_omelette),
                    "prepare",
                )
            ),
            exists_action(put_on_table, "required_table"),
        )

        # Serving requires a dish-specific prep/cook action to have happened.
        ifL(
            getattr(planned_dish, "cookie")("cookie_plan"),
            ifL(
                exists_action(serve, "cookie_serve_step"),
                andL(
                    exists_action(mix_dough, "cookie_mix_step"),
                    exists_action(bake_cookies, "cookie_bake_step"),
                ),
            ),
        )
        ifL(
            getattr(planned_dish, "omelette")("omelette_plan"),
            ifL(
                exists_action(serve, "omelette_serve_step"),
                exists_action(cook_omelette, "omelette_cook_step"),
            ),
        )
        ifL(
            getattr(planned_dish, "salad")("salad_plan"),
            ifL(
                exists_action(serve, "salad_serve_step"),
                exists_action(chop_lettuce, "salad_chop_step"),
            ),
        )

    return graph, (
        plan,
        step,
        contains,
        planned_action,
        precedes,
        earlier,
        later,
        planned_dish,
        dish_concept,
        action,
        plan_phase,
    )
