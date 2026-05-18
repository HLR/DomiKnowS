"""Declarative DomiKnowS graph for the package shipping planner domain."""
from __future__ import annotations

from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, atMostAL, existsAL, ifL, notL, orL


def build_graph(task: str = "ship_book", *, max_steps: int = 7):
    """Build the declarative package shipping graph."""

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("package_shipping") as graph:
        plan = Concept(name="plan")
        step = Concept(name="step")
        shipping_task = Concept(name="shipping_task")
        action = Concept(name="action")
        plan_phase = Concept(name="plan_phase")
        reference_position = Concept(name="reference_position")
        count_limit = Concept(name="count_limit")

        contains, = plan.contains(step)

        precedes = Concept(name="precedes")
        earlier, later = precedes.has_a(earlier=step, later=step)

        box_action = action(name="box_action")
        protection_action = action(name="protection_action")
        item_action = action(name="item_action")
        label_action = action(name="label_action")
        seal_action = action(name="seal_action")
        delivery_action = action(name="delivery_action")
        terminal_action = action(name="terminal_action")

        done = terminal_action(name="done")
        choose_box = box_action(name="choose_box")
        wrap_item = protection_action(name="wrap_item")
        add_padding = protection_action(name="add_padding")
        insert_item = item_action(name="insert_item")
        print_label = label_action(name="print_label")
        print_return_label = label_action(name="print_return_label")
        seal_box = seal_action(name="seal_box")
        drop_off = delivery_action(name="drop_off")
        request_pickup = delivery_action(name="request_pickup")

        actions = (
            done,
            choose_box,
            wrap_item,
            add_padding,
            insert_item,
            print_label,
            print_return_label,
            seal_box,
            drop_off,
            request_pickup,
        )

        planned_action = step(
            name="planned_action",
            ConceptClass=EnumConcept,
            values=[action_concept.name for action_concept in actions],
        )

        ship_book = shipping_task(name="ship_book")
        ship_fragile_vase = shipping_task(name="ship_fragile_vase")
        return_item = shipping_task(name="return_item")
        tasks = (ship_book, ship_fragile_vase, return_item)

        planned_shipping_task = plan(
            name="planned_shipping_task",
            ConceptClass=EnumConcept,
            values=[task_concept.name for task_concept in tasks],
        )

        if task not in planned_shipping_task.enum:
            raise ValueError(
                f"unknown shipping task {task!r}; expected one of {tuple(planned_shipping_task.enum)!r}"
            )

        start = plan_phase(name="start")
        box_ready = plan_phase(name="box_ready")
        item_protected = plan_phase(name="item_protected")
        item_inserted = plan_phase(name="item_inserted")
        labeled = plan_phase(name="labeled")
        sealed = plan_phase(name="sealed")
        shipped = plan_phase(name="shipped")
        done_phase = plan_phase(name="done_phase")

        pos_0 = reference_position(name="pos_0")
        pos_1 = reference_position(name="pos_1")
        pos_2 = reference_position(name="pos_2")
        pos_3 = reference_position(name="pos_3")
        pos_4 = reference_position(name="pos_4")
        pos_5 = reference_position(name="pos_5")
        pos_6 = reference_position(name="pos_6")
        pos_7 = reference_position(name="pos_7")

        max_1 = count_limit(name="max_1")
        max_plan_steps = count_limit(name=f"max_{max_steps}")

        task_requires_action = Concept(name="task_requires_action")
        task_requires_action.has_a(task=shipping_task, action=action)

        reference_plan_step = Concept(name="reference_plan_step")
        reference_plan_step.has_a(task=shipping_task, position=reference_position, action=action)

        phase_allows_action = Concept(name="phase_allows_action")
        phase_allows_action.has_a(phase=plan_phase, action=action)

        phase_transition = Concept(name="phase_transition")
        phase_transition.has_a(source_phase=plan_phase, action=action, target_phase=plan_phase)

        action_count_limit = Concept(name="action_count_limit")
        action_count_limit.has_a(action=action, limit=count_limit)

        non_terminal_action_count_limit = Concept(name="non_terminal_action_count_limit")
        non_terminal_action_count_limit.has_a(terminal_action=action, limit=count_limit)

        seal_box_at_most_once = action_count_limit(name="seal_box_at_most_once")
        seal_box_at_most_once.has_a(action=seal_box, limit=max_1)

        non_terminal_steps_at_most_max = non_terminal_action_count_limit(
            name="non_terminal_steps_at_most_max"
        )
        non_terminal_steps_at_most_max.has_a(terminal_action=done, limit=max_plan_steps)

        def required_fact(name, task_value, action_value):
            fact = task_requires_action(name=name)
            fact.has_a(task=task_value, action=action_value)
            return fact

        required_fact("ship_book_requires_box", ship_book, choose_box)
        required_fact("ship_book_requires_insert", ship_book, insert_item)
        required_fact("ship_book_requires_label", ship_book, print_label)
        required_fact("ship_book_requires_seal", ship_book, seal_box)
        required_fact("ship_book_requires_dropoff", ship_book, drop_off)

        required_fact("fragile_vase_requires_box", ship_fragile_vase, choose_box)
        required_fact("fragile_vase_requires_wrap", ship_fragile_vase, wrap_item)
        required_fact("fragile_vase_requires_padding", ship_fragile_vase, add_padding)
        required_fact("fragile_vase_requires_insert", ship_fragile_vase, insert_item)
        required_fact("fragile_vase_requires_label", ship_fragile_vase, print_label)
        required_fact("fragile_vase_requires_seal", ship_fragile_vase, seal_box)
        required_fact("fragile_vase_requires_dropoff", ship_fragile_vase, drop_off)

        required_fact("return_item_requires_box", return_item, choose_box)
        required_fact("return_item_requires_insert", return_item, insert_item)
        required_fact("return_item_requires_return_label", return_item, print_return_label)
        required_fact("return_item_requires_seal", return_item, seal_box)
        required_fact("return_item_requires_pickup", return_item, request_pickup)

        def reference_fact(name, task_value, position, action_value):
            fact = reference_plan_step(name=name)
            fact.has_a(task=task_value, position=position, action=action_value)
            return fact

        reference_fact("ship_book_step_0_choose_box", ship_book, pos_0, choose_box)
        reference_fact("ship_book_step_1_insert_item", ship_book, pos_1, insert_item)
        reference_fact("ship_book_step_2_print_label", ship_book, pos_2, print_label)
        reference_fact("ship_book_step_3_seal_box", ship_book, pos_3, seal_box)
        reference_fact("ship_book_step_4_drop_off", ship_book, pos_4, drop_off)
        reference_fact("ship_book_step_5_done", ship_book, pos_5, done)

        reference_fact("fragile_vase_step_0_choose_box", ship_fragile_vase, pos_0, choose_box)
        reference_fact("fragile_vase_step_1_wrap_item", ship_fragile_vase, pos_1, wrap_item)
        reference_fact("fragile_vase_step_2_add_padding", ship_fragile_vase, pos_2, add_padding)
        reference_fact("fragile_vase_step_3_insert_item", ship_fragile_vase, pos_3, insert_item)
        reference_fact("fragile_vase_step_4_print_label", ship_fragile_vase, pos_4, print_label)
        reference_fact("fragile_vase_step_5_seal_box", ship_fragile_vase, pos_5, seal_box)
        reference_fact("fragile_vase_step_6_drop_off", ship_fragile_vase, pos_6, drop_off)
        reference_fact("fragile_vase_step_7_done", ship_fragile_vase, pos_7, done)

        reference_fact("return_item_step_0_choose_box", return_item, pos_0, choose_box)
        reference_fact("return_item_step_1_insert_item", return_item, pos_1, insert_item)
        reference_fact("return_item_step_2_print_return_label", return_item, pos_2, print_return_label)
        reference_fact("return_item_step_3_seal_box", return_item, pos_3, seal_box)
        reference_fact("return_item_step_4_request_pickup", return_item, pos_4, request_pickup)
        reference_fact("return_item_step_5_done", return_item, pos_5, done)

        def phase_action(name, source_phase, action_value, target_phase):
            fact = phase_transition(name=name)
            fact.has_a(source_phase=source_phase, action=action_value, target_phase=target_phase)
            allowed = phase_allows_action(name=f"{name}_allowed")
            allowed.has_a(phase=source_phase, action=action_value)
            return fact

        phase_action("start_choose_box", start, choose_box, box_ready)
        phase_action("box_ready_wrap_item", box_ready, wrap_item, item_protected)
        phase_action("box_ready_insert_item", box_ready, insert_item, item_inserted)
        phase_action("item_protected_add_padding", item_protected, add_padding, item_protected)
        phase_action("item_protected_insert_item", item_protected, insert_item, item_inserted)
        phase_action("item_inserted_print_label", item_inserted, print_label, labeled)
        phase_action("item_inserted_print_return_label", item_inserted, print_return_label, labeled)
        phase_action("labeled_seal_box", labeled, seal_box, sealed)
        phase_action("sealed_drop_off", sealed, drop_off, shipped)
        phase_action("sealed_request_pickup", sealed, request_pickup, shipped)
        phase_action("shipped_done", shipped, done, done_phase)
        phase_action("done_stays_done", done_phase, done, done_phase)

        def action_value(action_concept):
            return getattr(planned_action, action_concept.name)

        def exists_action(action_concept, variable):
            return existsAL(action_value(action_concept)(variable))

        def any_action(action_concepts, variable):
            return orL(
                *[
                    action_value(action_concept)(f"{variable}_{i}")
                    for i, action_concept in enumerate(action_concepts)
                ]
            )

        def require_actions_for_task(task_value, required_actions):
            ifL(
                getattr(planned_shipping_task, task_value.name)("p"),
                andL(
                    *[
                        exists_action(action_concept, f"{task_value.name}_{action_concept.name}")
                        for action_concept in required_actions
                    ]
                ),
            )

        require_actions_for_task(ship_book, (choose_box, insert_item, print_label, seal_box, drop_off))
        require_actions_for_task(
            ship_fragile_vase,
            (choose_box, wrap_item, add_padding, insert_item, print_label, seal_box, drop_off),
        )
        require_actions_for_task(
            return_item,
            (choose_box, insert_item, print_return_label, seal_box, request_pickup),
        )

        ifL(
            precedes("order"),
            ifL(
                action_value(done)("earlier_step", path=("order", earlier)),
                action_value(done)("later_step", path=("order", later)),
            ),
        )

        atMostAL(notL(action_value(done)("non_terminal_step")), max_steps)
        atMostAL(action_value(seal_box)("seal_step"), 1)

        ifL(
            getattr(planned_shipping_task, "ship_fragile_vase")("fragile_plan"),
            exists_action(add_padding, "fragile_padding_step"),
        )
        ifL(
            getattr(planned_shipping_task, "return_item")("return_plan"),
            exists_action(print_return_label, "return_label_step"),
        )
        ifL(
            existsAL(any_action((drop_off, request_pickup), "delivery")),
            exists_action(seal_box, "required_seal_step"),
        )
        ifL(
            exists_action(insert_item, "insert_step"),
            exists_action(choose_box, "required_box_step"),
        )
        ifL(
            existsAL(any_action((print_label, print_return_label), "label")),
            exists_action(insert_item, "required_insert_step"),
        )

    return graph, (
        plan,
        step,
        contains,
        planned_action,
        precedes,
        earlier,
        later,
        planned_shipping_task,
        shipping_task,
        action,
        plan_phase,
    )
