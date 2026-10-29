"""Entity-order augmentation that preserves scene names and target meaning."""
from copy import deepcopy
from dataclasses import replace
import random


def permute_entities(context, plan, order):
    result = dict(context)
    entities = tuple(context.get("entity_table", context.get("entities", ())))
    if sorted(order) != list(range(len(entities))):
        raise ValueError("entity permutation must contain every original index exactly once")
    inverse = {old: new for new, old in enumerate(order)}
    if context.get("planning_facts") is not None:
        from domiknows.generation.grounded import PlanningFacts
        from .semantics import resolve_pointer
        def resolve(value):
            name, valid = resolve_pointer(value, dict(enumerate(entities)))
            if not valid:
                raise ValueError(f"planning entity {value!r} is absent")
            return name
        raw = context["planning_facts"]
        result["planning_facts"] = PlanningFacts.parse(raw, entities=entities, resolve=resolve, constants=("left", "right"))
        if isinstance(raw, dict) and raw.get("required_skill_sequence") and not result.get("task_skill_sequence"):
            result["task_skill_sequence"] = raw["required_skill_sequence"]
    result["entity_table"] = tuple(entities[i] for i in order)
    result.pop("entities", None)
    output = deepcopy(list(plan))
    for op in output:
        for key in ("target_entity_name", "target_container_name"):
            if key in op["params"]:
                value = op["params"][key]
                if isinstance(value, int) or str(value).isdigit():
                    op["params"][key] = inverse[int(value)]
    for key in ("task_target_entity", "task_target_container"):
        value = result.get(key)
        if value is not None and (isinstance(value, int) or str(value).isdigit()):
            result[key] = inverse[int(value)]
    return result, output


def diagnostic_examples(examples, mode="original", seed=42):
    """Input ablations; permutations remap labels solely to preserve meaning."""
    if mode == "original":
        return examples
    rng = random.Random(seed)
    result = []
    donor_indices = list(range(len(examples)))
    rng.shuffle(donor_indices)
    if len(donor_indices) > 1:
        for index in range(len(donor_indices)):
            if donor_indices[index] == index:
                other = (index + 1) % len(donor_indices)
                donor_indices[index], donor_indices[other] = donor_indices[other], donor_indices[index]
    donors = [examples[index] for index in donor_indices]
    for example, donor in zip(examples, donors):
        if mode == "permuted-entities":
            order = list(range(len(example.entities)))
            rng.shuffle(order)
            if len(order) > 1 and order == list(range(len(order))):
                order = order[1:] + order[:1]
            context, plan = permute_entities(example.as_reward_item(), example.operation_sequence, order)
            result.append(replace(example, entities=context["entity_table"], operation_sequence=tuple(plan),
                                  planning_facts=context.get("planning_facts"), task_skill_sequence=context.get("task_skill_sequence"),
                                  task_target_entity=context.get("task_target_entity"), task_target_container=context.get("task_target_container")))
        elif mode == "no-images":
            result.append(replace(example, image_paths=(), segmented_image_paths=()))
        elif mode == "shuffled-images":
            result.append(replace(example, image_paths=donor.image_paths, segmented_image_paths=donor.segmented_image_paths))
        elif mode == "shuffled-instructions":
            result.append(replace(example, instruction=donor.instruction))
        else:
            raise ValueError(f"unknown diagnostic input mode {mode!r}")
    return result
