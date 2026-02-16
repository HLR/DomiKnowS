from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# ---------- Internal expression trees ----------
@dataclass(frozen=True)
class Cond:  # boolean condition over variable x
    op: str
    args: tuple


def AND(*conds: Cond) -> Cond:
    # flatten nested ands and drop trivial singletons
    flat = []
    for c in conds:
        if c.op == "and":
            flat.extend(c.args)
        else:
            flat.append(c)
    if len(flat) == 1:
        return flat[0]
    return Cond("and", tuple(flat))


def OR(*conds: Cond) -> Cond:
    flat = []
    for c in conds:
        if c.op == "or":
            flat.extend(c.args)
        else:
            flat.append(c)
    if len(flat) == 1:
        return flat[0]
    return Cond("or", tuple(flat))


def LESS(*conds: Cond) -> Cond:
    flat = []
    for c in conds:
        if c.op == "count":
            flat.append(c.args)
        else:
            flat.append(c)
    return Cond("less", tuple(flat))


def MORE(*conds: Cond) -> Cond:
    flat = []
    for c in conds:
        if c.op == "count":
            flat.append(c.args)
        else:
            flat.append(c)
    return Cond("greater", tuple(flat))


def EQUAL(*conds: Cond) -> Cond:
    flat = []
    for c in conds:
        if c.op == "count":
            flat.append(c.args)
        else:
            flat.append(c)
    return Cond("equalCounts", tuple(flat))


def PRED(name: str) -> Cond:
    return Cond("pred", (name,))


def RELATE(name: str) -> Cond:
    return Cond("relate", (name,))


# ---------- Rendering to YOUR DSL ----------
var = 0
use_count = 0
relation_val = 0
need_relation2 = False


def render_cond_to_your_dsl(cond: Cond) -> str:
    """
    Renders Cond to:
      - pred: is_<name>('x') or is_<name>(path='x') depending on `use_path_kw`
      - and/or: andL(...), orL(...)
    """
    global relation_val
    global var
    global use_count
    global need_relation2
    if cond.op == "pred":
        if cond.args[0] == "scene":
            return ""
        pred_name = cond.args[0]
        # your examples mix is_large('x') and is_gray(path='x').
        # We'll default to using path='x' for most, but allow special-casing below.
        var_name = chr(var + 97)
        use_count += 1
        suffix = ""
        if use_count > 1:
            return f"{pred_name}(path='{var_name}')"
        if need_relation2:
            suffix = f", path=('rel{relation_val}', obj2)"
            need_relation2 = False
        return f"{pred_name}('{var_name}'{suffix})"

    if cond.op == "and" or cond.op == "or":
        all_str = [render_cond_to_your_dsl(c) for c in cond.args[:]]
        all_str = [c for c in all_str if c != ""]
        if len(all_str) == 1:
            return all_str[0]
        return f"{cond.op}L(" + ", ".join(all_str) + ")"

    if cond.op == "unique":
        expr = render_cond_to_your_dsl(cond.args)
        return f"{expr}"

    if cond.op == "count":
        var += 1
        use_count = 0
        expr = render_cond_to_your_dsl(cond.args)
        return f"sumL({expr})"

    if cond.op == "equalCounts" or cond.op == "less" or cond.op == "greater":
        var += 1
        use_count = 0
        x_vars = render_cond_to_your_dsl(cond.args[0])
        var += 1
        use_count = 0
        y_vars = render_cond_to_your_dsl(cond.args[1])
        return f"{cond.op}L({x_vars}, {y_vars})"

    if cond.op == "exist":
        var += 1
        use_count = 0
        expr = render_cond_to_your_dsl(cond.args)
        return f"existsL(andL({expr}))"

    if cond.op == "relate":
        var_name = chr(var + 97)
        pred_name = cond.args[0]
        relation_val += 1
        str_path = f"{pred_name}('rel{relation_val}', path=('{var_name}', obj1.reversed))"

        need_relation2 = True

        return str_path

    raise ValueError(f"Unknown Cond op: {cond.op}")


# ---------- Translator from their step program ----------
def translate_left_steps_to_your_dsl(all_program, current_idx, first_initial, indent=0):
    """
    Supports the common CLEVR-like operators:
      scene, filter_color, filter_material, filter_shape, filter_size,
      union, intersect (optional), count

    Assumption: these "set" values correspond to a condition over a single object variable `x`.
    """

    # Each step i produces either:
    #   - a Cond (representing a set/filter condition over x)
    #   - a final aggregation marker (count)
    global relation_val
    global var
    global use_count
    global need_relation2
    if current_idx == len(all_program) - 1:
        var = 0
        use_count = 0
        relation_val = 0
        need_relation2 = False

    values: List[Union[Cond, str]] = []
    step = all_program[current_idx]

    fn = step["function"]
    ins = step.get("inputs", [])
    vins = step.get("value_inputs", [])
    if fn == "scene":
        # "all objects" -> True condition (neutral for AND).
        # We'll represent as a special predicate TRUE and remove it later by simplification
        # OR simply use an empty AND identity; easiest: use pred "__true__" and drop in render.
        if not first_initial:
            return ""
        suffix = ""
        var_name = chr(var + 97)
        var += 1
        if need_relation2:
            suffix = f", path=('rel{relation_val - 1}', obj2)"
            need_relation2 = False
        return f"obj('{var_name}'{suffix})"

    elif fn.startswith("filter_"):
        if len(ins) != 1 or len(vins) != 1:
            raise ValueError(f"{fn} expects 1 input and 1 value_input at step {current_idx}")

        attr_value = str(vins[0])

        if first_initial:
            var_name = chr(var + 97)
            var += 1
            relation_suffix = ""
            if need_relation2:
                relation_suffix = f", path=('rel{relation_val - 1}', obj2)"
                need_relation2 = False
            init_str = f"{attr_value}('{var_name}'{relation_suffix})"
        else:
            var_name = chr(var + 96)
            init_str = f"{attr_value}(path=('{var_name}'))"

        filter_str = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial=False)

        return f"{init_str}, {filter_str}" if filter_str != "" else f"{init_str}"

    elif fn == "union":
        if len(ins) != 2:
            raise ValueError(f"union expects 2 inputs at step {current_idx}")
        base_ins_l = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial)
        base_ins_r = translate_left_steps_to_your_dsl(all_program, ins[1], first_initial)

        return f"orL({base_ins_l}, {base_ins_r})"

    elif fn == "count":
        if len(ins) != 1:
            raise ValueError(f"count expects 1 input at step {current_idx}")
        base_ins = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial=True)

        return f"sumL(andL({base_ins}))"

    elif fn == "less_than":
        if len(ins) != 2:
            raise ValueError(f"less than expects 2 input at step {current_idx}")
        base_ins_l = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial=True)
        var += 1
        base_ins_r = translate_left_steps_to_your_dsl(all_program, ins[1], first_initial=True)
        var += 1

        return f"lessL(andL({base_ins_l}), andL({base_ins_r}))"

    elif fn == "greater_than":
        if len(ins) != 2:
            raise ValueError(f"greater than expects 2 input at step {current_idx}")
        base_ins_l = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial=True)
        var += 1
        base_ins_r = translate_left_steps_to_your_dsl(all_program, ins[1], first_initial=True)
        var += 1

        return f"greaterL(andL({base_ins_l}), andL({base_ins_r}))"

    elif fn == "unique":
        if len(ins) != 1:
            raise ValueError(f"exist expects 1 input at step {current_idx}")
        base_ins = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial=True)
        var += 1

        return f"{base_ins}"

    elif fn == "exist":
        if len(ins) != 1:
            raise ValueError(f"exist expects 1 input at step {current_idx}")
        base_ins = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial=True)
        var += 1
        return f"existsL(andL({base_ins}))"

    elif fn == "intersect":
        if len(ins) != 2:
            raise ValueError(f"intersect expects 2 inputs at step {current_idx}")
        base_ins_l = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial)
        base_ins_r = translate_left_steps_to_your_dsl(all_program, ins[1], first_initial)

        return f"andL({base_ins_l}, {base_ins_r})"

    elif fn == "equal_integer":
        if len(ins) != 2:
            raise ValueError(f"union expects 2 inputs at step {current_idx}")

        base_ins_l = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial=True)
        var += 1
        base_ins_r = translate_left_steps_to_your_dsl(all_program, ins[1], first_initial=True)
        var += 1

        return f"equalCounts(andL({base_ins_l}), andL({base_ins_r}))"

    elif fn == "relate":
        if len(ins) != 1:
            raise ValueError(f"relate expects 1 input at step {current_idx}")

        attr_value = str(vins[0])

        init_relation_val = relation_val

        if first_initial:
            var_name = chr(var + 97)
            var += 1
            suffix = ""
            if need_relation2:
                suffix = f", path=('rel{relation_val - 1}', obj2)"
                need_relation2 = False
            str_path = f"obj('{var_name}'{suffix}), {attr_value}('rel{init_relation_val}', path=('{var_name}', obj1.reversed))"
        else:
            var_name = chr(var + 96)
            str_path = f"{attr_value}('rel{init_relation_val}', path=('{var_name}', obj1.reversed))"

        need_relation2 = True
        relation_val += 1
        next_obj = translate_left_steps_to_your_dsl(all_program, ins[0], first_initial=True)

        return f"{str_path}, {next_obj}"
    else:
        raise NotImplementedError(f"Unsupported function '{fn}' at step {current_idx}")


if __name__ == "__main__":
    import json

    with open("convert_CLEVR_program_manual_10_first_translation.json", 'rb') as file:
        results = json.load(file)

    # for program in results:
    #     for program_info in program["CLEVR_program"]:
    #         print(program_info)
    #     print(translate_left_steps_to_your_dsl(program["CLEVR_program"], len(program["CLEVR_program"]) - 1,
    #                                            first_initial=True))
    #     break

    program = [
        {
            "inputs": [],
            "function": "scene",
            "value_inputs": []
        },
        {
            "inputs": [
                0
            ],
            "function": "filter_shape",
            "value_inputs": [
                "cube"
            ]
        },
        {
            "inputs": [
                1
            ],
            "function": "unique",
            "value_inputs": []
        },
        {
            "inputs": [
                2
            ],
            "function": "relate",
            "value_inputs": [
                "front"
            ]
        },
        {
            "inputs": [
                3
            ],
            "function": "unique",
            "value_inputs": []
        },
        {
            "inputs": [
                4
            ],
            "function": "relate",
            "value_inputs": [
                "left"
            ]
        },
        {
            "inputs": [
                5
            ],
            "function": "exist",
            "value_inputs": []
        }
    ]
    print(translate_left_steps_to_your_dsl(program, len(program) - 1, first_initial=True))
    # for result in results:
    #     str_op = translate_left_steps_to_your_dsl(result["program"])