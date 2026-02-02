from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


def translate_left_domiknows(all_program, current_idx, first_initial, indent=0):
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

    values = []
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

        filter_str = translate_left_domiknows(all_program, ins[0], first_initial=False)

        return f"{init_str}, {filter_str}" if filter_str != "" else f"{init_str}"

    elif fn == "union":
        if len(ins) != 2:
            raise ValueError(f"union expects 2 inputs at step {current_idx}")
        base_ins_l = translate_left_domiknows(all_program, ins[0], first_initial)
        base_ins_r = translate_left_domiknows(all_program, ins[1], first_initial)

        return f"orL({base_ins_l}, {base_ins_r})"

    elif fn == "count":
        if len(ins) != 1:
            raise ValueError(f"count expects 1 input at step {current_idx}")
        base_ins = translate_left_domiknows(all_program, ins[0], first_initial=True)

        return f"sumL(andL({base_ins}))"

    elif fn == "less_than":
        if len(ins) != 2:
            raise ValueError(f"less than expects 2 input at step {current_idx}")
        base_ins_l = translate_left_domiknows(all_program, ins[0], first_initial=True)
        var += 1
        base_ins_r = translate_left_domiknows(all_program, ins[1], first_initial=True)
        var += 1

        return f"lessL(andL({base_ins_l}), andL({base_ins_r}))"

    elif fn == "greater_than":
        if len(ins) != 2:
            raise ValueError(f"greater than expects 2 input at step {current_idx}")
        base_ins_l = translate_left_domiknows(all_program, ins[0], first_initial=True)
        var += 1
        base_ins_r = translate_left_domiknows(all_program, ins[1], first_initial=True)
        var += 1

        return f"greaterL(andL({base_ins_l}), andL({base_ins_r}))"

    elif fn == "unique":
        if len(ins) != 1:
            raise ValueError(f"exist expects 1 input at step {current_idx}")
        base_ins = translate_left_domiknows(all_program, ins[0], first_initial=True)
        var += 1

        return f"{base_ins}"

    elif fn == "exist":
        if len(ins) != 1:
            raise ValueError(f"exist expects 1 input at step {current_idx}")
        base_ins = translate_left_domiknows(all_program, ins[0], first_initial=True)
        var += 1
        return f"existsL(andL({base_ins}))"

    elif fn == "intersect":
        if len(ins) != 2:
            raise ValueError(f"intersect expects 2 inputs at step {current_idx}")
        base_ins_l = translate_left_domiknows(all_program, ins[0], first_initial)
        base_ins_r = translate_left_domiknows(all_program, ins[1], first_initial)

        return f"andL({base_ins_l}, {base_ins_r})"

    elif fn == "equal_integer":
        if len(ins) != 2:
            raise ValueError(f"union expects 2 inputs at step {current_idx}")

        base_ins_l = translate_left_domiknows(all_program, ins[0], first_initial=True)
        var += 1
        base_ins_r = translate_left_domiknows(all_program, ins[1], first_initial=True)
        var += 1

        return f"equalCounts(andL({base_ins_l}), andL({base_ins_r}))"

    elif "query" in fn:
        if len(ins) != 1:
            raise ValueError(f"query expects 1 inputs at step {current_idx}")
        query_type = fn.split("_")[1]
        target_obj = translate_left_domiknows(all_program, ins[0], first_initial=True)
        return f"queryL({query_type}, iota(andL({target_obj})))"

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
        next_obj = translate_left_domiknows(all_program, ins[0], first_initial=True)
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

    program = [{'inputs': [], 'function': 'scene', 'value_inputs': []},
               {'inputs': [0], 'function': 'filter_size', 'value_inputs': ['large']},
               {'inputs': [1], 'function': 'filter_color', 'value_inputs': ['brown']},
               {'inputs': [2], 'function': 'filter_shape', 'value_inputs': ['sphere']},
               {'inputs': [3], 'function': 'unique', 'value_inputs': []},
               {'inputs': [4], 'function': 'relate', 'value_inputs': ['left']},
               {'inputs': [], 'function': 'scene', 'value_inputs': []},
               {'inputs': [6], 'function': 'filter_color', 'value_inputs': ['brown']},
               {'inputs': [7], 'function': 'filter_shape', 'value_inputs': ['cylinder']},
               {'inputs': [8], 'function': 'unique', 'value_inputs': []},
               {'inputs': [9], 'function': 'relate', 'value_inputs': ['right']},
               {'inputs': [5, 10], 'function': 'intersect', 'value_inputs': []},
               {'inputs': [11], 'function': 'filter_size', 'value_inputs': ['large']},
               {'inputs': [12], 'function': 'unique', 'value_inputs': []},
               {'inputs': [13], 'function': 'query_material', 'value_inputs': []}]
    print(translate_left_domiknows(program, len(program) - 1, first_initial=True))
    # for result in results:
    #     str_op = translate_left_steps_to_your_dsl(result["program"])
