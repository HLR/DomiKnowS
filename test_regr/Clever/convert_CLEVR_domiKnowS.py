from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import re


def _extract_first_var_symbol(expr: str) -> Optional[str]:
    """Return the first single-letter variable symbol found in a predicate call."""
    match = re.search(r"\('([a-z])'", expr)
    return match.group(1) if match else None


def translate_left_domiknows(
    all_program,
    current_idx,
    first_initial,
    apply_sum=True,
    indent=0,
    relation_syntax="legacy",
):
    """
    Supports the common CLEVR-like operators:
      scene, filter_color, filter_material, filter_shape, filter_size,
      union, intersect (optional), count.

    relation_syntax:
      "legacy"  – reified path-based:  left('rel0', path=('a', obj1.reversed))
      "binary"  – direct binary:       left('a', 'b')
    """

    # Each step i produces either:
    #   - a Cond (representing a set/filter condition over x)
    #   - a final aggregation marker (count)
    global relation_val
    global var
    global use_count
    global need_relation2
    global pending_arg2_nav
    if current_idx == len(all_program) - 1:
        var = 0
        use_count = 0
        relation_val = 0
        need_relation2 = False
        pending_arg2_nav = "obj2"

    values = []
    step = all_program[current_idx]

    fn = step["function"]
    ins = step.get("inputs", [])
    vins = step.get("value_inputs", [])
    if fn == "scene":
        if not first_initial:
            var_name = chr(var + 96)
            if relation_syntax == "binary":
                return f"obj('{var_name}')", 0
            return f"obj(path=('{var_name}'))", 0
        suffix = ""
        var += 1
        var_name = chr(var + 96)
        if need_relation2:
            suffix = f", path=('rel{relation_val - 1}', {pending_arg2_nav})"
            need_relation2 = False
        return f"obj('{var_name}'{suffix})", 0

    elif fn.startswith("filter_"):
        if len(ins) != 1 or len(vins) != 1:
            raise ValueError(f"{fn} expects 1 input and 1 value_input at step {current_idx}")

        attr_value = str(vins[0])

        if first_initial:
            var += 1
            var_name = chr(var + 96)
            relation_suffix = ""
            if need_relation2:
                if relation_syntax == "binary":
                    need_relation2 = False
                else:
                    relation_suffix = f", path=('rel{relation_val - 1}', {pending_arg2_nav})"
                    need_relation2 = False
            init_str = f"{attr_value}('{var_name}'{relation_suffix})"
        else:
            var_name = chr(var + 96)
            if relation_syntax == "binary":
                # In binary mode, use direct variable reference instead of path
                init_str = f"{attr_value}('{var_name}')"
            else:
                init_str = f"{attr_value}(path=('{var_name}'))"

        filter_str, depth_ins = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=False,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )
        if depth_ins == 0:
            filter_str = ""

        return (f"{init_str}, {filter_str}", depth_ins + 1) if filter_str != "" else (f"{init_str}", 1)

    elif fn == "union":
        if len(ins) != 2:
            raise ValueError(f"union expects 2 inputs at step {current_idx}")
        base_ins_l, depth_ins_l = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )
        base_ins_l = f"andL({base_ins_l})" if depth_ins_l > 1 else base_ins_l
        base_ins_r, depth_ins_r = translate_left_domiknows(
            all_program,
            ins[1],
            first_initial,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )
        base_ins_r = f"andL({base_ins_r})" if depth_ins_r > 1 else base_ins_r
        return f"orL({base_ins_l}, {base_ins_r})", 1

    elif fn == "count":
        if len(ins) != 1:
            raise ValueError(f"count expects 1 input at step {current_idx}")
        base_ins, depth_ins = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )

        final_ins = f"andL({base_ins})" if depth_ins > 1 else base_ins

        return f"sumL({final_ins})" if apply_sum else f"{final_ins}", depth_ins + 1

    elif fn == "less_than":
        if len(ins) != 2:
            raise ValueError(f"less than expects 2 input at step {current_idx}")
        base_ins_l, depth_l = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=False,
            relation_syntax=relation_syntax,
        )
        base_ins_r, depth_r = translate_left_domiknows(
            all_program,
            ins[1],
            first_initial=True,
            apply_sum=False,
            relation_syntax=relation_syntax,
        )

        return f"lessL({base_ins_l}, {base_ins_r})", max(depth_l, depth_l) + 1

    elif fn == "greater_than":
        if len(ins) != 2:
            raise ValueError(f"greater than expects 2 input at step {current_idx}")
        base_ins_l, depth_l = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=False,
            relation_syntax=relation_syntax,
        )
        base_ins_r, depth_r = translate_left_domiknows(
            all_program,
            ins[1],
            first_initial=True,
            apply_sum=False,
            relation_syntax=relation_syntax,
        )

        return f"greaterL({base_ins_l}, {base_ins_r})", max(depth_l, depth_l) + 1

    elif fn == "unique":
        if len(ins) != 1:
            raise ValueError(f"exist expects 1 input at step {current_idx}")
        base_ins, depth = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )

        return f"{base_ins}", depth + 1

    elif fn == "exist":
        if len(ins) != 1:
            raise ValueError(f"exist expects 1 input at step {current_idx}")
        base_ins, depth = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )

        final_ins = f"andL({base_ins})" if depth > 1 else base_ins

        return f"existsL({final_ins})", depth + 1

    elif fn == "intersect":
        if len(ins) != 2:
            raise ValueError(f"intersect expects 2 inputs at step {current_idx}")
        base_ins_l, depth_l = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )
        base_ins_r, depth_r = translate_left_domiknows(
            all_program,
            ins[1],
            first_initial,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )

        return f"andL({base_ins_l}, {base_ins_r})", max(depth_l, depth_l) + 1

    elif fn.startswith("equal_") and fn != "equal_integer":
        # equal_color, equal_shape, equal_material, equal_size
        # Binary comparison: do two objects have the same attribute?
        # Uses sameL(attribute_concept, 'x', 'y') — no relation node needed.
        if len(ins) != 2:
            raise ValueError(f"{fn} expects 2 inputs at step {current_idx}")

        attr_suffix = fn.replace("equal_", "")

        base_ins_l, depth_l = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )
        base_ins_r, depth_r = translate_left_domiknows(
            all_program,
            ins[1],
            first_initial=True,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )

        var_l = _extract_first_var_symbol(base_ins_l)
        var_r = _extract_first_var_symbol(base_ins_r)

        same_term = f"sameL({attr_suffix}, '{var_l}', '{var_r}')"

        return f"existsL(andL({base_ins_l}, {base_ins_r}, {same_term}))", max(depth_l, depth_r) + 1

    elif fn == "equal_integer":
        if len(ins) != 2:
            raise ValueError(f"union expects 2 inputs at step {current_idx}")

        base_ins_l, depth_l = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=False,
            relation_syntax=relation_syntax,
        )
        base_ins_r, depth_r = translate_left_domiknows(
            all_program,
            ins[1],
            first_initial=True,
            apply_sum=False,
            relation_syntax=relation_syntax,
        )

        return f"equalCountsL({base_ins_l}, {base_ins_r})", max(depth_l, depth_l) + 1

    elif "query" in fn:
        if len(ins) != 1:
            raise ValueError(f"query expects 1 inputs at step {current_idx}")
        query_type = fn.split("_")[1]
        target_obj, depth = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )
        return f"queryL({query_type}, iotaL({target_obj}))", depth + 1

    elif fn.startswith("same_") or fn.startswith("different_"):
        # same_shape, same_color, same_material, same_size
        # different_shape, different_color, different_material, different_size
        # Semantics: given input object, find objects with the same / different attribute.
        # Uses sameL / differentL — no relation node or pair concept needed.
        if len(ins) != 1:
            raise ValueError(f"{fn} expects 1 input at step {current_idx}")

        is_different = fn.startswith("different_")
        attr_suffix = fn.replace("different_", "") if is_different else fn.replace("same_", "")
        constraint_name = "differentL" if is_different else "sameL"

        if first_initial:
            var_name = chr(var + 97)
            var += 1
            suffix = ""
            if need_relation2:
                if relation_syntax == "binary":
                    need_relation2 = False
                else:
                    suffix = f", path=('rel{relation_val - 1}', {pending_arg2_nav})"
                    need_relation2 = False
            obj_term = f"obj('{var_name}'{suffix})"
        else:
            var_name = chr(var + 96)
            obj_term = None

        next_obj, depth = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )
        related_var = _extract_first_var_symbol(next_obj)

        same_term = f"{constraint_name}({attr_suffix}, '{var_name}', '{related_var}')"

        if obj_term is not None:
            return f"{obj_term}, {next_obj}, {same_term}", depth + 1
        return f"{next_obj}, {same_term}", depth + 1

    elif fn == "relate":
        if len(ins) != 1:
            raise ValueError(f"relate expects 1 input at step {current_idx}")

        attr_value = str(vins[0])

        # Determine whether this is a reverse-direction relation (_rev suffix).
        # Forward relations (left, right, front, behind) live on pair_forward and
        # are navigated via obj1 / obj2.
        # Reverse relations (left_rev, right_rev, …) live on pair_reverse and
        # are navigated via obj1_rev / obj2_rev.
        is_reverse = attr_value.endswith("_rev")
        arg1_nav = "obj1_rev" if is_reverse else "obj1"
        arg2_nav = "obj2_rev" if is_reverse else "obj2"

        init_relation_val = relation_val

        if first_initial:
            var_name = chr(var + 97)
            var += 1
            suffix = ""
            if need_relation2:
                if relation_syntax == "binary":
                    need_relation2 = False
                else:
                    suffix = f", path=('rel{relation_val - 1}', {pending_arg2_nav})"
                    need_relation2 = False
            obj_term = f"obj('{var_name}'{suffix})"
        else:
            var_name = chr(var + 96)
            obj_term = None

        next_obj, depth = translate_left_domiknows(
            all_program,
            ins[0],
            first_initial=True,
            apply_sum=apply_sum,
            relation_syntax=relation_syntax,
        )
        related_var = _extract_first_var_symbol(next_obj)

        relation_legacy = f"{attr_value}('rel{init_relation_val}', path=('{var_name}', {arg1_nav}.reversed))"

        # Binary syntax is emitted here when requested.
        relation_binary = None
        if related_var is not None and relation_syntax == "binary":
            relation_binary = f"{attr_value}('{var_name}', '{related_var}')"

        if relation_syntax == "legacy":
            relation_term = relation_legacy
        elif relation_syntax == "binary":
            relation_term = relation_binary if relation_binary is not None else relation_legacy
        else:
            raise ValueError("relation_syntax must be one of: legacy, binary")

        need_relation2 = True
        pending_arg2_nav = arg2_nav
        relation_val += 1

        # For binary, ensure the RHS object variable is introduced first.
        # This avoids evaluating relation(a,b) before predicates that bind b.
        if relation_syntax == "binary" and relation_binary is not None:
            if obj_term is not None:
                return f"{obj_term}, {next_obj}, {relation_term}", depth + 1
            return f"{next_obj}, {relation_term}", depth + 1

        if obj_term is not None:
            return f"{obj_term}, {relation_term}, {next_obj}", depth + 1
        return f"{relation_term}, {next_obj}", depth + 1
    else:
        raise NotImplementedError(f"Unsupported function '{fn}' at step {current_idx}")


if __name__ == "__main__":
    import json

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

    print("=== Legacy syntax ===")
    print(translate_left_domiknows(program, len(program) - 1, first_initial=True,
                                   relation_syntax="legacy"))
    print()
    print("=== Binary syntax ===")
    print(translate_left_domiknows(program, len(program) - 1, first_initial=True,
                                   relation_syntax="binary"))

    # Simple one-relation test
    simple_program = [
        {"inputs": [], "function": "scene", "value_inputs": []},
        {"inputs": [0], "function": "filter_shape", "value_inputs": ["cube"]},
        {"inputs": [1], "function": "filter_color", "value_inputs": ["red"]},
        {"inputs": [2], "function": "unique", "value_inputs": []},
        {"inputs": [3], "function": "relate", "value_inputs": ["left"]},
        {"inputs": [4], "function": "filter_size", "value_inputs": ["large"]},
        {"inputs": [5], "function": "exist", "value_inputs": []},
    ]

    print()
    print("=== Simple one-relation (legacy) ===")
    print(translate_left_domiknows(simple_program, len(simple_program) - 1,
                                   first_initial=True, relation_syntax="legacy"))
    print()
    print("=== Simple one-relation (binary) ===")
    print(translate_left_domiknows(simple_program, len(simple_program) - 1,
                                   first_initial=True, relation_syntax="binary"))