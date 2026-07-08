from __future__ import annotations

import re
from typing import Any

g_attribute_concepts: dict[str, list[str]] = {
    "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "material": ["rubber", "metal"],
    "shape": ["cube", "sphere", "cylinder"],
    "size": ["small", "large"],
}

g_relational_concepts: dict[str, list[str]] = {
    "spatial_relation": ["left", "right", "front", "behind"],
}


def _extract_first_var_symbol(expr: str) -> str | None:
    match = re.search(r"\(['\"]([a-z])['\"]", expr)
    return match.group(1) if match else None


class _ProgramTranslator:
    """Recursive CLEVR functional-program translator with explicit state."""

    def __init__(self, program: list[dict[str, Any]], relation_syntax: str = "legacy"):
        if relation_syntax not in {"legacy", "binary"}:
            raise ValueError("relation_syntax must be one of: legacy, binary")
        self.program = program
        self.relation_syntax = relation_syntax
        self.var = 0
        self.relation_val = 0
        self.need_relation2 = False
        self.pending_arg2_nav = "obj2"

    def translate(self) -> str:
        if not self.program:
            return 'existsL(obj("a"))'
        expression, _depth = self._translate_at(len(self.program) - 1, first_initial=True)
        return expression

    def _translate_at(
        self,
        current_idx: int,
        *,
        first_initial: bool,
        apply_sum: bool = True,
    ) -> tuple[str, int]:
        step = self.program[current_idx]
        fn = step.get("function", step.get("type"))
        ins = step.get("inputs", [])
        vins = step.get("value_inputs", [])

        if fn == "scene":
            if not first_initial:
                var_name = chr(self.var + 96)
                if self.relation_syntax == "binary":
                    return f"obj('{var_name}')", 0
                return f"obj(path=('{var_name}'))", 0
            suffix = ""
            self.var += 1
            var_name = chr(self.var + 96)
            if self.need_relation2:
                suffix = f", path=('rel{self.relation_val - 1}', {self.pending_arg2_nav})"
                self.need_relation2 = False
            return f"obj('{var_name}'{suffix})", 0

        if fn and fn.startswith("filter_"):
            if len(ins) != 1 or len(vins) != 1:
                raise ValueError(f"{fn} expects 1 input and 1 value_input at step {current_idx}")
            attr_value = str(vins[0])
            if first_initial:
                self.var += 1
                var_name = chr(self.var + 96)
                relation_suffix = ""
                if self.need_relation2:
                    if self.relation_syntax == "binary":
                        self.need_relation2 = False
                    else:
                        relation_suffix = f", path=('rel{self.relation_val - 1}', {self.pending_arg2_nav})"
                        self.need_relation2 = False
                init_str = f"{attr_value}('{var_name}'{relation_suffix})"
            else:
                var_name = chr(self.var + 96)
                if self.relation_syntax == "binary":
                    init_str = f"{attr_value}('{var_name}')"
                else:
                    init_str = f"{attr_value}(path=('{var_name}'))"
            filter_str, depth_ins = self._translate_at(ins[0], first_initial=False, apply_sum=apply_sum)
            if depth_ins == 0:
                filter_str = ""
            return (f"{init_str}, {filter_str}", depth_ins + 1) if filter_str else (init_str, 1)

        if fn == "union":
            if len(ins) != 2:
                raise ValueError(f"union expects 2 inputs at step {current_idx}")
            left, depth_left = self._translate_at(ins[0], first_initial=first_initial, apply_sum=apply_sum)
            right, depth_right = self._translate_at(ins[1], first_initial=first_initial, apply_sum=apply_sum)
            left = f"andL({left})" if depth_left > 1 else left
            right = f"andL({right})" if depth_right > 1 else right
            return f"orL({left}, {right})", 1

        if fn == "intersect":
            if len(ins) != 2:
                raise ValueError(f"intersect expects 2 inputs at step {current_idx}")
            left, depth_left = self._translate_at(ins[0], first_initial=first_initial, apply_sum=apply_sum)
            right, depth_right = self._translate_at(ins[1], first_initial=first_initial, apply_sum=apply_sum)
            return f"andL({left}, {right})", max(depth_left, depth_right) + 1

        if fn == "count":
            if len(ins) != 1:
                raise ValueError(f"count expects 1 input at step {current_idx}")
            base, depth = self._translate_at(ins[0], first_initial=True, apply_sum=apply_sum)
            final = f"andL({base})" if depth > 1 else base
            return (f"sumL({final})" if apply_sum else final), depth + 1

        if fn in {"less_than", "greater_than"}:
            if len(ins) != 2:
                raise ValueError(f"{fn} expects 2 inputs at step {current_idx}")
            left, depth_left = self._translate_at(ins[0], first_initial=True, apply_sum=False)
            right, depth_right = self._translate_at(ins[1], first_initial=True, apply_sum=False)
            op = "lessL" if fn == "less_than" else "greaterL"
            return f"{op}({left}, {right})", max(depth_left, depth_right) + 1

        if fn == "equal_integer":
            if len(ins) != 2:
                raise ValueError(f"equal_integer expects 2 inputs at step {current_idx}")
            left, depth_left = self._translate_at(ins[0], first_initial=True, apply_sum=False)
            right, depth_right = self._translate_at(ins[1], first_initial=True, apply_sum=False)
            return f"equalCountsL({left}, {right})", max(depth_left, depth_right) + 1

        if fn and fn.startswith("equal_"):
            if len(ins) != 2:
                raise ValueError(f"{fn} expects 2 inputs at step {current_idx}")
            attr_suffix = fn.replace("equal_", "")
            left, depth_left = self._translate_at(ins[0], first_initial=True, apply_sum=apply_sum)
            right, depth_right = self._translate_at(ins[1], first_initial=True, apply_sum=apply_sum)
            var_left = _extract_first_var_symbol(left)
            var_right = _extract_first_var_symbol(right)
            same_term = f"sameL({attr_suffix}, '{var_left}', '{var_right}')"
            return f"existsL(andL({left}, {right}, {same_term}))", max(depth_left, depth_right) + 1

        if fn == "unique":
            if len(ins) != 1:
                raise ValueError(f"unique expects 1 input at step {current_idx}")
            base, depth = self._translate_at(ins[0], first_initial=True, apply_sum=apply_sum)
            return base, depth + 1

        if fn == "exist":
            if len(ins) != 1:
                raise ValueError(f"exist expects 1 input at step {current_idx}")
            base, depth = self._translate_at(ins[0], first_initial=True, apply_sum=apply_sum)
            final = f"andL({base})" if depth > 1 else base
            return f"existsL({final})", depth + 1

        if fn and "query" in fn:
            if len(ins) != 1:
                raise ValueError(f"{fn} expects 1 input at step {current_idx}")
            query_type = fn.split("_", 1)[1]
            target_obj, depth = self._translate_at(ins[0], first_initial=True, apply_sum=apply_sum)
            return f"queryL({query_type}, iotaL({target_obj}))", depth + 1

        if fn and (fn.startswith("same_") or fn.startswith("different_")):
            if len(ins) != 1:
                raise ValueError(f"{fn} expects 1 input at step {current_idx}")
            is_different = fn.startswith("different_")
            attr_suffix = fn.replace("different_", "") if is_different else fn.replace("same_", "")
            constraint_name = "differentL" if is_different else "sameL"
            if first_initial:
                var_name = chr(self.var + 97)
                self.var += 1
                suffix = ""
                if self.need_relation2:
                    if self.relation_syntax == "binary":
                        self.need_relation2 = False
                    else:
                        suffix = f", path=('rel{self.relation_val - 1}', {self.pending_arg2_nav})"
                        self.need_relation2 = False
                obj_term = f"obj('{var_name}'{suffix})"
            else:
                var_name = chr(self.var + 96)
                obj_term = None
            next_obj, depth = self._translate_at(ins[0], first_initial=True, apply_sum=apply_sum)
            related_var = _extract_first_var_symbol(next_obj)
            same_term = f"{constraint_name}({attr_suffix}, '{var_name}', '{related_var}')"
            if obj_term is not None:
                return f"{obj_term}, {next_obj}, {same_term}", depth + 1
            return f"{next_obj}, {same_term}", depth + 1

        if fn == "relate":
            if len(ins) != 1 or len(vins) != 1:
                raise ValueError(f"relate expects 1 input and 1 value_input at step {current_idx}")
            attr_value = str(vins[0])
            is_reverse = attr_value.endswith("_rev")
            arg1_nav = "obj1_rev" if is_reverse else "obj1"
            arg2_nav = "obj2_rev" if is_reverse else "obj2"
            init_relation_val = self.relation_val
            if first_initial:
                var_name = chr(self.var + 97)
                self.var += 1
                suffix = ""
                if self.need_relation2:
                    if self.relation_syntax == "binary":
                        self.need_relation2 = False
                    else:
                        suffix = f", path=('rel{self.relation_val - 1}', {self.pending_arg2_nav})"
                        self.need_relation2 = False
                obj_term = f"obj('{var_name}'{suffix})"
            else:
                var_name = chr(self.var + 96)
                obj_term = None
            next_obj, depth = self._translate_at(ins[0], first_initial=True, apply_sum=apply_sum)
            related_var = _extract_first_var_symbol(next_obj)
            relation_legacy = (
                f"{attr_value}('rel{init_relation_val}', path=('{var_name}', {arg1_nav}.reversed))"
            )
            relation_binary = (
                f"{attr_value}('{var_name}', '{related_var}')" if related_var is not None else None
            )
            relation_term = relation_binary if self.relation_syntax == "binary" and relation_binary else relation_legacy
            self.need_relation2 = True
            self.pending_arg2_nav = arg2_nav
            self.relation_val += 1
            if self.relation_syntax == "binary" and relation_binary:
                if obj_term is not None:
                    return f"{obj_term}, {next_obj}, {relation_term}", depth + 1
                return f"{next_obj}, {relation_term}", depth + 1
            if obj_term is not None:
                return f"{obj_term}, {relation_term}, {next_obj}", depth + 1
            return f"{relation_term}, {next_obj}", depth + 1

        raise NotImplementedError(f"Unsupported CLEVR function {fn!r} at step {current_idx}")


def detect_query_type(program: list[dict[str, Any]] | None) -> str | None:
    if not program:
        return None
    fn = program[-1].get("function", program[-1].get("type", ""))
    return fn.split("_", 1)[1] if fn.startswith("query_") else None


def translate_program_to_constraint(
    program: list[dict[str, Any]],
    relation_syntax: str = "legacy",
) -> str:
    return _ProgramTranslator(program, relation_syntax=relation_syntax).translate()


def translate_program_to_answer_constraint(
    program: list[dict[str, Any]],
    answer: Any,
    relation_syntax: str = "legacy",
) -> str:
    """Return a boolean constraint that pins a query target to its gold answer."""
    query_type = detect_query_type(program)
    if query_type is None:
        return translate_program_to_constraint(program, relation_syntax=relation_syntax)
    translator = _ProgramTranslator(program, relation_syntax=relation_syntax)
    query_step = program[-1]
    target_obj, depth = translator._translate_at(
        query_step["inputs"][0],
        first_initial=True,
        apply_sum=True,
    )
    selected_var = _extract_first_var_symbol(target_obj) or "a"
    normalized_answer = _normalize_answer_name(answer)
    answer_term = f"{normalized_answer}(path=('{selected_var}'))"
    content = f"andL({target_obj}, {answer_term})" if depth > 1 else f"andL({target_obj}, {answer_term})"
    return f"existsL({content})"


def create_execution_for_question(
    program: list[dict[str, Any]],
    question_index: int,
    question_type: str | None = None,
    relation_syntax: str = "legacy",
) -> tuple[str, str | None]:
    del question_index, question_type
    return translate_program_to_constraint(program, relation_syntax=relation_syntax), detect_query_type(program)


def answer_to_query_label(answer: Any, query_type: str) -> int:
    values = g_attribute_concepts.get(query_type)
    if values is None:
        raise ValueError(f"Unsupported query type {query_type!r}")
    normalized = _normalize_answer_name(answer)
    try:
        return values.index(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Answer {answer!r} is not valid for query type {query_type!r}; expected one of {values}"
        ) from exc


def _normalize_answer_name(answer: Any) -> str:
    normalized = str(answer).lower()
    if normalized == "metallic":
        normalized = "metal"
    if normalized == "matte":
        normalized = "rubber"
    if normalized == "big":
        normalized = "large"
    if normalized == "tiny":
        normalized = "small"
    if normalized == "block":
        normalized = "cube"
    if normalized == "ball":
        normalized = "sphere"
    return normalized


def prepare_logic_fields(
    items: list[dict[str, Any]],
    *,
    device: str = "cpu",
    relation_syntax: str = "legacy",
    executions: list[str] | None = None,
    query_types: list[str | None] | None = None,
    pin_query_answers: bool = False,
) -> list[dict[str, Any]]:
    import torch

    for i, item in enumerate(items):
        program = item.get("program", [])
        query_type = query_types[i] if query_types is not None else detect_query_type(program)
        if query_type is not None:
            item["query_type"] = query_type
            if pin_query_answers:
                item["logic_str"] = translate_program_to_answer_constraint(
                    program,
                    item.get("answer"),
                    relation_syntax=relation_syntax,
                )
                item["logic_label"] = torch.tensor([1.0], dtype=torch.float32, device=device)
            else:
                item["logic_str"] = (
                    executions[i]
                    if executions is not None
                    else translate_program_to_constraint(program, relation_syntax=relation_syntax)
                )
                item["logic_label"] = torch.tensor(
                    [answer_to_query_label(item.get("answer"), query_type)],
                    dtype=torch.long,
                    device=device,
                )
        else:
            item["logic_str"] = (
                executions[i]
                if executions is not None
                else translate_program_to_constraint(program, relation_syntax=relation_syntax)
            )
            answer = item.get("answer")
            if isinstance(answer, str):
                label = answer.strip().lower() == "yes"
            else:
                label = bool(answer)
            item["query_type"] = None
            item["logic_label"] = torch.tensor([float(label)], dtype=torch.float32, device=device)
    return items
