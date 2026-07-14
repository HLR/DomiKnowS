from collections import defaultdict

from .graph import OBJECT_SYMBOL_RELATIONS, collect_kb_relations, collect_object_relations, safe_name


SUPPORTED_ANSWER_TYPES = {"object"}


def materialize_bounded_facts(instance, max_depth=2):
    """
    Materialize ObjectClass/ObjectCategory from Name and Hypernym with a fixed depth.

    This is deliberately bounded and non-recursive:
      Name(o, x) and Hypernym(x, y) -> ObjectClass(o, y)
      ObjectClass(o, x) and Hypernym(x, y) -> ObjectCategory(o, y)
    """
    facts = list(instance.get("visual_facts", [])) + list(instance.get("kb_facts", []))
    hypernym_by_src = defaultdict(set)
    names_by_level = {0: set(), 1: set(), 2: set()}
    level_predicates = {0: "Name", 1: "ObjectClass", 2: "ObjectCategory"}

    for pred, left, right in facts:
        if pred == "Hypernym":
            hypernym_by_src[left].add(right)
        elif pred == "Name":
            names_by_level[0].add((left, right))

    for depth in range(1, max_depth + 1):
        prev = names_by_level[depth - 1]
        for obj, src_symbol in prev:
            for dst_symbol in hypernym_by_src.get(src_symbol, set()):
                names_by_level[depth].add((obj, dst_symbol))

    for depth in range(1, max_depth + 1):
        for obj, symbol in sorted(names_by_level[depth]):
            facts.append((level_predicates[depth], obj, symbol))

    return facts


def is_convertible(instance):
    try:
        create_query_logic(instance)
    except ValueError:
        return False
    return True


def assert_convertible(instance):
    create_query_logic(instance)
    return True


def validate_dataset_convertible(instances):
    failures = []
    for index, instance in enumerate(instances):
        try:
            create_query_logic(instance)
        except ValueError as exc:
            failures.append((index, str(exc)))
    return failures


def compile_graphqa_dataset(instances, graph_context):
    executable_instances = [create_executable_instance(instance) for instance in instances]
    return graph_context.graph.compile_executable(
        executable_instances,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=graph_context.namespace,
    )


def create_query_logic(instance):
    query = instance["query"]
    answer_type = query.get("answer_type")
    if answer_type not in SUPPORTED_ANSWER_TYPES:
        raise ValueError(f"Unsupported GraphQA answer_type: {answer_type!r}")

    target_type = query.get("target_type")
    if target_type != "__any_object__" and target_type not in instance.get("symbols", []):
        raise ValueError(f"Unknown target_type symbol: {target_type!r}")

    base_predicates = ['obj("o")']
    if target_type != "__any_object__":
        base_predicates.extend(_object_symbol_predicate("ObjectCategory", "o", target_type, "target_type"))

    alternatives = query.get("alternatives")
    if alternatives:
        branches = []
        for alt_index, conditions in enumerate(alternatives):
            predicates = list(base_predicates)
            for condition_index, condition in enumerate(conditions):
                predicates.extend(_condition_predicates(instance, condition, alt_index * 100 + condition_index))
            branches.append(_and_body(predicates, indent="                "))
        body = "orL(\n" + ",\n".join(branches) + "\n            )"
    else:
        predicates = list(base_predicates)
        for index, condition in enumerate(query.get("conditions", [])):
            predicates.extend(_condition_predicates(instance, condition, index))
        body = _and_body(predicates, indent="            ")

    return f"iotaL(\n        {body}\n    )"


def _and_body(predicates, indent="            "):
    return "andL(\n" + indent + (",\n" + indent).join(predicates) + "\n" + indent[:-4] + ")"

def create_executable_instance(instance, label=0):
    converted = dict(instance)
    converted["facts"] = materialize_bounded_facts(instance)
    converted["logic_str"] = create_query_logic(instance)
    converted["logic_label"] = label
    return converted


def _condition_predicates(instance, condition, index):
    if len(condition) != 3:
        raise ValueError(f"Invalid GraphQA condition: {condition!r}")
    pred, left, right = condition
    if left != "o":
        raise ValueError(f"Only target variable 'o' is supported, got {left!r}")
    if pred in OBJECT_SYMBOL_RELATIONS:
        _require_symbol(instance, right)
        return _object_symbol_predicate(pred, "o", right, f"{pred.lower()}{index}")
    if pred == "SemanticClass":
        _require_symbol(instance, right)
        return [_semantic_class_predicate(right, f"semantic{index}")]
    if pred == "KG":
        return _kg_condition_predicates(instance, right, index)
    if pred == "OneOf":
        return _one_of_predicate(instance, right, index)
    if pred in collect_kb_relations(instance):
        raise ValueError(f"KB relation {pred!r} cannot be used directly as an object query condition")
    if pred in collect_object_relations(instance):
        _require_object(instance, right)
        return _object_relation_predicate(pred, "o", right, f"{safe_name(pred).lower()}{index}")
    raise ValueError(f"Unsupported GraphQA predicate: {pred!r}")


def _require_symbol(instance, value):
    if value not in instance.get("symbols", []):
        raise ValueError(f"Unknown symbol in condition: {value!r}")


def _require_object(instance, value):
    if value not in instance.get("objects", []):
        raise ValueError(f"Unknown object in condition: {value!r}")


def _object_symbol_predicate(pred, obj_var, symbol, var_prefix):
    symbol_name = safe_name(symbol)
    pair_var = f"{var_prefix}_pair"
    return [
        f'{pred}("{pair_var}", path=("{obj_var}", object_symbol_object.reversed))',
        f'{symbol_name}(path=("{pair_var}", object_symbol_symbol))',
    ]


def _semantic_class_predicate(symbol, var_prefix):
    type_body = _and_body(
        _object_symbol_predicate("ObjectClass", "o", symbol, f"{var_prefix}_class"),
        indent="        ",
    )
    category_body = _and_body(
        _object_symbol_predicate("ObjectCategory", "o", symbol, f"{var_prefix}_object_category"),
        indent="        ",
    )
    return "orL(\n" + type_body + ",\n" + category_body + "\n    )"


def _object_relation_predicate(pred, obj_var, other_obj, var_prefix):
    other_name = safe_name(other_obj)
    pred_name = safe_name(pred)
    pair_var = f"{var_prefix}_pair"
    return [
        f'{pred_name}("{pair_var}", path=("{obj_var}", object_pair_src.reversed))',
        f'{other_name}(path=("{pair_var}", object_pair_dst))',
    ]



def _kg_condition_predicates(instance, value, index):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid KG condition payload: {value!r}")
    rel, dst_symbol = value
    if rel not in collect_kb_relations(instance):
        raise ValueError(f"Unknown KG relation in condition: {rel!r}")
    _require_symbol(instance, dst_symbol)
    name_pair = f"kg_name{index}_pair"
    kg_pair = f"kg{index}_pair"
    rel_name = safe_name(rel)
    dst_name = safe_name(dst_symbol)
    return [
        f'ObjectCategory("{name_pair}", path=("o", object_symbol_object.reversed))',
        f'{rel_name}("{kg_pair}", path=("{name_pair}", object_symbol_symbol, symbol_pair_src.reversed))',
        f'{dst_name}(path=("{kg_pair}", symbol_pair_dst))',
    ]


def _one_of_predicate(instance, object_ids, index):
    candidates = [str(obj) for obj in object_ids if str(obj) in instance.get("objects", [])]
    if not candidates:
        raise ValueError("OneOf condition has no candidate objects in this instance")
    calls = [f'{safe_name(obj)}(path="o")' for obj in candidates]
    if len(calls) == 1:
        return calls
    return ["orL(" + ", ".join(calls) + ")"]
