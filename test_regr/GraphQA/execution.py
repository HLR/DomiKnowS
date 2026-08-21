from collections import defaultdict

from .graph import OBJECT_SYMBOL_RELATIONS, alias_values, canonical_relation, collect_kb_relations, collect_object_relations, safe_name


SUPPORTED_ANSWER_TYPES = {"object"}


def materialize_bounded_facts(instance, max_depth=2):
    """
    Materialize ObjectType/ObjectCategory from Name and TypeOf with a fixed depth.

    This is deliberately bounded and non-recursive:
      Name(o, x) and TypeOf(x, y) -> ObjectType(o, y)
      ObjectType(o, x) and TypeOf(x, y) -> ObjectCategory(o, y)
    """
    facts = [
        (canonical_relation(pred), left, right)
        for pred, left, right in list(instance.get("visual_facts", [])) + list(instance.get("kb_facts", []))
    ]
    type_of_by_src = defaultdict(set)
    names_by_level = {0: set(), 1: set(), 2: set()}
    level_predicates = {0: "Name", 1: "ObjectType", 2: "ObjectCategory"}

    for pred, left, right in facts:
        if pred == "TypeOf":
            type_of_by_src[left].add(right)
        elif pred == "Name":
            names_by_level[0].add((left, right))

    for depth in range(1, max_depth + 1):
        prev = names_by_level[depth - 1]
        for obj, src_symbol in prev:
            for dst_symbol in type_of_by_src.get(src_symbol, set()):
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
    object_label_space = [str(value) for value in getattr(graph_context, "object_values", [])]
    executable_instances = [
        create_executable_instance(instance, answer_label_space=object_label_space)
        for instance in instances
    ]
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

    body = create_object_query_body(instance)
    selector = create_object_selector_logic(instance, body=body)
    return f"queryL(\n        answer_object,\n        {selector}\n    )"


def create_object_query_body(instance):
    query = instance["query"]
    target_type = query.get("target_type")
    # As in CLEVR, iotaL first binds the entity that queryL classifies.
    # Learned relation predicates are then reached from this object by paths.
    base_predicates = ['object_domain("o")']
    if target_type != "__any_object__":
        base_predicates.append(_bounded_type_predicate(instance, target_type, "target_type"))

    alternatives = query.get("alternatives")
    if alternatives:
        branches = []
        for alt_index, conditions in enumerate(alternatives):
            predicates = list(base_predicates)
            for condition_index, condition in enumerate(conditions):
                predicates.extend(_condition_predicates(instance, condition, alt_index * 100 + condition_index))
            branches.append(_and_body(predicates, indent="                "))
        return "orL(\n" + ",\n".join(branches) + "\n            )"

    predicates = list(base_predicates)
    for index, condition in enumerate(query.get("conditions", [])):
        predicates.extend(_condition_predicates(instance, condition, index))
    return _and_body(predicates, indent="            ")


def create_candidate_membership_logic(instance, candidate_object):
    body = create_object_query_body(instance)
    # Ground the query variable directly to the candidate object subtype. This
    # avoids a second equality-style atom, which this DomiKnowS constructor can
    # leave unbound inside nested executable expressions.
    candidate_root = f'{safe_name(candidate_object)}("o")'
    body = body.replace('obj("o")', candidate_root, 1)
    return f"existsL(\n        {body}\n    )"


def create_object_selector_logic(instance, body=None):
    if body is None:
        body = create_object_query_body(instance)
    return f"iotaL(\n        {body}\n    )"


def _and_body(predicates, indent="            "):
    return "andL(\n" + indent + (",\n" + indent).join(predicates) + "\n" + indent[:-4] + ")"

def create_executable_instance(instance, label=None, answer_label_space=None):
    converted = dict(instance)
    converted["facts"] = materialize_bounded_facts(instance)
    converted["logic_str"] = create_query_logic(instance)
    converted["logic_label"] = _resolve_answer_label(instance, label, answer_label_space)
    converted["answer_mode"] = "single_object_query"
    return converted


def create_candidate_membership_instance(instance, candidate_object, label):
    converted = dict(instance)
    converted["facts"] = materialize_bounded_facts(instance)
    converted["logic_str"] = create_candidate_membership_logic(instance, candidate_object)
    converted["logic_label"] = bool(label)
    converted["answer_mode"] = "candidate_membership"
    converted["candidate_object"] = str(candidate_object)
    return converted


def _resolve_answer_label(instance, label=None, answer_label_space=None):
    if label is not None:
        return label
    answer = instance.get("expected_answer")
    if answer is None:
        return None
    answer = str(answer)
    if answer_label_space is None:
        answer_label_space = [str(obj) for obj in instance.get("objects", [])]
    else:
        answer_label_space = [str(obj) for obj in answer_label_space]
    if answer not in answer_label_space:
        raise ValueError(f"Answer object {answer!r} is not in the queryL(obj, ...) label space")
    return answer_label_space.index(answer)


def _condition_predicates(instance, condition, index):
    if len(condition) != 3:
        raise ValueError(f"Invalid GraphQA condition: {condition!r}")
    pred, left, right = condition
    pred = canonical_relation(pred)
    if left != "o":
        raise ValueError(f"Only target variable 'o' is supported, got {left!r}")
    if pred == "Name":
        return _name_predicate(instance, right, f"name{index}")
    if pred in OBJECT_SYMBOL_RELATIONS:
        return _aliased_object_symbol_predicate(instance, pred, right, f"{pred.lower()}{index}")
    if pred == "SemanticClass":
        return [_semantic_class_predicate(instance, right, f"semantic{index}")]
    if pred == "KG":
        return _kg_condition_predicates(instance, right, index)
    if pred in {"RelationFrom", "RelationTo"}:
        return _candidate_relation_predicate(instance, pred, right, index)
    if pred == "OneOf":
        return _one_of_predicate(instance, right, index)
    if _is_object_grounded_kb_relation(instance, pred):
        return _aliased_object_symbol_predicate(instance, pred, right, f"{pred.lower()}{index}")
    if pred in collect_kb_relations(instance):
        raise ValueError(f"KB relation {pred!r} cannot be used directly as an object query condition")
    if pred in collect_object_relations(instance):
        _require_object(instance, right)
        return _object_relation_predicate(pred, "o", right, f"{safe_name(pred).lower()}{index}")
    raise ValueError(f"Unsupported GraphQA predicate: {pred!r}")


def _is_object_grounded_kb_relation(instance, pred):
    pred = canonical_relation(pred)
    objects = {str(obj) for obj in instance.get("objects", [])}
    return any(
        canonical_relation(fact_pred) == pred and str(left) in objects
        for fact_pred, left, _right in instance.get("kb_facts", [])
    )


def _require_symbol(instance, value):
    if value not in instance.get("symbols", []):
        raise ValueError(f"Unknown symbol in condition: {value!r}")


def _require_object(instance, value):
    if value not in instance.get("objects", []):
        raise ValueError(f"Unknown object in condition: {value!r}")


def _type_sources_for_targets(instance, targets, max_depth=2):
    """Return bounded reverse TypeOf sources that can reach any target symbol."""
    reverse_type = defaultdict(set)
    for pred, src, dst in instance.get("kb_facts", []):
        if canonical_relation(pred) == "TypeOf":
            reverse_type[str(dst)].add(str(src))

    out = set(str(target) for target in targets if target is not None)
    frontier = set(out)
    for _depth in range(max(0, int(max_depth))):
        next_frontier = set()
        for dst in frontier:
            for src in reverse_type.get(dst, set()):
                if src not in out:
                    out.add(src)
                    next_frontier.add(src)
        frontier = next_frontier
        if not frontier:
            break
    return list(dict.fromkeys(str(value) for value in out))


def _semantic_class_symbols(instance, symbol):
    aliases = alias_values("SemanticClass", symbol)
    return _type_sources_for_targets(instance, aliases, max_depth=2)


def _has_exact_scene_name(instance, symbol):
    symbol = str(symbol)
    return any(canonical_relation(pred) == "Name" and str(right) == symbol for pred, _left, right in instance.get("visual_facts", []))


def _name_symbols(instance, symbol):
    # VQAR Find_Name is exact when the queried surface name appears in the
    # scene. If not, fall back to the bounded inherited-name closure used by
    # Scallop's name/is_a rules. This prevents plural scene names such as
    # "windows" from also selecting unrelated singular "window" objects.
    if str(symbol).endswith("s") and _has_exact_scene_name(instance, symbol):
        return [str(symbol)]
    aliases = alias_values("SemanticClass", symbol)
    return _type_sources_for_targets(instance, aliases, max_depth=2)


def _kg_destination_symbols(instance, symbol):
    aliases = alias_values("SemanticClass", symbol)
    aliases.extend(alias_values("Attribute", symbol))
    return list(dict.fromkeys(str(value) for value in aliases if value is not None))


def _object_symbol_predicate(pred, obj_var, symbol, var_prefix):
    symbol_name = safe_name(symbol)
    pair_var = f"{var_prefix}_pair"
    return [
        f'{pred}("{pair_var}", path=("{obj_var}", object_symbol_object.reversed))',
        f'{symbol_name}(path=("{pair_var}", object_symbol_symbol))',
    ]


def _aliased_object_symbol_predicate(instance, pred, symbol, var_prefix):
    predicate_sets = []
    for alias_index, alias in enumerate(alias_values(pred, symbol)):
        if alias not in instance.get("symbols", []):
            continue
        predicate_sets.append(_object_symbol_predicate(pred, "o", alias, f"{var_prefix}_{alias_index}"))
    if not predicate_sets:
        _require_symbol(instance, symbol)
        return _object_symbol_predicate(pred, "o", symbol, var_prefix)
    if len(predicate_sets) == 1:
        return predicate_sets[0]
    bodies = [_and_body(predicates, indent="        ") for predicates in predicate_sets]
    return ["orL(\n" + ",\n".join(bodies) + "\n    )"]


def _semantic_class_predicate(instance, symbol, var_prefix):
    return _bounded_type_predicate(instance, symbol, var_prefix)


def _name_predicate(instance, symbol, var_prefix):
    direct_predicates = []
    bodies = []
    for alias_index, alias in enumerate(_name_symbols(instance, symbol)):
        if alias not in instance.get("symbols", []):
            continue
        predicates = _object_symbol_predicate("Name", "o", alias, f"{var_prefix}_name_{alias_index}")
        direct_predicates.append(predicates)
        bodies.append(_and_body(predicates, indent="        "))
    if not _has_exact_scene_name(instance, symbol):
        bodies.extend(_bounded_type_bodies(instance, symbol, var_prefix, include_direct=False))
    if not bodies:
        _require_symbol(instance, symbol)
        predicates = _object_symbol_predicate("Name", "o", symbol, f"{var_prefix}_exact")
        direct_predicates.append(predicates)
        bodies.append(_and_body(predicates, indent="        "))
    if len(bodies) == 1 and len(direct_predicates) == 1:
        return direct_predicates[0]
    return ["orL(\n" + ",\n".join(bodies) + "\n    )"]


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
    rel = canonical_relation(rel)
    if rel not in collect_kb_relations(instance):
        raise ValueError(f"Unknown KG relation in condition: {rel!r}")
    dst_aliases = [alias for alias in _kg_destination_symbols(instance, dst_symbol) if alias in instance.get("symbols", [])]
    if not dst_aliases:
        _require_symbol(instance, dst_symbol)
        dst_aliases = [dst_symbol]
    source_bodies = []
    for body_index, dst_alias in enumerate(dst_aliases):
        source_bodies.extend(
            _bounded_kb_relation_bodies(rel, dst_alias, f"kg{index}_{body_index}", max_type_depth=2)
        )
    return ["orL(\n" + ",\n".join(source_bodies) + "\n    )"]


def _bounded_type_predicate(instance, symbol, var_prefix):
    bodies = _bounded_type_bodies(instance, symbol, var_prefix, include_direct=True)
    if not bodies:
        _require_symbol(instance, symbol)
        bodies = _bounded_kb_relation_bodies("TypeOf", symbol, var_prefix, max_type_depth=1)
    if len(bodies) == 1:
        return bodies[0]
    return "orL(\n" + ",\n".join(bodies) + "\n    )"


def _bounded_type_bodies(instance, symbol, var_prefix, include_direct=True):
    """Ground an object name, then follow fixed TypeOf edges up to depth two."""
    targets = [
        alias for alias in alias_values("SemanticClass", symbol)
        if alias in instance.get("symbols", [])
    ]
    if not targets and symbol in instance.get("symbols", []):
        targets = [symbol]
    bodies = []
    for target_index, target in enumerate(targets):
        if include_direct:
            bodies.append(_and_body(
                _object_symbol_predicate("Name", "o", target, f"{var_prefix}_{target_index}_direct"),
                indent="        ",
            ))
        bodies.extend(
            _bounded_kb_relation_bodies(
                "TypeOf", target, f"{var_prefix}_{target_index}", max_type_depth=1,
            )
        )
    return bodies


def _bounded_kb_relation_bodies(rel, dst_symbol, var_prefix, max_type_depth=2):
    """Join learned Name grounding with fixed KB edges, optionally through TypeOf."""
    rel_name = safe_name(rel)
    dst_name = safe_name(dst_symbol)
    bodies = []
    for type_depth in range(max(0, int(max_type_depth)) + 1):
        name_pair = f"{var_prefix}_d{type_depth}_name_pair"
        predicates = [
            f'Name("{name_pair}", path=("o", object_symbol_object.reversed))',
        ]
        source_pair = name_pair
        source_relation = "object_symbol_symbol"
        for hop in range(type_depth):
            type_pair = f"{var_prefix}_d{type_depth}_type{hop}_pair"
            predicates.append(
                f'TypeOf("{type_pair}", path=("{source_pair}", {source_relation}, symbol_pair_src.reversed))'
            )
            source_pair = type_pair
            source_relation = "symbol_pair_dst"
        relation_pair = f"{var_prefix}_d{type_depth}_rel_pair"
        predicates.extend([
            f'{rel_name}("{relation_pair}", path=("{source_pair}", {source_relation}, symbol_pair_src.reversed))',
            f'{dst_name}(path=("{relation_pair}", symbol_pair_dst))',
        ])
        bodies.append(_and_body(predicates, indent="        "))
    return bodies


def _candidate_relation_predicate(instance, pred, value, index):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid relation condition payload: {value!r}")
    rel, object_ids = value
    rel = canonical_relation(rel)
    if rel not in collect_object_relations(instance):
        raise ValueError(f"Unknown object relation in condition: {rel!r}")
    candidates = [str(obj) for obj in object_ids if str(obj) in instance.get("objects", [])]
    if not candidates:
        raise ValueError("Relation condition has no candidate anchor objects in this instance")
    calls = []
    rel_name = safe_name(rel)
    for candidate_index, candidate in enumerate(candidates):
        pair_var = f"rel{index}_{candidate_index}_pair"
        candidate_name = safe_name(candidate)
        if pred == "RelationFrom":
            calls.append(_and_body([
                f'{rel_name}("{pair_var}", path=("o", object_pair_dst.reversed))',
                f'{candidate_name}(path=("{pair_var}", object_pair_src))',
            ], indent="        "))
        else:
            calls.append(_and_body([
                f'{rel_name}("{pair_var}", path=("o", object_pair_src.reversed))',
                f'{candidate_name}(path=("{pair_var}", object_pair_dst))',
            ], indent="        "))
    if len(calls) == 1:
        return calls
    return ["orL(\n" + ",\n".join(calls) + "\n    )"]


def _one_of_predicate(instance, object_ids, index):
    candidates = [str(obj) for obj in object_ids if str(obj) in instance.get("objects", [])]
    if not candidates:
        raise ValueError("OneOf condition has no candidate objects in this instance")
    calls = [f'{safe_name(obj)}(path="o")' for obj in candidates]
    if len(calls) == 1:
        return calls
    return ["orL(" + ", ".join(calls) + ")"]
