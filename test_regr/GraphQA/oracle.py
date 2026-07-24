from .execution import materialize_bounded_facts
from .graph import alias_values, canonical_relation


def answer_object(instance):
    answers = answer_objects(instance)
    return answers[0] if len(answers) == 1 else None


def answer_objects(instance):
    facts = set(_hashable_fact(fact) for fact in materialize_bounded_facts(instance))
    query = instance["query"]
    target_type = query["target_type"]

    condition_groups = query.get("alternatives") or [query.get("conditions", [])]
    answers = []
    for obj in instance["objects"]:
        if target_type != "__any_object__" and ("ObjectCategory", obj, target_type) not in facts:
            continue
        if any(all(_satisfies_condition(facts, obj, condition) for condition in conditions) for conditions in condition_groups):
            answers.append(str(obj))
    return answers


def check_oracle(instance, expected_answer):
    return answer_object(instance) == str(expected_answer)


def check_oracle_set(instance, expected_answers):
    return set(answer_objects(instance)) == {str(answer) for answer in expected_answers}


def _hashable_fact(fact):
    pred, left, right = fact
    if isinstance(right, list):
        right = tuple(right)
    return (pred, left, right)


def _satisfies_condition(facts, obj, condition):
    pred, left, right = condition
    if isinstance(right, list):
        right = tuple(right)
    pred = canonical_relation(pred)
    if left != "o":
        return False
    if pred == "SemanticClass":
        return any(
            ("Name", obj, alias) in facts or ("ObjectType", obj, alias) in facts or ("ObjectCategory", obj, alias) in facts
            for alias in alias_values("SemanticClass", right)
        )
    if pred == "OneOf":
        return str(obj) in {str(candidate) for candidate in right}
    if pred == "KG":
        if not isinstance(right, (list, tuple)) or len(right) != 2:
            return False
        rel, dst = right
        rel = canonical_relation(rel)
        sources = {
            source
            for fact_pred, fact_obj, source in facts
            if fact_pred in {"Name", "ObjectType", "ObjectCategory"} and fact_obj == obj
        }
        return any((rel, source, dst) in facts for source in sources)
    if pred in {"RelationFrom", "RelationTo"}:
        if not isinstance(right, (list, tuple)) or len(right) != 2:
            return False
        rel, candidates = right
        rel = canonical_relation(rel)
        candidate_set = {str(candidate) for candidate in candidates}
        if pred == "RelationFrom":
            return any((rel, candidate, obj) in facts for candidate in candidate_set)
        return any((rel, obj, candidate) in facts for candidate in candidate_set)
    if pred == "Attribute":
        return any((pred, obj, alias) in facts for alias in alias_values("Attribute", right))
    try:
        return (pred, obj, right) in facts
    except TypeError:
        return False

