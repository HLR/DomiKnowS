from .execution import materialize_bounded_facts


def answer_object(instance):
    facts = set(materialize_bounded_facts(instance))
    query = instance["query"]
    target_type = query["target_type"]

    answers = []
    for obj in instance["objects"]:
        if target_type != "__any_object__" and ("ObjectCategory", obj, target_type) not in facts:
            continue
        if all(_satisfies_condition(facts, obj, condition) for condition in query.get("conditions", [])):
            answers.append(obj)
    return answers[0] if len(answers) == 1 else None


def check_oracle(instance, expected_answer):
    return answer_object(instance) == expected_answer


def _satisfies_condition(facts, obj, condition):
    pred, left, right = condition
    if left != "o":
        return False
    if pred == "SemanticClass":
        return ("ObjectClass", obj, right) in facts or ("ObjectCategory", obj, right) in facts
    return (pred, obj, right) in facts

