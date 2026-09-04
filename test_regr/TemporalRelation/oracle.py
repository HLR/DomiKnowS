from .graph import TEMPORAL_LABELS, unpack_pair


INVERSE = {"Before": "After", "After": "Before", "Equal": "Equal", "Vague": "Vague"}


def answer_label(instance):
    query_pair = instance.get("query_pair")
    if query_pair is not None:
        e1, e2, label = unpack_pair(query_pair)
        if label is not None:
            return label
        for pair in instance.get("event_pairs", []):
            pair_e1, pair_e2, pair_label = unpack_pair(pair)
            if pair_e1 == e1 and pair_e2 == e2:
                return pair_label
        return None

    pairs = instance.get("event_pairs", [])
    if len(pairs) != 1:
        return None
    _e1, _e2, label = unpack_pair(pairs[0])
    return label


def check_oracle(instance, expected_label):
    return answer_label(instance) == expected_label


def consistency_violations(instances):
    labels = {}
    violations = []
    for index, instance in enumerate(instances):
        for pair in instance.get("event_pairs", []):
            e1, e2, label = unpack_pair(pair)
            if label not in TEMPORAL_LABELS:
                violations.append(("unsupported_label", index, e1, e2, label))
                continue
            key = (e1, e2)
            if key in labels and labels[key] != label:
                violations.append(("mutual_exclusion", key, labels[key], label))
            labels[key] = label

    for (e1, e2), label in labels.items():
        inverse = labels.get((e2, e1))
        if inverse is not None and inverse != INVERSE[label]:
            violations.append(("inverse", (e1, e2), label, inverse))
        if label == "Before" and labels.get((e2, e1)) == "Before":
            violations.append(("cycle", e1, e2))
        if label == "Equal" and labels.get((e2, e1)) not in (None, "Equal"):
            violations.append(("equality_symmetry", e1, e2))

    events = sorted({event for pair in labels for event in pair})
    for x in events:
        for y in events:
            for z in events:
                if labels.get((x, y)) == "Before" and labels.get((y, z)) == "Before":
                    if labels.get((z, x)) == "Before":
                        violations.append(("no_cycle", x, y, z))
                    if labels.get((x, z)) not in (None, "Before"):
                        violations.append(("transitivity_conflict", x, y, z, labels.get((x, z))))
    return violations


def infer_transitive_before(instances):
    before = set()
    for instance in instances:
        for pair in instance.get("event_pairs", []):
            e1, e2, label = unpack_pair(pair)
            if label == "Before":
                before.add((e1, e2))
    changed = True
    while changed:
        changed = False
        additions = set()
        for x, y in before:
            for y2, z in before:
                if y == y2 and (x, z) not in before:
                    additions.add((x, z))
        if additions:
            before.update(additions)
            changed = True
    return before
