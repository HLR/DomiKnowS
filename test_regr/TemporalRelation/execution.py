from .graph import TEMPORAL_LABELS, unpack_pair


SUPPORTED_LABELS = set(TEMPORAL_LABELS)


def create_query_logic(instance):
    _resolve_query_pair(instance)
    _validate_document_pairs(instance)

    return """queryL(
        temporal_relation,
        iotaL(
            andL(
                EventPair("p"),
                event(path=("p", pair_event1)),
                event(path=("p", pair_event2)),
                query_event1(path=("p", pair_event1)),
                query_event2(path=("p", pair_event2))
            )
        )
    )"""


def create_executable_instance(instance, label=None):
    query_pair = _query_pair(instance)
    expected = label if label is not None else query_pair.get("label")
    converted = dict(instance)
    converted["query_event_groundings"] = create_query_event_groundings(instance)
    converted["candidate_event_pairs"] = create_candidate_event_pairs(instance)
    converted["pair_learner_examples"] = create_pair_learner_examples(instance)
    converted["logic_str"] = create_query_logic(instance)
    converted["logic_label"] = label_to_index(expected) if expected is not None else None
    return converted



def create_query_event_groundings(instance):
    """Return oracle labels for learned query_event1/query_event2 marker predicates.

    In a CLEVR-style learned setting, an LLM/question interpreter predicts these
    marker predicates over event nodes. MATRES gives the target pair, so the
    adapter can provide perfect labels for training or oracle execution.
    """
    resolved = _resolve_query_pair(instance)
    query_e1 = resolved["pair"]["e1"]
    query_e2 = resolved["pair"]["e2"]
    groundings = []
    for event in instance.get("events", []):
        event_id = _event_id(event)
        groundings.append(
            {
                "event_id": event_id,
                "query_event1": event_id == query_e1,
                "query_event2": event_id == query_e2,
            }
        )
    return groundings


def create_pair_learner_examples(instance):
    """Create local-classifier examples for every ordered event pair.

    Each example is the natural input for a small LM: document text with event
    markers plus a multiclass label over Before/After/Equal/Vague when MATRES
    supervises that pair.
    """
    examples = []
    for pair in create_candidate_event_pairs(instance):
        e1, e2, label = unpack_pair(pair)
        examples.append(
            {
                "e1": e1,
                "e2": e2,
                "label": label,
                "label_index": label_to_index(label) if label is not None else None,
                "text_with_event_markers": mark_text_for_pair(instance, e1, e2),
                "target_concept": "temporal_relation",
                "label_concepts": TEMPORAL_LABELS,
            }
        )
    return examples


def mark_text_for_pair(instance, e1, e2):
    """Return text with [E1]/[E2] markers around the pair event mentions."""
    event1 = _require_event(instance, e1)
    event2 = _require_event(instance, e2)
    token1 = _require_event_token(instance, event1)
    token2 = _require_event_token(instance, event2)
    tokens = instance.get("tokens", [])
    if not tokens:
        return _fallback_marked_text(instance, event1, event2)

    pieces = []
    for token in tokens:
        token_id = token.get("id") if isinstance(token, dict) else token
        token_text = token.get("text") if isinstance(token, dict) else str(token)
        if token_id == token1:
            token_text = f"[E1]{token_text}[/E1]"
        if token_id == token2:
            token_text = f"[E2]{token_text}[/E2]"
        pieces.append(token_text)
    marked = " ".join(pieces).strip()
    return marked or _fallback_marked_text(instance, event1, event2)


def _fallback_marked_text(instance, event1, event2):
    text = instance.get("text") or ""
    e1_text = event1.get("text", event1.get("id"))
    e2_text = event2.get("text", event2.get("id"))
    return f"{text} [E1]{e1_text}[/E1] [E2]{e2_text}[/E2]".strip()

def create_candidate_event_pairs(instance, include_self=False):
    """Return all ordered document-level event pairs for pairwise classification."""
    event_ids = [_event_id(event) for event in instance.get("events", [])]
    candidate_pairs = []
    supervised = {_pair_key(pair): pair for pair in instance.get("event_pairs", [])}
    for e1 in event_ids:
        for e2 in event_ids:
            if not include_self and e1 == e2:
                continue
            pair = supervised.get((e1, e2))
            if pair is None:
                pair = {"e1": e1, "e2": e2, "label": None}
            candidate_pairs.append(pair)
    return candidate_pairs


def compile_temporal_dataset(instances, graph_context):
    executable_instances = [create_executable_instance(instance) for instance in instances]
    return graph_context.graph.compile_executable(
        executable_instances,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=graph_context.namespace,
    )


def validate_dataset_convertible(instances):
    failures = []
    for index, instance in enumerate(instances):
        try:
            create_query_logic(instance)
        except ValueError as exc:
            failures.append((index, str(exc)))
    return failures


def label_to_index(label):
    if label not in SUPPORTED_LABELS:
        raise ValueError(f"Unsupported temporal label: {label!r}")
    return TEMPORAL_LABELS.index(label)


def index_to_label(index):
    return TEMPORAL_LABELS[index]


def _resolve_query_pair(instance):
    pair = _query_pair(instance)
    event1 = _require_event(instance, pair["e1"])
    event2 = _require_event(instance, pair["e2"])
    return {
        "pair": pair,
        "event1": event1,
        "event2": event2,
        "event1_token_id": _require_event_token(instance, event1),
        "event2_token_id": _require_event_token(instance, event2),
    }


def _validate_document_pairs(instance):
    if not instance.get("event_pairs"):
        raise ValueError("Expected at least one labeled event pair")
    for pair in instance.get("event_pairs", []):
        e1, e2, label = unpack_pair(pair)
        if e1 is None or e2 is None:
            raise ValueError(f"Event pair must include e1 and e2: {pair!r}")
        _require_event(instance, e1)
        _require_event(instance, e2)
        if label is not None and label not in SUPPORTED_LABELS:
            raise ValueError(f"Unsupported temporal label: {label!r}")


def _query_pair(instance):
    if instance.get("query_pair") is not None:
        e1, e2, label = unpack_pair(instance["query_pair"])
        if label is None:
            label = _label_for_pair(instance, e1, e2)
        pair = {"e1": e1, "e2": e2, "label": label}
    else:
        pairs = instance.get("event_pairs", [])
        if not pairs:
            raise ValueError("Expected query_pair or at least one event pair")
        e1, e2, label = unpack_pair(pairs[0])
        pair = {"e1": e1, "e2": e2, "label": label}
    if pair["e1"] is None or pair["e2"] is None:
        raise ValueError(f"Query pair must include e1 and e2: {pair!r}")
    if pair["label"] is not None and pair["label"] not in SUPPORTED_LABELS:
        raise ValueError(f"Unsupported temporal label: {pair['label']!r}")
    return pair


def _label_for_pair(instance, e1, e2):
    for pair in instance.get("event_pairs", []):
        pair_e1, pair_e2, label = unpack_pair(pair)
        if pair_e1 == e1 and pair_e2 == e2:
            return label
    return None


def _require_event(instance, event_id):
    for event in instance.get("events", []):
        if isinstance(event, dict) and event.get("id") == event_id:
            return event
        if event == event_id:
            return {"id": event_id, "token_id": event_id}
    raise ValueError(f"Unknown event id in query pair: {event_id!r}")


def _require_event_token(instance, event):
    token_id = event.get("token_id") or event.get("token") or event.get("id")
    token_ids = {token.get("id") if isinstance(token, dict) else token for token in instance.get("tokens", [])}
    if token_ids and token_id not in token_ids:
        raise ValueError(f"Event {event.get('id')!r} points to unknown token id: {token_id!r}")
    return token_id


def _event_id(event):
    return event.get("id") if isinstance(event, dict) else event


def _pair_key(pair):
    e1, e2, _label = unpack_pair(pair)
    return e1, e2
