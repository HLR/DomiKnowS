import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

from .config import TEMPORAL_CONFIG


DEFAULT_TEMPORAL_DATA_ROOT = TEMPORAL_CONFIG.data_root


class TemporalDatasetNotFound(FileNotFoundError):
    pass


def discover_temporal_datasets(root=DEFAULT_TEMPORAL_DATA_ROOT):
    root = Path(root)
    matres = sorted(root.glob("**/*matres*")) + sorted(root.glob("**/*MATRES*"))
    matres += sorted(root.glob("**/timebank.txt"))
    matres += sorted(root.glob("**/aquaint.txt"))
    matres += sorted(root.glob("**/platinum.txt"))
    tbdense = sorted(root.glob("**/*tbdense*")) + sorted(root.glob("**/*TB*Dense*"))
    return {"root": root, "matres": sorted(set(matres)), "tbdense": sorted(set(tbdense))}


def load_temporal_instances(path, limit=None, group_by_document=True,
                            dataset_name="auto"):
    path = Path(path)
    if not path.is_file():
        raise TemporalDatasetNotFound(f"Temporal dataset file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    elif path.suffix.lower() in {".tsv", ".tab"}:
        rows = _read_tsv(path)
    elif path.suffix.lower() == ".txt":
        rows = _read_matres_txt(path)
    else:
        raise ValueError(f"Unsupported temporal dataset format: {path.suffix}")

    resolved_dataset = _resolve_dataset_name(path, dataset_name)
    rows = _apply_dataset_name(rows, resolved_dataset, path)
    document_rows = [
        row for row in rows
        if isinstance(row.get("event_pairs"), list)
        and isinstance(row.get("events"), list)
    ]
    if document_rows:
        if len(document_rows) != len(rows):
            raise ValueError(
                f"{path}: cannot mix document-level and relation-row JSON records")
        documents = _normalize_document_rows(document_rows, limit=limit)
        if group_by_document:
            return documents
        instances = []
        for document in documents:
            for pair in document["event_pairs"]:
                instance = dict(document)
                instance["event_pairs"] = [pair]
                instance["query_pair"] = dict(pair)
                instances.append(instance)
                if limit is not None and len(instances) >= limit:
                    return instances
        return instances
    if group_by_document:
        return _normalize_grouped_rows(rows, limit=limit)
    instances = [_normalize_row(row) for row in rows]
    return instances[:limit] if limit is not None else instances


def _read_jsonl(path):
    rows = []
    with open(path, "r") as data_file:
        for line in data_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_tsv(path):
    with open(path, "r", newline="") as data_file:
        return list(csv.DictReader(data_file, delimiter="\t"))


def _read_matres_txt(path):
    rows = []
    with open(path, "r", newline="") as data_file:
        reader = csv.reader(data_file, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError(f"Expected 6 tab-separated fields in {path}:{line_number}, got {len(row)}: {row!r}")
            doc_id, verb1, verb2, eiid1, eiid2, relation = row
            rows.append(
                {
                    "doc_id": doc_id,
                    "text": doc_id,
                    "e1": f"eiid{eiid1}",
                    "e2": f"eiid{eiid2}",
                    "e1_text": verb1,
                    "e2_text": verb2,
                    "e1_token_id": f"eiid{eiid1}",
                    "e2_token_id": f"eiid{eiid2}",
                    "tokens": [
                        {"id": f"eiid{eiid1}", "text": verb1},
                        {"id": f"eiid{eiid2}", "text": verb2},
                    ],
                    "label": relation,
                    "dataset": "matres",
                }
            )
    return rows


def _normalize_grouped_rows(rows, limit=None):
    grouped = OrderedDict()
    for row in rows:
        e1 = _row_value(row, "e1", "event1", "source_event")
        e2 = _row_value(row, "e2", "event2", "target_event")
        label = _row_value(row, "label", "relation", "temporal_relation")
        text = _row_value(row, "text", "sentence", default="")
        doc_id = _row_value(row, "doc_id", "document_id", default=text)
        dataset = _row_value(row, "dataset")
        split = _row_value(row, "split")
        if e1 is None or e2 is None or label is None:
            raise ValueError(f"Could not normalize temporal row: {row!r}")

        key = doc_id if doc_id not in (None, "") else text
        if key not in grouped:
            grouped[key] = {
                "doc_id": doc_id,
                "text": text,
                "tokens": OrderedDict(),
                "events": OrderedDict(),
                "event_pairs": [],
                "dataset": dataset,
                "split": split,
            }
        instance = grouped[key]
        if dataset is not None and instance["dataset"] not in (None, dataset):
            raise ValueError(
                f"Conflicting dataset identities for document {doc_id!r}: "
                f"{instance['dataset']!r} versus {dataset!r}")
        if split is not None and instance["split"] not in (None, split):
            raise ValueError(
                f"Conflicting split identities for document {doc_id!r}: "
                f"{instance['split']!r} versus {split!r}")
        instance["dataset"] = instance["dataset"] or dataset
        instance["split"] = instance["split"] or split
        _merge_tokens(instance["tokens"], _tokens_from_row(row, text))
        tokens = list(instance["tokens"].values())
        e1_id = str(e1)
        e2_id = str(e2)
        instance["events"][e1_id] = _event_from_row(row, "e1", e1_id, tokens)
        instance["events"][e2_id] = _event_from_row(row, "e2", e2_id, tokens)
        instance["event_pairs"].append({"e1": e1_id, "e2": e2_id, "label": _normalize_label(label)})

    instances = []
    for instance in grouped.values():
        pairs = instance["event_pairs"]
        instances.append(
            {
                "doc_id": instance["doc_id"],
                "text": instance["text"],
                "tokens": list(instance["tokens"].values()),
                "events": list(instance["events"].values()),
                "event_pairs": pairs,
                "query_pair": {"e1": pairs[0]["e1"], "e2": pairs[0]["e2"]} if pairs else None,
                "dataset": instance["dataset"],
                "split": instance["split"],
            }
        )
        if limit is not None and len(instances) >= limit:
            break
    return instances


def _normalize_document_rows(rows, limit=None):
    instances = []
    seen_documents = set()
    for row in rows:
        doc_id = _row_value(row, "doc_id", "document_id")
        if doc_id in (None, ""):
            raise ValueError(f"Document-level temporal record lacks doc_id: {row!r}")
        if doc_id in seen_documents:
            raise ValueError(f"Duplicate document-level temporal record: {doc_id!r}")
        seen_documents.add(doc_id)

        tokens = list(row.get("tokens") or [])
        events = list(row.get("events") or [])
        event_ids = {
            str(event.get("id") if isinstance(event, dict) else event)
            for event in events
        }
        pairs = []
        for pair in row.get("event_pairs", []):
            e1 = _row_value(pair, "e1", "event1", "source_event")
            e2 = _row_value(pair, "e2", "event2", "target_event")
            label = _row_value(pair, "label", "relation", "temporal_relation")
            if e1 is None or e2 is None or label is None:
                raise ValueError(
                    f"Could not normalize temporal pair in {doc_id!r}: {pair!r}")
            e1, e2 = str(e1), str(e2)
            if e1 not in event_ids or e2 not in event_ids:
                raise ValueError(
                    f"Temporal pair {e1!r}->{e2!r} in {doc_id!r} references "
                    "an event absent from the document event list")
            pairs.append({"e1": e1, "e2": e2, "label": _normalize_label(label)})
        if not pairs:
            raise ValueError(f"Document-level temporal record has no pairs: {doc_id!r}")
        instances.append({
            "doc_id": doc_id,
            "text": row.get("text") or "",
            "tokens": tokens,
            "events": events,
            "event_pairs": pairs,
            "query_pair": dict(pairs[0]),
            "dataset": _row_value(row, "dataset"),
            "split": _row_value(row, "split"),
        })
        if limit is not None and len(instances) >= limit:
            break
    return instances


def _normalize_row(row):
    e1 = _row_value(row, "e1", "event1", "source_event")
    e2 = _row_value(row, "e2", "event2", "target_event")
    label = _row_value(row, "label", "relation", "temporal_relation")
    text = _row_value(row, "text", "sentence", default="")
    doc_id = _row_value(row, "doc_id", "document_id")
    if e1 is None or e2 is None or label is None:
        raise ValueError(f"Could not normalize temporal row: {row!r}")

    tokens = _tokens_from_row(row, text)
    e1_id = str(e1)
    e2_id = str(e2)
    return {
        "doc_id": doc_id,
        "text": text,
        "tokens": tokens,
        "events": [_event_from_row(row, "e1", e1_id, tokens), _event_from_row(row, "e2", e2_id, tokens)],
        "event_pairs": [{"e1": e1_id, "e2": e2_id, "label": _normalize_label(label)}],
        "query_pair": {"e1": e1_id, "e2": e2_id},
        "dataset": _row_value(row, "dataset"),
        "split": _row_value(row, "split"),
    }


def _resolve_dataset_name(path, dataset_name):
    if dataset_name not in (None, "", "auto"):
        value = str(dataset_name).lower()
        if value not in {"matres", "tbdense"}:
            raise ValueError(f"Unsupported temporal dataset name: {dataset_name!r}")
        return value
    lowered = str(path).lower()
    if "tbdense" in lowered or "tb-dense" in lowered or "timebank-dense" in lowered:
        return "tbdense"
    if path.suffix.lower() == ".txt" or "matres" in lowered:
        return "matres"
    return None


def _apply_dataset_name(rows, dataset_name, path):
    normalized = []
    for row_number, row in enumerate(rows, start=1):
        row = dict(row)
        existing = row.get("dataset")
        if existing not in (None, ""):
            existing = str(existing).lower()
            if dataset_name is not None and existing != dataset_name:
                raise ValueError(
                    f"{path}:{row_number}: row dataset {existing!r} conflicts "
                    f"with requested dataset {dataset_name!r}")
            row["dataset"] = existing
        elif dataset_name is not None:
            row["dataset"] = dataset_name
        normalized.append(row)
    return normalized


def _tokens_from_row(row, text):
    raw_tokens = row.get("tokens")
    if isinstance(raw_tokens, str):
        try:
            parsed = json.loads(raw_tokens)
            if isinstance(parsed, list):
                raw_tokens = parsed
        except json.JSONDecodeError:
            raw_tokens = raw_tokens.split()
    if not raw_tokens:
        raw_tokens = re.findall(r"\w+|[^\w\s]", text)
    return [
        token if isinstance(token, dict) else {"id": f"t{index}", "text": str(token)}
        for index, token in enumerate(raw_tokens)
    ]


def _merge_tokens(target, tokens):
    for token in tokens:
        token_id = token.get("id") if isinstance(token, dict) else token
        if token_id not in target:
            target[token_id] = token if isinstance(token, dict) else {"id": token_id, "text": str(token)}


def _event_from_row(row, prefix, event_id, tokens):
    return {
        "id": event_id,
        "token_id": _event_token_id(row, prefix, event_id, tokens),
        "text": row.get(f"{prefix}_text") or event_id,
    }


def _event_token_id(row, prefix, event_id, tokens):
    for key in (f"{prefix}_token_id", f"{prefix}_token", f"{prefix}_index", f"{prefix}_idx"):
        value = row.get(key)
        if value not in (None, ""):
            value = str(value)
            if value.isdigit():
                index = int(value)
                if index < len(tokens):
                    return tokens[index].get("id") if isinstance(tokens[index], dict) else f"t{index}"
            return value
    token_ids = {token.get("id") if isinstance(token, dict) else token for token in tokens}
    return event_id if event_id in token_ids else (tokens[0].get("id") if tokens else event_id)


def _row_value(row, *keys, default=None):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


#: Canonical spelling for every relation any supported corpus can express.
#:
#: ``SIMULTANEOUS`` used to be folded into ``Equal``. That is wrong for TB-Dense:
#: MATRES's EQUAL compares start-points under its multi-axis scheme, while
#: TB-Dense's SIMULTANEOUS asserts interval identity — and collapsing them
#: silently destroys the distinction the symmetry/containment constraints need.
#: ``INCLUDES``/``IS_INCLUDED`` previously fell through unmapped and only failed
#: later, inside ``TEMPORAL_LABELS.index``, as an opaque ValueError.
_LABEL_ALIASES = {
    "BEFORE": "Before", "B": "Before",
    "AFTER": "After", "A": "After",
    "EQUAL": "Equal",
    "VAGUE": "Vague", "V": "Vague", "NONE": "Vague",
    "INCLUDES": "Includes", "I": "Includes",
    "IS_INCLUDED": "IsIncluded", "ISINCLUDED": "IsIncluded", "II": "IsIncluded",
    "SIMULTANEOUS": "Simultaneous", "S": "Simultaneous",
}


def _normalize_label(label):
    """Map a corpus relation onto its canonical name.

    Raises on anything unrecognised: a silently passed-through label only fails
    much later as an index error, with no indication of which corpus or row
    produced it.
    """
    text = str(label).strip()
    canonical = _LABEL_ALIASES.get(text.upper())
    if canonical is not None:
        return canonical
    if text in set(_LABEL_ALIASES.values()):
        return text
    raise ValueError(
        f"Unrecognised temporal relation {label!r}. Known relations: "
        f"{sorted(set(_LABEL_ALIASES.values()))}. Add an alias to "
        f"_LABEL_ALIASES if a new corpus uses a different spelling.")
