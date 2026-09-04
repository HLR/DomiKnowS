"""Convert TB-Dense releases into loader-ready JSONL.

The canonical ``muk343/TimeBank-dense`` mirror contains TimeML documents in
``train/``, ``dev/`` and ``test/``.  Some other mirrors expose a delimited
relation table instead, so both layouts remain supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .dataset import _normalize_label


DEFAULT_COLUMNS = ("doc_id", "e1", "e2", "label")
SPLITS = ("train", "dev", "test")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def convert_rows(reader, columns=DEFAULT_COLUMNS, dataset_name="tbdense"):
    """Yield loader-ready records from a delimited relation table."""
    index_of = {name: position for position, name in enumerate(columns)}
    required = ("doc_id", "e1", "e2", "label")
    missing = [name for name in required if name not in index_of]
    if missing:
        raise ValueError(f"--columns must include {required}; missing {missing}")

    for line_number, row in enumerate(reader, start=1):
        if not row or all(not str(cell).strip() for cell in row):
            continue
        if len(row) <= max(index_of.values()):
            raise ValueError(
                f"line {line_number}: expected at least {max(index_of.values()) + 1} "
                f"columns for layout {columns}, got {len(row)}: {row!r}")

        doc_id = str(row[index_of["doc_id"]]).strip()
        e1 = str(row[index_of["e1"]]).strip()
        e2 = str(row[index_of["e2"]]).strip()
        label = _normalize_label(row[index_of["label"]])
        e1_text = str(row[index_of["e1_text"]]).strip() if "e1_text" in index_of else e1
        e2_text = str(row[index_of["e2_text"]]).strip() if "e2_text" in index_of else e2
        text = str(row[index_of["text"]]).strip() if "text" in index_of else doc_id

        yield {
            "doc_id": doc_id,
            "text": text,
            "e1": e1,
            "e2": e2,
            "e1_text": e1_text,
            "e2_text": e2_text,
            "e1_token_id": e1,
            "e2_token_id": e2,
            "tokens": [{"id": e1, "text": e1_text}, {"id": e2, "text": e2_text}],
            "label": label,
            "dataset": dataset_name,
        }


def _local_name(tag):
    return str(tag).rsplit("}", 1)[-1]


def _elements(root, name):
    return [element for element in root.iter() if _local_name(element.tag) == name]


def _tokenize_text_element(text_element):
    """Return document tokens and the first token occupied by every EVENT."""
    tokens = []
    event_tokens = {}
    event_text = {}

    def append_text(value):
        for piece in TOKEN_RE.findall(value or ""):
            tokens.append({"id": f"t{len(tokens)}", "text": piece})

    def visit(element):
        append_text(element.text)
        for child in element:
            if _local_name(child.tag) == "EVENT":
                eid = child.get("eid")
                if not eid:
                    raise ValueError("EVENT is missing required eid")
                start = len(tokens)
                visit(child)
                if len(tokens) == start:
                    raise ValueError(f"EVENT {eid!r} contains no token text")
                if eid in event_tokens:
                    raise ValueError(f"duplicate EVENT eid {eid!r}")
                event_tokens[eid] = tokens[start]["id"]
                event_text[eid] = " ".join(
                    token["text"] for token in tokens[start:len(tokens)])
            else:
                visit(child)
            append_text(child.tail)

    visit(text_element)
    return tokens, event_tokens, event_text


def convert_timeml_file(path, split=None, dataset_name="tbdense",
                        conflict_policy="error", stats=None):
    """Convert one TimeML document, retaining only event-to-event TLINKs."""
    path = Path(path)
    if conflict_policy not in {"error", "last"}:
        raise ValueError(f"unsupported conflict_policy={conflict_policy!r}")
    stats = stats if stats is not None else Counter()
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{path}: malformed TimeML XML: {exc}") from exc

    doc_nodes = _elements(root, "DOCID")
    if len(doc_nodes) != 1:
        raise ValueError(f"{path}: expected exactly one DOCID, found {len(doc_nodes)}")
    doc_id = "".join(doc_nodes[0].itertext()).strip()
    if not doc_id:
        raise ValueError(f"{path}: DOCID is empty")

    text_nodes = _elements(root, "TEXT")
    if len(text_nodes) != 1:
        raise ValueError(f"{path}: expected exactly one TEXT, found {len(text_nodes)}")
    try:
        tokens, event_tokens, event_text = _tokenize_text_element(text_nodes[0])
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    text = " ".join(token["text"] for token in tokens)

    instance_to_events = {}
    for element in _elements(root, "MAKEINSTANCE"):
        eiid, eid = element.get("eiid"), element.get("eventID")
        if not eiid or not eid:
            raise ValueError(f"{path}: MAKEINSTANCE must contain eiid and eventID")
        if eid not in event_tokens:
            raise ValueError(f"{path}: MAKEINSTANCE {eiid!r} references unknown EVENT {eid!r}")
        candidates = instance_to_events.setdefault(eiid, set())
        if candidates and eid not in candidates:
            stats["ambiguous_makeinstance_ids_seen"] += 1
        candidates.add(eid)

    seen = {}
    records = []
    for element in _elements(root, "TLINK"):
        source_iid = element.get("eventInstanceID")
        target_iid = element.get("relatedToEventInstance")
        if not source_iid or not target_iid:
            continue  # event-time and time-time relations are outside EventPair
        lid = element.get("lid", "<unknown>")
        if source_iid not in instance_to_events or target_iid not in instance_to_events:
            missing = [
                value for value in (source_iid, target_iid)
                if value not in instance_to_events
            ]
            raise ValueError(
                f"{path}: TLINK {lid} references unknown event instance(s) {missing}")
        ambiguous = [
            value for value in (source_iid, target_iid)
            if len(instance_to_events[value]) != 1
        ]
        if ambiguous:
            details = {value: sorted(instance_to_events[value]) for value in ambiguous}
            raise ValueError(
                f"{path}: TLINK {lid} references ambiguous event instance(s) {details}")
        try:
            label = _normalize_label(element.get("relType"))
        except ValueError as exc:
            raise ValueError(f"{path}: TLINK {lid}: {exc}") from exc

        e1 = next(iter(instance_to_events[source_iid]))
        e2 = next(iter(instance_to_events[target_iid]))
        key = (e1, e2)
        previous = seen.get(key)
        if previous is not None and previous[0] != label:
            if conflict_policy == "error":
                raise ValueError(
                    f"{path}: conflicting TLINKs for {e1!r}->{e2!r}: "
                    f"{previous[0]!r} at {previous[2]} versus {label!r} at {lid}; "
                    "use --conflict-policy last only after inspecting the source")
            stats["conflicting_tlinks_resolved"] += 1
            records[previous[1]]["label"] = label
            seen[key] = (label, previous[1], lid)
            continue
        if previous is not None:
            stats["exact_duplicate_tlinks_skipped"] += 1
            continue
        seen[key] = (label, len(records), lid)
        records.append({
            "e1": e1,
            "e2": e2,
            "label": label,
        })
    event_ids = {
        event_id
        for candidates in instance_to_events.values()
        for event_id in candidates
    }
    token_position = {
        token["id"]: index for index, token in enumerate(tokens)
    }
    events = [
        {
            "id": event_id,
            "text": event_text[event_id],
            "token_id": event_tokens[event_id],
        }
        for event_id in sorted(
            event_ids,
            key=lambda value: (token_position[event_tokens[value]], value),
        )
    ]
    if not records:
        return []
    return [{
        "doc_id": doc_id,
        "text": text,
        "tokens": tokens,
        "events": events,
        "event_pairs": records,
        "dataset": dataset_name,
        "split": split,
    }]


def _write_jsonl(records, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    with open(output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def convert_delimited_file(source, output, columns=DEFAULT_COLUMNS, delimiter="\t",
                           dataset_name="tbdense"):
    with open(source, "r", newline="", encoding="utf-8") as handle:
        return _write_jsonl(
            convert_rows(csv.reader(handle, delimiter=delimiter), columns=columns,
                         dataset_name=dataset_name),
            output,
        )


def convert_timeml_source(source, output=None, output_dir=None,
                          dataset_name="tbdense", conflict_policy="error",
                          stats=None):
    """Convert a .tml file, a split directory, or a clone root."""
    source = Path(source)
    stats = stats if stats is not None else Counter()
    split_dirs = {name: source / name for name in SPLITS}
    is_clone_root = source.is_dir() and all(path.is_dir() for path in split_dirs.values())
    if is_clone_root:
        if output_dir is None:
            raise ValueError("--output-dir is required when --source is a clone root")
        outputs = {}
        for split, directory in split_dirs.items():
            records = []
            for path in sorted(directory.glob("*.tml")):
                records.extend(convert_timeml_file(path, split=split,
                                                   dataset_name=dataset_name,
                                                   conflict_policy=conflict_policy,
                                                   stats=stats))
            outputs[split] = _write_jsonl(records, Path(output_dir) / f"{split}.jsonl")
        return outputs

    paths = [source] if source.is_file() else sorted(source.glob("*.tml"))
    if not paths:
        raise ValueError(f"no .tml files found under {source}")
    if output is None:
        raise ValueError("--output is required for a TimeML file or split directory")
    split = source.name.lower() if source.is_dir() and source.name.lower() in SPLITS else None
    records = []
    for path in paths:
        records.extend(convert_timeml_file(path, split=split,
                                           dataset_name=dataset_name,
                                           conflict_policy=conflict_policy,
                                           stats=stats))
    return _write_jsonl(records, output)


def _format_summary(name, records, output):
    labels = Counter()
    relation_rows = 0
    for row in records:
        pairs = row.get("event_pairs")
        if pairs is None:
            pairs = [row]
        relation_rows += len(pairs)
        labels.update(pair["label"] for pair in pairs)
    documents = {row["doc_id"] for row in records}
    size = Path(output).stat().st_size
    return (
        f"{name}: documents={len(documents)} relations={relation_rows} bytes={size} "
        f"labels={dict(sorted(labels.items()))} output={output}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", help="destination JSONL for a file/split source")
    parser.add_argument("--output-dir", help="destination directory for clone-root conversion")
    parser.add_argument("--format", choices=("auto", "timeml", "delimited"), default="auto")
    parser.add_argument(
        "--columns", default=",".join(DEFAULT_COLUMNS),
        help=("Comma-separated delimited source order. Must include "
              "doc_id,e1,e2,label; e1_text,e2_text,text are optional."))
    parser.add_argument("--delimiter", default="\t")
    parser.add_argument("--dataset-name", default="tbdense")
    parser.add_argument(
        "--conflict-policy",
        choices=("error", "last"),
        default="error",
        help=(
            "How to handle duplicate event pairs with different labels. "
            "'error' is strict; 'last' explicitly selects the later TLINK and "
            "reports the number resolved."
        ),
    )
    args = parser.parse_args(argv)

    source = Path(args.source)
    source_format = args.format
    if source_format == "auto":
        source_format = "timeml" if (
            source.suffix.lower() == ".tml"
            or source.is_dir()
        ) else "delimited"

    if source_format == "delimited":
        if not args.output:
            parser.error("--output is required for delimited conversion")
        columns = tuple(name.strip() for name in args.columns.split(",") if name.strip())
        records = convert_delimited_file(
            source, args.output, columns=columns, delimiter=args.delimiter,
            dataset_name=args.dataset_name)
        print(_format_summary("delimited", records, args.output))
        return 0

    try:
        stats = Counter()
        converted = convert_timeml_source(
            source, output=args.output, output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            conflict_policy=args.conflict_policy,
            stats=stats)
    except ValueError as exc:
        parser.error(str(exc))
    if isinstance(converted, dict):
        for split, records in converted.items():
            print(_format_summary(split, records, Path(args.output_dir) / f"{split}.jsonl"))
    else:
        print(_format_summary(source.name, converted, args.output))
    if stats:
        print(f"source_anomalies={dict(sorted(stats.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
