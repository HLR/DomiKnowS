"""Convert TimeBank-Dense TimeML files into DomiKnowS TemporalRelation JSONL.

The converter keeps the methodology shape used by the TemporalRelation adapter:
event nodes are extracted from TimeML EVENT/MAKEINSTANCE annotations, each
selected TLINK becomes a query pair, and the audit fields can be passed through
the existing executable query construction.

By default we project TimeBank-Dense labels to the four MATRES labels:

    BEFORE -> Before
    AFTER -> After
    SIMULTANEOUS -> Equal
    NONE/VAGUE -> Vague

INCLUDES and IS_INCLUDED are skipped unless a downstream graph adds those
concepts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from .execution import create_executable_instance
from .graph import TEMPORAL_LABELS


LABEL_MAP = {
    "BEFORE": "Before",
    "AFTER": "After",
    "SIMULTANEOUS": "Equal",
    "EQUAL": "Equal",
    "NONE": "Vague",
    "VAGUE": "Vague",
}


def _clean_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _append_text_tokens(text, tokens):
    for piece in re.findall(r"\w+|[^\w\s]", text or ""):
        tokens.append({"id": f"t{len(tokens)}", "text": piece})


def _parse_text_node(text_node):
    tokens = []
    event_by_eid = {}

    def visit(node):
        _append_text_tokens(node.text, tokens)
        if node.tag == "EVENT":
            token_id = tokens[-1]["id"] if tokens else f"t{len(tokens)}"
            eid = node.attrib.get("eid")
            if eid:
                event_by_eid[eid] = {
                    "eid": eid,
                    "token_id": token_id,
                    "text": _clean_space(" ".join(token["text"] for token in tokens if token["id"] == token_id)),
                }
        for child in list(node):
            visit(child)
            _append_text_tokens(child.tail, tokens)

    visit(text_node)
    text = " ".join(token["text"] for token in tokens)
    return tokens, event_by_eid, _clean_space(text)


def parse_tml(path):
    root = ET.parse(path).getroot()
    doc_id = root.findtext("DOCID") or path.stem
    text_node = root.find("TEXT")
    if text_node is None:
        return []
    tokens, event_by_eid, text = _parse_text_node(text_node)

    eiid_to_event = {}
    for instance in root.findall("MAKEINSTANCE"):
        eiid = instance.attrib.get("eiid")
        eid = instance.attrib.get("eventID")
        event = event_by_eid.get(eid)
        if eiid and event:
            eiid_to_event[eiid] = {"id": eiid, "token_id": event["token_id"], "text": event["text"] or eid}

    rows = []
    for link in root.findall("TLINK"):
        e1 = link.attrib.get("eventInstanceID")
        e2 = link.attrib.get("relatedToEventInstance")
        raw_label = (link.attrib.get("relType") or "").upper()
        label = LABEL_MAP.get(raw_label)
        if not e1 or not e2 or label is None:
            continue
        if e1 not in eiid_to_event or e2 not in eiid_to_event:
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "text": text,
                "tokens": tokens,
                "events": [eiid_to_event[e1], eiid_to_event[e2]],
                "event_pairs": [{"e1": e1, "e2": e2, "label": label, "raw_label": raw_label}],
                "query_pair": {"e1": e1, "e2": e2},
                "source_path": str(path),
            }
        )
    return rows


def load_tbdense_rows(root):
    rows = []
    for path in sorted(Path(root).glob("*/*.tml")):
        rows.extend(parse_tml(path))
    return rows


def balanced_sample(rows, total, seed=13, labels=None):
    labels = labels or list(TEMPORAL_LABELS)
    if total % len(labels) != 0:
        raise ValueError(f"--total must be divisible by {len(labels)} labels")
    per_label = total // len(labels)
    buckets = defaultdict(list)
    for row in rows:
        label = row["event_pairs"][0]["label"]
        if label in labels:
            buckets[label].append(row)
    rng = random.Random(seed)
    sampled = []
    for label in labels:
        candidates = list(buckets[label])
        rng.shuffle(candidates)
        if len(candidates) < per_label:
            raise ValueError(f"Label {label} has {len(candidates)} rows, need {per_label}")
        sampled.extend(candidates[:per_label])
    rng.shuffle(sampled)
    return sampled


def write_jsonl(path, rows, include_logic=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as data_file:
        for row in rows:
            output = dict(row)
            if include_logic:
                converted = create_executable_instance(row)
                output["logic_str"] = converted["logic_str"]
                output["logic_label"] = int(converted["logic_label"])
                output["pair_learner_examples"] = converted["pair_learner_examples"]
            data_file.write(json.dumps(output, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args(argv)

    rows = load_tbdense_rows(args.root)
    counts = Counter(row["event_pairs"][0]["label"] for row in rows)
    raw_counts = Counter(row["event_pairs"][0].get("raw_label") for row in rows)
    if args.stats_only:
        print(json.dumps({"total": len(rows), "labels": dict(counts), "raw_labels": dict(raw_counts)}, sort_keys=True))
        return 0

    sampled = balanced_sample(rows, total=args.total, seed=args.seed)
    write_jsonl(args.output, sampled)
    sampled_counts = Counter(row["event_pairs"][0]["label"] for row in sampled)
    print(json.dumps({"output": str(args.output), "total": len(sampled), "labels": dict(sampled_counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
