import json
import os
import pickle
from pathlib import Path

from .graph import alias_values


DEFAULT_VQAR_ROOT = Path(os.environ.get("GRAPHQA_VQAR_ROOT", "/egr/research-hlr2/premsrit/VQAR_data"))
FALLBACK_VQAR_ROOT = Path("/localscratch2/VQAR/VQAR_all/VQAR")
DEFAULT_DATA_DIR = DEFAULT_VQAR_ROOT / "data"
DEFAULT_FEATURE_KB_DIR = Path("/localscratch2/VQAR/feature_file")
TASK_GLOB = "dataset/task_list/*tasks*.pkl"
SUPPORTED_FUNCTIONS = {
    "Initial",
    "Find_Name",
    "Find_Attr",
    "Hypernym_Find",
    "Relate",
    "Relate_Reverse",
    "And",
    "Or",
    "KG_Find",
}


class GraphQADatasetNotFound(FileNotFoundError):
    """Raised when the VQAR code is present but the downloaded dataset is not."""


def discover_vqar_dataset(root=None):
    """
    Locate the real VQAR dataset expected by the Scallop implementation.

    The VQAR README/download script puts the Zenodo payload under
    ``VQAR_all/VQAR/data``. The actual executable question files are pickled
    task lists under ``data/dataset/task_list``.
    """
    root = Path(root) if root is not None else _default_existing_root()
    data_dir = root / "data"
    task_paths = sorted(data_dir.glob(TASK_GLOB))
    return {
        "root": root,
        "data_dir": data_dir,
        "meta_info": data_dir / "gqa_info.json",
        "knowledge_base": data_dir / "knowledge_base",
        "task_paths": task_paths,
    }


def require_vqar_dataset(root=None):
    discovered = discover_vqar_dataset(root)
    missing = []
    if not discovered["data_dir"].is_dir():
        missing.append(str(discovered["data_dir"]))
    if not discovered["meta_info"].is_file():
        missing.append(str(discovered["meta_info"]))
    if not discovered["task_paths"]:
        missing.append(str(discovered["data_dir"] / TASK_GLOB))
    if missing:
        raise GraphQADatasetNotFound(
            "VQAR dataset files are not installed. Expected the Zenodo data.zip "
            f"payload under {discovered['data_dir']}. Missing: {missing}"
        )
    return discovered


def load_vqar_tasks(path, limit=None, offset=0):
    with open(path, "rb") as task_file:
        tasks = list(pickle.load(task_file))
    start = max(0, int(offset))
    stop = None if limit is None else start + int(limit)
    return tasks[start:stop]


def load_vqar_graphqa_instances(path, limit=None, kb_dir=None, offset=0):
    tasks = load_vqar_tasks(path, limit=limit, offset=offset)
    kb_facts = load_kb_facts(kb_dir=kb_dir)
    instances = []
    failures = []
    for index, task in enumerate(tasks):
        try:
            instances.append(vqar_task_to_graphqa_instance(task, kb_facts=kb_facts))
        except ValueError as exc:
            failures.append((index, str(exc)))
    return instances, failures


def load_kb_facts(kb_dir=None):
    """Load TypeOf and open-attribute KG facts from VQAR knowledge_base files.

    When an explicit VQAR ``kb_dir`` is provided, use only that directory.
    Mixing in fallback ConceptNet-style files pollutes the oracle hierarchy
    with noisy paths such as ``kite -> bird -> animal`` and makes C2 oracle
    execution diverge from the Scallop/Tadalog setting.
    """
    facts = []
    if kb_dir is not None:
        kb_dir = Path(kb_dir)
        candidates = [kb_dir / "is_a.facts", kb_dir / "in_oa_rel.facts"]
    else:
        candidates = [
            DEFAULT_DATA_DIR / "knowledge_base" / "is_a.facts",
            DEFAULT_DATA_DIR / "knowledge_base" / "in_oa_rel.facts",
            FALLBACK_VQAR_ROOT / "data" / "knowledge_base" / "is_a.facts",
            FALLBACK_VQAR_ROOT / "data" / "knowledge_base" / "in_oa_rel.facts",
        ]
        if not any(path.is_file() for path in candidates):
            candidates.append(DEFAULT_FEATURE_KB_DIR / "isa_relation.csv")

    for path in candidates:
        if not path.is_file():
            continue
        if path.name == "in_oa_rel.facts":
            facts.extend(_read_open_attribute_path(path))
        else:
            facts.extend(_read_isa_path(path))
    return facts


def load_isa_facts(kb_dir=None):
    return [fact for fact in load_kb_facts(kb_dir=kb_dir) if fact[0] == "TypeOf"]



_SCENE_GRAPH_BY_IMAGE_CACHE = None


def _ensure_joined_vqar_task(task):
    """Accept both joined task-list records and official VQAR question records."""
    if not isinstance(task, dict):
        return task
    if "question" in task and "scene_graph" in task:
        return task
    if "clauses" not in task or "image_id" not in task:
        return task
    image_id = task.get("image_id")
    scene_graph = _scene_graph_by_image().get(str(image_id), {})
    object_ids = task.get("input", []) or sorted(scene_graph.get("names", {}).keys())
    return {
        "image_id": image_id,
        "scene_graph": scene_graph,
        "question": task,
        "object_ids": object_ids,
    }


def _scene_graph_by_image():
    global _SCENE_GRAPH_BY_IMAGE_CACHE
    if _SCENE_GRAPH_BY_IMAGE_CACHE is not None:
        return _SCENE_GRAPH_BY_IMAGE_CACHE
    _SCENE_GRAPH_BY_IMAGE_CACHE = {}
    candidates = [
        DEFAULT_DATA_DIR / "dataset" / "scene_graph" / "gqa_train_formatted_scene_graph.pkl",
        DEFAULT_DATA_DIR / "dataset" / "scene_graph" / "gqa_val_formatted_scene_graph.pkl",
        DEFAULT_DATA_DIR / "dataset" / "scene_graph" / "gqa_test_formatted_scene_graph.pkl",
        FALLBACK_VQAR_ROOT / "data" / "dataset" / "scene_graph" / "gqa_train_formatted_scene_graph.pkl",
        FALLBACK_VQAR_ROOT / "data" / "dataset" / "scene_graph" / "gqa_val_formatted_scene_graph.pkl",
        FALLBACK_VQAR_ROOT / "data" / "dataset" / "scene_graph" / "gqa_test_formatted_scene_graph.pkl",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        with open(path, "rb") as scene_file:
            scene_graphs = pickle.load(scene_file)
        iterable = scene_graphs.values() if isinstance(scene_graphs, dict) else scene_graphs
        for scene_graph in iterable:
            if isinstance(scene_graph, dict) and "image_id" in scene_graph:
                _SCENE_GRAPH_BY_IMAGE_CACHE[str(scene_graph["image_id"])] = scene_graph
    return _SCENE_GRAPH_BY_IMAGE_CACHE

def vqar_task_to_graphqa_instance(task, kb_facts=None):
    task = _ensure_joined_vqar_task(task)
    question = task.get("question", {})
    scene_graph = task.get("scene_graph", {})
    objects = [str(obj) for obj in question.get("input", []) or task.get("object_ids", [])]
    if not objects:
        objects = sorted({str(obj) for obj in scene_graph.get("names", {}).keys()})

    visual_facts, symbols = _scene_graph_to_facts(scene_graph)
    query = _clauses_to_query(question.get("clauses", []), question.get("output", []))
    symbols.update(_symbols_from_query(query))
    if query["target_type"] not in (None, "__any_object__"):
        symbols.add(query["target_type"])

    kb_facts = list(kb_facts or [])
    symbols.update(symbol for _pred, left, right in kb_facts for symbol in (left, right))

    return {
        "objects": objects,
        "object_metadata": _object_metadata(task, objects, scene_graph),
        "symbols": sorted(symbols),
        "visual_facts": visual_facts,
        "kb_facts": kb_facts,
        "query": query,
        "expected_answer": _single_answer(question.get("output", [])),
        "expected_answers": _answer_list(question.get("output", [])),
        "source_question_id": question.get("question_id"),
        "source_image_id": task.get("image_id") or question.get("image_id"),
    }


def _object_metadata(task, objects, scene_graph):
    """Keep non-label object inputs for learned grounding prompts.

    Scene-graph names/attributes/relations are the labels for visual predicates,
    so they should not be fed back into the predicate learner. Bounding boxes
    and precomputed region-feature summaries are safe inputs for the scorer.
    """
    metadata = {}
    bboxes = scene_graph.get("bboxes", {}) if isinstance(scene_graph, dict) else {}
    object_ids = [str(obj) for obj in task.get("object_ids", []) or objects]
    object_features = task.get("object_feature", []) or []
    feature_by_object = {}
    for index, obj in enumerate(object_ids):
        if index < len(object_features):
            full_feature = _feature_vector(object_features[index])
            feature_by_object[str(obj)] = {
                "summary": _feature_summary(object_features[index]),
                "vector": full_feature,
            }

    for obj in objects:
        key = str(obj)
        raw_bbox = bboxes.get(obj, bboxes.get(key))
        if raw_bbox is None:
            try:
                raw_bbox = bboxes.get(int(key))
            except (TypeError, ValueError):
                raw_bbox = None
        item = {}
        image_id = task.get("image_id") or (task.get("question", {}) or {}).get("image_id")
        if image_id is not None:
            item["image_id"] = str(image_id)
        if task.get("url"):
            item["image_url"] = str(task.get("url"))
        if raw_bbox is not None:
            item["bbox"] = [float(value) for value in raw_bbox]
        if key in feature_by_object:
            item["feature"] = feature_by_object[key]["summary"]
            item["feature_vector"] = feature_by_object[key]["vector"]
        metadata[key] = item
    return metadata


def _feature_vector(feature):
    try:
        return [float(value) for value in feature]
    except TypeError:
        return []


def _feature_summary(feature, head=8):
    try:
        values = [float(value) for value in feature]
    except TypeError:
        return {}
    if not values:
        return {}
    length = len(values)
    mean = sum(values) / length
    maximum = max(values)
    nonzero = sum(1 for value in values if value != 0.0)
    return {
        "dim": length,
        "mean": round(mean, 6),
        "max": round(maximum, 6),
        "nonzero": nonzero,
        "head": [round(value, 6) for value in values[:head]],
    }


def _scene_graph_to_facts(scene_graph):
    facts = []
    symbols = set()

    for obj, name in scene_graph.get("names", {}).items():
        name = _normalize_gqa_symbol("name", name)
        if name is None:
            continue
        facts.append(("Name", str(obj), name))
        symbols.add(name)

    for obj, attrs in scene_graph.get("attributes", {}).items():
        for attr in _as_list(attrs):
            attr = _normalize_gqa_symbol("attr", attr)
            if attr is None:
                continue
            facts.append(("Attribute", str(obj), attr))
            symbols.add(attr)

    for subject, object_info in scene_graph.get("relations", {}).items():
        for obj, rels in object_info.items():
            for rel in _as_list(rels):
                rel = _normalize_gqa_relation(rel)
                if rel is None:
                    continue
                facts.append((rel, str(subject), str(obj)))

    return facts, symbols


def _clauses_to_query(clauses, output):
    if not clauses:
        raise ValueError("VQAR task has no question clauses")

    expressions = {}
    last_expression = None
    last_program = None
    for clause in clauses:
        function = clause.get("function")
        if function not in SUPPORTED_FUNCTIONS:
            raise ValueError(f"Unsupported VQAR clause function: {function!r}")

        clause_id = clause.get("clause_id")
        inputs = clause.get("input_clause_id")
        text_input = clause.get("text_input")

        if function == "Initial":
            expression = [[]]
            program = {"op": "all"}
        elif function == "And":
            input_ids = _as_list(inputs)
            branches = [_expression_for_input(expressions, input_id) for input_id in input_ids]
            expression = [[]]
            for branch in branches:
                expression = [left + right for left in expression for right in branch]
            program = {
                "op": "and",
                "inputs": [_program_for_input(expressions, input_id) for input_id in input_ids],
            }
        elif function == "Or":
            input_ids = _as_list(inputs)
            expression = []
            for input_id in input_ids:
                expression.extend(_expression_for_input(expressions, input_id))
            program = {
                "op": "or",
                "inputs": [_program_for_input(expressions, input_id) for input_id in input_ids],
            }
        else:
            if inputs in (None, "") and last_expression is not None:
                expression = [list(conditions) for conditions in last_expression]
                input_program = last_program
            else:
                expression = [list(conditions) for conditions in _expression_for_input(expressions, inputs)]
                input_program = _program_for_input(expressions, inputs)
            condition = _clause_condition(clause, function, text_input)
            if condition is not None:
                for conditions in expression:
                    conditions.append(condition)
            if function in {"Relate", "Relate_Reverse"} and condition is not None:
                relation, object_ids = condition[2]
                program = {
                    "op": "relate",
                    "direction": condition[0],
                    "relation": relation,
                    "candidates": object_ids,
                    "input": input_program or {"op": "all"},
                }
            elif condition is not None:
                program = {
                    "op": "filter",
                    "condition": condition,
                    "input": input_program or {"op": "all"},
                }
            else:
                program = input_program or {"op": "all"}

        if clause_id is not None:
            expressions[clause_id] = {"expression": expression, "program": program}
        last_expression = expression
        last_program = program

    alternatives = last_expression or [[]]
    final_program = last_program or {"op": "all"}
    target_type = "__any_object__"
    if len(alternatives) == 1:
        remaining = []
        for condition in alternatives[0]:
            if condition[0] == "ObjectCategory" and target_type == "__any_object__":
                target_type = condition[2]
            else:
                remaining.append(condition)
        alternatives = [remaining]

    answer = _single_answer(output)
    if answer is not None:
        alternatives = [
            [condition for condition in conditions if not (condition[0] == "Attribute" and condition[2] is None)]
            for conditions in alternatives
        ]

    query = {
        "target_type": target_type,
        "conditions": alternatives[0] if len(alternatives) == 1 else [],
        "answer_type": "object",
    }
    if len(alternatives) > 1:
        query["alternatives"] = alternatives
    query["program"] = final_program
    return query


def _expression_for_input(expressions, input_id):
    if input_id in (None, ""):
        return [[]]
    if isinstance(input_id, (list, tuple)):
        expression = [[]]
        for item in input_id:
            branch = _expression_for_input(expressions, item)
            expression = [left + right for left in expression for right in branch]
        return expression
    item = expressions.get(input_id)
    if isinstance(item, dict):
        return [list(conditions) for conditions in item.get("expression", [[]])]
    return [list(conditions) for conditions in item or [[]]]


def _program_for_input(expressions, input_id):
    if input_id in (None, ""):
        return {"op": "all"}
    if isinstance(input_id, (list, tuple)):
        return {"op": "and", "inputs": [_program_for_input(expressions, item) for item in input_id]}
    item = expressions.get(input_id)
    if isinstance(item, dict):
        return item.get("program", {"op": "all"})
    return {"op": "all"}


def _clause_condition(clause, function, text_input):
    if function == "Find_Name":
        symbol = _normalize_symbol(text_input)
        return ("Name", "o", symbol) if symbol is not None else None
    if function == "Hypernym_Find":
        symbol = _normalize_symbol(text_input)
        if symbol in {"thing"}:
            return None
        return ("SemanticClass", "o", symbol) if symbol is not None else None
    if function == "Find_Attr":
        return ("Attribute", "o", _normalize_symbol(text_input))
    if function == "KG_Find":
        return _kg_find_condition(text_input)
    if function in {"Relate", "Relate_Reverse"}:
        rel = _normalize_relation(text_input)
        object_ids = [str(obj) for obj in clause.get("object_ids", [])]
        if rel is None or not object_ids:
            return None
        if function == "Relate":
            return ("RelationFrom", "o", (rel, object_ids))
        return ("RelationTo", "o", (rel, object_ids))
    return None

def _read_isa_path(path):
    facts = []
    with open(path, "r") as isa_file:
        for line in isa_file:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[0] == "isa":
                src, dst = parts[1], parts[2]
            elif len(parts) >= 2:
                src, dst = parts[0], parts[1]
            else:
                continue
            src = _normalize_symbol(src)
            dst = _normalize_symbol(dst)
            if src is not None and dst is not None:
                facts.append(("TypeOf", src, dst))
    return facts


def _read_open_attribute_path(path):
    facts = []
    with open(path, "r") as kg_file:
        for line in kg_file:
            parts = line.strip().split("	")
            if len(parts) < 3:
                continue
            rel, src, dst = parts[0], parts[1], parts[2]
            rel = _normalize_relation(rel)
            src = _normalize_symbol(src)
            dst = _normalize_symbol(dst)
            if rel is None or src is None or dst is None:
                continue
            facts.append((rel, src, dst))
    return facts


def _symbols_from_query(query):
    symbols = set()
    condition_groups = [query.get("conditions", [])]
    condition_groups.extend(query.get("alternatives", []))
    for conditions in condition_groups:
        for pred, _left, right in conditions:
            if pred in {"Name", "ObjectType", "ObjectCategory"}:
                symbols.add(right)
            elif pred == "Attribute":
                symbols.update(alias_values("Attribute", right))
            elif pred == "SemanticClass":
                symbols.update(alias_values("SemanticClass", right))
            elif pred == "KG":
                _rel, dst = right
                symbols.add(dst)
    return {symbol for symbol in symbols if symbol is not None and not str(symbol).startswith("__")}


def _kg_find_condition(text_input):
    if not isinstance(text_input, (list, tuple)) or len(text_input) < 3:
        return None
    left, rel, right = text_input[0], text_input[1], text_input[2]
    if left not in ("BLANK", "", None):
        return None
    rel = _normalize_relation(rel)
    right = _normalize_symbol(right)
    if rel is None or right is None:
        return None
    return ("KG", "o", (rel, right))


def _single_answer(output):
    answers = _answer_list(output)
    return answers[0] if len(answers) == 1 else None


def _answer_list(output):
    if output is None:
        return []
    if isinstance(output, (list, tuple, set)):
        return [str(item) for item in output]
    return [str(output)]


_GQA_INFO_CACHE = None
_GQA_REVERSE_INDEX_CACHE = {}


def _load_gqa_info():
    global _GQA_INFO_CACHE
    if _GQA_INFO_CACHE is not None:
        return _GQA_INFO_CACHE
    candidates = [
        DEFAULT_DATA_DIR / "gqa_info.json",
        FALLBACK_VQAR_ROOT / "data" / "gqa_info.json",
    ]
    for path in candidates:
        if path.is_file():
            with open(path, "r") as info_file:
                _GQA_INFO_CACHE = json.load(info_file)
            return _GQA_INFO_CACHE
    _GQA_INFO_CACHE = {}
    return _GQA_INFO_CACHE


def _reverse_gqa_index(kind):
    if kind in _GQA_REVERSE_INDEX_CACHE:
        return _GQA_REVERSE_INDEX_CACHE[kind]
    info = _load_gqa_info().get(kind, {})
    index = info.get("idx", {}) if isinstance(info, dict) else {}
    reverse = {str(idx): label for label, idx in index.items()}
    _GQA_REVERSE_INDEX_CACHE[kind] = reverse
    return reverse


def _normalize_gqa_symbol(kind, value):
    if isinstance(value, (list, tuple)):
        value = next((part for part in value if part not in ("", "BLANK", None)), None)
    mapped = _reverse_gqa_index(kind).get(str(value))
    return _normalize_symbol(mapped if mapped is not None else value)


def _normalize_gqa_relation(value):
    if isinstance(value, (list, tuple)):
        value = next((part for part in value if part not in ("", "BLANK", None)), None)
    mapped = _reverse_gqa_index("rel").get(str(value))
    return _normalize_relation(mapped if mapped is not None else value)


def _normalize_symbol(value):
    if isinstance(value, (list, tuple)):
        value = next((part for part in value if part not in ("", "BLANK", None)), None)
    if value is None or value == -1:
        return None
    return str(value).strip().replace(" ", "_")


RELATION_TEXT_ALIASES = {
    "left": "to_the_left_of",
    "right": "to_the_right_of",
    "near": "hanging_near",
    "next_to": "hanging_near",
    "beside": "hanging_near",
    "in": "sleeping_in",
    "above": "at_top_of",
    "over": "at_top_of",
    "below": "at_bottom_of",
    "under": "at_bottom_of",
    "front": "in_front_of",
    "behind": "standing_behind",
    "on": "at_top_of",
    "under": "supporting",
}


def _normalize_relation(value):
    value = _normalize_symbol(value)
    if value is None or str(value).isdigit():
        return None
    value = RELATION_TEXT_ALIASES.get(value.lower(), value)
    return "".join(part.capitalize() for part in value.replace("-", "_").split("_") if part)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _default_existing_root():
    if DEFAULT_VQAR_ROOT.exists():
        return DEFAULT_VQAR_ROOT
    return FALLBACK_VQAR_ROOT
