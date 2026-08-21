from dataclasses import dataclass
import keyword
import re

from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, ifL


OBJECT_SYMBOL_RELATIONS = {"Name", "ObjectType", "ObjectCategory", "Attribute", "Capable"}
BASE_KB_RELATIONS = {"TypeOf"}
SYMMETRIC_OBJECT_RELATIONS = {"HangingNear", "Near", "By", "CloseTo"}
RELATION_ALIASES = {
    "Hypernym": "TypeOf",
    "ObjectClass": "ObjectType",
    "watching": "LookingAt",
    "Watching": "LookingAt",
    "looking_at": "LookingAt",
}
RESERVED_CONCEPT_NAMES = {
    "path", "graph", "andL", "orL", "iotaL", "queryL", "existsL", "notL",
    # Graph schema names must never be shadowed by VQAR object/symbol values.
    "scene", "obj", "symbol", "object_symbol_pair", "symbol_pair", "object_pair",
    "answer_object", "object_symbol_relation", "symbol_pair_relation", "object_pair_relation",
}

ATTRIBUTE_ALIASES = {
    "round": ["round", "circular"],
    "wooden": ["wooden", "hardwood", "wood"],
    "large": ["large", "giant", "big"],
    "small": ["small", "tiny", "little"],
    "plaid": ["plaid", "checkered"],
    "light_blue": ["light_blue", "bright_blue"],
    "metal": ["metal", "metallic"],
    "concrete": ["concrete", "cement"],
    "uncooked": ["uncooked", "raw"],
}

SEMANTIC_CLASS_ALIASES = {
    # Keep these aliases intentionally narrow. Broad concepts such as
    # ``food`` or ``container`` often make iotaL ambiguous in VQAR scenes.
    "drinks": ["drinks", "drink", "water", "beverage"],
    "kitchenware": ["kitchenware", "sink", "plate", "tray"],
    "canidae": ["canidae", "dog", "canid", "canine"],
    "tableware": ["tableware", "plate", "tray", "dish"],
    "shoes": ["shoes", "shoe", "footwear"],
    "odd-toed_ungulate": ["odd-toed_ungulate", "horse", "equine"],
    "plant": ["plant", "tree", "flower", "flowers"],
    "mammal": ["mammal", "person", "people", "dog", "horse", "zebra", "elephant", "sheep", "cow"],
    "object": ["object", "sign", "generic_artifact", "physical_object", "tangible_thing"],
    # Official VQAR question files contain WordNet/GQA class labels that do
    # not always match the KG surface form used by scene graph names.
    "tops": ["tops", "top"],
    "person.type.01": ["person.type.01", "person", "people"],
    "part_of_body": ["part_of_body", "body_part", "external_anatomical_part"],
    "part_of_vehicle": ["part_of_vehicle", "vehicle_part", "auto_part"],
    "clothing": ["pants", "shirt", "coat", "jeans", "jacket", "sweater", "socks", "sock", "skirt"],
    "place": ["sky", "building", "sidewalk", "station", "bridge", "ocean", "street", "runway", "tower", "balcony", "road"],
    "accessory": ["shoe", "shoes", "cap", "wristband", "goggles", "mask", "glove", "belt", "hat", "glasses"],
    "symbol": ["letters", "numbers", "arrow"],
    "sports_equipment": ["frisbee", "ball", "racket"],
    "hat": ["hat", "cap"],
    "public_transports": ["airplane", "train", "bus"],
    "bedding": ["pillow", "blanket", "bed"],
    "electrical_appliance": ["oven", "microwave", "stove"],
    "electronic_device": ["keyboard", "computer", "monitor", "laptop", "phone"],
    "vehicle": ["train", "bicycle", "car", "bus", "truck", "airplane", "boat", "motorcycle"],
    "kitchen_utensil": ["knife", "spoon", "fork"],
}


def alias_values(kind, value):
    value = str(value)
    if kind == "Attribute":
        aliases = [value] + list(ATTRIBUTE_ALIASES.get(value, []))
        return list(dict.fromkeys(aliases))
    if kind == "SemanticClass":
        aliases = [value] + list(SEMANTIC_CLASS_ALIASES.get(value, []))
        if value.endswith("s") and len(value) > 3:
            aliases.append(value[:-1])
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(aliases))
    return [value]


def canonical_relation(value):
    if value is None:
        return None
    raw = str(value).strip()
    mapped = RELATION_ALIASES.get(raw, RELATION_ALIASES.get(raw.lower(), raw))
    if any(sep in mapped for sep in ("_", "-", " ")):
        return "".join(part.capitalize() for part in mapped.replace("-", "_").replace(" ", "_").split("_") if part)
    if mapped.islower():
        return mapped.capitalize()
    return mapped


def safe_name(value):
    """Convert instance object/symbol ids into valid DomiKnowS variable names."""
    value = str(value)
    safe = re.sub(r"\W", "_", value)
    if not safe or safe[0].isdigit() or keyword.iskeyword(safe) or safe in RESERVED_CONCEPT_NAMES:
        safe = f"v_{safe}"
    return safe


@dataclass
class GraphQAContext:
    graph: Graph
    scene: Concept
    obj: Concept
    answer_object: Concept
    object_domain: Concept
    symbol: Concept
    object_values: list
    symbol_values: list
    scene_contains_obj: Relation
    scene_contains_symbol: Relation
    object_symbol_pair: Concept
    symbol_pair: Concept
    object_pair: Concept
    object_symbol_object: Relation
    object_symbol_symbol: Relation
    symbol_pair_src: Relation
    symbol_pair_dst: Relation
    object_pair_src: Relation
    object_pair_dst: Relation
    object_symbol_relation: Concept
    symbol_pair_relation: Concept
    object_pair_relation: Concept
    object_symbol_relations: dict
    symbol_relations: dict
    object_relations: dict
    object_concepts: dict
    symbol_concepts: dict
    concepts: dict
    namespace: dict


def create_graphqa_graph(instance_or_dataset, graph_name="graphqa", include_global_constraints=False):
    """
    Build the bounded GraphQA DomiKnowS graph.

    Like CLEVR, relation concepts are created by iterating over the relation
    vocabulary. For GraphQA that vocabulary is discovered from all facts and
    query conditions in the instance or dataset passed in.
    """
    instances = _as_instances(instance_or_dataset)
    objects = _collect_values(instances, "objects")
    symbols = _collect_values(instances, "symbols")
    object_symbol_relation_names = collect_object_symbol_relations(instances)
    object_relation_names = collect_object_relations(instances)

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph(graph_name) as graph:
        scene = Concept(name="scene")
        obj = Concept(name="obj")
        answer_object = obj(name="answer_object")
        object_domain = obj(name="object_domain")
        symbol = Concept(name="symbol")
        scene_contains_obj, = scene.contains(obj)
        scene_contains_symbol, = scene.contains(symbol)

        object_concepts = {safe_name(o): answer_object(name=safe_name(o)) for o in objects}
        symbol_concepts = {safe_name(s): symbol(name=safe_name(s)) for s in symbols}

        object_symbol_pair = Concept(name="object_symbol_pair")
        object_symbol_object, object_symbol_symbol = object_symbol_pair.has_a(
            object_arg=obj,
            symbol_arg=symbol,
        )
        object_symbol_relation = object_symbol_pair(name="object_symbol_relation")
        object_symbol_relations = {
            rel_name: object_symbol_pair(name=rel_name)
            for rel_name in object_symbol_relation_names
        }

        symbol_pair = Concept(name="symbol_pair")
        symbol_pair_src, symbol_pair_dst = symbol_pair.has_a(
            src_arg=symbol,
            dst_arg=symbol,
        )
        symbol_pair_relation = symbol_pair(name="symbol_pair_relation")
        symbol_relations = {
            rel_name: symbol_pair(name=rel_name)
            for rel_name in collect_kb_relations(instances)
        }

        object_pair = Concept(name="object_pair")
        object_pair_src, object_pair_dst = object_pair.has_a(
            src_arg=obj,
            dst_arg=obj,
        )
        object_pair_relation = object_pair(name="object_pair_relation")
        object_relations = {
            rel_name: object_pair(name=rel_name)
            for rel_name in object_relation_names
        }

        if include_global_constraints:
            _apply_graphqa_consistency_constraints(
                obj=obj,
                symbol=symbol,
                object_symbol_object=object_symbol_object,
                object_symbol_symbol=object_symbol_symbol,
                symbol_pair_src=symbol_pair_src,
                symbol_pair_dst=symbol_pair_dst,
                object_symbol_relations=object_symbol_relations,
                symbol_relations=symbol_relations,
            )

    concepts = {
        "scene": scene,
        "obj": obj,
        "answer_object": answer_object,
        "object_domain": object_domain,
        "symbol": symbol,
        "scene_contains_obj": scene_contains_obj,
        "scene_contains_symbol": scene_contains_symbol,
        "object_symbol_pair": object_symbol_pair,
        "object_symbol_object": object_symbol_object,
        "object_symbol_symbol": object_symbol_symbol,
        "symbol_pair": symbol_pair,
        "symbol_pair_src": symbol_pair_src,
        "symbol_pair_dst": symbol_pair_dst,
        "object_pair": object_pair,
        "object_pair_src": object_pair_src,
        "object_pair_dst": object_pair_dst,
        "object_symbol_relation": object_symbol_relation,
        "symbol_pair_relation": symbol_pair_relation,
        "object_pair_relation": object_pair_relation,
        **object_symbol_relations,
        **symbol_relations,
        **object_relations,
        **object_concepts,
        **symbol_concepts,
    }

    namespace = dict(concepts)
    namespace.update({"graph": graph, "iota_target": obj})
    _register_namespace(graph, namespace)

    return GraphQAContext(
        graph=graph,
        scene=scene,
        obj=obj,
        answer_object=answer_object,
        object_domain=object_domain,
        symbol=symbol,
        object_values=objects,
        symbol_values=symbols,
        scene_contains_obj=scene_contains_obj,
        scene_contains_symbol=scene_contains_symbol,
        object_symbol_pair=object_symbol_pair,
        symbol_pair=symbol_pair,
        object_pair=object_pair,
        object_symbol_object=object_symbol_object,
        object_symbol_symbol=object_symbol_symbol,
        symbol_pair_src=symbol_pair_src,
        symbol_pair_dst=symbol_pair_dst,
        object_pair_src=object_pair_src,
        object_pair_dst=object_pair_dst,
        object_symbol_relation=object_symbol_relation,
        symbol_pair_relation=symbol_pair_relation,
        object_pair_relation=object_pair_relation,
        object_symbol_relations=object_symbol_relations,
        symbol_relations=symbol_relations,
        object_relations=object_relations,
        object_concepts=object_concepts,
        symbol_concepts=symbol_concepts,
        concepts=concepts,
        namespace=namespace,
    )



def _apply_graphqa_consistency_constraints(
    *,
    obj,
    symbol,
    object_symbol_object,
    object_symbol_symbol,
    symbol_pair_src,
    symbol_pair_dst,
    object_symbol_relations,
    symbol_relations,
):
    """Global constraints relating learned grounding to fixed KB evidence."""
    name = object_symbol_relations.get("Name")
    object_type = object_symbol_relations.get("ObjectType")
    object_category = object_symbol_relations.get("ObjectCategory")
    type_of = symbol_relations.get("TypeOf")
    if name is None:
        return

    if type_of is None or object_type is None or object_category is None:
        return

    # Learned Name(o,x) and fixed TypeOf(x,y) entail ObjectType(o,y).
    ifL(
        andL(
            name("name_pair"),
            obj("o", path=("name_pair", object_symbol_object)),
            symbol("x", path=("name_pair", object_symbol_symbol)),
            type_of("type_pair", path=("x", symbol_pair_src.reversed)),
            symbol("y", path=("type_pair", symbol_pair_dst)),
        ),
        object_type(
            "object_type_pair",
            path=(("o", object_symbol_object.reversed), ("y", object_symbol_symbol.reversed)),
        ),
        name="graphqa_name_typeof_implies_object_type",
    )

    # A second fixed TypeOf edge yields the bounded category prediction.
    ifL(
        andL(
            object_type("object_type_pair"),
            obj("o2", path=("object_type_pair", object_symbol_object)),
            symbol("x2", path=("object_type_pair", object_symbol_symbol)),
            type_of("type_pair2", path=("x2", symbol_pair_src.reversed)),
            symbol("y2", path=("type_pair2", symbol_pair_dst)),
        ),
        object_category(
            "object_category_pair",
            path=(("o2", object_symbol_object.reversed), ("y2", object_symbol_symbol.reversed)),
        ),
        name="graphqa_object_type_typeof_implies_category",
    )

def collect_object_symbol_relations(instance_or_dataset):
    instances = _as_instances(instance_or_dataset)
    relation_names = set(OBJECT_SYMBOL_RELATIONS)
    for instance in instances:
        objects = {str(obj) for obj in instance.get("objects", [])}
        for pred, left, _right in instance.get("kb_facts", []):
            pred = canonical_relation(pred)
            if str(left) in objects:
                relation_names.add(pred)
        query = instance.get("query", {})
        condition_groups = [query.get("conditions", [])]
        condition_groups.extend(query.get("alternatives", []))
        for conditions in condition_groups:
            for pred, left, _right in conditions:
                pred = canonical_relation(pred)
                if left == "o" and any(
                    canonical_relation(fact_pred) == pred and str(fact_left) in objects
                    for fact_pred, fact_left, _fact_right in instance.get("kb_facts", [])
                ):
                    relation_names.add(pred)
    return sorted(relation_names)


def collect_kb_relations(instance_or_dataset):
    instances = _as_instances(instance_or_dataset)
    relation_names = set(BASE_KB_RELATIONS)
    for instance in instances:
        objects = {str(obj) for obj in instance.get("objects", [])}
        for pred, left, _right in instance.get("kb_facts", []):
            if str(left) in objects:
                continue
            relation_names.add(canonical_relation(pred))
        query = instance.get("query", {})
        condition_groups = [query.get("conditions", [])]
        condition_groups.extend(query.get("alternatives", []))
        for conditions in condition_groups:
            for pred, _left, right in conditions:
                pred = canonical_relation(pred)
                if pred == "KG":
                    relation_names.add(canonical_relation(right[0]))
    return sorted(relation_names)


KB_RELATIONS = collect_kb_relations


def collect_object_relations(instance_or_dataset):
    instances = _as_instances(instance_or_dataset)
    relation_names = set()
    for instance in instances:
        for pred, _left, _right in instance.get("visual_facts", []):
            pred = canonical_relation(pred)
            if _is_object_relation(pred):
                relation_names.add(pred)
        query = instance.get("query", {})
        condition_groups = [query.get("conditions", [])]
        condition_groups.extend(query.get("alternatives", []))
        for conditions in condition_groups:
            for pred, _left, right in conditions:
                pred = canonical_relation(pred)
                if pred in {"RelationFrom", "RelationTo"}:
                    relation_names.add(canonical_relation(right[0]))
                elif _is_object_relation(pred):
                    relation_names.add(pred)
    return sorted(relation_names)


def _is_object_relation(pred):
    return pred not in {"KG", "OneOf", "SemanticClass", "RelationFrom", "RelationTo"} and pred not in OBJECT_SYMBOL_RELATIONS and pred not in collect_kb_relations([])


def _as_instances(instance_or_dataset):
    if isinstance(instance_or_dataset, dict):
        return [instance_or_dataset]
    return list(instance_or_dataset)


def _collect_values(instances, key):
    values = []
    seen = set()
    for instance in instances:
        for value in instance.get(key, []):
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


def _register_namespace(graph, namespace):
    var_map = graph.varNameReversedMap
    for name, value in namespace.items():
        var_map[name] = value
        if hasattr(value, "reversed"):
            var_map[f"{name}.reversed"] = value.reversed
