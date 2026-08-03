"""Object-centered single-answer GraphQA pipeline.

Visual names and attributes are binary concepts on Object. Visual relations are
binary concepts on ObjectPair. Every learned/oracle atomic predicate is attached
through ModuleLearner, matching the CLEVR program layout.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from domiknows.graph import Concept, Graph, Relation
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor, FunctionalSensor

from .dataset import load_kb_facts, load_vqar_tasks, vqar_task_to_graphqa_instance
from .graph import canonical_relation, safe_name

DEFAULT_IMAGE_CACHE = Path("/egr/research-hlr2/premsrit/VQAR_data/image_cache")


class BinaryOracleLearner(torch.nn.Module):
    """Convert fixed 0/1 predicate evidence to [No, Yes] logits."""

    def __init__(self, confidence=8.0):
        super().__init__()
        self.confidence = float(confidence)

    def forward(self, labels):
        labels = labels.long().view(-1)
        return torch.stack(
            [
                (1 - labels).float() * self.confidence,
                labels.float() * self.confidence,
            ],
            dim=-1,
        )


@dataclass
class ObjectCenteredContext:
    graph: Graph
    image: Concept
    obj: Concept
    pair: Concept
    answer_object: Concept
    image_contains_obj: Relation
    pair_src: Relation
    pair_dst: Relation
    answer_slots: list
    object_predicates: dict
    relation_predicates: dict
    namespace: dict


def predicate_name(kind, value):
    prefix = {"Name": "name", "Attribute": "attr"}[kind]
    return f"{prefix}_{safe_name(value)}"


def required_visual_predicates(instances):
    object_predicates = set()
    relation_predicates = set()
    for instance in instances:
        query = instance.get("query", {})
        if query.get("target_type") not in (None, "__any_object__"):
            raise ValueError("Object-centered VLM smoke does not yet translate semantic target_type")
        branches = query.get("alternatives") or [query.get("conditions", [])]
        for conditions in branches:
            for pred, _left, right in conditions:
                pred = canonical_relation(pred)
                if pred in {"Name", "Attribute"}:
                    object_predicates.add((pred, str(right)))
                elif pred in {"RelationFrom", "RelationTo"}:
                    relation, _objects = right
                    relation_predicates.add(canonical_relation(relation))
                elif pred == "OneOf":
                    continue
                else:
                    raise ValueError(
                        f"Object-centered VLM smoke does not yet support predicate {pred!r}"
                    )
    return sorted(object_predicates), sorted(relation_predicates)


def create_object_centered_graph(instances):
    object_specs, relation_specs = required_visual_predicates(instances)
    max_objects = max(len(instance.get("objects", [])) for instance in instances)

    Graph.clear()
    Concept.clear()
    Relation.clear()
    from domiknows.graph.dataNode import DataNode, DataNodeBuilder
    from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()
    with Graph("graphqa_object_centered") as graph:
        image = Concept(name="image")
        obj = Concept(name="object")
        image_contains_obj, = image.contains(obj)

        answer_object = obj(name="answer_object")
        answer_slots = [
            answer_object(name=f"answer_slot_{index}") for index in range(max_objects)
        ]

        object_predicates = {
            spec: obj(name=predicate_name(*spec)) for spec in object_specs
        }

        pair = Concept(name="object_pair")
        pair_src, pair_dst = pair.has_a(src_arg=obj, dst_arg=obj)
        relation_predicates = {
            relation: pair(name=f"relation_{safe_name(relation)}")
            for relation in relation_specs
        }

    namespace = {
        "answer_object": answer_object,
        "pair_src": pair_src,
        "pair_dst": pair_dst,
        **{concept.name: concept for concept in answer_slots},
        **{concept.name: concept for concept in object_predicates.values()},
        **{concept.name: concept for concept in relation_predicates.values()},
    }
    return ObjectCenteredContext(
        graph=graph,
        image=image,
        obj=obj,
        pair=pair,
        answer_object=answer_object,
        image_contains_obj=image_contains_obj,
        pair_src=pair_src,
        pair_dst=pair_dst,
        answer_slots=answer_slots,
        object_predicates=object_predicates,
        relation_predicates=relation_predicates,
        namespace=namespace,
    )


def _condition_atoms(instance, context, condition, index):
    pred, left, right = condition
    pred = canonical_relation(pred)
    if left != "o":
        raise ValueError(f"Expected target variable 'o', got {left!r}")

    if pred in {"Name", "Attribute"}:
        concept = context.object_predicates[(pred, str(right))]
        return [f'{concept.name}(path="o")']

    if pred == "OneOf":
        slots = []
        object_index = {str(value): i for i, value in enumerate(instance["objects"])}
        for value in right:
            if str(value) in object_index:
                slots.append(f'{context.answer_slots[object_index[str(value)]].name}(path="o")')
        if not slots:
            raise ValueError("OneOf has no candidate object in this instance")
        return [slots[0] if len(slots) == 1 else "orL(" + ", ".join(slots) + ")"]

    if pred in {"RelationFrom", "RelationTo"}:
        relation, candidate_objects = right
        relation = canonical_relation(relation)
        relation_concept = context.relation_predicates[relation]
        object_index = {str(value): i for i, value in enumerate(instance["objects"])}
        branches = []
        for candidate_index, candidate in enumerate(candidate_objects):
            if str(candidate) not in object_index:
                continue
            slot = context.answer_slots[object_index[str(candidate)]]
            pair_var = f"r{index}_{candidate_index}"
            if pred == "RelationFrom":
                atoms = [
                    f'{relation_concept.name}("{pair_var}", path=("o", pair_dst.reversed))',
                    f'{slot.name}(path=("{pair_var}", pair_src))',
                ]
            else:
                atoms = [
                    f'{relation_concept.name}("{pair_var}", path=("o", pair_src.reversed))',
                    f'{slot.name}(path=("{pair_var}", pair_dst))',
                ]
            branches.append("andL(" + ", ".join(atoms) + ")")
        if not branches:
            raise ValueError("Relation condition has no candidate anchor")
        return [branches[0] if len(branches) == 1 else "orL(" + ", ".join(branches) + ")"]

    raise ValueError(f"Unsupported object-centered condition: {condition!r}")


def create_logic(instance, context):
    query = instance["query"]
    conditions = query.get("conditions", [])
    if not conditions:
        raise ValueError("Object-centered query requires at least one visual condition")

    first = conditions[0]
    first_pred, first_left, first_right = first
    first_pred = canonical_relation(first_pred)
    if first_left != "o" or first_pred not in {"Name", "Attribute"}:
        raise ValueError("The first object-centered atom must be Name or Attribute")

    first_concept = context.object_predicates[(first_pred, str(first_right))]
    atoms = [f'{first_concept.name}("o")']
    for index, condition in enumerate(conditions[1:], start=1):
        atoms.extend(_condition_atoms(instance, context, condition, index))

    body = atoms[0] if len(atoms) == 1 else "andL(\n            " + ",\n            ".join(atoms) + "\n        )"
    return (
        "queryL(\n"
        "    answer_object,\n"
        "    iotaL(\n"
        f"        {body}\n"
        "    )\n"
        ")"
    )


def _same_image(_property=None, src_arg=None, dst_arg=None, **kwargs):
    if src_arg is not None and dst_arg is not None:
        return src_arg.getAttribute("image_id") == dst_arg.getAttribute("image_id")
    nodes = [value for value in kwargs.values() if hasattr(value, "getAttribute")]
    return len(nodes) < 2 or nodes[0].getAttribute("image_id") == nodes[1].getAttribute("image_id")


def _qwen_module(model_path, device, relation, attr):
    clever_dir = Path(__file__).resolve().parents[1] / "Clever"
    if str(clever_dir) not in sys.path:
        sys.path.insert(0, str(clever_dir))
    from qwen_vl_hf import QwenVLSharedHF

    return QwenVLSharedHF(
        model_path=str(model_path),
        device=device,
        relation=relation,
        attr=attr,
        use_llm_lora=False,
    )


def attach_sensors(context, mode="oracle", model_path=None, device="cpu"):
    context.image["index"] = FunctionalReaderSensor(
        keyword="image_index",
        forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
    )
    context.image["pil_image"] = FunctionalReaderSensor(
        keyword="pil_image", forward=lambda data: [data]
    )
    context.image["image_filename"] = FunctionalReaderSensor(
        keyword="image_filename", forward=lambda data: [data]
    )
    context.obj["index"] = FunctionalReaderSensor(
        keyword="object_indices",
        forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
    )
    context.obj["ids"] = FunctionalReaderSensor(
        keyword="object_ids", forward=lambda data: list(data)
    )
    context.obj["boxes"] = FunctionalReaderSensor(
        keyword="object_boxes",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device),
    )
    context.obj["image_id"] = FunctionalSensor(
        context.image["index"],
        context.obj["boxes"],
        forward=lambda image_index, boxes: image_index.repeat(len(boxes)),
    )
    context.obj[context.image_contains_obj] = EdgeSensor(
        context.obj["index"],
        context.image["index"],
        relation=context.image_contains_obj,
        forward=lambda objects, _image: torch.ones_like(objects).unsqueeze(-1),
    )

    for slot_index, slot in enumerate(context.answer_slots):
        key = f"answer_slot_{slot_index}_label"
        context.obj[key] = FunctionalSensor(
            context.obj["index"],
            forward=lambda indices, expected=slot_index: (indices.view(-1) == expected).long(),
        )
        context.obj[slot] = ModuleLearner(
            key, module=BinaryOracleLearner(), device=device
        )

    for (kind, value), concept in context.object_predicates.items():
        if mode == "oracle":
            key = f"{concept.name}_oracle"
            context.obj[key] = FunctionalReaderSensor(
                keyword=key,
                forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
            )
            module = BinaryOracleLearner()
            context.obj[concept] = ModuleLearner(key, module=module, device=device)
        else:
            module = _qwen_module(model_path, device, relation=1, attr=value)
            context.obj[concept] = ModuleLearner(
                context.image["pil_image"],
                context.image["image_filename"],
                context.obj["boxes"],
                module=module,
                device=device,
            )

    context.pair[
        context.pair_src.reversed,
        context.pair_dst.reversed,
    ] = CompositionCandidateSensor(
        context.obj["image_id"],
        relations=(context.pair_src.reversed, context.pair_dst.reversed),
        forward=_same_image,
    )
    for relation, concept in context.relation_predicates.items():
        if mode == "oracle":
            key = f"{concept.name}_oracle"
            context.pair[key] = FunctionalReaderSensor(
                keyword=key,
                forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
            )
            context.pair[concept] = ModuleLearner(
                key, module=BinaryOracleLearner(), device=device
            )
        else:
            module = _qwen_module(model_path, device, relation=2, attr=relation)
            context.pair[concept] = ModuleLearner(
                context.image["pil_image"],
                context.image["image_filename"],
                context.obj["boxes"],
                module=module,
                device=device,
            )


def _image_and_boxes(instance, image_cache):
    image_id = instance.get("source_image_id")
    image_path = Path(image_cache) / f"{image_id}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    boxes = []
    metadata = instance.get("object_metadata", {})
    for object_id in instance["objects"]:
        box = metadata.get(str(object_id), {}).get("bbox")
        if box is None:
            raise ValueError(f"Missing bounding box for object {object_id}")
        x, y, w, h = [float(value) for value in box]
        boxes.append([x * width, y * height, (x + w) * width, (y + h) * height])
    return image, image_path, boxes


def _attach_object_metadata(instance, task):
    """Attach non-label visual inputs without changing the shared dataset adapter."""
    scene_graph = task.get("scene_graph", {})
    bboxes = scene_graph.get("bboxes", {}) if isinstance(scene_graph, dict) else {}
    metadata = {}
    for object_id in instance["objects"]:
        key = str(object_id)
        box = bboxes.get(object_id, bboxes.get(key))
        if box is None:
            try:
                box = bboxes.get(int(key))
            except (TypeError, ValueError):
                pass
        metadata[key] = {
            "bbox": None if box is None else [float(value) for value in box]
        }
    instance["object_metadata"] = metadata
    return instance


def populate_example(instance, context, image_cache=DEFAULT_IMAGE_CACHE, device="cpu"):
    objects = [str(value) for value in instance["objects"]]
    object_index = {value: index for index, value in enumerate(objects)}
    facts = {
        (canonical_relation(pred), str(left), str(right))
        for pred, left, right in instance.get("visual_facts", [])
    }
    image, image_path, boxes = _image_and_boxes(instance, image_cache)
    row = {
        "image_index": torch.tensor([0], dtype=torch.long, device=device),
        "pil_image": image,
        "image_filename": str(image_path),
        "object_indices": torch.arange(len(objects), dtype=torch.long, device=device),
        "object_ids": objects,
        "object_boxes": boxes,
        "logic_str": create_logic(instance, context),
        "logic_label": torch.tensor(
            [object_index[str(instance["expected_answer"])]],
            dtype=torch.long,
            device=device,
        ),
    }
    for (kind, value), concept in context.object_predicates.items():
        row[f"{concept.name}_oracle"] = [
            int((kind, object_id, value) in facts) for object_id in objects
        ]
    for relation, concept in context.relation_predicates.items():
        row[f"{concept.name}_oracle"] = [
            int((relation, src, dst) in facts)
            for src in objects
            for dst in objects
        ]
    return row


def build_program(instances, mode="oracle", model_path=None, image_cache=DEFAULT_IMAGE_CACHE, device="cpu"):
    if any(len(instance.get("expected_answers", [])) != 1 for instance in instances):
        raise ValueError("Object-centered pipeline currently supports single-answer instances only")
    context = create_object_centered_graph(instances)
    attach_sensors(context, mode=mode, model_path=model_path, device=device)
    rows = [
        populate_example(instance, context, image_cache=image_cache, device=device)
        for instance in instances
    ]
    dataset = context.graph.compile_executable(
        rows,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=context.namespace,
    )
    poi = [
        context.image,
        context.obj,
        context.pair,
        context.answer_object,
        *context.answer_slots,
        *context.object_predicates.values(),
        *context.relation_predicates.values(),
        context.graph.constraint,
    ]
    program = InferenceProgram(
        context.graph,
        SolverModel,
        poi=poi,
        device=device,
        tnorm="P",
        inferTypes=["local/argmax"],
        beta=1.0,
    )
    return context, dataset, program


def load_instances(task_path, kb_dir, limit):
    kb_facts = load_kb_facts(kb_dir)
    instances = []
    failures = []
    for index, task in enumerate(load_vqar_tasks(task_path, limit=limit)):
        try:
            instance = vqar_task_to_graphqa_instance(task, kb_facts=kb_facts)
            _attach_object_metadata(instance, task)
            if len(instance.get("expected_answers", [])) != 1:
                continue
            required_visual_predicates([instance])
            instances.append(instance)
        except Exception as exc:
            failures.append((index, type(exc).__name__, str(exc)))
    return instances, failures


def evaluate_executable(dataset, program, device):
    with torch.no_grad():
        return program.evaluate_condition(dataset, device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--kb-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--mode", choices=["oracle", "qwen-vl"], default="oracle")
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "qwen-vl" and not args.model_path:
        raise ValueError("--model-path is required for --mode qwen-vl")
    instances, failures = load_instances(args.task_path, args.kb_dir, args.limit)
    if not instances:
        raise ValueError(f"No supported single-answer instances; failures={failures[:5]}")
    context, dataset, program = build_program(
        instances,
        mode=args.mode,
        model_path=args.model_path,
        image_cache=args.image_cache,
        device=args.device,
    )
    accuracy = evaluate_executable(dataset, program, args.device)
    print(json.dumps({
        "mode": args.mode,
        "loaded": len(instances),
        "failures": len(failures),
        "accuracy": accuracy,
        "logic": dataset[0].get("logic_str", instances[0].get("logic_str")),
    }, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
