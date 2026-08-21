"""Object-centered single-answer GraphQA pipeline.

Visual names and attributes are binary concepts on Object. Visual relations are
binary concepts on ObjectPair. Every learned/oracle atomic predicate is attached
through ModuleLearner, matching the CLEVR program layout.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.logicalConstrain import existsL, ifL, miotaL, nandL
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor, FunctionalSensor

from .dataset import load_kb_facts, load_vqar_tasks, vqar_task_to_graphqa_instance
from .execution import materialize_bounded_facts
from .graph import alias_values, canonical_relation, safe_name
from .modules import (
    GraphQAPredicateClassifier,
    NO_RELATION_LABEL,
    _object_pair_feature_prompt,
    _object_symbol_feature_prompt,
)

DEFAULT_IMAGE_CACHE = Path("/egr/research-hlr2/premsrit/VQAR_data/image_cache")


_OPPOSITE_RELATIONS = {
    "Left": "Right",
    "Right": "Left",
    "ToLeftOf": "ToRightOf",
    "ToRightOf": "ToLeftOf",
    "ToTheLeftOf": "ToTheRightOf",
    "ToTheRightOf": "ToTheLeftOf",
    "Above": "Below",
    "Below": "Above",
    "InFrontOf": "Behind",
    "Behind": "InFrontOf",
    "Over": "Under",
    "Under": "Over",
    "Inside": "Contain",
    "Contain": "Inside",
}




def _derived_or_logits(*source_logits):
    """Deterministically materialize a KB-derived object concept.

    KB facts stay as graph constraints, but executable queries also need local
    scores for the derived concept. We compute those scores from the source name
    predicates without adding a learnable module for the KB concept itself.
    """

    if not source_logits:
        raise ValueError("A derived KB concept needs at least one source predicate")
    margins = [logits[..., 1] - logits[..., 0] for logits in source_logits]
    yes_margin = torch.stack(margins, dim=0).amax(dim=0)
    return torch.stack([-yes_margin, yes_margin], dim=-1)


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


class NonSelfPairLogitWrapper(torch.nn.Module):
    """Adapt full-square VLM relation scores to DomiKnowS ObjectPair rows."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        logits = self.module(*args, **kwargs)
        logits = logits.view(-1, logits.shape[-1])
        boxes = args[2] if len(args) >= 3 else kwargs.get("bounding_boxes")
        if boxes is None:
            return logits
        object_count = int(boxes.shape[0]) if torch.is_tensor(boxes) else len(boxes)
        if object_count <= 0:
            return logits
        full_count = object_count * object_count
        non_self_count = object_count * (object_count - 1)
        if logits.shape[0] == full_count:
            keep = [
                src * object_count + dst
                for src in range(object_count)
                for dst in range(object_count)
                if src != dst
            ]
            index = torch.as_tensor(keep, dtype=torch.long, device=logits.device)
            return logits.index_select(0, index)
        if logits.shape[0] == non_self_count:
            return logits
        raise ValueError(
            f"Expected square or non-self relation logits for {object_count} objects, "
            f"got {logits.shape[0]} rows"
        )


class ScallopLocalFactLearner(torch.nn.Module):
    """Convert local scene-graph evidence into probabilistic fact logits.

    Scallop's VQAR setup feeds probabilistic local facts to the symbolic
    executor. This module gives DomiKnowS the same interface: each predicate
    remains a ModuleLearner, but its input is a compact local-fact evidence bit
    rather than a VLM prompt.
    """

    def __init__(self, confidence=8.0, learnable_scale=False):
        super().__init__()
        confidence = torch.tensor(float(confidence), dtype=torch.float32)
        if learnable_scale:
            self.logit_scale = torch.nn.Parameter(confidence)
        else:
            self.register_buffer("logit_scale", confidence)

    def forward(self, evidence):
        evidence = evidence.float().view(-1)
        scale = self.logit_scale.clamp_min(0.0)
        yes_margin = (evidence * 2.0 - 1.0) * scale
        return torch.stack([-yes_margin, yes_margin], dim=-1)


class ScallopTrainedPredicateView(torch.nn.Module):
    """Binary view of a trained GraphQA atomic-predicate classifier.

    The shared classifier predicts the multiclass local predicate family
    (Name/Attribute/NoRelation or a visual relation label).  Each DomiKnowS
    concept receives the corresponding yes/no logit pair, matching Scallop's
    neural-probabilistic input relations while staying compatible with
    ModuleLearner concepts.
    """

    def __init__(self, shared, kind, label, prompt_builder, device="cpu"):
        super().__init__()
        self.shared = shared
        self.kind = kind
        self.label = label
        self.prompt_builder = prompt_builder
        self.device_name = device

    def forward(self, prompts):
        prompts = list(prompts)
        if not prompts:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device_name)
        examples = [
            {"kind": self.kind, "label": self.label, "prompt": prompt}
            for prompt in prompts
        ]
        family_logits = self.shared.forward_examples(examples)
        labels = (
            self.shared.object_symbol_labels
            if self.kind == "object_symbol"
            else self.shared.object_pair_labels
        )
        label_to_index = {value: index for index, value in enumerate(labels)}
        yes_index = label_to_index.get(self.label)
        no_index = label_to_index.get(NO_RELATION_LABEL)
        if yes_index is None:
            raise ValueError(f"Trained predicate checkpoint has no label {self.label!r} for {self.kind}")
        if no_index is None:
            no_logit = torch.logsumexp(family_logits, dim=-1)
        else:
            no_logit = family_logits[:, no_index]
        return torch.stack([no_logit, family_logits[:, yes_index]], dim=-1)


def load_scallop_trained_classifier(model_path, checkpoint_path, device, options):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    spaces = checkpoint.get("label_spaces") or {}
    classifier = GraphQAPredicateClassifier(
        model_path=model_path,
        object_symbol_labels=spaces.get("object_symbol"),
        symbol_pair_labels=spaces.get("symbol_pair"),
        object_pair_labels=spaces.get("object_pair"),
        device=device,
        freeze_backbone=bool(options.get("freeze_backbone", True)),
        lora_r=int(options.get("lora_r", 0)),
        lora_alpha=int(options.get("lora_alpha", 16)),
        lora_dropout=float(options.get("lora_dropout", 0.05)),
        lora_target_modules=options.get("lora_target_modules"),
        max_length=int(options.get("max_length", 128)),
        encode_batch_size=options.get("encode_batch_size"),
    )
    classifier.object_symbol_head.load_state_dict(checkpoint["object_symbol_head"])
    classifier.symbol_pair_head.load_state_dict(checkpoint["symbol_pair_head"])
    classifier.object_pair_head.load_state_dict(checkpoint["object_pair_head"])
    if "backbone_lora" in checkpoint:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(classifier.backbone, checkpoint["backbone_lora"])
    classifier.eval()
    return classifier


class ScallopObjectMLP(torch.nn.Module):
    """VQAR Section 5.2 MLP over a 2048-D object-region feature."""

    def __init__(self, input_dim, output_dim, hidden_dim=1024, dropout=0.3, hidden_layers=1):
        super().__init__()
        layers = []
        width = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.extend(
                [
                    torch.nn.Linear(width, int(hidden_dim)),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(int(hidden_dim)),
                    torch.nn.Dropout(float(dropout)),
                ]
            )
            width = int(hidden_dim)
        layers.append(torch.nn.Linear(width, int(output_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features.float())


class ScallopRelationMLP(torch.nn.Module):
    """VQAR relation MLP over two object features and two bounding boxes."""

    def __init__(self, object_input_dim, output_dim, hidden_dim=1024, dropout=0.5):
        super().__init__()
        pair_dim = int(object_input_dim) * 2 + 8
        self.net = torch.nn.Sequential(
            torch.nn.Linear(pair_dim, int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(int(hidden_dim)),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, pair_features):
        return self.net(pair_features.float())


# DomiKnowS's InferenceModel defaults to torch.nn.BCELoss (not
# BCEWithLogitsLoss), which requires its input already in [0, 1] -- it applies
# its own sigmoid/softmax to whatever logits our concept-view modules return,
# then asserts the result lands in range. An extreme logit (e.g. from
# ScallopObjectMLP/ScallopRelationMLP's BatchNorm1d producing an unstable
# output on a small pair-batch) can saturate that internal sigmoid so hard
# the floating-point result lands a hair outside [0, 1] (1.0000001 instead of
# 1.0), which crashes as a CUDA device-side assert deep in framework code
# (dataNode.py's inferLocal) that we're not able to edit directly. Sigmoid is
# already fully saturated by +-30 (sigmoid(30) is ~1 - 9e-14), so clamping
# logits to that range here costs no real signal while keeping every
# downstream probability computation safely away from the boundary.
_LOGIT_CLAMP = 30.0


def _clamp_logits(tensor):
    # clamp() leaves NaN untouched (comparisons with NaN are always false), so an
    # unstable BatchNorm1d output on a small batch can still produce a NaN logit
    # that reaches DomiKnowS's inferLocal()/torch.argmax() and trips a CUDA
    # device-side assert. Scrub NaN/Inf before clamping.
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=_LOGIT_CLAMP, neginf=-_LOGIT_CLAMP)
    return tensor.clamp(min=-_LOGIT_CLAMP, max=_LOGIT_CLAMP)


class ScallopMLPConceptView(torch.nn.Module):
    """Expose one multiclass MLP output as binary [No, Yes] concept logits."""

    def __init__(self, shared, yes_index, no_index=None):
        super().__init__()
        self.shared = shared
        self.yes_index = int(yes_index)
        self.no_index = None if no_index is None else int(no_index)

    def forward(self, features):
        logits = self.shared(features)
        yes = logits[:, self.yes_index]
        if self.no_index is None:
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            mask[self.yes_index] = False
            no = torch.logsumexp(logits[:, mask], dim=-1)
        else:
            no = logits[:, self.no_index]
        return _clamp_logits(torch.stack([no, yes], dim=-1))


class ScallopSigmoidConceptView(torch.nn.Module):
    """Expose one independent attribute logit as binary [No, Yes] logits."""

    def __init__(self, shared, yes_index):
        super().__init__()
        self.shared = shared
        self.yes_index = int(yes_index)

    def forward(self, features):
        yes = self.shared(features)[:, self.yes_index]
        return _clamp_logits(torch.stack([torch.zeros_like(yes), yes], dim=-1))


def _multiclass_binary_logits(logits, yes_index):
    yes_index = int(yes_index)
    yes = logits[:, yes_index]
    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask[yes_index] = False
    no = torch.logsumexp(logits[:, mask], dim=-1)
    return _clamp_logits(torch.stack([no, yes], dim=-1))


def _sigmoid_binary_logits(logits, yes_index):
    yes = logits[:, int(yes_index)]
    return _clamp_logits(torch.stack([torch.zeros_like(yes), yes], dim=-1))


def _objects_for_vocab(instances):
    names = set()
    attributes = set()
    for instance in instances:
        for pred, _left, right in instance.get("visual_facts", []):
            pred = canonical_relation(pred)
            if pred in {"Name", "ObjectType", "ObjectCategory"}:
                names.add(str(right))
            elif pred == "Attribute":
                attributes.add(str(right))
        for kind, value in required_visual_predicates([instance])[0]:
            if kind == "Name":
                names.add(str(value))
            elif kind == "Attribute":
                attributes.add(str(value))
    return sorted(names), sorted(attributes)


def _object_feature_dim(instances):
    for instance in instances:
        for metadata in (instance.get("object_metadata") or {}).values():
            vector = metadata.get("feature_vector") or []
            if vector:
                return len(vector)
    return 8


def _build_scallop_mlp_modules(context, instances, device, options):
    checkpoint = options.get("scallop_mlp_checkpoint")
    state = torch.load(checkpoint, map_location=device, weights_only=False) if checkpoint else None
    if state:
        if state.get("architecture_version") != "vqar_scallop_5_2":
            raise ValueError("Checkpoint predates the faithful VQAR Section 5.2 architecture")
        name_to_index = dict(state["indices"]["name"])
        attr_to_index = dict(state["indices"]["attribute"])
        relation_to_index = dict(state["indices"]["relation"])
        name_classes = int(state["num_classes"]["name"])
        attr_classes = int(state["num_classes"]["attribute"])
        relation_classes = int(state["num_classes"]["relation"])
        feature_dim = int(state.get("feature_dim", _object_feature_dim(instances)))
    elif options.get("scallop_mlp_name_indices") is not None:
        name_to_index = dict(options.get("scallop_mlp_name_indices") or {})
        attr_to_index = dict(options.get("scallop_mlp_attr_indices") or {})
        relation_to_index = dict(options.get("scallop_mlp_relation_indices") or {})
        name_classes = int(options.get("scallop_mlp_name_classes", 0))
        attr_classes = int(options.get("scallop_mlp_attr_classes", 0))
        relation_classes = int(options.get("scallop_mlp_relation_classes", 0))
        feature_dim = int(options.get("scallop_mlp_feature_dim", _object_feature_dim(instances)))
    else:
        name_vocab, attr_vocab = _objects_for_vocab(instances)
        relation_vocab = sorted(context.relation_predicates)
        name_to_index = {value: index for index, value in enumerate(name_vocab)}
        attr_to_index = {value: index for index, value in enumerate(attr_vocab)}
        relation_to_index = {value: index for index, value in enumerate(relation_vocab)}
        name_classes = len(name_vocab)
        attr_classes = len(attr_vocab)
        relation_classes = len(relation_vocab)
        feature_dim = _object_feature_dim(instances)
    hidden = int((state or {}).get("hidden_dim", options.get("scallop_mlp_hidden_dim", 1024)))
    modules = {
        "feature_dim": feature_dim,
        "hidden_dim": hidden,
        "name_to_index": name_to_index,
        "attr_to_index": attr_to_index,
        "relation_to_index": relation_to_index,
        "num_classes": {
            "name": name_classes,
            "attribute": attr_classes,
            "relation": relation_classes,
        },
        "name_mlp": ScallopObjectMLP(feature_dim, name_classes, hidden, 0.3, hidden_layers=2).to(device),
        "attr_mlp": ScallopObjectMLP(feature_dim, attr_classes, hidden, 0.3, hidden_layers=1).to(device),
        "relation_mlp": ScallopRelationMLP(feature_dim, relation_classes + 1, hidden, 0.5).to(device),
    }
    if state:
        for key in ("name_mlp", "attr_mlp", "relation_mlp"):
            if key in state:
                modules[key].load_state_dict(state[key])
    return modules


@dataclass
class ObjectCenteredContext:
    graph: Graph
    image: Concept
    obj: Concept
    pair: Concept
    answer_object: Concept
    object_domain: Concept
    image_contains_obj: Relation
    pair_src: Relation
    pair_dst: Relation
    answer_slots: list
    object_predicates: dict
    relation_predicates: dict
    relation_object_predicates: dict
    knowledge_predicates: dict
    knowledge_sources: dict
    namespace: dict


def predicate_name(kind, value):
    prefix = {"Name": "name", "Attribute": "attr"}[kind]
    return f"{prefix}_{safe_name(value)}"


def knowledge_key(pred, right):
    pred = canonical_relation(pred)
    if pred == "SemanticClass":
        return ("SemanticClass", "TypeOf", str(right))
    if pred == "KG":
        relation, value = right
        return ("KG", canonical_relation(relation), str(value))
    raise ValueError(f"Not a knowledge condition: {pred!r}")


def knowledge_predicate_name(key):
    kind, relation, value = key
    prefix = "semantic_class" if kind == "SemanticClass" else f"kg_{safe_name(relation)}"
    return f"{prefix}_{safe_name(value)}"


def relation_object_predicate_name(direction, relation):
    direction_name = "from" if direction == "RelationFrom" else "to"
    return f"has_relation_{direction_name}_{safe_name(relation)}"


def _relation_reachability_logits(direction, pair_logits):
    pair_logits = pair_logits.view(-1, 2)
    pair_count = pair_logits.shape[0]
    margins = pair_logits[:, 1] - pair_logits[:, 0]

    # DomiKnowS ObjectPair datanodes are non-self pairs, ordered row-major with
    # src != dst.  VLM relation scorers receive object boxes and naturally score
    # the full square object grid, including self-pairs.  Support both shapes so
    # relation-derived object predicates can be used with either scorer family.
    square_count = int(pair_count ** 0.5)
    if square_count * square_count == pair_count:
        object_count = square_count
        yes_margin = margins.view(object_count, object_count).clone()
        diagonal = torch.arange(object_count, device=yes_margin.device)
        yes_margin[diagonal, diagonal] = -1e9
    else:
        object_count = int((1 + (1 + 4 * pair_count) ** 0.5) / 2)
        if object_count * (object_count - 1) != pair_count:
            raise ValueError(
                f"Expected square or non-self object-pair logits, got {pair_count} rows"
            )
        yes_margin = margins.new_full((object_count, object_count), -1e9)
        cursor = 0
        for src in range(object_count):
            for dst in range(object_count):
                if src == dst:
                    continue
                yes_margin[src, dst] = margins[cursor]
                cursor += 1
    if direction == "RelationFrom":
        # RelationFrom(o, R, all objects): exists c such that R(c, o).
        yes_margin = yes_margin.amax(dim=0)
    else:
        # RelationTo(o, R, all objects): exists c such that R(o, c).
        yes_margin = yes_margin.amax(dim=1)
    return torch.stack([-yes_margin, yes_margin], dim=-1)


def _knowledge_sources(instance, key, max_depth=2):
    kind, relation, value = key
    facts = [
        (canonical_relation(pred), str(left), str(right))
        for pred, left, right in instance.get("kb_facts", [])
    ]
    if kind == "KG":
        return sorted({left for pred, left, right in facts if pred == relation and right == value})

    # Broad semantic classes in VQAR, especially object, often require one
    # additional reverse TypeOf hop beyond the original KB depth used for KG
    # predicates. This stays bounded while matching the scene-graph gold labels.
    max_depth = max(max_depth, 3)
    targets = set(alias_values("SemanticClass", value))
    sources = set()
    frontier = set(targets)
    for _depth in range(max_depth):
        parents = {
            left for pred, left, right in facts
            if pred == "TypeOf" and right in frontier
        }
        sources.update(parents)
        frontier = parents
        if not frontier:
            break
    sources.update(targets)
    return sorted(sources)


def _register_visual_condition(instance, condition, object_predicates, relation_predicates, relation_object_predicates, knowledge_sources):
    pred, _left, right = condition
    pred = canonical_relation(pred)
    if pred in {"Name", "Attribute"}:
        object_predicates.add((pred, str(right)))
    elif pred in {"RelationFrom", "RelationTo"}:
        relation, candidate_objects = right
        relation = canonical_relation(relation)
        relation_predicates.add(relation)
        objects = {str(value) for value in instance.get("objects", [])}
        candidates = {str(value) for value in candidate_objects}
        if candidates == objects:
            relation_object_predicates.add((pred, relation))
    elif pred in {"KG", "SemanticClass"}:
        key = knowledge_key(pred, right)
        sources = _knowledge_sources(instance, key)
        allowed_names = instance.get("_allowed_visual_names")
        if allowed_names is not None:
            sources = [source for source in sources if source in allowed_names]
        if not sources:
            raise ValueError(f"No KB grounding found for {key!r}")
        knowledge_sources.setdefault(key, set()).update(sources)
        object_predicates.update(("Name", source) for source in sources)
    elif pred == "OneOf":
        return
    else:
        raise ValueError(
            f"Object-centered VLM smoke does not yet support predicate {pred!r}"
        )


def _register_visual_program(instance, program, object_predicates, relation_predicates, relation_object_predicates, knowledge_sources):
    if not program:
        return
    op = program.get("op")
    if op in {"all", None}:
        return
    if op == "filter":
        _register_visual_condition(
            instance, program["condition"], object_predicates, relation_predicates,
            relation_object_predicates, knowledge_sources,
        )
        _register_visual_program(
            instance, program.get("input"), object_predicates, relation_predicates,
            relation_object_predicates, knowledge_sources,
        )
        return
    if op in {"and", "or"}:
        for child in program.get("inputs", []) or []:
            _register_visual_program(
                instance, child, object_predicates, relation_predicates,
                relation_object_predicates, knowledge_sources,
            )
        return
    if op == "relate":
        relation = canonical_relation(program["relation"])
        relation_predicates.add(relation)
        objects = {str(value) for value in instance.get("objects", [])}
        candidates = {str(value) for value in program.get("candidates", [])}
        if candidates and candidates == objects:
            relation_object_predicates.add((program["direction"], relation))
        _register_visual_program(
            instance, program.get("input"), object_predicates, relation_predicates,
            relation_object_predicates, knowledge_sources,
        )
        return
    raise ValueError(f"Unsupported structured GraphQA op: {op!r}")


def required_visual_predicates(instances):
    object_predicates = set()
    relation_predicates = set()
    relation_object_predicates = set()
    knowledge_sources = {}
    for instance in instances:
        query = instance.get("query", {})
        if query.get("target_type") not in (None, "__any_object__"):
            raise ValueError("Object-centered VLM smoke does not yet translate semantic target_type")
        if query.get("program") is not None:
            _register_visual_program(
                instance, query["program"], object_predicates, relation_predicates,
                relation_object_predicates, knowledge_sources,
            )
        branches = query.get("alternatives") or [query.get("conditions", [])]
        for conditions in branches:
            for condition in conditions:
                _register_visual_condition(
                    instance, condition, object_predicates, relation_predicates,
                    relation_object_predicates, knowledge_sources,
                )
    return (
        sorted(object_predicates),
        sorted(relation_predicates),
        sorted(relation_object_predicates),
        {key: sorted(values) for key, values in sorted(knowledge_sources.items())},
    )


def create_object_centered_graph(instances, include_global_consistency=True):
    object_specs, relation_specs, relation_object_specs, knowledge_sources = required_visual_predicates(instances)
    if include_global_consistency:
        relation_specs = sorted(set(relation_specs) | {
            _OPPOSITE_RELATIONS[relation]
            for relation in relation_specs
            if relation in _OPPOSITE_RELATIONS
        })
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
        object_domain = obj(name="object_domain")
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
        if include_global_consistency:
            seen_opposites = set()
            for relation, opposite in _OPPOSITE_RELATIONS.items():
                if relation not in relation_predicates or opposite not in relation_predicates:
                    continue
                pair_key = frozenset((relation, opposite))
                if pair_key in seen_opposites:
                    continue
                seen_opposites.add(pair_key)
                nandL(
                    relation_predicates[relation]("p"),
                    relation_predicates[opposite]("p"),
                    name=f"opposite_{safe_name(relation)}_{safe_name(opposite)}",
                )
        relation_object_predicates = {
            spec: obj(name=relation_object_predicate_name(*spec))
            for spec in relation_object_specs
        }
        knowledge_predicates = {
            key: obj(name=knowledge_predicate_name(key)) for key in knowledge_sources
        }
        if include_global_consistency:
            for key, sources in knowledge_sources.items():
                derived = knowledge_predicates[key]
                for source in sources:
                    ifL(object_predicates[("Name", source)]("o"), derived("o"))

    namespace = {
        "answer_object": answer_object,
        "object_domain": object_domain,
        "object": obj,
        "miotaL": miotaL,
        "existsL": existsL,
        "pair_src": pair_src,
        "pair_dst": pair_dst,
        **{concept.name: concept for concept in answer_slots},
        **{concept.name: concept for concept in object_predicates.values()},
        **{concept.name: concept for concept in relation_predicates.values()},
        **{concept.name: concept for concept in relation_object_predicates.values()},
        **{concept.name: concept for concept in knowledge_predicates.values()},
    }
    return ObjectCenteredContext(
        graph=graph,
        image=image,
        obj=obj,
        pair=pair,
        answer_object=answer_object,
        object_domain=object_domain,
        image_contains_obj=image_contains_obj,
        pair_src=pair_src,
        pair_dst=pair_dst,
        answer_slots=answer_slots,
        object_predicates=object_predicates,
        relation_predicates=relation_predicates,
        relation_object_predicates=relation_object_predicates,
        knowledge_predicates=knowledge_predicates,
        knowledge_sources=knowledge_sources,
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

    if pred in {"KG", "SemanticClass"}:
        key = knowledge_key(pred, right)
        if key not in context.knowledge_predicates:
            raise ValueError(f"Knowledge condition has no derived concept: {key!r}")
        return [f'{context.knowledge_predicates[key].name}(path="o")']

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
        valid_candidates = [str(candidate) for candidate in candidate_objects if str(candidate) in object_index]
        if not valid_candidates:
            raise ValueError("Relation condition has no candidate anchor")

        # Keep relation selection object-aligned, following
        # test_regr.tiny_multi_answer.example_relationAnswers:
        #   miotaL(andL(object("x"), rel("r", path=("x", src.reversed)), ...))
        # The first bound variable remains answer object ``o``; pair rows are
        # traversed only to test whether a qualifying endpoint exists.
        #
        # The reference example's andL is always FLAT (red("x"), left("r", ...),
        # ball("y", ...)) — every atom is a direct argument of the same andL, so
        # DomiKnowS can align the "o"-shaped and pair-shaped rows itself. Nesting
        # a second andL(relation_atom, endpoint_atom) as one argument of the
        # outer andL (as this branch used to) hides that alignment from the
        # outer call: the outer andL sees an N-shaped atom next to an opaque
        # sub-expression it can't broadcast, and later fails inside
        # lcLossBooleanMethods.andVar with an N vs N*N tensor-size mismatch once
        # DomiKnowS expands the nested pair variable. Returning the flat atom
        # list (instead of pre-wrapping it in its own andL) lets the caller's
        # single top-level andL(...) see every atom directly, matching the
        # working reference pattern. Only the OR case genuinely needs its own
        # nested andL per branch, since andL can't distribute over orL.
        pair_var = f"r{index}"
        if pred == "RelationFrom":
            relation_atom = f'{relation_concept.name}("{pair_var}", path=("o", pair_dst.reversed))'
            endpoint_role = "pair_src"
        else:
            relation_atom = f'{relation_concept.name}("{pair_var}", path=("o", pair_src.reversed))'
            endpoint_role = "pair_dst"

        if set(valid_candidates) == set(object_index):
            endpoint_atom = (
                f'{context.obj.name}("y{index}", '
                f'path=("{pair_var}", {endpoint_role}))'
            )
            return [relation_atom, endpoint_atom]

        if len(valid_candidates) == 1:
            slot = context.answer_slots[object_index[str(valid_candidates[0])]]
            endpoint_atom = f'{slot.name}("y{index}", path=("{pair_var}", {endpoint_role}))'
            return [relation_atom, endpoint_atom]

        branches = []
        for candidate in valid_candidates:
            slot = context.answer_slots[object_index[str(candidate)]]
            endpoint_atom = f'{slot.name}("y{index}", path=("{pair_var}", {endpoint_role}))'
            branches.append("andL(" + ", ".join([relation_atom, endpoint_atom]) + ")")
        return ["orL(" + ", ".join(branches) + ")"]

    raise ValueError(f"Unsupported object-centered condition: {condition!r}")


def _create_branch_body(instance, context, conditions):
    if not conditions:
        return f'{context.object_domain.name}("o")'
    first_pred, first_left, first_right = conditions[0]
    first_pred = canonical_relation(first_pred)
    if first_left != "o":
        raise ValueError("The first object-centered atom must select object variable 'o'")
    if first_pred in {"Name", "Attribute"}:
        first_concept = context.object_predicates[(first_pred, str(first_right))]
        atoms = [f'{first_concept.name}("o")']
        remaining = conditions[1:]
    elif first_pred in {"KG", "SemanticClass"}:
        key = knowledge_key(first_pred, first_right)
        if key not in context.knowledge_predicates:
            raise ValueError(f"Knowledge condition has no derived concept: {key!r}")
        atoms = [f'{context.knowledge_predicates[key].name}("o")']
        remaining = conditions[1:]
    else:
        atoms = [f'{context.object_domain.name}("o")']
        remaining = conditions
    for index, condition in enumerate(remaining, start=1):
        atoms.extend(_condition_atoms(instance, context, condition, index))
    return atoms[0] if len(atoms) == 1 else "andL(" + ", ".join(atoms) + ")"


def active_concepts_for_instances(context, instances):
    """Return the minimal active concepts needed for the given examples.

    The graph may contain the union vocabulary for a whole shard.  This helper
    selects only the concepts used by the current example/chunk before DomiKnowS
    populate/inference, following ``tiny_dynamic_graph/example_dynamic_graph.py``.
    """

    requested = {
        context.image,
        context.obj,
        context.pair,
        context.answer_object,
        context.object_domain,
    }
    max_objects = max(len(instance.get("objects", [])) for instance in instances)
    requested.update(context.answer_slots[:max_objects])

    object_specs, relation_specs, relation_object_specs, knowledge_sources = required_visual_predicates(instances)
    for spec in object_specs:
        requested.add(context.object_predicates[spec])
    for relation in relation_specs:
        requested.add(context.relation_predicates[relation])
    for spec in relation_object_specs:
        if spec in context.relation_object_predicates:
            requested.add(context.relation_object_predicates[spec])
    for key, sources in knowledge_sources.items():
        if key in context.knowledge_predicates:
            requested.add(context.knowledge_predicates[key])
        for source in sources:
            source_spec = ("Name", source)
            if source_spec in context.object_predicates:
                requested.add(context.object_predicates[source_spec])

    # Defensive: ``context.object_predicates``/etc. can hold a Concept object
    # that is no longer (or never was) the exact object registered on
    # ``context.graph``'s own tree -- e.g. a stale reference surviving past a
    # ``Graph.clear()`` elsewhere in the same process. ``set_active_concepts``
    # resolves Concept instances by identity (``is``), not by name, so a
    # mismatched-but-same-named entry raises "Concept ... does not belong to
    # graph ..." and previously crashed the whole instance. Drop those here
    # instead of letting every caller of this helper hit that crash; this
    # only narrows what gets activated for one instance, it does not change
    # graph construction.
    # Concept.__eq__/__hash__ are overridden (NamedTreeNode-based) to compare
    # by name/structure, not identity, so a plain ``in``/set check here would
    # silently accept a stale-but-same-named concept -- exactly what needs
    # rejecting. Match ``_resolve_activation_concept``'s own check: identity
    # (``is``) against the graph's live concept objects.
    live_concepts = list(context.graph._activation_concepts().values())
    live, dropped = [], []
    for concept in requested:
        (live if any(concept is candidate for candidate in live_concepts) else dropped).append(concept)
    if dropped:
        context.stale_concepts_dropped = getattr(context, "stale_concepts_dropped", 0) + len(dropped)
    return tuple(live)


_OBJECT_VAR_NAMES = tuple("xyzabcdefghijklmnopqrstuvw")


def _object_var_for_depth(depth):
    if depth < len(_OBJECT_VAR_NAMES):
        return _OBJECT_VAR_NAMES[depth]
    return f"x{depth}"


def _concept_atom_for_var(context, concept, var, intro_path=None):
    if intro_path is None:
        return f'{concept.name}(path="{var}")'
    return f'{concept.name}("{var}", path={intro_path})'


def _domain_atom_for_var(context, var, intro_path=None):
    if intro_path is None:
        return f'{context.obj.name}("{var}")'
    return f'{context.obj.name}("{var}", path={intro_path})'


def _condition_atom_for_var(instance, context, condition, var, intro_path=None):
    pred, _left, right = condition
    pred = canonical_relation(pred)
    if pred in {"Name", "Attribute"}:
        concept = context.object_predicates[(pred, str(right))]
        return _concept_atom_for_var(context, concept, var, intro_path=intro_path)
    if pred in {"KG", "SemanticClass"}:
        key = knowledge_key(pred, right)
        if key not in context.knowledge_predicates:
            raise ValueError(f"Knowledge condition has no derived concept: {key!r}")
        return _concept_atom_for_var(context, context.knowledge_predicates[key], var, intro_path=intro_path)
    if pred == "OneOf":
        object_index = {str(value): i for i, value in enumerate(instance["objects"])}
        slots = []
        for value in right:
            if str(value) in object_index:
                slot = context.answer_slots[object_index[str(value)]]
                slots.append(_concept_atom_for_var(context, slot, var, intro_path=intro_path))
        if not slots:
            raise ValueError("OneOf has no candidate object in this instance")
        return slots[0] if len(slots) == 1 else "orL(" + ", ".join(slots) + ")"
    raise ValueError(f"Expected unary object condition, got {condition!r}")


def _relation_qualifying_objects(instance, relation, direction, candidates):
    """Objects satisfying a RelationFrom/RelationTo condition, read directly
    off ``instance["visual_facts"]`` instead of a symbolic pair traversal.

    Facts are ``(relation, subject, obj)`` with subject as the pair's src and
    obj as its dst (``dataset._scene_graph_to_facts``). This mirrors exactly
    what the symbolic construction in the "relate" branch below computes:
    RelationFrom binds ``var`` to the pair's dst (endpoint = src);
    RelationTo binds ``var`` to the pair's src (endpoint = dst).
    """
    candidate_set = {str(value) for value in candidates}
    qualifying = set()
    for rel, subject, obj in instance.get("visual_facts", []):
        if canonical_relation(rel) != relation:
            continue
        if direction == "RelationFrom":
            if str(subject) in candidate_set:
                qualifying.add(str(obj))
        else:
            if str(obj) in candidate_set:
                qualifying.add(str(subject))
    return qualifying


def _relation_oneof_atom_for_var(instance, context, program, var, intro_path=None):
    """OneOf-style atom for a leaf "relate" op, or ``None`` if unresolvable."""
    relation = canonical_relation(program["relation"])
    if relation not in context.relation_predicates:
        return None
    direction = program["direction"]
    object_index = {str(value): i for i, value in enumerate(instance["objects"])}
    candidates = [str(value) for value in program.get("candidates", []) if str(value) in object_index]
    if not candidates:
        candidates = list(object_index)
    qualifying = _relation_qualifying_objects(instance, relation, direction, candidates)
    slots = [
        _concept_atom_for_var(context, context.answer_slots[object_index[obj_id]], var, intro_path=intro_path)
        for obj_id in sorted(qualifying, key=lambda value: object_index[value])
        if obj_id in object_index
    ]
    if not slots:
        return None
    return slots[0] if len(slots) == 1 else "orL(" + ", ".join(slots) + ")"


def _and_body(atoms):
    atoms = [atom for atom in atoms if atom]
    if not atoms:
        raise ValueError("Cannot build empty andL body")
    return atoms[0] if len(atoms) == 1 else "andL(" + ", ".join(atoms) + ")"


def _structured_atoms(instance, context, program, var="x", intro_path=None, counter=None, depth=0, bound=False):
    """Create flat CLEVR-style atoms for structured GraphQA execution.

    Relation chains must look like the tiny_multi_answer/CLEVR pattern:
    ``obj(a), rel0(path=(a,...)), concept(b, path=(rel0,...)), rel1(path=(b,...))``.
    We therefore introduce endpoint variables with a real concept whenever the
    input program provides one, and only fall back to ``object`` for unconstrained
    intermediate endpoints.
    """

    if counter is None:
        counter = {"value": 0}
    if not program:
        program = {"op": "all"}
    op = program.get("op")

    if op == "all":
        if bound and intro_path is None:
            return []
        return [_domain_atom_for_var(context, var, intro_path=intro_path)]

    if op == "filter":
        input_program = program.get("input") or {"op": "all"}
        if intro_path is not None:
            atoms = [_condition_atom_for_var(instance, context, program["condition"], var, intro_path=intro_path)]
            if input_program.get("op") != "all":
                atoms.extend(_structured_atoms(
                    instance, context, input_program, var=var,
                    intro_path=intro_path, counter=counter, depth=depth, bound=True,
                ))
            return atoms
        atoms = _structured_atoms(
            instance, context, input_program, var=var,
            intro_path=intro_path, counter=counter, depth=depth, bound=bound,
        )
        atoms.append(_condition_atom_for_var(instance, context, program["condition"], var))
        return atoms

    if op == "and":
        atoms = []
        children = program.get("inputs", []) or [{"op": "all"}]
        for child in children:
            atoms.extend(_structured_atoms(
                instance, context, child, var=var,
                intro_path=intro_path,
                counter=counter, depth=depth, bound=bound,
            ))
        return atoms

    if op == "or":
        # Each branch is built independently and may (or may not) introduce its
        # own pair/endpoint sub-variables via a "relate" op, so sibling
        # branches can end up at different shapes: a plain per-``var`` atom vs.
        # one carrying a relation-pair traversal. Neither a flat andL nor an
        # existsL(...) wrapper gets DomiKnowS's andVar/orVar to reconcile that
        # (both still crash with an N vs N*pairs tensor-size mismatch; verified
        # empirically, not just in theory). A direct "relate" branch with no
        # further nested condition is resolved from the instance's own known
        # scene-graph facts instead: which objects actually qualify is fully
        # determined by ``visual_facts`` already, so there's no need to make
        # DomiKnowS reduce a symbolic pair traversal at all. This turns the
        # branch into the same OneOf-style per-``var`` atom the other branches
        # use (see _condition_atom_for_var's "OneOf" case), matching shape with
        # any sibling. Anything else (nested/chained relate) falls back to the
        # existing symbolic construction.
        bodies = []
        children = program.get("inputs", []) or [{"op": "all"}]
        for child in children:
            if child.get("op") == "relate" and (child.get("input") or {"op": "all"}).get("op") == "all":
                resolved = _relation_oneof_atom_for_var(instance, context, child, var, intro_path=intro_path)
                if resolved is not None:
                    bodies.append(resolved)
                    continue
            child_atoms = _structured_atoms(
                instance, context, child, var=var,
                intro_path=intro_path,
                counter=counter, depth=depth, bound=bound,
            )
            bodies.append(_and_body(child_atoms))
        return [bodies[0] if len(bodies) == 1 else "orL(" + ", ".join(bodies) + ")"]

    if op == "relate":
        relation = canonical_relation(program["relation"])
        relation_concept = context.relation_predicates[relation]
        counter["value"] += 1
        pair_var = f"r{counter['value']}"
        # ``depth`` alone doesn't distinguish sibling "relate" conditions under
        # a plain "and" (they're all called with the same depth, since depth
        # only increases along a chained/nested path, not across siblings), so
        # two unrelated relations both produced an endpoint variable named "y"
        # -- DomiKnowS then conflated them, grounding the cross-product of both
        # relations' candidates instead of each independently (miotaL label
        # count mismatched the grounded candidate count). ``counter`` already
        # guarantees a fresh value per relate hop regardless of AND/OR
        # structure (it drives ``pair_var`` above), so reuse it here too.
        endpoint_var = _object_var_for_depth(counter["value"])
        direction = program["direction"]
        if direction == "RelationFrom":
            relation_atom = f'{relation_concept.name}("{pair_var}", path=("{var}", pair_dst.reversed))'
            endpoint_role = "pair_src"
        elif direction == "RelationTo":
            relation_atom = f'{relation_concept.name}("{pair_var}", path=("{var}", pair_src.reversed))'
            endpoint_role = "pair_dst"
        else:
            raise ValueError(f"Unknown relation direction: {direction!r}")

        endpoint_intro = f'("{pair_var}", {endpoint_role})'
        atoms = ([] if bound and intro_path is None else [_domain_atom_for_var(context, var, intro_path=intro_path)])
        atoms.append(relation_atom)

        object_index = {str(value): i for i, value in enumerate(instance["objects"])}
        candidates = [str(value) for value in program.get("candidates", []) if str(value) in object_index]
        input_program = program.get("input") or {"op": "all"}
        if input_program.get("op") != "all":
            atoms.extend(_structured_atoms(
                instance,
                context,
                input_program,
                var=endpoint_var,
                intro_path=endpoint_intro,
                counter=counter,
                depth=depth + 1,
                bound=False,
            ))
        elif not candidates or set(candidates) == set(object_index):
            atoms.append(_domain_atom_for_var(
                context, endpoint_var, intro_path=endpoint_intro
            ))
        if candidates and set(candidates) != set(object_index):
            atoms.append(_condition_atom_for_var(
                instance, context, ("OneOf", endpoint_var, candidates), endpoint_var,
                intro_path=endpoint_intro,
            ))
        return atoms

    raise ValueError(f"Unsupported structured GraphQA op: {op!r}")


def _structured_body(instance, context, program, var="x", intro_path=None, counter=None):
    return _and_body(_structured_atoms(
        instance, context, program, var=var, intro_path=intro_path, counter=counter,
    ))


def create_query_body(instance, context):
    query = instance["query"]
    if query.get("program") is not None:
        return _structured_body(instance, context, query["program"])
    branches = query.get("alternatives") or [query.get("conditions", [])]
    bodies = [
        _create_branch_body(instance, context, conditions)
        for conditions in branches
    ]
    return bodies[0] if len(bodies) == 1 else "orL(" + ", ".join(bodies) + ")"


def create_logic(instance, context, selector="iotaL"):
    body = create_query_body(instance, context)
    if selector == "miotaL":
        return (
            "miotaL(\n"
            f"    {body}\n"
            ")"
        )
    if selector != "iotaL":
        raise ValueError(f"Unknown object selector: {selector!r}")
    return (
        "queryL(\n"
        "    answer_object,\n"
        "    iotaL(\n"
        f"        {body}\n"
        "    )\n"
        ")"
    )


def create_membership_logic(instance, context, candidate_index):
    """Ask whether one candidate belongs to the executable answer set."""

    body = create_query_body(instance, context)
    candidate = context.answer_slots[candidate_index]
    return (
        "existsL(\n"
        "    andL(\n"
        f"        {body},\n"
        f'        {candidate.name}(path="o")\n'
        "    )\n"
        ")"
    )


def _same_image(_property=None, src_arg=None, dst_arg=None, **kwargs):
    if src_arg is not None and dst_arg is not None:
        return src_arg is not dst_arg and src_arg.getAttribute("image_id") == dst_arg.getAttribute("image_id")
    nodes = [value for value in kwargs.values() if hasattr(value, "getAttribute")]
    return len(nodes) < 2 or nodes[0].getAttribute("image_id") == nodes[1].getAttribute("image_id")


def _qwen_module(model_path, device, relation, attr, **options):
    clever_dir = Path(__file__).resolve().parents[1] / "Clever"
    if str(clever_dir) not in sys.path:
        sys.path.insert(0, str(clever_dir))
    from qwen_vl_hf import QwenVLSharedHF

    return QwenVLSharedHF(
        model_path=str(model_path),
        device=device,
        relation=relation,
        attr=attr,
        **options,
    )


def _internvl_module(model_path, device, relation, attr, **options):
    clever_dir = Path(__file__).resolve().parents[1] / "Clever"
    if str(clever_dir) not in sys.path:
        sys.path.insert(0, str(clever_dir))
    from peftvllm import InternVLSharedHF

    # These options belong to other predicate backends (Qwen's trained text
    # scorer, scallop-trained, scallop-mlp) that share this call site's options
    # dict in evaluate_object_centered_c2.py. InternVL manages its own token
    # and image batching and forwards unknown keywords to nn.Module, where
    # they are rejected.
    options.pop("max_length", None)
    options.pop("encode_batch_size", None)
    options.pop("grouped_unary", None)
    options.pop("choice_max_options", None)
    options.pop("scallop_confidence", None)
    options.pop("scallop_learnable_scale", None)
    options.pop("scallop_checkpoint", None)
    options.pop("lora_target_modules", None)
    options.pop("scallop_mlp_hidden_dim", None)
    options.pop("scallop_mlp_dropout", None)
    options.pop("scallop_mlp_checkpoint", None)

    return InternVLSharedHF(
        model_path=str(model_path),
        device=device,
        relation=relation,
        attr=attr,
        **options,
    )


def _learned_module(mode, model_path, device, relation, kind, value, factory, options):
    if factory is not None:
        module = factory(kind, value, relation)
        return module.to(device)
    if mode == "qwen-vl":
        return _qwen_module(
            model_path, device, relation=relation, attr=value, **options
        )
    if mode == "internvl":
        return _internvl_module(
            model_path, device, relation=relation, attr=value, **options
        )
    raise ValueError(f"Unknown learned predicate mode: {mode!r}")


_ATTRIBUTE_CHOICE_GROUPS = (
    ("white", "black", "green", "blue", "brown", "red", "gray", "yellow", "orange", "silver", "pink", "tan", "purple", "gold"),
    ("large", "small", "tall", "long", "short", "little", "thin", "thick"),
    ("wooden", "metal", "plastic", "brick", "concrete", "stone", "steel", "glass", "leather", "ceramic"),
    ("round", "square", "rectangular", "pointy"),
    ("open", "closed", "empty", "full"),
    ("dark", "clear", "bright", "light", "shiny", "colorful"),
    ("wet", "dry", "dirty", "clean"),
    ("standing", "sitting", "walking", "hanging", "parked", "flying", "playing", "eating", "lying", "surfing"),
)


def _attribute_choice_group(value):
    value = str(value)
    for group in _ATTRIBUTE_CHOICE_GROUPS:
        if value in group:
            # Treat grouped attributes as closed-choice classification.  Adding
            # a generic none option made small VLMs over-select rejection and
            # destroyed recall for executable predicates.
            return list(group)
    return None


def _choice_groups_for_object_predicates(context, max_options=None):
    """Build per-kind multiple-choice groups for unary visual predicates.

    Name and Attribute are mutually exclusive enough to be scored as closed
    choices within the active graph.  KG-derived predicates are intentionally
    excluded: KB facts remain symbolic and are derived from visual Name concepts.
    """

    if max_options is None:
        max_options = int(os.environ.get("DOMIKNOWS_GRAPHQA_CHOICE_MAX_OPTIONS", "25"))
    max_options = max(2, min(int(max_options), 26))
    values_by_kind = {}
    for kind, value in context.object_predicates:
        values_by_kind.setdefault(kind, set()).add(str(value))

    groups = {}
    for kind, values in values_by_kind.items():
        if kind not in {"Name", "Attribute"}:
            continue
        sorted_values = sorted(values)
        for value in sorted_values:
            if kind == "Attribute":
                choices = _attribute_choice_group(value)
                if choices is not None and len(choices) <= max_options:
                    groups[(kind, value)] = choices
                    continue
        if len(sorted_values) < 2 or len(sorted_values) > max_options:
            # Do not invent a target-specific alphabetic menu for huge KG-driven
            # vocabularies.  Those fall back to the pairwise yes/no scorer until
            # we add taxonomy-aware batching.  Single-option menus are also
            # avoided because softmax would force probability 1.0.
            continue
        for value in sorted_values:
            if (kind, value) in groups:
                continue
            groups[(kind, value)] = list(sorted_values)
    return groups


def _vlm_module_options(qwen_options):
    options = dict(qwen_options)
    options.pop("grouped_unary", None)
    options.pop("choice_max_options", None)
    return options


def _object_choice_options(qwen_options, kind, value, choice_groups):
    options = _vlm_module_options(qwen_options)
    if not qwen_options.get("grouped_unary", True):
        return options
    choices = choice_groups.get((kind, str(value)))
    if not choices:
        return options
    options["choice_group"] = choices
    options["choice_prompt_kind"] = "object name" if kind == "Name" else "object attribute"
    return options


def attach_sensors(
    context,
    mode="oracle",
    model_path=None,
    device="cpu",
    module_factory=None,
    qwen_options=None,
    instances=None,
):
    qwen_options = dict(qwen_options or {})
    scallop_confidence = float(qwen_options.get(
        "scallop_confidence",
        os.environ.get("DOMIKNOWS_GRAPHQA_SCALLOP_CONFIDENCE", "8.0"),
    ))
    scallop_learnable_scale = bool(qwen_options.get("scallop_learnable_scale", False))
    scallop_trained = None
    if mode == "scallop-trained":
        checkpoint = qwen_options.get("scallop_checkpoint")
        if checkpoint is None:
            raise ValueError("--scallop-checkpoint is required for mode=scallop-trained")
        scallop_trained = load_scallop_trained_classifier(
            model_path, checkpoint, device, qwen_options
        )
    scallop_mlp = None
    if mode == "scallop-mlp":
        scallop_mlp = _build_scallop_mlp_modules(context, instances or [], device, qwen_options)
        context.scallop_mlp_modules = scallop_mlp
    choice_groups = _choice_groups_for_object_predicates(
        context, max_options=qwen_options.get("choice_max_options")
    )
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
    context.obj["scallop_mlp_feature"] = FunctionalReaderSensor(
        keyword="object_scallop_features",
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
    if mode == "scallop-mlp":
        context.obj["scallop_name_logits"] = ModuleLearner(
            context.obj["scallop_mlp_feature"],
            module=scallop_mlp["name_mlp"], device=device,
        )
        context.obj["scallop_attr_logits"] = ModuleLearner(
            context.obj["scallop_mlp_feature"],
            module=scallop_mlp["attr_mlp"], device=device,
        )

    context.obj[context.object_domain] = FunctionalSensor(
        context.obj["index"],
        forward=lambda indices: torch.stack(
            (-torch.ones_like(indices, dtype=torch.float32) * 8.0,
             torch.ones_like(indices, dtype=torch.float32) * 8.0),
            dim=-1,
        ),
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
        elif mode == "scallop-local":
            key = f"{concept.name}_scallop"
            context.obj[key] = FunctionalReaderSensor(
                keyword=key,
                forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device).view(-1, 1),
            )
            module = ScallopLocalFactLearner(
                confidence=scallop_confidence,
                learnable_scale=scallop_learnable_scale,
            )
            context.obj[concept] = ModuleLearner(key, module=module, device=device)
        elif mode == "scallop-trained":
            key = f"{concept.name}_scallop_trained_prompt"
            context.obj[key] = FunctionalReaderSensor(keyword=key, forward=lambda data: list(data))
            label = "Name" if kind == "Name" else "Attribute"
            module = ScallopTrainedPredicateView(
                scallop_trained, "object_symbol", label, None, device=device
            )
            context.obj[concept] = ModuleLearner(key, module=module, device=device)
        elif mode == "scallop-mlp":
            if kind == "Name":
                yes_index = scallop_mlp["name_to_index"].get(str(value))
                source = context.obj["scallop_name_logits"]
                projection = _multiclass_binary_logits
            else:
                yes_index = scallop_mlp["attr_to_index"].get(str(value))
                source = context.obj["scallop_attr_logits"]
                projection = _sigmoid_binary_logits
            if yes_index is None:
                raise ValueError(f"Scallop MLP vocabulary has no {kind} value {value!r}")
            context.obj[concept] = FunctionalSensor(
                source,
                forward=functools.partial(projection, yes_index=yes_index),
            )
        else:
            module_options = _object_choice_options(qwen_options, kind, value, choice_groups)
            module = _learned_module(
                mode, model_path, device, 1, kind, value, module_factory, module_options
            )
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
    context.pair["scallop_mlp_feature"] = FunctionalSensor(
        context.obj["scallop_mlp_feature"],
        context.obj["boxes"],
        forward=lambda features, boxes: _pair_scallop_features(features, boxes),
    )
    if mode == "scallop-mlp":
        context.pair["scallop_relation_logits"] = ModuleLearner(
            context.pair["scallop_mlp_feature"],
            module=scallop_mlp["relation_mlp"], device=device,
        )

    for key, derived in context.knowledge_predicates.items():
        source_sensors = [
            context.obj[context.object_predicates[("Name", source)]]
            for source in context.knowledge_sources[key]
        ]
        context.obj[derived] = FunctionalSensor(
            *source_sensors,
            forward=lambda *logits: _derived_or_logits(*logits),
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
        elif mode == "scallop-local":
            key = f"{concept.name}_scallop"
            context.pair[key] = FunctionalReaderSensor(
                keyword=key,
                forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device).view(-1, 1),
            )
            context.pair[concept] = ModuleLearner(
                key,
                module=ScallopLocalFactLearner(
                    confidence=scallop_confidence,
                    learnable_scale=scallop_learnable_scale,
                ),
                device=device,
            )
        elif mode == "scallop-trained":
            key = f"{concept.name}_scallop_trained_prompt"
            context.pair[key] = FunctionalReaderSensor(keyword=key, forward=lambda data: list(data))
            context.pair[concept] = ModuleLearner(
                key,
                module=ScallopTrainedPredicateView(
                    scallop_trained, "object_pair", relation, None, device=device
                ),
                device=device,
            )
        elif mode == "scallop-mlp":
            yes_index = scallop_mlp["relation_to_index"].get(str(relation))
            if yes_index is None:
                raise ValueError(f"Scallop MLP relation vocabulary has no {relation!r}")
            context.pair[concept] = FunctionalSensor(
                context.pair["scallop_relation_logits"],
                forward=functools.partial(_multiclass_binary_logits, yes_index=yes_index),
            )
        else:
            module = _learned_module(
                mode, model_path, device, 2, "Relation", relation, module_factory, _vlm_module_options(qwen_options)
            )
            if mode in {"qwen-vl", "internvl"}:
                module = NonSelfPairLogitWrapper(module)
            context.pair[concept] = ModuleLearner(
                context.image["pil_image"],
                context.image["image_filename"],
                context.obj["boxes"],
                module=module,
                device=device,
            )

    for (direction, relation), derived in context.relation_object_predicates.items():
        relation_concept = context.relation_predicates[relation]
        context.obj[derived] = FunctionalSensor(
            context.pair[relation_concept],
            forward=lambda logits, d=direction: _relation_reachability_logits(d, logits),
        )


def _normalize_box(box):
    if box is None:
        return [0.0, 0.0, 0.0, 0.0]
    values = [float(value) for value in box]
    if len(values) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    return values


def _object_scallop_feature_vector(instance, object_id, box):
    metadata = (instance.get("object_metadata") or {}).get(str(object_id), {})
    vector = metadata.get("feature_vector") or []
    if not vector:
        summary = metadata.get("feature") or {}
        vector = list(summary.get("head") or [])
    return [float(value) for value in vector]


def _pad_feature_rows(rows):
    width = max((len(row) for row in rows), default=4)
    return [row + [0.0] * (width - len(row)) for row in rows]


def _pair_scallop_features(features, boxes):
    object_count = int(features.shape[0])
    rows = []
    for src in range(object_count):
        for dst in range(object_count):
            if src == dst:
                continue
            rows.append(torch.cat([features[src], features[dst], boxes[src], boxes[dst]], dim=0))
    if not rows:
        width = int(features.shape[-1]) * 2 + 8
        return torch.empty((0, width), dtype=features.dtype, device=features.device)
    return torch.stack(rows, dim=0)


def _ensure_image_cached(instance, image_cache):
    """Return the cached image path, downloading it first if missing.

    Mirrors ``scallop_style_qwen_executor._load_image_for_instance``: VQAR
    tasks carry the source ``image_url`` on their object metadata, so a
    cache miss is usually recoverable rather than a hard failure. Raises
    ``FileNotFoundError`` only when the file is absent AND no instance
    object metadata has a usable URL to fetch it from.
    """
    image_id = instance.get("source_image_id")
    image_path = Path(image_cache) / f"{image_id}.jpg"
    if image_path.is_file():
        return image_path
    url = None
    for item in (instance.get("object_metadata") or {}).values():
        if item.get("image_url"):
            url = item["image_url"]
            break
    if not url:
        raise FileNotFoundError(image_path)
    import requests
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(response.content)
    return image_path


def _image_and_boxes(instance, image_cache):
    image_path = _ensure_image_cached(instance, image_cache)
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
    existing_metadata = instance.get("object_metadata", {}) or {}
    metadata = {}
    for object_id in instance["objects"]:
        key = str(object_id)
        item = dict(existing_metadata.get(key, {}))
        box = bboxes.get(object_id, bboxes.get(key))
        if box is None:
            try:
                box = bboxes.get(int(key))
            except (TypeError, ValueError):
                pass
        if box is not None:
            item["bbox"] = [float(value) for value in box]
        elif "bbox" not in item:
            item["bbox"] = None
        metadata[key] = item
    instance["object_metadata"] = metadata
    return instance



def _oracle_object_predicate_label(facts, kind, object_id, value):
    """Return oracle evidence for an object concept with VQAR aliases.

    Scene graph facts sometimes use an alias of the queried predicate, e.g.
    ``large`` is stored as ``giant``. The symbolic oracle already expands these
    aliases, so the DomiKnowS oracle ModuleLearner should receive the same
    labels.
    """

    object_id = str(object_id)
    value = str(value)
    if kind == "Attribute":
        return any(("Attribute", object_id, alias) in facts for alias in alias_values("Attribute", value))
    if kind == "Name":
        aliases = alias_values("SemanticClass", value)
        return any(
            (predicate, object_id, alias) in facts
            for predicate in ("Name", "ObjectType", "ObjectCategory")
            for alias in aliases
        )
    return (kind, object_id, value) in facts

def populate_example(instance, context, image_cache=DEFAULT_IMAGE_CACHE, device="cpu"):
    objects = [str(value) for value in instance["objects"]]
    object_index = {value: index for index, value in enumerate(objects)}
    facts = {
        (canonical_relation(pred), str(left), str(right))
        for pred, left, right in materialize_bounded_facts(instance)
    }
    image, image_path, boxes = _image_and_boxes(instance, image_cache)
    row = {
        "image_index": torch.tensor([0], dtype=torch.long, device=device),
        "pil_image": image,
        "image_filename": str(image_path),
        "object_indices": torch.arange(len(objects), dtype=torch.long, device=device),
        "object_ids": objects,
        "object_boxes": boxes,
        "object_scallop_features": _pad_feature_rows([
            _object_scallop_feature_vector(instance, object_id, box)
            for object_id, box in zip(objects, boxes)
        ]),
        "logic_str": create_logic(instance, context),
        "logic_label": torch.tensor(
            [object_index[str(instance["expected_answer"])]],
            dtype=torch.long,
            device=device,
        ),
    }
    for (kind, value), concept in context.object_predicates.items():
        labels = [
            int(_oracle_object_predicate_label(facts, kind, object_id, value))
            for object_id in objects
        ]
        row[f"{concept.name}_oracle"] = labels
        row[f"{concept.name}_scallop"] = labels
        row[f"{concept.name}_scallop_trained_prompt"] = [
            _object_symbol_feature_prompt(
                instance, object_id, value, instance.get("query", {}),
                labels=["Attribute", "Name", NO_RELATION_LABEL],
            )
            for object_id in objects
        ]
    for relation, concept in context.relation_predicates.items():
        labels = [
            int((relation, src, dst) in facts)
            for src in objects
            for dst in objects
            if src != dst
        ]
        row[f"{concept.name}_oracle"] = labels
        row[f"{concept.name}_scallop"] = labels
        row[f"{concept.name}_scallop_trained_prompt"] = [
            _object_pair_feature_prompt(
                instance, src, dst, instance.get("query", {}),
                labels=sorted(context.relation_predicates) + [NO_RELATION_LABEL],
            )
            for src in objects
            for dst in objects
            if src != dst
        ]
    return row


def populate_miota_example(instance, context, image_cache=DEFAULT_IMAGE_CACHE, device="cpu"):
    """Create one executable multi-hot label for the full answer set."""

    objects = [str(value) for value in instance["objects"]]
    answers = instance.get("expected_answers")
    if not answers and instance.get("expected_answer") is not None:
        answers = [instance["expected_answer"]]
    expected = {str(value) for value in (answers or [])}
    seed = dict(instance)
    seed["expected_answer"] = objects[0]
    row = populate_example(seed, context, image_cache=image_cache, device=device)
    row["logic_str"] = create_logic(instance, context, selector="miotaL")
    row["logic_label"] = torch.tensor(
        [int(object_id in expected) for object_id in objects],
        dtype=torch.float32,
        device=device,
    )
    return row


def populate_membership_examples(
    instance, context, image_cache=DEFAULT_IMAGE_CACHE, device="cpu"
):
    """Create one executable Boolean label for every candidate object."""

    objects = [str(value) for value in instance["objects"]]
    answers = instance.get("expected_answers")
    if not answers and instance.get("expected_answer") is not None:
        answers = [instance["expected_answer"]]
    expected = {str(value) for value in (answers or [])}
    seed = dict(instance)
    seed["expected_answer"] = objects[0]
    base = populate_example(seed, context, image_cache=image_cache, device=device)
    rows = []
    for candidate_index, object_id in enumerate(objects):
        row = dict(base)
        row["logic_str"] = create_membership_logic(instance, context, candidate_index)
        row["logic_label"] = torch.tensor(
            [int(object_id in expected)], dtype=torch.long, device=device
        )
        row["candidate_object_id"] = object_id
        rows.append(row)
    return rows


def build_program(
    instances,
    mode="oracle",
    model_path=None,
    image_cache=DEFAULT_IMAGE_CACHE,
    device="cpu",
    answer_mode="iota",
    module_factory=None,
    qwen_options=None,
    include_global_consistency=True,
    beta=1.0,
    executable_constraint_loss_weight=1.0,
    global_constraint_loss_weight=1.0,
    compile_lc=False,
):
    instances = list(instances)
    qwen_options = dict(qwen_options or {})
    if mode == "scallop-mlp":
        name_indices = qwen_options.get("scallop_mlp_name_indices")
        checkpoint = qwen_options.get("scallop_mlp_checkpoint")
        if not name_indices and checkpoint:
            checkpoint_state = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            name_indices = checkpoint_state.get("indices", {}).get("name", {})
            qwen_options["scallop_mlp_name_indices"] = name_indices
        allowed_names = set(name_indices or {})
        instances = [
            dict(instance, _allowed_visual_names=allowed_names) for instance in instances
        ]
    if answer_mode not in {"iota", "membership", "miota", "mixed"}:
        raise ValueError(f"Unknown answer_mode: {answer_mode!r}")
    if answer_mode == "iota" and any(
        len(instance.get("expected_answers", [])) != 1 for instance in instances
    ):
        raise ValueError("Object-centered pipeline currently supports single-answer instances only")
    context = create_object_centered_graph(
        instances, include_global_consistency=include_global_consistency
    )
    attach_sensors(
        context,
        mode=mode,
        model_path=model_path,
        device=device,
        module_factory=module_factory,
        qwen_options=qwen_options,
        instances=instances,
    )
    if answer_mode == "membership":
        rows = [
            row
            for instance in instances
            for row in populate_membership_examples(
                instance, context, image_cache=image_cache, device=device
            )
        ]
    elif answer_mode == "miota":
        rows = [
            populate_miota_example(instance, context, image_cache=image_cache, device=device)
            for instance in instances
        ]
    elif answer_mode == "mixed":
        rows = [
            (
                populate_miota_example(instance, context, image_cache=image_cache, device=device)
                if len(instance.get("expected_answers", [])) > 1
                else populate_example(instance, context, image_cache=image_cache, device=device)
            )
            for instance in instances
        ]
    else:
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
        context.object_domain,
        *context.answer_slots,
        *context.object_predicates.values(),
        *context.relation_predicates.values(),
        *context.relation_object_predicates.values(),
        *context.knowledge_predicates.values(),
        context.graph.constraint,
    ]
    if compile_lc:
        from .compiled_global_loss import enable_compiled_global_constraint_loss

        enable_compiled_global_constraint_loss()
    program = InferenceProgram(
        context.graph,
        SolverModel,
        poi=poi,
        device=device,
        tnorm="P",
        inferTypes=["local/argmax"],
        beta=beta,
        include_global_constraint_loss=include_global_consistency,
        executable_constraint_loss_weight=executable_constraint_loss_weight,
        global_constraint_loss_weight=global_constraint_loss_weight,
        query_loss=NBCrossEntropyLoss,
    )
    if compile_lc:
        # InferenceModel.__init__ never forwards compile_lc to its own
        # super().__init__() call, so passing compile_lc as an InferenceProgram
        # kwarg gets silently dropped -- it never reaches the cmodel instance's
        # constructor at all. Set the attribute directly on the constructed
        # cmodel instead; the patched _calculate_global_constraint_loss reads
        # self.compile_lc at call time, so this is equivalent to having passed
        # it through construction, just applied after the fact.
        program.cmodel.compile_lc = True
    return context, dataset, program


def train_dynamic_instances(
    instances,
    module_factory,
    image_cache=DEFAULT_IMAGE_CACHE,
    device="cpu",
    epochs=1,
    learning_rate=1e-4,
    optimizer_factory=None,
):
    """Train fresh query-specific graphs while retaining shared learner state."""

    if optimizer_factory is None:
        optimizer_factory = functools.partial(torch.optim.AdamW, lr=learning_rate)
    optimizer = None
    summaries = []
    for instance_index, instance in enumerate(instances):
        context, dataset, program = build_program(
            [instance],
            mode="learner",
            image_cache=image_cache,
            device=device,
            answer_mode="membership",
            module_factory=module_factory,
        )
        program.to(device)
        trainable = [
            parameter for parameter in program.model.parameters()
            if parameter.requires_grad
        ]
        if optimizer is None:
            if not trainable:
                raise ValueError("Dynamic GraphQA learner has no trainable parameters")
            optimizer = optimizer_factory(trainable)
        else:
            optimized = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            if any(id(parameter) not in optimized for parameter in trainable):
                raise ValueError(
                    "module_factory created graph-local parameters; dynamic training requires "
                    "one shared learner across every graph"
                )
        program.opt = optimizer
        program.train(
            dataset,
            warmup_epochs=0,
            constraint_epochs=epochs,
            device=device,
            c_lr=learning_rate,
        )
        summaries.append({
            "instance_index": instance_index,
            "rows": len(dataset),
            "object_predicates": tuple(sorted(context.object_predicates)),
            "relation_predicates": tuple(sorted(context.relation_predicates)),
            "relation_object_predicates": tuple(sorted(context.relation_object_predicates)),
        })
    return optimizer, summaries


def load_instances(
    task_path, kb_dir, limit, single_answer_only=True, offset=0, max_objects=None,
    image_cache=DEFAULT_IMAGE_CACHE,
):
    """Load GraphQA instances.

    ``max_objects``, when set, drops instances whose object count exceeds the
    threshold. ``create_object_centered_graph`` sizes its ``answer_slots``
    (and other per-instance structures) to ``max(len(objects))`` across the
    whole loaded batch, so a handful of outlier high-object-count instances
    inflate DomiKnowS's per-instance candidate/constraint work for every
    instance in the batch, not just the outliers. Filtering the tail keeps
    that shared bound small without touching the DomiKnowS framework itself.

    ``image_cache``, when set, downloads each instance's source image into
    the cache up front (via ``_ensure_image_cached``, using the task's own
    ``image_url``) and drops only the instances where that's impossible
    (nothing cached and no URL, or the download fails). Otherwise a single
    missing image crashes the whole ``build_program`` batch it belongs to
    (``_image_and_boxes`` raises while populating every row in one
    listcomp), losing every other instance in that batch along with it.
    Pass ``None`` to skip this check entirely.
    """
    kb_facts = load_kb_facts(kb_dir)
    instances = []
    failures = []
    oversized = 0
    missing_image = 0
    for index, task in enumerate(
        load_vqar_tasks(task_path, limit=limit, offset=offset), start=offset
    ):
        try:
            instance = vqar_task_to_graphqa_instance(task, kb_facts=kb_facts)
            _attach_object_metadata(instance, task)
            if single_answer_only and len(instance.get("expected_answers", [])) != 1:
                continue
            if max_objects is not None and len(instance.get("objects", [])) > max_objects:
                oversized += 1
                continue
            if image_cache is not None:
                try:
                    _ensure_image_cached(instance, image_cache)
                except (FileNotFoundError, OSError):
                    missing_image += 1
                    continue
            required_visual_predicates([instance])
            instances.append(instance)
        except Exception as exc:
            failures.append((index, type(exc).__name__, str(exc)))
    if oversized:
        failures.append((None, "OversizedInstanceFilter", f"dropped {oversized} instances with >{max_objects} objects"))
    if missing_image:
        failures.append((None, "MissingImageFilter", f"dropped {missing_image} instances with no cached image file"))
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
    parser.add_argument("--mode", choices=["oracle", "scallop-local", "scallop-trained", "scallop-mlp", "qwen-vl", "internvl"], default="oracle")
    parser.add_argument("--answer-mode", choices=["iota", "membership", "miota", "mixed"], default="iota")
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode not in {"oracle", "scallop-local"} and not args.model_path:
        raise ValueError("--model-path is required for learned predicate modes")
    instances, failures = load_instances(
        args.task_path,
        args.kb_dir,
        args.limit,
        single_answer_only=args.answer_mode == "iota",
    )
    if not instances:
        raise ValueError(f"No supported instances; failures={failures[:5]}")
    context, dataset, program = build_program(
        instances,
        mode=args.mode,
        model_path=args.model_path,
        image_cache=args.image_cache,
        device=args.device,
        answer_mode=args.answer_mode,
    )
    accuracy = evaluate_executable(dataset, program, args.device)
    print(json.dumps({
        "mode": args.mode,
        "answer_mode": args.answer_mode,
        "loaded": len(instances),
        "failures": len(failures),
        "accuracy": accuracy,
        "logic": dataset[0].get("logic_str", instances[0].get("logic_str")),
    }, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
