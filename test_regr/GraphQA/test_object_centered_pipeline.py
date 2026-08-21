import copy
import unittest
from pathlib import Path

import torch
from PIL import Image

from domiknows.sensor.pytorch import ModuleLearner

from .evaluate_object_centered_c2 import miota_prediction_records
from .object_centered_pipeline import (
    ScallopLocalFactLearner,
    ScallopObjectMLP,
    ScallopRelationMLP,
    _object_feature_dim,
    _object_scallop_feature_vector,
    _pair_scallop_features,
    _sigmoid_binary_logits,
    build_program,
    evaluate_executable,
    train_dynamic_instances,
)


class SharedBoxClassifier(torch.nn.Module):
    """Small stand-in for one prompt-conditioned VLM shared by fresh graphs."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)


class SharedPredicateView(torch.nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, _images, _filenames, boxes):
        device = self.shared.linear.weight.device
        return self.shared.linear(boxes.to(device=device, dtype=torch.float32) / 100.0)


class SharedModuleFactory:
    def __init__(self):
        self.shared = SharedBoxClassifier()

    def __call__(self, _kind, _value, _arity):
        return SharedPredicateView(self.shared)


class TestObjectCenteredPipeline(unittest.TestCase):
    def setUp(self):
        self.image_cache = Path("/tmp/graphqa_object_centered_images")
        self.image_cache.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), "white").save(self.image_cache / "smoke.jpg")
        self.instance = {
            "objects": ["o1", "o2", "o3"],
            "object_metadata": {
                "o1": {"bbox": [0.55, 0.20, 0.30, 0.30]},
                "o2": {"bbox": [0.10, 0.20, 0.30, 0.30]},
                "o3": {"bbox": [0.10, 0.60, 0.20, 0.20]},
            },
            "symbols": ["van", "red", "car"],
            "visual_facts": [
                ("Name", "o1", "van"),
                ("Attribute", "o1", "red"),
                ("Name", "o2", "car"),
                ("RightOf", "o1", "o2"),
            ],
            "kb_facts": [],
            "query": {
                "target_type": "__any_object__",
                "conditions": [
                    ("Name", "o", "van"),
                    ("Attribute", "o", "red"),
                    ("RelationTo", "o", ("RightOf", ["o2"])),
                ],
                "answer_type": "object",
            },
            "expected_answer": "o1",
            "expected_answers": ["o1"],
            "source_image_id": "smoke",
        }
    def test_scallop_52_feature_shapes_and_heads(self):
        instance = copy.deepcopy(self.instance)
        for metadata in instance["object_metadata"].values():
            metadata["feature_vector"] = [0.0] * 2048
        self.assertEqual(_object_feature_dim([instance]), 2048)
        feature = _object_scallop_feature_vector(
            instance, "o1", instance["object_metadata"]["o1"]["bbox"]
        )
        self.assertEqual(len(feature), 2048)

        features = torch.randn(3, 2048)
        boxes = torch.randn(3, 4)
        pair_features = _pair_scallop_features(features, boxes)
        self.assertEqual(tuple(pair_features.shape), (6, 4104))

        name_head = ScallopObjectMLP(2048, 500, hidden_dim=32, hidden_layers=2)
        relation_head = ScallopRelationMLP(2048, 230, hidden_dim=32)
        self.assertEqual(tuple(name_head(features).shape), (3, 500))
        self.assertEqual(tuple(relation_head(pair_features).shape), (6, 230))
        self.assertEqual(
            sum(isinstance(module, torch.nn.BatchNorm1d) for module in name_head.modules()),
            2,
        )

    def test_attribute_projection_is_independent_sigmoid(self):
        logits = torch.tensor([[4.0, -2.0, 0.75]])
        binary = _sigmoid_binary_logits(logits, yes_index=2)
        self.assertTrue(torch.allclose(
            binary.softmax(dim=-1)[:, 1], torch.sigmoid(logits[:, 2])
        ))

    def test_oracle_predicates_are_direct_module_learners(self):
        context, _dataset, _program = build_program(
            [self.instance], mode="oracle", image_cache=self.image_cache, device="cuda"
        )
        learners = [
            sensor for sensor in context.graph.get_sensors()
            if isinstance(sensor, ModuleLearner)
        ]
        properties = {sensor.prop.fullname for sensor in learners}
        for concept in context.object_predicates.values():
            self.assertIn(context.obj[concept].fullname, properties)
        for concept in context.relation_predicates.values():
            self.assertIn(context.pair[concept].fullname, properties)

    def test_scallop_local_predicates_execute_like_probabilistic_facts(self):
        context, dataset, program = build_program(
            [self.instance], mode="scallop-local", image_cache=self.image_cache, device="cpu"
        )
        learners = [
            sensor for sensor in context.graph.get_sensors()
            if isinstance(sensor, ModuleLearner)
        ]
        self.assertTrue(any(isinstance(sensor.module, ScallopLocalFactLearner) for sensor in learners))
        self.assertEqual(evaluate_executable(dataset, program, "cpu"), 100.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required by this SolverModel")
    def test_compositional_oracle_query_returns_o1(self):
        _context, dataset, program = build_program(
            [self.instance], mode="oracle", image_cache=self.image_cache, device="cuda"
        )
        logic = dataset[0]["logic_str"]
        self.assertIn('name_van("o")', logic)
        self.assertIn('attr_red(path="o")', logic)
        self.assertIn('pair_src.reversed', logic)
        self.assertEqual(evaluate_executable(dataset, program, "cuda"), 100.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required by this SolverModel")
    def test_kg_property_is_derived_and_returns_o1(self):
        instance = copy.deepcopy(self.instance)
        instance["visual_facts"] = [
            ("Name", "o1", "zebra"),
            ("Name", "o2", "car"),
            ("Attribute", "o1", "red"),
        ]
        instance["kb_facts"] = [("Is", "zebra", "herbivorous")]
        instance["query"]["conditions"] = [
            ("KG", "o", ("Is", "herbivorous")),
            ("Attribute", "o", "red"),
        ]
        context, dataset, program = build_program(
            [instance], mode="oracle", image_cache=self.image_cache, device="cuda"
        )
        logic = dataset[0]["logic_str"]
        self.assertIn('kg_Is_herbivorous("o")', logic)
        self.assertNotIn('name_zebra("o")', logic)
        self.assertIn('attr_red(path="o")', logic)
        sensor_properties = {
            sensor.prop.fullname for sensor in context.graph.get_sensors()
        }
        derived = next(iter(context.knowledge_predicates.values()))
        self.assertNotIn(derived.fullname, sensor_properties)
        self.assertGreaterEqual(len(context.graph.logicalConstrains), 1)
        self.assertEqual(evaluate_executable(dataset, program, "cuda"), 100.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required by this SolverModel")
    def test_semantic_class_is_derived_and_returns_o1(self):
        instance = copy.deepcopy(self.instance)
        instance["visual_facts"] = [
            ("Name", "o1", "water"),
            ("Name", "o2", "car"),
        ]
        instance["kb_facts"] = [("TypeOf", "water", "drinks")]
        instance["query"]["conditions"] = [
            ("SemanticClass", "o", "drinks"),
        ]
        context, dataset, program = build_program(
            [instance], mode="oracle", image_cache=self.image_cache, device="cuda"
        )
        logic = dataset[0]["logic_str"]
        self.assertIn('semantic_class_drinks("o")', logic)
        self.assertNotIn('name_water("o")', logic)
        sensor_properties = {
            sensor.prop.fullname for sensor in context.graph.get_sensors()
        }
        derived = next(iter(context.knowledge_predicates.values()))
        self.assertNotIn(derived.fullname, sensor_properties)
        self.assertGreaterEqual(len(context.graph.logicalConstrains), 1)
        self.assertEqual(evaluate_executable(dataset, program, "cuda"), 100.0)

    def test_semantic_class_miota_supports_multiple_answers(self):
        instance = copy.deepcopy(self.instance)
        instance["visual_facts"] = [
            ("Name", "o1", "water"),
            ("Name", "o2", "water"),
            ("Name", "o3", "car"),
        ]
        instance["kb_facts"] = [("TypeOf", "water", "drinks")]
        instance["query"]["conditions"] = [
            ("SemanticClass", "o", "drinks"),
        ]
        instance["expected_answer"] = None
        instance["expected_answers"] = ["o1", "o2"]
        context, dataset, program = build_program(
            [instance], mode="oracle", image_cache=self.image_cache, device="cpu",
            answer_mode="miota",
        )
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["logic_label"].tolist(), [1.0, 1.0, 0.0])
        self.assertIn("semantic_class_drinks", dataset[0]["logic_str"])
        derived = next(iter(context.knowledge_predicates.values()))
        sensor_properties = {
            sensor.prop.fullname for sensor in context.graph.get_sensors()
        }
        self.assertNotIn(derived.fullname, sensor_properties)
        self.assertGreaterEqual(len(context.graph.logicalConstrains), 1)
        records = miota_prediction_records([instance], dataset, program, "cpu")
        self.assertEqual(records[0]["predicted_answers"], ["o1", "o2"])
        self.assertTrue(records[0]["correct"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required by this SolverModel")
    def test_relation_only_query_uses_relation_aligned_object_root(self):
        instance = copy.deepcopy(self.instance)
        instance["query"]["conditions"] = [
            ("RelationTo", "o", ("RightOf", ["o2"])),
        ]
        instance["query"]["program"] = {
            "op": "relate",
            "direction": "RelationTo",
            "relation": "RightOf",
            "candidates": ["o2"],
            "input": {"op": "all"},
        }
        _context, dataset, program = build_program(
            [instance], mode="oracle", image_cache=self.image_cache, device="cuda"
        )
        logic = dataset[0]["logic_str"]
        self.assertIn('object("x")', logic)
        self.assertIn('relation_RightOf("r1", path=("x", pair_src.reversed))', logic)
        self.assertEqual(evaluate_executable(dataset, program, "cuda"), 100.0)

    def test_nested_relation_miota_uses_tiny_relation_chain_shape(self):
        instance = {
            "objects": ["o1", "o2", "o3"],
            "object_metadata": {
                "o1": {"bbox": [0.10, 0.10, 0.20, 0.20]},
                "o2": {"bbox": [0.40, 0.10, 0.20, 0.20]},
                "o3": {"bbox": [0.70, 0.10, 0.20, 0.20]},
            },
            "symbols": ["ball"],
            "visual_facts": [
                ("Name", "o2", "ball"),
                ("Name", "o3", "ball"),
                ("LeftOf", "o1", "o2"),
                ("LeftOf", "o2", "o3"),
            ],
            "kb_facts": [],
            "query": {
                "target_type": "__any_object__",
                "conditions": [],
                "answer_type": "object",
                "program": {
                    "op": "relate",
                    "direction": "RelationTo",
                    "relation": "LeftOf",
                    "candidates": ["o1", "o2", "o3"],
                    "input": {
                        "op": "filter",
                        "condition": ("Name", "o", "ball"),
                        "input": {
                            "op": "relate",
                            "direction": "RelationTo",
                            "relation": "LeftOf",
                            "candidates": ["o1", "o2", "o3"],
                            "input": {
                                "op": "filter",
                                "condition": ("Name", "o", "ball"),
                                "input": {"op": "all"},
                            },
                        },
                    },
                },
            },
            "expected_answer": None,
            "expected_answers": ["o1"],
            "source_image_id": "smoke",
        }
        _context, dataset, _program = build_program(
            [instance], mode="oracle", image_cache=self.image_cache, device="cpu",
            answer_mode="miota",
        )
        logic = dataset[0]["logic_str"].replace("\n", " ")
        self.assertIn('miotaL', logic)
        self.assertIn('object("x")', logic)
        self.assertIn('relation_LeftOf("r1", path=("x", pair_src.reversed))', logic)
        self.assertIn('name_ball("y", path=("r1", pair_dst))', logic)
        self.assertIn('relation_LeftOf("r2", path=("y", pair_src.reversed))', logic)
        self.assertIn('name_ball("z", path=("r2", pair_dst))', logic)

    def test_relation_candidate_constraint_stays_on_endpoint_path(self):
        instance = copy.deepcopy(self.instance)
        instance["query"]["conditions"] = []
        instance["query"]["program"] = {
            "op": "relate",
            "direction": "RelationTo",
            "relation": "LeftOf",
            "candidates": ["o3"],
            "input": {"op": "all"},
        }
        instance["visual_facts"] = [
            ("LeftOf", "o1", "o3"),
            ("LeftOf", "o2", "o1"),
        ]
        instance["expected_answer"] = None
        instance["expected_answers"] = ["o1"]

        _context, dataset, _program = build_program(
            [instance], mode="oracle", image_cache=self.image_cache, device="cpu",
            answer_mode="miota",
        )
        logic = dataset[0]["logic_str"].replace("\n", " ")
        self.assertIn('relation_LeftOf("r1", path=("x", pair_src.reversed))', logic)
        self.assertIn('answer_slot_2("y", path=("r1", pair_dst))', logic)
        self.assertNotIn('answer_slot_2(path="y")', logic)
        self.assertNotIn('object("y", path=("r1", pair_dst))', logic)

    def test_iota_still_rejects_multiple_answers(self):
        instance = copy.deepcopy(self.instance)
        instance["expected_answers"] = ["o1", "o2"]
        instance["expected_answer"] = None
        with self.assertRaisesRegex(ValueError, "single-answer"):
            build_program(
                [instance], mode="oracle", image_cache=self.image_cache, device="cpu"
            )

    def test_membership_execution_supports_multiple_answers(self):
        instance = copy.deepcopy(self.instance)
        instance["visual_facts"].append(("Attribute", "o2", "red"))
        instance["query"]["conditions"] = [("Attribute", "o", "red")]
        instance["expected_answers"] = ["o1", "o2"]
        instance["expected_answer"] = None
        _context, dataset, program = build_program(
            [instance], mode="oracle", image_cache=self.image_cache, device="cpu",
            answer_mode="membership",
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual([row["logic_label"].item() for row in dataset], [1, 1, 0])
        self.assertTrue(all("existsL" in row["logic_str"] for row in dataset))
        self.assertEqual(evaluate_executable(dataset, program, "cpu"), 100.0)

    def test_miota_execution_supports_multiple_answers(self):
        instance = copy.deepcopy(self.instance)
        instance["visual_facts"].append(("Attribute", "o2", "red"))
        instance["query"]["conditions"] = [("Attribute", "o", "red")]
        instance["expected_answers"] = ["o1", "o2"]
        instance["expected_answer"] = None
        _context, dataset, program = build_program(
            [instance], mode="oracle", image_cache=self.image_cache, device="cpu",
            answer_mode="miota",
        )
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["logic_label"].tolist(), [1.0, 1.0, 0.0])
        self.assertIn("miotaL", dataset[0]["logic_str"])
        self.assertNotIn("queryL", dataset[0]["logic_str"])
        self.assertEqual(evaluate_executable(dataset, program, "cpu"), 100.0)

    def test_dynamic_graphs_reuse_one_trainable_learner(self):
        red = copy.deepcopy(self.instance)
        red["query"]["conditions"] = [("Attribute", "o", "red")]
        red["expected_answer"] = "o1"
        red["expected_answers"] = ["o1"]
        car = copy.deepcopy(self.instance)
        car["query"]["conditions"] = [("Name", "o", "car")]
        car["expected_answer"] = "o2"
        car["expected_answers"] = ["o2"]
        factory = SharedModuleFactory()
        before = factory.shared.linear.weight.detach().clone()
        optimizer, summaries = train_dynamic_instances(
            [red, car], module_factory=factory, image_cache=self.image_cache,
            device="cpu", epochs=1, learning_rate=1e-2,
        )
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual([summary["rows"] for summary in summaries], [3, 3])
        self.assertNotEqual(
            summaries[0]["object_predicates"], summaries[1]["object_predicates"]
        )
        self.assertFalse(torch.equal(before, factory.shared.linear.weight.detach()))


if __name__ == "__main__":
    unittest.main()
