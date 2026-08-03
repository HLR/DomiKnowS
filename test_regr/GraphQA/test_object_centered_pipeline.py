import unittest
from pathlib import Path

import torch
from PIL import Image

from domiknows.sensor.pytorch import ModuleLearner

from .object_centered_pipeline import build_program, evaluate_executable


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

    def test_multiple_answers_are_explicitly_rejected(self):
        instance = dict(self.instance)
        instance["expected_answers"] = ["o1", "o2"]
        instance["expected_answer"] = None
        with self.assertRaisesRegex(ValueError, "single-answer"):
            build_program(
                [instance], mode="oracle", image_cache=self.image_cache, device="cpu"
            )


if __name__ == "__main__":
    unittest.main()
