import unittest

import torch

from domiknows.sensor.pytorch import ModuleLearner

from .object_centered_smoke import build_smoke, run_smoke, smoke_logic


class TestObjectCenteredGraphQASmoke(unittest.TestCase):
    def test_atomic_visual_concepts_use_direct_module_learners(self):
        context = build_smoke(device="cpu")

        learners = [
            sensor for sensor in context.graph.get_sensors()
            if isinstance(sensor, ModuleLearner)
        ]
        learned_properties = {sensor.prop.fullname for sensor in learners}
        for concept in (context.van, context.red, context.car):
            self.assertIn(context.obj[concept].fullname, learned_properties)
        self.assertIn(context.pair[context.right_of].fullname, learned_properties)

    def test_query_uses_object_and_relation_paths(self):
        logic = smoke_logic()

        self.assertIn('van("o")', logic)
        self.assertIn('red(path="o")', logic)
        self.assertIn('path=("o", pair_src.reversed)', logic)
        self.assertIn('car(path=("r", pair_dst))', logic)
        self.assertIn("queryL", logic)
        self.assertIn("iotaL", logic)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by this DomiKnowS SolverModel")
    def test_execution_loss_updates_all_binary_predicates(self):
        result = run_smoke(device="cuda", epochs=1)

        self.assertGreater(result["trainable_count"], 0)
        self.assertEqual(result["updated_count"], result["trainable_count"])
        self.assertTrue(result["all_updated"])


if __name__ == "__main__":
    unittest.main()
