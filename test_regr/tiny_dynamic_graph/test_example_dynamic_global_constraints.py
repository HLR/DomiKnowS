import unittest

from .example_dynamic_global_constraints import (
    active_rule_names,
    build_dynamic_constraint_example,
    global_constraint_gradient_norm,
    train_to_overfit,
)


class TinyDynamicGlobalConstraintTest(unittest.TestCase):
    def test_dynamic_global_constraints_train_shared_mlp(self):
        context = build_dynamic_constraint_example()

        active_rules = {
            example.example_id: active_rule_names(context, example)
            for example in context.examples
        }
        self.assertEqual(active_rules["color_rule"], ("red_implies_colored",))
        self.assertEqual(active_rules["animal_rule"], ("dog_implies_animal",))
        self.assertEqual(active_rules["plant_rule"], ("tree_implies_plant",))
        self.assertGreater(global_constraint_gradient_norm(context), 0.0)

        context, before, after, parameters_changed = train_to_overfit(context)

        self.assertTrue(parameters_changed)
        self.assertLess(after["loss"]["executable"], before["loss"]["executable"])
        self.assertLess(after["loss"]["global"], before["loss"]["global"])
        self.assertEqual(after["concept_accuracy"], 100.0)
        self.assertEqual(after["constraint_accuracy"], 100.0)
        self.assertIsNone(context.graph._active_concepts)


if __name__ == "__main__":
    unittest.main()
