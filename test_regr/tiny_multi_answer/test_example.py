import unittest

from domiknows.sensor.pytorch import ModuleLearner

from .example import GOLD_ANSWERS, build_multi_answer_example, predict_answer_set


class TinyMultiAnswerTest(unittest.TestCase):
    def test_multi_answer_membership_queries(self):
        example = build_multi_answer_example()

        self.assertEqual(predict_answer_set(example), GOLD_ANSWERS)
        self.assertEqual(example.program.evaluate_condition(example.dataset, device="cpu"), 100.0)
        self.assertTrue(all("existsL" in logic for logic in example.logic_strings))
        self.assertEqual(set(example.concept_learners), {"red", "dog", "cat"})
        self.assertTrue(
            all(isinstance(learner, ModuleLearner) for learner in example.concept_learners.values())
        )


if __name__ == "__main__":
    unittest.main()
