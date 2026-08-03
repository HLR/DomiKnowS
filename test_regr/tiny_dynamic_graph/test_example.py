import unittest

from .example import EXAMPLES, infer_each_dynamic_graph, train_each_dynamic_graph


class TinyDynamicGraphTest(unittest.TestCase):
    def test_training_and_inference_across_fresh_graphs(self):
        model, trained_graphs, parameters_changed = train_each_dynamic_graph()
        accuracies, inferred_graphs = infer_each_dynamic_graph(model)

        self.assertTrue(parameters_changed)
        self.assertEqual(set(accuracies), {spec.example_id for spec in EXAMPLES})
        self.assertTrue(all(accuracy == 100.0 for accuracy in accuracies.values()))
        self.assertEqual(trained_graphs, inferred_graphs)
        self.assertNotEqual(trained_graphs[0][0], trained_graphs[1][0])
        self.assertEqual(trained_graphs[0][1], ("red", "dog", "cat"))
        self.assertEqual(trained_graphs[1][1], ("red", "cat", "tree"))


if __name__ == "__main__":
    unittest.main()
