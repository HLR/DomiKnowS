"""Focused, mocked tests for the wall-clock timing instrumentation added to
train_epoch()/evaluate_dev() in train_scallop_mlp.py. Mocks out program.train
and miota_prediction_records so these run without a real DomiKnowS program,
model, or dataset -- only the timing math and dict-shape/back-compat behavior
are under test.
"""
import time
import unittest
from unittest.mock import MagicMock, patch

from . import train_scallop_mlp


class TestTrainEpochTiming(unittest.TestCase):
    def test_wall_seconds_and_seconds_per_example_reported(self):
        program = MagicMock()
        sleep_for = 0.05

        def fake_train(*_args, **_kwargs):
            time.sleep(sleep_for)

        program.train.side_effect = fake_train
        dataset = list(range(5))
        optimizer = MagicMock()

        result = train_scallop_mlp.train_epoch(
            program, dataset, optimizer, device="cpu", epoch=3, lr=1e-4, batch_size=2,
        )

        # Pre-existing fields must be unaffected.
        self.assertEqual(result["examples"], 5)
        self.assertEqual(result["domiknows_train_epoch"], 3)
        # New timing fields.
        self.assertGreaterEqual(result["wall_seconds"], sleep_for)
        self.assertAlmostEqual(
            result["seconds_per_example"], result["wall_seconds"] / 5, places=6,
        )

    def test_seconds_per_example_none_when_no_examples(self):
        program = MagicMock()
        result = train_scallop_mlp.train_epoch(
            program, [], MagicMock(), device="cpu", epoch=1, lr=1e-4, batch_size=2,
        )
        self.assertEqual(result["examples"], 0)
        self.assertIsNone(result["seconds_per_example"])
        self.assertGreaterEqual(result["wall_seconds"], 0.0)


class TestEvaluateDevTiming(unittest.TestCase):
    def _canned_records(self):
        return [
            {
                "correct": True,
                "scores": [0.9, 0.1],
                "objects": ["o1", "o2"],
                "gold_answers": ["o1"],
            },
            {
                "correct": False,
                "scores": [0.2, 0.8],
                "objects": ["o1", "o2"],
                "gold_answers": [],
            },
        ]

    def test_wall_seconds_reported_and_metrics_unchanged(self):
        sleep_for = 0.05

        def fake_records(*_args, **_kwargs):
            time.sleep(sleep_for)
            return self._canned_records()

        program = MagicMock()
        with patch.object(
            train_scallop_mlp, "miota_prediction_records", side_effect=fake_records
        ):
            result = train_scallop_mlp.evaluate_dev(
                instances=[object(), object()],
                dataset=[object(), object()],
                program=program,
                device="cpu",
                threshold=0.5,
                decode_policy="family-top1",
            )

        program.model.eval.assert_called_once()

        # Pre-existing metric behavior must be unaffected by the timing change.
        self.assertEqual(result["examples"], 2)
        self.assertAlmostEqual(result["exact_answer_acc"], 0.5)
        self.assertAlmostEqual(result["top1_gold_hit"], 0.5)
        self.assertAlmostEqual(result["recall_at_5"], 1.0)
        self.assertAlmostEqual(result["recall_at_10"], 1.0)

        # New timing fields.
        self.assertGreaterEqual(result["wall_seconds"], sleep_for)
        self.assertAlmostEqual(
            result["seconds_per_example"], result["wall_seconds"] / 2, places=6,
        )

    def test_seconds_per_example_none_when_no_records(self):
        with patch.object(
            train_scallop_mlp, "miota_prediction_records", return_value=[]
        ):
            result = train_scallop_mlp.evaluate_dev(
                instances=[], dataset=[], program=MagicMock(), device="cpu",
                threshold=0.5, decode_policy="family-top1",
            )
        self.assertEqual(result["examples"], 0)
        self.assertIsNone(result["exact_answer_acc"])
        self.assertIsNone(result["seconds_per_example"])
        self.assertGreaterEqual(result["wall_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
