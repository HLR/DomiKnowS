import json
import os
import sys
import unittest
from pathlib import Path

import torch
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .config import CONFIG_ENV_VAR, load_temporal_config
from .dataset import discover_temporal_datasets, load_temporal_instances
from .execution import (
    compile_temporal_dataset,
    create_candidate_event_pairs,
    create_executable_instance,
    create_pair_learner_examples,
    create_query_event_groundings,
    create_query_logic,
    label_to_index,
    validate_dataset_convertible,
)
from .graph import TEMPORAL_LABELS, create_temporal_graph
from .llm_inference import (
    StaticChoiceBackend,
    build_query_event_choice_examples,
    build_temporal_relation_choice_examples,
    parse_choice,
    run_llm_inference,
)
from .modules import (
    OracleTemporalPredicateClassifier,
    event_prompt,
    pair_prompt,
    predictions_from_logits,
)
from .oracle import answer_label, check_oracle, consistency_violations, infer_transitive_before
from .program import evaluate_packed_left_example
from .smoke_test_dataset import consistency_failures_for_each_sample


SAMPLE_INSTANCE = {
    "doc_id": "toy_doc",
    "text": "John packed his bag before he left the house and arrived at work.",
    "tokens": [
        {"id": "t0", "text": "John"},
        {"id": "t1", "text": "packed"},
        {"id": "t2", "text": "his"},
        {"id": "t3", "text": "bag"},
        {"id": "t4", "text": "before"},
        {"id": "t5", "text": "he"},
        {"id": "t6", "text": "left"},
        {"id": "t7", "text": "the"},
        {"id": "t8", "text": "house"},
        {"id": "t9", "text": "and"},
        {"id": "t10", "text": "arrived"},
        {"id": "t11", "text": "at"},
        {"id": "t12", "text": "work"},
    ],
    "events": [
        {"id": "packed_bag", "token_id": "t1", "text": "packed"},
        {"id": "left_house", "token_id": "t6", "text": "left"},
        {"id": "arrived_work", "token_id": "t10", "text": "arrived"},
    ],
    "event_pairs": [
        {"e1": "packed_bag", "e2": "left_house", "label": "Before"},
        {"e1": "left_house", "e2": "packed_bag", "label": "After"},
        {"e1": "left_house", "e2": "arrived_work", "label": "Before"},
        {"e1": "arrived_work", "e2": "left_house", "label": "After"},
        {"e1": "packed_bag", "e2": "arrived_work", "label": "Before"},
        {"e1": "arrived_work", "e2": "packed_bag", "label": "After"},
    ],
    "query_pair": {"e1": "packed_bag", "e2": "left_house"},
}


CONSISTENT_DATASET = [SAMPLE_INSTANCE]


class TestTemporalRelationAdapter(unittest.TestCase):
    def test_graph_defines_event_pair_and_label_concepts(self):
        context = create_temporal_graph(SAMPLE_INSTANCE)

        self.assertEqual(tuple(context.label_concepts.keys()), TEMPORAL_LABELS)
        self.assertIn("sentence", context.namespace)
        self.assertIn("token", context.namespace)
        self.assertIn("event", context.namespace)
        self.assertIn("query_event1", context.namespace)
        self.assertIn("query_event2", context.namespace)
        self.assertIn("EventPair", context.namespace)
        self.assertIn("temporal_relation", context.namespace)
        self.assertEqual(context.event_concepts, {})

    def test_query_uses_queryL_iotaL_event_pair(self):
        logic = create_query_logic(SAMPLE_INSTANCE)

        self.assertIn("queryL", logic)
        self.assertIn("temporal_relation", logic)
        self.assertIn("iotaL", logic)
        self.assertIn("EventPair", logic)
        self.assertIn('event("p1", path=("p", pair_event1))', logic)
        self.assertIn('event("p2", path=("p", pair_event2))', logic)
        self.assertIn("query_event1", logic)
        self.assertIn("query_event2", logic)
        self.assertNotIn("packed_bag", logic)
        self.assertNotIn("left_house", logic)


    def test_query_event_markers_are_learner_predicates_over_events(self):
        groundings = create_query_event_groundings(SAMPLE_INSTANCE)

        self.assertEqual(
            groundings,
            [
                {"event_id": "packed_bag", "query_event1": True, "query_event2": False},
                {"event_id": "left_house", "query_event1": False, "query_event2": True},
                {"event_id": "arrived_work", "query_event1": False, "query_event2": False},
            ],
        )

    def test_pair_learner_examples_mark_text_and_target_temporal_relation(self):
        examples = create_pair_learner_examples(SAMPLE_INSTANCE)
        first = next(example for example in examples if example["e1"] == "packed_bag" and example["e2"] == "left_house")

        self.assertEqual(len(examples), 6)
        self.assertEqual(first["label"], "Before")
        self.assertEqual(first["label_index"], label_to_index("Before"))
        self.assertEqual(first["target_concept"], "temporal_relation")
        self.assertEqual(first["label_concepts"], TEMPORAL_LABELS)
        self.assertIn("[E1]packed[/E1]", first["text_with_event_markers"])
        self.assertIn("[E2]left[/E2]", first["text_with_event_markers"])


    def test_parse_choice_accepts_llm_letter_text_and_embedded_answer(self):
        choices = ["packed_bag: packed (token=t1)", "left_house: left (token=t6)"]

        self.assertEqual(parse_choice("B", choices), choices[1])
        self.assertEqual(parse_choice("Answer: left_house: left", choices), choices[1])
        self.assertEqual(parse_choice("1", choices), choices[0])
        self.assertEqual(parse_choice("Before", list(TEMPORAL_LABELS)), "Before")

    def test_query_event_choice_examples_ask_for_one_candidate_event(self):
        examples = build_query_event_choice_examples(SAMPLE_INSTANCE)

        self.assertEqual([example.task for example in examples], ["query_event1", "query_event2"])
        self.assertEqual(len(examples[0].choices), 3)
        self.assertIn("packed_bag: packed", examples[0].answer)
        self.assertIn("left_house: left", examples[1].answer)
        self.assertIn("Candidate", examples[0].prompt + " Candidate")

    def test_small_llm_output_predicts_query_events_and_temporal_relation(self):
        backend = StaticChoiceBackend([
            "packed_bag: packed",
            "left_house: left",
            "Before",
            "After",
            "Before",
            "After",
            "Before",
            "After",
        ])

        result = run_llm_inference(SAMPLE_INSTANCE, backend)

        self.assertEqual(
            result["query_event_groundings"],
            [
                {"event_id": "packed_bag", "query_event1": True, "query_event2": False},
                {"event_id": "left_house", "query_event1": False, "query_event2": True},
                {"event_id": "arrived_work", "query_event1": False, "query_event2": False},
            ],
        )
        self.assertEqual(len(result["event_pair_predictions"]), 6)
        self.assertEqual(result["event_pair_predictions"][0]["label"], "Before")
        self.assertEqual(len(backend.prompts), 8)

    def test_temporal_relation_choice_examples_are_multiple_choice_labels(self):
        examples = build_temporal_relation_choice_examples(SAMPLE_INSTANCE)
        first = examples[0]

        self.assertEqual(len(examples), 6)
        self.assertEqual(first.task, "temporal_relation")
        self.assertEqual(first.choices, list(TEMPORAL_LABELS))
        self.assertIn("[E1]", first.prompt)
        self.assertIn("[E2]", first.prompt)

    def test_oracle_predicate_classifier_outputs_domiknows_concept_logits(self):
        classifier = OracleTemporalPredicateClassifier()
        batch = classifier(SAMPLE_INSTANCE)

        self.assertEqual(batch.event_logits.shape, (3, 2))
        self.assertEqual(batch.query_event1_logits.shape, (3, 2))
        self.assertEqual(batch.query_event2_logits.shape, (3, 2))
        self.assertEqual(batch.temporal_relation_logits.shape, (6, len(TEMPORAL_LABELS)))
        self.assertEqual(batch.event_ids, ["packed_bag", "left_house", "arrived_work"])
        self.assertEqual(batch.pair_ids[0], ("packed_bag", "left_house"))

    def test_oracle_predicate_classifier_predictions_match_all_supervised_pairs(self):
        result = predictions_from_logits(OracleTemporalPredicateClassifier()(SAMPLE_INSTANCE))

        self.assertEqual(
            result["query_event_groundings"],
            [
                {"event_id": "packed_bag", "query_event1": True, "query_event2": False},
                {"event_id": "left_house", "query_event1": False, "query_event2": True},
                {"event_id": "arrived_work", "query_event1": False, "query_event2": False},
            ],
        )
        labels_by_pair = {
            (prediction["e1"], prediction["e2"]): prediction["label"]
            for prediction in result["event_pair_predictions"]
        }
        for pair in SAMPLE_INSTANCE["event_pairs"]:
            self.assertEqual(labels_by_pair[(pair["e1"], pair["e2"])], pair["label"])

    def test_qwen_classifier_inputs_are_fixed_concept_classification_prompts(self):
        event_text = event_prompt(SAMPLE_INSTANCE, SAMPLE_INSTANCE["events"][0])
        pair_text = pair_prompt(SAMPLE_INSTANCE, "packed_bag", "left_house")

        self.assertIn("TemporalRelation query pair", event_text)
        self.assertIn("Candidate event: packed_bag", event_text)
        self.assertIn("Labels: Before, After, Equal, Vague", pair_text)
        self.assertIn("[E1]packed[/E1]", pair_text)
        self.assertIn("[E2]left[/E2]", pair_text)

    def test_candidate_pairs_cover_all_ordered_document_events(self):
        candidates = create_candidate_event_pairs(SAMPLE_INSTANCE)
        candidate_keys = {(pair["e1"], pair["e2"]) for pair in candidates}

        self.assertEqual(len(candidates), 6)
        self.assertEqual(
            candidate_keys,
            {
                ("packed_bag", "left_house"),
                ("packed_bag", "arrived_work"),
                ("left_house", "packed_bag"),
                ("left_house", "arrived_work"),
                ("arrived_work", "packed_bag"),
                ("arrived_work", "left_house"),
            },
        )

    def test_compile_temporal_dataset(self):
        context = create_temporal_graph(SAMPLE_INSTANCE)
        executable = create_executable_instance(SAMPLE_INSTANCE)

        self.assertEqual(executable["logic_label"], label_to_index("Before"))
        self.assertEqual(len(executable["query_event_groundings"]), 3)
        self.assertEqual(len(executable["candidate_event_pairs"]), 6)
        self.assertEqual(len(executable["pair_learner_examples"]), 6)
        compiled = compile_temporal_dataset([SAMPLE_INSTANCE], context)
        self.assertEqual(len(compiled), 1)
        self.assertEqual(validate_dataset_convertible([SAMPLE_INSTANCE]), [])

    def test_oracle_returns_expected_query_label(self):
        self.assertEqual(answer_label(SAMPLE_INSTANCE), "Before")
        self.assertTrue(check_oracle(SAMPLE_INSTANCE, "Before"))

    def test_inference_program_executes_temporal_query(self):
        self.assertEqual(evaluate_packed_left_example(device="cpu"), 100.0)


    def test_oracle_passes_for_all_labeled_event_pairs(self):
        for pair in SAMPLE_INSTANCE["event_pairs"]:
            with self.subTest(e1=pair["e1"], e2=pair["e2"]):
                instance = {
                    **SAMPLE_INSTANCE,
                    "query_pair": {"e1": pair["e1"], "e2": pair["e2"]},
                }

                self.assertEqual(answer_label(instance), pair["label"])
                self.assertTrue(check_oracle(instance, pair["label"]))

    def test_consistency_rules_accept_inverse_and_transitive_graph(self):
        self.assertEqual(consistency_violations(CONSISTENT_DATASET), [])
        self.assertIn(("packed_bag", "arrived_work"), infer_transitive_before(CONSISTENT_DATASET))

    def test_consistency_rules_reject_inverse_violation_and_cycle(self):
        inconsistent = [
            {
                **SAMPLE_INSTANCE,
                "event_pairs": [
                    {"e1": "packed_bag", "e2": "left_house", "label": "Before"},
                    {"e1": "left_house", "e2": "packed_bag", "label": "Before"},
                ],
            }
        ]
        violation_types = {violation[0] for violation in consistency_violations(inconsistent)}

        self.assertIn("inverse", violation_types)
        self.assertIn("cycle", violation_types)


    def test_smoke_consistency_checks_each_sample_separately(self):
        same_event_ids_other_doc = {
            **SAMPLE_INSTANCE,
            "doc_id": "toy_doc_2",
            "event_pairs": [
                {"e1": "packed_bag", "e2": "left_house", "label": "Before"},
                {"e1": "left_house", "e2": "packed_bag", "label": "After"},
            ],
        }

        self.assertEqual(consistency_failures_for_each_sample([SAMPLE_INSTANCE, same_event_ids_other_doc]), [])

    def test_loader_groups_rows_by_document_for_whole_dataset(self):
        rows = (
            "doc_id\ttext\te1\te2\tlabel\te1_text\te2_text\te1_index\te2_index\n"
            "d1\tA packed left arrived\tpacked_bag\tleft_house\tBEFORE\tpacked\tleft\t1\t2\n"
            "d1\tA packed left arrived\tleft_house\tarrived_work\tBEFORE\tleft\tarrived\t2\t3\n"
        )
        with TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "temporal_rows.tsv"
            data_file.write_text(rows)
            instances = load_temporal_instances(data_file)

        self.assertEqual(len(instances), 1)
        self.assertEqual(len(instances[0]["events"]), 3)
        self.assertEqual(len(instances[0]["event_pairs"]), 2)
        self.assertEqual(len(create_candidate_event_pairs(instances[0])), 6)

    def test_dataset_discovery_reports_candidate_roots(self):
        discovered = discover_temporal_datasets()
        self.assertIn("matres", discovered)
        self.assertIn("tbdense", discovered)

    def test_config_resolves_portable_paths_relative_to_config_file(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "temporal-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "python_path": "../source",
                        "data_root": "datasets",
                        "training_model": "./models/qwen",
                        "inference_model": "Qwen/Qwen2.5-0.5B-Instruct",
                        "output_dir": "checkpoints",
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {CONFIG_ENV_VAR: str(config_path)}):
                config = load_temporal_config()

        self.assertEqual(config.data_root, config_path.parent / "datasets")
        self.assertEqual(config.python_path, config_path.parent.parent / "source")
        self.assertEqual(config.training_model, str(config_path.parent / "models" / "qwen"))
        self.assertEqual(config.inference_model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(config.output_path("run.pt"), config_path.parent / "checkpoints" / "run.pt")


class ConstraintsActuallyTrainTest(unittest.TestCase):
    """Guards the four ways this example's constraints have been silently inert.

    Each of these failed at some point while training completed normally and
    reported a plausible accuracy: no constraint label sensor, the global
    constraint loss defaulted off, an IndexError on an empty variable, and — the
    subtlest — a candidate sensor that built *zero* EventPair datanodes, so every
    rule quantified over EventPair was vacuous.

    Runs on CPU with a stub in place of the 8B backbone, so it needs no GPU.
    """

    @staticmethod
    def _build(extra_argv=()):
        import torch
        import test_regr.TemporalRelation.program_qwen_train as P
        from test_regr.TemporalRelation.dataset import load_temporal_instances

        class _StubLearner(torch.nn.Module):
            def __init__(self, *a, **k):
                super().__init__()
                self.lin = torch.nn.Linear(1, 4)

            def forward(self, prompts):
                n = 1 if isinstance(prompts, str) else len(list(prompts))
                return self.lin(torch.zeros(n, 1))

        original = P.QwenTemporalRelationLearner
        P.QwenTemporalRelationLearner = _StubLearner
        try:
            data_path = (Path(__file__).parent / "data" / "MATRES" / "timebank.txt")
            instances = P.expand_document_query_instances(
                load_temporal_instances(data_path))[:3]
            argv = sys.argv
            sys.argv = ["test", *extra_argv]
            try:
                args = P.parse_args()
            finally:
                sys.argv = argv
            args.device = "cpu"
            args.max_events_per_instance = 8
            args.pair_selection = "target"
            args.max_pairs_per_instance = 2
            args.lora_r = 0
            return P.build_temporal_program(instances, args)
        finally:
            P.QwenTemporalRelationLearner = original

    def test_event_pair_datanodes_are_created(self):
        """Zero EventPair groundings makes every constraint vacuous."""
        dataset, ctx, program = self._build(["--no-transitivity"])
        datanode = next(iter(program.populate([dataset[0]], device="cpu")))
        pairs = datanode.findDatanodes(select=ctx.event_pair)
        self.assertEqual(len(pairs), 2, "EventPair candidates were not built")

    def test_every_global_rule_grounds(self):
        """All five rules must produce a lossTensor, not None."""
        dataset, ctx, program = self._build(["--no-transitivity"])
        datanode = next(iter(program.populate([dataset[0]], device="cpu")))
        datanode.inferLocal(keys=("softmax",))
        losses = datanode.calculateLcLoss(tnorm="P")
        heads = [lc.lcName for _k, lc in ctx.graph.logicalConstrainsRecursive
                 if getattr(lc, "headLC", False)]
        self.assertEqual(len(heads), 5)
        for name in heads:
            with self.subTest(rule=name):
                self.assertIsNotNone(losses[name]["lossTensor"],
                                     f"{name} produced no grounding")

    def test_constraint_loss_reaches_model_parameters(self):
        """closs must be differentiable *and* touch a learnable weight."""
        import test_regr.TemporalRelation.program_qwen_train as P
        dataset, _ctx, program = self._build(
            ["--no-transitivity", "--constraint-epochs", "1"])
        args = type("A", (), {"constraint_epochs": 1})()
        reached = P.verify_constraint_gradient_flow(program, dataset, args)
        self.assertIsNotNone(reached)
        self.assertGreater(reached, 0)

    def test_exactly_one_loss_is_not_constant(self):
        """A constant loss carries no gradient, however plausible its value."""
        dataset, ctx, program = self._build(["--no-transitivity"])
        seen = set()
        for bias in ([0.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]):
            with torch.no_grad():
                for module in program.model.modules():
                    if isinstance(module, torch.nn.Linear) and module.out_features == 4:
                        module.bias.copy_(torch.tensor(bias))
            datanode = next(iter(program.populate([dataset[0]], device="cpu")))
            datanode.inferLocal(keys=("softmax",))
            tensor = datanode.calculateLcLoss(tnorm="P")["LC1"]["lossTensor"]
            seen.add(round(float(tensor.detach().flatten()[0]), 6))
        self.assertGreater(len(seen), 1,
                           "exactly-one loss did not respond to the prediction")

    def test_splitLossColumns_tolerates_an_empty_variable(self):
        """An ungroundable variable must not raise (pre-existing IndexError)."""
        from domiknows.solver.logicalConstraintConstructor import (
            LogicalConstraintConstructor)
        self.assertEqual(LogicalConstraintConstructor.splitLossColumns([]), [])


if __name__ == "__main__":
    unittest.main()
