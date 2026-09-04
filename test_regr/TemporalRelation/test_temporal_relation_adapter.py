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


class TBDenseTimeMLConverterTest(unittest.TestCase):
    @staticmethod
    def _timeml(extra_tlinks="", second_relation="BEFORE"):
        return f"""<?xml version="1.0" encoding="utf-8"?>
<TimeML>
  <DOCID>dense-1</DOCID>
  <TEXT>Alice <EVENT eid="e1">arrived</EVENT> before Bob
    <EVENT eid="e2">left</EVENT> on <TIMEX3 tid="t1">Monday</TIMEX3>.</TEXT>
  <MAKEINSTANCE eventID="e1" eiid="ei1" />
  <MAKEINSTANCE eventID="e2" eiid="ei2" />
  <TLINK lid="l1" eventInstanceID="ei1"
         relatedToEventInstance="ei2" relType="{second_relation}" />
  <TLINK lid="l2" eventInstanceID="ei1"
         relatedToTime="t1" relType="IS_INCLUDED" />
  <TLINK lid="l3" timeID="t1" relatedToEventInstance="ei2" relType="BEFORE" />
  {extra_tlinks}
</TimeML>"""

    def test_timeml_parser_maps_instances_and_filters_time_links(self):
        from test_regr.TemporalRelation.convert_tbdense import convert_timeml_file

        with TemporaryDirectory() as directory:
            path = Path(directory) / "doc.tml"
            path.write_text(self._timeml(), encoding="utf-8")
            rows = convert_timeml_file(path, split="train")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(
            (row["event_pairs"][0]["e1"], row["event_pairs"][0]["e2"],
             row["event_pairs"][0]["label"]),
                         ("e1", "e2", "Before"))
        events = {event["id"]: event for event in row["events"]}
        self.assertEqual((events["e1"]["text"], events["e2"]["text"]),
                         ("arrived", "left"))
        self.assertEqual(row["dataset"], "tbdense")
        self.assertEqual(row["split"], "train")
        token_ids = {token["id"] for token in row["tokens"]}
        self.assertIn(events["e1"]["token_id"], token_ids)
        self.assertIn(events["e2"]["token_id"], token_ids)
        self.assertIn("Alice arrived before Bob left on Monday .", row["text"])

    def test_clone_root_conversion_preserves_splits(self):
        from test_regr.TemporalRelation.convert_tbdense import convert_timeml_source

        with TemporaryDirectory() as directory:
            root = Path(directory) / "TimeBank-dense"
            output = Path(directory) / "converted"
            for split in ("train", "dev", "test"):
                split_dir = root / split
                split_dir.mkdir(parents=True)
                xml = self._timeml().replace(
                    "<DOCID>dense-1</DOCID>", f"<DOCID>{split}-doc</DOCID>")
                (split_dir / f"{split}.tml").write_text(xml, encoding="utf-8")

            converted = convert_timeml_source(root, output_dir=output)
            self.assertEqual(set(converted), {"train", "dev", "test"})
            doc_sets = []
            for split in ("train", "dev", "test"):
                path = output / f"{split}.jsonl"
                self.assertTrue(path.is_file())
                loaded = load_temporal_instances(path, dataset_name="tbdense")
                self.assertEqual(loaded[0]["dataset"], "tbdense")
                self.assertEqual(loaded[0]["split"], split)
                doc_sets.append({item["doc_id"] for item in loaded})
            self.assertTrue(doc_sets[0].isdisjoint(doc_sets[1]))
            self.assertTrue(doc_sets[0].isdisjoint(doc_sets[2]))
            self.assertTrue(doc_sets[1].isdisjoint(doc_sets[2]))

    def test_timeml_parser_rejects_unknown_instance_conflict_and_label(self):
        from test_regr.TemporalRelation.convert_tbdense import convert_timeml_file

        cases = {
            "missing": self._timeml().replace(
                'relatedToEventInstance="ei2"', 'relatedToEventInstance="ei404"', 1),
            "conflict": self._timeml(
                '<TLINK lid="l4" eventInstanceID="ei1" '
                'relatedToEventInstance="ei2" relType="AFTER" />'),
            "unknown": self._timeml(second_relation="OVERLAP"),
        }
        with TemporaryDirectory() as directory:
            for name, xml in cases.items():
                with self.subTest(case=name):
                    path = Path(directory) / f"{name}.tml"
                    path.write_text(xml, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, str(path).replace("\\", "\\\\")):
                        convert_timeml_file(path)

    def test_timeml_conflict_policy_last_is_explicit_and_counted(self):
        from collections import Counter
        from test_regr.TemporalRelation.convert_tbdense import convert_timeml_file

        xml = self._timeml(
            '<TLINK lid="l4" eventInstanceID="ei1" '
            'relatedToEventInstance="ei2" relType="AFTER" />')
        with TemporaryDirectory() as directory:
            path = Path(directory) / "conflict.tml"
            path.write_text(xml, encoding="utf-8")
            stats = Counter()
            rows = convert_timeml_file(
                path, conflict_policy="last", stats=stats)
        self.assertEqual(rows[0]["event_pairs"][0]["label"], "After")
        self.assertEqual(stats["conflicting_tlinks_resolved"], 1)

    def test_timeml_parser_reports_malformed_xml(self):
        from test_regr.TemporalRelation.convert_tbdense import convert_timeml_file

        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.tml"
            path.write_text("<TimeML><DOCID>x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed TimeML XML"):
                convert_timeml_file(path)


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
    def _build(extra_argv=(), instances=None, pair_selection="target",
               max_pairs_per_instance=2):
        import torch
        import test_regr.TemporalRelation.program_qwen_train as P
        from test_regr.TemporalRelation.dataset import load_temporal_instances

        class _StubLearner(torch.nn.Module):
            def __init__(self, *a, **k):
                super().__init__()
                self.lin = torch.nn.Linear(1, len(P.TEMPORAL_LABELS))

            def forward(self, prompts, dataset_mask=None):
                n = 1 if isinstance(prompts, str) else len(list(prompts))
                logits = self.lin(torch.zeros(n, 1))
                return P.apply_dataset_mask(logits, dataset_mask)

        original = P.QwenTemporalRelationLearner
        P.QwenTemporalRelationLearner = _StubLearner
        try:
            data_path = (Path(__file__).parent / "data" / "MATRES" / "timebank.txt")
            if instances is None:
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
            args.pair_selection = pair_selection
            args.max_pairs_per_instance = max_pairs_per_instance
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

    def test_transitivity_has_positive_groundings(self):
        import copy

        instance = copy.deepcopy(SAMPLE_INSTANCE)
        instance["dataset"] = "matres"
        dataset, ctx, program = self._build(
            ["--no-constraint-gradient-check"],
            instances=[instance],
            pair_selection="all",
            max_pairs_per_instance=None,
        )
        datanode = next(iter(program.populate([dataset[0]], device="cpu")))
        datanode.inferLocal(keys=("softmax",))
        losses = datanode.calculateLcLoss(tnorm="P")
        key = next(
            lc.lcName for _key, lc in ctx.graph.logicalConstrainsRecursive
            if getattr(lc, "name", None) == "temporal_before_transitive"
        )
        tensor = losses[key]["lossTensor"]
        self.assertIsNotNone(tensor)
        self.assertGreater(tensor.numel(), 0)

    def test_constraint_loss_reaches_model_parameters(self):
        """closs must be differentiable *and* touch a learnable weight."""
        import test_regr.TemporalRelation.program_qwen_train as P
        dataset, _ctx, program = self._build(
            ["--no-transitivity", "--constraint-epochs", "1"])
        args = type("A", (), {"constraint_epochs": 1})()
        reached = P.verify_constraint_gradient_flow(program, dataset, args)
        self.assertIsNotNone(reached)
        self.assertGreater(reached, 0)

    def test_epoch_checkpoint_round_trips_trainable_and_session_state(self):
        import types
        import test_regr.TemporalRelation.program_qwen_train as P

        dataset, _ctx, program = self._build(
            ["--no-transitivity", "--no-constraint-gradient-check"])
        program.opt = torch.optim.AdamW(program.model.parameters(), lr=1e-3)
        trainable = {
            name: parameter.detach().clone()
            for name, parameter in program.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)

        with TemporaryDirectory() as directory:
            args = types.SimpleNamespace(
                output=Path(directory) / "temporal.pt",
                device="cpu",
                _active_dataset_names={"matres"},
                training_style="simple",
                model_path="stub-model",
            )
            path = P.save_training_checkpoint(
                program, args, completed_epoch=1, phase="warmup",
                c_session={"iter": len(dataset)}, warmup_epochs=1,
                constraint_epochs=1)
            self.assertTrue(path.is_file())
            self.assertLess(path.stat().st_size, 1_000_000)

            with torch.no_grad():
                for parameter in program.model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(10.0)
            args.tnorm = "G"
            with self.assertRaisesRegex(ValueError, "configuration conflicts"):
                P.load_training_checkpoint(
                    path, program, args, warmup_epochs=1,
                    constraint_epochs=1)
            args.tnorm = None
            restored = P.load_training_checkpoint(
                path, program, args, warmup_epochs=1,
                constraint_epochs=1)

        self.assertEqual(restored["completed_epoch"], 1)
        self.assertEqual(restored["c_session"], {"iter": len(dataset)})
        for name, parameter in program.model.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.equal(parameter, trainable[name]))

    def test_phased_resume_skips_completed_epoch_and_calls_checkpoint_hook(self):
        dataset, _ctx, program = self._build(
            ["--no-transitivity", "--no-constraint-gradient-check"])
        program.opt = torch.optim.AdamW(program.model.parameters(), lr=1e-3)
        completed = []

        program.train(
            [dataset[0]], warmup_epochs=1, constraint_epochs=1,
            start_epoch=1, resume_c_session={"iter": 1},
            epoch_end_callback=(
                lambda _program, epoch, phase, session:
                completed.append((epoch, phase, dict(session)))
            ),
        )

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][0:2], (2, "constraint"))
        self.assertGreater(completed[0][2]["iter"], 1)

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

    def test_dev_split_constraints_are_evaluated(self):
        """A split compiled after construction must still reach the metrics.

        ``poi`` is snapshotted when the program is built, so the dev split's
        executable-constraint properties are missing from it and every dev
        constraint counter silently reads zero — while the same evaluation on
        the training split reports real numbers.
        """
        import test_regr.TemporalRelation.program_qwen_train as P

        train_data, ctx, program = self._build(["--no-transitivity"])
        instances = P.expand_document_query_instances(
            load_temporal_instances(
                Path(__file__).parent / "data" / "MATRES" / "timebank.txt"))[3:6]
        dev_data = P.compile_program_train_dataset(
            instances, ctx, device="cpu", max_events_per_instance=8,
            pair_selection="target", max_pairs_per_instance=2)

        before = program.evaluate_condition(dev_data, device="cpu", return_dict=True)
        self.assertEqual(before["query_total"], 0, "expected the stale-POI symptom")

        added = P.refresh_constraint_poi(program, ctx)
        self.assertEqual(added, len(instances))

        after = program.evaluate_condition(dev_data, device="cpu", return_dict=True)
        self.assertEqual(after["query_total"], len(dev_data))

    def test_matres_label_space_and_rules_are_unchanged(self):
        """The extended vocabulary must be opt-in, not the new default."""
        from test_regr.TemporalRelation.graph import MATRES_LABELS, TEMPORAL_LABELS

        ctx = create_temporal_graph()
        self.assertEqual(tuple(TEMPORAL_LABELS), MATRES_LABELS)
        rules = [lc.name for _k, lc in ctx.graph.logicalConstrainsRecursive
                 if getattr(lc, "headLC", False)]
        self.assertEqual(len(rules), 6)

    def test_extended_label_space_adds_containment_rules(self):
        from test_regr.TemporalRelation.graph import (
            EXTENDED_LABELS, MATRES_LABELS, TEMPORAL_LABELS)

        base = {lc.name for _k, lc in create_temporal_graph().graph.logicalConstrainsRecursive
                if getattr(lc, "headLC", False)}
        ctx = create_temporal_graph(labels=EXTENDED_LABELS)
        try:
            self.assertEqual(len(TEMPORAL_LABELS), 7)
            extended = {lc.name for _k, lc in ctx.graph.logicalConstrainsRecursive
                        if getattr(lc, "headLC", False)}
            added = extended - base
            self.assertIn("temporal_includes_inverse_is_included", added)
            self.assertIn("temporal_simultaneous_symmetric", added)
            self.assertIn("temporal_includes_transitive", added)
            # Allen composition is disjunctive in general; the ambiguous entry
            # Includes o Before must NOT be encoded as a definite implication.
            self.assertNotIn("temporal_includes_before_is_before", added)
        finally:
            create_temporal_graph(labels=MATRES_LABELS)

    def test_simultaneous_is_not_collapsed_into_equal(self):
        """The old normaliser folded SIMULTANEOUS into Equal, losing the relation."""
        from test_regr.TemporalRelation.dataset import _normalize_label

        self.assertEqual(_normalize_label("SIMULTANEOUS"), "Simultaneous")
        self.assertEqual(_normalize_label("EQUAL"), "Equal")
        self.assertEqual(_normalize_label("IS_INCLUDED"), "IsIncluded")
        self.assertEqual(_normalize_label("i"), "Includes")
        self.assertEqual(_normalize_label("NONE"), "Vague")
        with self.assertRaises(ValueError):
            _normalize_label("OVERLAP")

    def test_matres_files_use_only_matres_relations(self):
        """Guards the assumption that dropping SIMULTANEOUS->Equal is safe."""
        from test_regr.TemporalRelation.dataset import _normalize_label

        path = Path(__file__).parent / "data" / "MATRES" / "timebank.txt"
        seen = set()
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) == 6:
                    seen.add(_normalize_label(fields[5]))
        self.assertEqual(seen, {"Before", "After", "Equal", "Vague"})

    def test_legal_label_mask_blocks_out_of_corpus_relations(self):
        """A MATRES row must never be trainable toward a TB-Dense-only relation."""
        import test_regr.TemporalRelation.program_qwen_train as P
        from test_regr.TemporalRelation.graph import EXTENDED_LABELS

        mask = P.legal_label_mask("matres", EXTENDED_LABELS)
        illegal = [label for label, ok in zip(EXTENDED_LABELS, mask.tolist()) if not ok]
        self.assertEqual(set(illegal), {"Includes", "IsIncluded", "Simultaneous"})

        loss = P.WeightedTemporalCrossEntropyLoss(None, dataset_mask=mask)
        logits = torch.zeros(1, len(EXTENDED_LABELS))
        target = torch.tensor([0])
        value = loss(logits, target)
        self.assertTrue(torch.isfinite(value))
        # softmax over the masked logits gives the illegal classes zero mass
        masked = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(masked, dim=-1)
        self.assertAlmostEqual(float(probs[0, EXTENDED_LABELS.index("Includes")]), 0.0)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

        trainable = torch.zeros(1, len(EXTENDED_LABELS), requires_grad=True)
        P.WeightedTemporalCrossEntropyLoss(None, dataset_mask=mask)(
            trainable, target).backward()
        self.assertEqual(
            float(trainable.grad[0, EXTENDED_LABELS.index("Includes")]), 0.0)

    def test_tbdense_converter_emits_loader_ready_rows(self):
        from test_regr.TemporalRelation.convert_tbdense import convert_rows
        from test_regr.TemporalRelation.dataset import _normalize_grouped_rows

        source = [
            ["doc1", "e1", "e2", "b"],
            ["doc1", "e2", "e3", "i"],
            ["doc1", "e1", "e3", "s"],
        ]
        rows = list(convert_rows(iter(source)))
        self.assertEqual([row["label"] for row in rows],
                         ["Before", "Includes", "Simultaneous"])
        self.assertTrue(all(row["dataset"] == "tbdense" for row in rows))

        # the existing loader consumes them with no new parser
        instances = _normalize_grouped_rows(rows)
        self.assertEqual(len(instances), 1)
        self.assertEqual(len(instances[0]["event_pairs"]), 3)
        self.assertEqual(instances[0]["dataset"], "tbdense")

    def test_mixed_corpus_rows_receive_different_masks(self):
        import copy
        import test_regr.TemporalRelation.program_qwen_train as P
        from test_regr.TemporalRelation.graph import EXTENDED_LABELS

        matres = copy.deepcopy(SAMPLE_INSTANCE)
        matres["dataset"] = "matres"
        tbdense = copy.deepcopy(SAMPLE_INSTANCE)
        tbdense["doc_id"] = "dense_doc"
        tbdense["dataset"] = "tbdense"
        tbdense["event_pairs"][0]["label"] = "Includes"
        tbdense["query_pair"] = {
            "e1": "packed_bag", "e2": "left_house", "label": "Includes"}

        labels, names = P._activate_labels_for_instances([matres, tbdense])
        self.assertEqual(labels, EXTENDED_LABELS)
        self.assertEqual(names, {"matres", "tbdense"})
        ctx = create_temporal_graph(labels=labels)
        rows = P.compile_program_train_dataset(
            [matres, tbdense], ctx, device="cpu",
            pair_selection="target", max_pairs_per_instance=2)
        matres_mask = rows[0]["dataset_mask"][0]
        dense_mask = rows[1]["dataset_mask"][0]
        self.assertFalse(bool(matres_mask[EXTENDED_LABELS.index("Includes")]))
        self.assertTrue(bool(dense_mask[EXTENDED_LABELS.index("Includes")]))
        self.assertTrue(bool(matres_mask[EXTENDED_LABELS.index("Equal")]))
        self.assertFalse(bool(dense_mask[EXTENDED_LABELS.index("Equal")]))

    def test_dataset_cli_alignment_and_loader_conflict_are_rejected(self):
        import test_regr.TemporalRelation.program_qwen_train as P

        with self.assertRaisesRegex(ValueError, "supplied 1 value"):
            P._parse_dataset_names("matres", expected=2)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({
                "doc_id": "d", "e1": "e1", "e2": "e2",
                "label": "Before", "dataset": "matres",
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicts"):
                load_temporal_instances(path, dataset_name="tbdense")

    def test_mixed_corpus_program_masks_logits_before_all_consumers(self):
        import copy
        from test_regr.TemporalRelation.graph import EXTENDED_LABELS

        matres = copy.deepcopy(SAMPLE_INSTANCE)
        matres["dataset"] = "matres"
        tbdense = copy.deepcopy(SAMPLE_INSTANCE)
        tbdense["doc_id"] = "dense_doc"
        tbdense["dataset"] = "tbdense"
        tbdense["event_pairs"][0]["label"] = "Includes"
        tbdense["query_pair"] = {
            "e1": "packed_bag", "e2": "left_house", "label": "Includes"}

        dataset, ctx, program = self._build(
            ["--no-transitivity", "--no-constraint-gradient-check"],
            instances=[matres, tbdense],
        )
        illegal = {
            "matres": ["Includes", "IsIncluded", "Simultaneous"],
            "tbdense": ["Equal"],
        }
        for row, dataset_name in zip(dataset, ("matres", "tbdense")):
            _mloss, _metric, *output = program.model(row)
            constraint_loss, *_ = program.cmodel(output[1])
            self.assertTrue(torch.isfinite(constraint_loss))
            logits = ctx.event_pair[ctx.temporal_relation](row)
            for label in illegal[dataset_name]:
                self.assertTrue(torch.isneginf(
                    logits[:, EXTENDED_LABELS.index(label)]).all())

    def test_splitLossColumns_tolerates_an_empty_variable(self):
        """An ungroundable variable must not raise (pre-existing IndexError)."""
        from domiknows.solver.logicalConstraintConstructor import (
            LogicalConstraintConstructor)
        self.assertEqual(LogicalConstraintConstructor.splitLossColumns([]), [])


if __name__ == "__main__":
    unittest.main()
