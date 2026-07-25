import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from .dataset import (
    GraphQADatasetNotFound,
    discover_vqar_dataset,
    require_vqar_dataset,
    vqar_task_to_graphqa_instance,
)
from .execution import (
    assert_convertible,
    compile_graphqa_dataset,
    create_candidate_membership_instance,
    create_candidate_membership_logic,
    create_executable_instance,
    create_query_logic,
    materialize_bounded_facts,
    validate_dataset_convertible,
)
from .graph import collect_object_relations, create_graphqa_graph
from .oracle import answer_object, check_oracle


SAMPLE_INSTANCE = {
    "objects": ["o1", "o2"],
    "symbols": ["dog", "mammal", "animal", "white"],
    "visual_facts": [
        ("Name", "o1", "dog"),
        ("Attribute", "o1", "white"),
        ("LeftOf", "o1", "o2"),
    ],
    "kb_facts": [
        ("TypeOf", "dog", "mammal"),
        ("TypeOf", "mammal", "animal"),
    ],
    "query": {
        "target_type": "animal",
        "conditions": [
            ("Attribute", "o", "white"),
            ("LeftOf", "o", "o2"),
        ],
        "answer_type": "object",
    },
    "expected_answer": "o1",
}


MULTI_RELATION_DATASET = [
    SAMPLE_INSTANCE,
    {
        "objects": ["o1", "o2", "o3"],
        "symbols": ["dog", "mammal", "animal", "white", "black"],
        "visual_facts": [
            ("Name", "o2", "dog"),
            ("Attribute", "o2", "black"),
            ("RightOf", "o2", "o1"),
            ("Above", "o2", "o3"),
            ("Below", "o3", "o2"),
            ("FrontOf", "o2", "o3"),
            ("Behind", "o3", "o2"),
        ],
        "kb_facts": [
            ("TypeOf", "dog", "mammal"),
            ("TypeOf", "mammal", "animal"),
        ],
        "query": {
            "target_type": "animal",
            "conditions": [
                ("Attribute", "o", "black"),
                ("RightOf", "o", "o1"),
                ("Above", "o", "o3"),
                ("FrontOf", "o", "o3"),
            ],
            "answer_type": "object",
        },
        "expected_answer": "o2",
    },
]


class TestGraphQAAdapter(unittest.TestCase):
    def test_bounded_type_of_propagation_materializes_object_category(self):
        facts = set(materialize_bounded_facts(SAMPLE_INSTANCE))

        self.assertIn(("ObjectType", "o1", "mammal"), facts)
        self.assertIn(("ObjectCategory", "o1", "animal"), facts)
        self.assertNotIn(("ObjectCategory", "o1", "mammal"), facts)

    def test_graph_defines_all_dataset_object_relations_like_clevr(self):
        context = create_graphqa_graph(MULTI_RELATION_DATASET)

        self.assertEqual(
            collect_object_relations(MULTI_RELATION_DATASET),
            ["Above", "Behind", "Below", "FrontOf", "LeftOf", "RightOf"],
        )
        for relation in ["Above", "Behind", "Below", "FrontOf", "LeftOf", "RightOf"]:
            self.assertIn(relation, context.object_relations)
            self.assertIn(relation, context.namespace)

    def test_graphqa_query_is_convertible_to_domiknows_execute_logic(self):
        context = create_graphqa_graph(SAMPLE_INSTANCE)
        executable = create_executable_instance(
            SAMPLE_INSTANCE,
            answer_label_space=[str(value) for value in context.object_values],
        )

        self.assertTrue(assert_convertible(SAMPLE_INSTANCE))
        self.assertEqual(create_query_logic(SAMPLE_INSTANCE), executable["logic_str"])
        self.assertIn("queryL", executable["logic_str"])
        self.assertIn("iotaL", executable["logic_str"])
        self.assertEqual(executable["logic_label"], 0)

        compiled = compile_graphqa_dataset([SAMPLE_INSTANCE], context)
        self.assertEqual(len(compiled), 1)

    def test_all_questions_in_dataset_compile_to_domiknows_execute_logic(self):
        context = create_graphqa_graph(MULTI_RELATION_DATASET)

        self.assertEqual(validate_dataset_convertible(MULTI_RELATION_DATASET), [])
        compiled = compile_graphqa_dataset(MULTI_RELATION_DATASET, context)
        self.assertEqual(len(compiled), len(MULTI_RELATION_DATASET))

    def test_oracle_answers_expected_objects(self):
        self.assertEqual(answer_object(SAMPLE_INSTANCE), "o1")
        self.assertTrue(check_oracle(SAMPLE_INSTANCE, "o1"))
        self.assertEqual(answer_object(MULTI_RELATION_DATASET[1]), "o2")

    def test_set_answer_candidate_membership_logic_compiles(self):
        multi_answer = {
            "objects": ["o1", "o2", "o3"],
            "symbols": ["dog"],
            "visual_facts": [
                ("Name", "o1", "dog"),
                ("Name", "o2", "dog"),
            ],
            "kb_facts": [],
            "query": {
                "target_type": "__any_object__",
                "conditions": [("Name", "o", "dog")],
                "answer_type": "object",
            },
            "expected_answer": None,
            "expected_answers": ["o1", "o2"],
        }
        context = create_graphqa_graph(multi_answer)
        executable_rows = [
            create_candidate_membership_instance(multi_answer, "o1", True),
            create_candidate_membership_instance(multi_answer, "o2", True),
            create_candidate_membership_instance(multi_answer, "o3", False),
        ]

        self.assertIn("existsL", create_candidate_membership_logic(multi_answer, "o1"))
        compiled = context.graph.compile_executable(
            executable_rows,
            logic_keyword="logic_str",
            logic_label_keyword="logic_label",
            extra_namespace_values=context.namespace,
        )
        self.assertEqual(len(compiled), 3)

    def test_converter_rejects_unbounded_or_unsupported_shapes(self):
        unsupported = {
            **SAMPLE_INSTANCE,
            "query": {**SAMPLE_INSTANCE["query"], "answer_type": "count"},
        }

        with self.assertRaises(ValueError):
            create_query_logic(unsupported)

    def test_vqar_dataset_discovery_reports_missing_optional_data(self):
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            discovered = discover_vqar_dataset(dataset_root)

            self.assertEqual(discovered["root"], dataset_root)
            self.assertEqual(discovered["data_dir"], dataset_root / "data")
            self.assertEqual(discovered["task_paths"], [])
            with self.assertRaises(GraphQADatasetNotFound):
                require_vqar_dataset(dataset_root)

    def test_vqar_task_can_convert_to_graphqa_instance(self):
        task = {
            "image_id": "img1",
            "scene_graph": {
                "names": {"1": "dog", "2": "cat"},
                "attributes": {"1": ["white"], "2": ["black"]},
                "relations": {"1": {"2": ["left of"]}},
            },
            "question": {
                "question_id": "q1",
                "input": ["1", "2"],
                "output": ["1"],
                "clauses": [
                    {"function": "Initial"},
                    {"function": "Hypernym_Find", "text_input": "animal"},
                    {"function": "Find_Attr", "text_input": "white"},
                ],
            },
        }

        instance = vqar_task_to_graphqa_instance(
            task,
            kb_facts=[("TypeOf", "dog", "mammal"), ("TypeOf", "mammal", "animal")],
        )

        instance["expected_answer"] = "1"

        self.assertEqual(instance["expected_answers"], ["1"])
        self.assertEqual(instance["query"]["target_type"], "__any_object__")
        self.assertIn(("SemanticClass", "o", "animal"), instance["query"]["conditions"])
        self.assertIn(("Attribute", "o", "white"), instance["query"]["conditions"])
        self.assertTrue(assert_convertible(instance))
        self.assertEqual(answer_object(instance), "1")

    def test_vqar_task_preserves_multi_answer_outputs(self):
        task = {
            "image_id": "img1",
            "scene_graph": {
                "names": {"1": "dog", "2": "dog"},
                "attributes": {"1": ["white"], "2": ["black"]},
                "relations": {},
            },
            "question": {
                "question_id": "q_multi",
                "input": ["1", "2"],
                "output": ["1", "2"],
                "clauses": [
                    {"function": "Initial"},
                    {"function": "Find_Name", "text_input": "dog"},
                ],
            },
        }

        instance = vqar_task_to_graphqa_instance(task)

        self.assertIsNone(instance["expected_answer"])
        self.assertEqual(instance["expected_answers"], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
