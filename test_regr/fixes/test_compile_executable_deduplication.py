import torch

from domiknows.graph import Concept, Graph
from domiknows.graph.executable import LogicDataset

from test_regr.tiny_dynamic_graph.example import reset_domiknows_state


def _compile_duplicate_rows(deduplicate):
    reset_domiknows_state()
    with Graph("deduplicated_executable_formulas") as graph:
        obj = Concept(name="object")
        red = obj(name="red")
        candidate = obj(name="candidate_0")

    rows = [
        {
            "logic_str": 'existsL(andL(red("o"), candidate_0(path="o")))',
            "logic_label": torch.tensor([0]),
        },
        {
            "logic_str": ('existsL( andL( red("o"), candidate_0(path = "o") ) )'),
            "logic_label": torch.tensor([1]),
        },
    ]
    dataset = graph.compile_executable(
        rows,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values={"red": red, "candidate_0": candidate},
        deduplicate=deduplicate,
    )
    return graph, dataset


def test_deduplication_reuses_canonical_formula_and_keeps_runtime_labels():
    graph, dataset = _compile_duplicate_rows(deduplicate=True)

    assert len(graph.executableLCs) == 1
    assert dataset.deduplicated is True
    assert dataset.unique_constraint_count == 1
    assert dataset.reused_constraint_count == 1
    assert dataset.lc_name_list[0] == dataset.lc_name_list[1]

    name = dataset.lc_name_list[0]
    label_key = LogicDataset.KEYWORD_FMT.format(lc_name=name)
    assert dataset[0][LogicDataset.curr_lc_key] == name
    assert dataset[1][LogicDataset.curr_lc_key] == name
    assert dataset[0][label_key].item() == 0
    assert dataset[1][label_key].item() == 1
    assert graph.executableLCsLabels[name].item() == 0


def test_distinct_per_row_identity_remains_the_compatibility_default():
    graph, dataset = _compile_duplicate_rows(deduplicate=False)

    assert len(graph.executableLCs) == 2
    assert dataset.deduplicated is False
    assert dataset.unique_constraint_count == 2
    assert dataset.reused_constraint_count == 0
    assert dataset.lc_name_list[0] != dataset.lc_name_list[1]


def test_parameterization_reuses_concept_slots_and_alpha_variable_literals():
    reset_domiknows_state()
    with Graph("parameterized_executable_formulas") as graph:
        obj = Concept(name="object")
        red = obj(name="red")
        blue = obj(name="blue")
        candidate_0 = obj(name="candidate_0")
        candidate_1 = obj(name="candidate_1")

    rows = [
        {
            "logic_str": 'existsL(andL(red("x"), candidate_0(path="x")))',
            "logic_label": torch.tensor([0]),
        },
        {
            "logic_str": 'existsL(andL(blue("y"), candidate_1(path="y")))',
            "logic_label": torch.tensor([1]),
        },
        {
            # The different equality pattern is not alpha-equivalent: "y"
            # and "z" denote distinct logical variables.
            "logic_str": 'existsL(andL(blue("y"), candidate_1(path="z")))',
            "logic_label": torch.tensor([0]),
        },
    ]
    dataset = graph.compile_executable(
        rows,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values={
            "red": red,
            "blue": blue,
            "candidate_0": candidate_0,
            "candidate_1": candidate_1,
        },
        parameterize=True,
    )

    assert dataset.parameterized is True
    assert dataset.unique_constraint_count == 2
    assert dataset.reused_constraint_count == 1
    assert dataset.lc_name_list[0] == dataset.lc_name_list[1]
    assert dataset.lc_name_list[2] != dataset.lc_name_list[0]

    template_name = dataset.lc_name_list[0]
    first_binding = dataset[0][LogicDataset.BINDINGS_KEY][template_name][0]
    second_binding = dataset[1][LogicDataset.BINDINGS_KEY][template_name][0]
    assert tuple(concept.name for concept in first_binding) == (
        "red",
        "candidate_0",
    )
    assert tuple(concept.name for concept in second_binding) == (
        "blue",
        "candidate_1",
    )
