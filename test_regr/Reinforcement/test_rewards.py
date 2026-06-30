import torch

from domiknows.reinforcement import (
    ReinforcementProgram,
    binary_label,
    binary_label_name,
    binary_match_reward,
    coerce_label_tensor,
    count_reward,
    flatten_generator_output,
    normalize_text,
)


def test_flatten_generator_output_common_shapes():
    nested = {
        "predictions": [
            "yes",
            torch.tensor([0, 1]),
            ("no", {"answer": torch.tensor(1)}),
        ]
    }

    assert flatten_generator_output(nested) == ["yes", 0, 1, "no", 1]
    assert flatten_generator_output(torch.tensor([[1, 2], [3, 4]])) == [1, 2, 3, 4]
    assert flatten_generator_output("yes") == ["yes"]


def test_label_normalization_and_coercion():
    assert normalize_text(" Yes!! ") == "yes"
    assert binary_label("true") == 1
    assert binary_label("zero") == 0
    assert binary_label(torch.tensor([1])) == 1
    assert binary_label_name("true") == "yes"
    assert binary_label_name("0") == "no"
    assert binary_label_name("positive", true_label="Y", false_label="N") == "Y"
    assert binary_label_name("unknown", default="missing") == "missing"
    assert coerce_label_tensor("yes", 3).tolist() == [1.0, 1.0, 1.0]
    assert coerce_label_tensor(["no", "yes"], 2).tolist() == [0.0, 1.0]


def test_binary_match_reward_matches_easy_and_hard_behavior():
    assert binary_match_reward(["yes", "true", "1"], 1).tolist() == [1.0, 1.0, 1.0]
    assert binary_match_reward(["no", "false", "0"], 0).tolist() == [1.0, 1.0, 1.0]
    assert binary_match_reward(["yes", "no"], [1, 0]).tolist() == [1.0, 1.0]
    assert binary_match_reward(["yes", "no"], [0, 1]).tolist() == [0.0, 0.0]


def test_count_reward_modes():
    outputs = ["zero", "one", 0, "yes", "zero"]

    assert count_reward(outputs, "zero", 3, mode="exact").tolist() == [1.0]
    assert count_reward(outputs, "zero", 2, mode="at_least").tolist() == [1.0]
    assert count_reward(outputs, "zero", 2, mode="at_most").tolist() == [0.0]
    assert count_reward(outputs, "one", 2, mode="exact").tolist() == [1.0]


def test_reinforcement_program_calls_old_style_reward_function():
    program = object.__new__(ReinforcementProgram)
    program.decoder = lambda samples, targets, datanode, data_item: ["yes"]
    program.reward_from_constraints = False

    reward = program._sample_reward(
        samples={},
        present_targets=[],
        datanode=object(),
        data_item={"logic_label": "yes"},
        reward_fn=lambda generator_output: binary_match_reward(generator_output, "yes"),
    )

    assert reward == 1.0


def test_reinforcement_program_calls_context_aware_reward_function():
    program = object.__new__(ReinforcementProgram)
    program.decoder = lambda samples, targets, datanode, data_item: data_item["generated"]
    program.reward_from_constraints = False
    seen = {}

    def reward_fn(generator_output, *, data_item=None, datanode=None, samples=None, targets=None):
        seen["generator_output"] = generator_output
        seen["data_item"] = data_item
        seen["datanode"] = datanode
        seen["samples"] = samples
        seen["targets"] = targets
        return torch.tensor([0.25, 0.75])

    datanode = object()
    samples = {"answer": torch.tensor([1])}
    targets = ["answer"]
    data_item = {"generated": "yes"}
    reward = program._sample_reward(samples, targets, datanode, data_item, reward_fn)

    assert reward == 0.5
    assert seen["generator_output"] == "yes"
    assert seen["data_item"] is data_item
    assert seen["datanode"] is datanode
    assert seen["samples"] is samples
    assert seen["targets"] is targets
