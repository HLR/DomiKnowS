import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import EOS_TOKEN, dummy_dataset, load_eai_dataset
from reward import (
    TokenVocabulary,
    abstract_state_from_tokens,
    evaluate_goal_satisfaction,
    make_eai_reward_function,
    state_recall,
)


def _sample(tl_goal, target_tokens, task_id="synthetic"):
    objects = tuple(
        token for token in target_tokens
        if token not in {EOS_TOKEN, "grab", "read", "right_grasp", "right_place_inside", "standup", "switchon", "putback"}
    )
    return {
        "task_id": task_id,
        "tl_goal": tl_goal,
        "target_action_tokens": target_tokens,
        "object_tokens": objects,
    }


def test_goal_is_read_from_tl_not_demonstration_side_effects():
    example = dummy_dataset(device="cpu", max_steps=8)[0]
    vocabulary = TokenVocabulary(example["generation_vocab"], eos_token=EOS_TOKEN)
    # The demonstration opens a cabinet first, but the TL goal only requires the
    # light to be on. A shorter valid plan must receive full reward.
    result = evaluate_goal_satisfaction(["switchon", "light_1", EOS_TOKEN], example, vocabulary)
    assert result["is_success"] == 1.0
    assert result["gold_state"] == {("on", "light_1")}


def test_wrong_relation_source_is_rejected():
    example = _sample(
        "inside(apple.1, fridge.1)",
        ["right_grasp", "apple_1", "right_place_inside", "fridge_1", EOS_TOKEN],
    )
    wrong = ["right_grasp", "banana_1", "right_place_inside", "fridge_1", EOS_TOKEN]
    result = evaluate_goal_satisfaction(wrong, example)
    assert result["is_success"] == 0.0
    assert result["recall"] == 0.0
    assert state_recall({("inside", "banana_1", "fridge_1")}, {("inside", "apple_1", "fridge_1")}) == 0.0


def test_empty_plan_cannot_satisfy_nonempty_goal():
    example = _sample(
        "ON(light.245) and PLUGGED_IN(light.245)",
        ["standup", "switchon", "light_245", EOS_TOKEN],
    )
    assert evaluate_goal_satisfaction([EOS_TOKEN], example)["is_success"] == 0.0
    assert evaluate_goal_satisfaction(example["target_action_tokens"], example)["is_success"] == 1.0


def test_objectless_actions_do_not_shift_following_actions():
    state = abstract_state_from_tokens(["standup", "walk", "kitchen_1", "switchon", "light_245", EOS_TOKEN])
    assert ("near", "kitchen_1") in state
    assert ("on", "light_245") in state


def test_virtualhome_putback_means_ontop():
    example = _sample(
        "ONTOP(soap.1002, washing_machine.1001)",
        ["grab", "soap_1002", "putback", "washing_machine_1001", EOS_TOKEN],
    )
    result = evaluate_goal_satisfaction(example["target_action_tokens"], example)
    assert result["is_success"] == 1.0
    assert ("ontop", "soap_1002", "washing_machine_1001") in result["predicted_state"]
    assert ("inside", "soap_1002", "washing_machine_1001") not in result["predicted_state"]


def test_then_requires_action_order():
    example = _sample(
        "(exists x0. (GRAB(x0))) then exists x0. (READ(x0))",
        ["grab", "novel_1000", "read", "novel_1000", EOS_TOKEN],
    )
    assert evaluate_goal_satisfaction(example["target_action_tokens"], example)["is_success"] == 1.0
    reversed_plan = ["read", "novel_1000", "grab", "novel_1000", EOS_TOKEN]
    assert evaluate_goal_satisfaction(reversed_plan, example)["is_success"] == 0.0


def test_reward_closure_tensor_contract():
    example = _sample("closed(fridge.1)", ["close", "fridge_1", EOS_TOKEN])
    reward = make_eai_reward_function(example)(example["target_action_tokens"])
    assert reward.shape == torch.Size([1])
    assert reward.dtype == torch.float32
    assert reward.item() == 1.0


def test_dummy_reference_plans_pass_and_empty_plans_fail():
    examples = dummy_dataset(device="cpu", max_steps=8)
    vocabulary = TokenVocabulary(examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
    for example in examples:
        assert evaluate_goal_satisfaction(
            example["target_action_tokens"], example, vocabulary
        )["is_success"] == 1.0
        assert evaluate_goal_satisfaction(
            [vocabulary.eos_label], example, vocabulary
        )["is_success"] == 0.0


def run_tests():
    print("=== Testing EAI Reward and Goal Satisfaction Module ===")
    focused = [
        test_goal_is_read_from_tl_not_demonstration_side_effects,
        test_wrong_relation_source_is_rejected,
        test_empty_plan_cannot_satisfy_nonempty_goal,
        test_objectless_actions_do_not_shift_following_actions,
        test_virtualhome_putback_means_ontop,
        test_then_requires_action_order,
        test_reward_closure_tensor_contract,
        test_dummy_reference_plans_pass_and_empty_plans_fail,
    ]
    for test in focused:
        test()
    print(f"Focused adversarial tests: {len(focused)}/{len(focused)} passed")

    print("\nTesting all EAI reference trajectories against their temporal goals:")
    examples = load_eai_dataset("all", limit=None, max_steps=135, device="cpu")
    vocabulary = TokenVocabulary(examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
    successes = 0
    total_facts = 0
    for example in examples:
        result = evaluate_goal_satisfaction(example["target_action_tokens"], example, vocabulary)
        assert result["parse_error"] is None, f"TL evaluation failed for {example['task_id']}: {result['parse_error']}"
        successes += int(result["is_success"] == 1.0)
        total_facts += len(result["gold_state"])
        empty_result = evaluate_goal_satisfaction([vocabulary.eos_label], example, vocabulary)
        assert empty_result["is_success"] == 0.0, f"Empty plan incorrectly satisfied {example['task_id']}"

    total = len(examples)
    assert total == 438, f"Expected the complete 438-example benchmark, got {total}"
    assert successes == total, f"Reference temporal-goal success was {successes}/{total}"
    print(f"Reference goal success: {successes}/{total} (100.0%)")
    print(f"Average grounded goal facts: {total_facts / total:.2f}")
    print("\nALL REWARD AND TEMPORAL-GOAL TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
