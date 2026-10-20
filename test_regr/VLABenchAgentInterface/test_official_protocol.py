import importlib.util
import json
import random
from pathlib import Path

import pytest

from test_regr.VLABenchAgentInterface import official_protocol as op
from test_regr.VLABenchAgentInterface.planning_metrics import BENCHMARK_PATTERNS, PlanningMetrics

UPSTREAM_SCORER = Path(__file__).parent / "outputs/native_refined_20261014/results_full/official_scoring_source.py"


def make_episode(root, task, number, instruction, plan, dimension="M&T"):
    folder = root / dimension / task / f"example{number}"
    (folder / "input").mkdir(parents=True)
    (folder / "output").mkdir()
    for name in ("input.png", "input_mask.png"):
        (folder / "input" / name).write_bytes(b"png")
    (folder / "input" / "instruction.txt").write_text(instruction, encoding="utf-8")
    (folder / "output" / "operation_sequence.json").write_text(json.dumps({"skill_sequence": plan}), encoding="utf-8")


def pick_place(entity, container):
    return [{"name": "pick", "params": {"target_entity_name": entity}},
            {"name": "place", "params": {"target_container_name": container}}]


@pytest.fixture
def dataset(tmp_path):
    for number in range(3):
        make_episode(tmp_path, "select_fruit", number, f"fruit {number}", pick_place(number, 9))
    make_episode(tmp_path, "select_billiards", 0, "ball", pick_place(1, 2))  # not in upstream dim2task
    make_episode(tmp_path, "cook_dishes", 0, "cook", pick_place(1, 2) * 2, "Complex")
    return tmp_path


def test_vendored_prompt_matches_the_pinned_upstream_hash():
    assert "Target Number" in op.load_official_prompt()


def test_loader_uses_directory_task_and_upstream_dimensions(dataset):
    examples = op.load_official_examples(dataset)
    by_task = {example.task: example for example in examples}
    assert len(examples) == 5
    assert by_task["select_fruit"].dimension == "M&T"
    assert by_task["cook_dishes"].dimension == "Complex"
    assert by_task["select_billiards"].dimension is None
    assert [e.task for e in op.load_official_examples(dataset, tasks=["cook_dishes"])] == ["cook_dishes"]
    assert op.official_dependency("cook_dishes") == "Seq-independent"
    assert op.official_dependency("select_fruit") == "Sequential"


def test_ti_list_follows_the_upstream_layout(dataset):
    examples = op.load_official_examples(dataset)
    target = examples[0]
    plain = op.build_ti_list("PROMPT", target)
    assert plain[0] == ["text", "PROMPT"]
    assert [kind for kind, _ in plain] == ["text", "text", "image", "text", "image", "text", "text", "text"]
    assert plain[-1] == ["text", "Please give the output skill sequence"]
    shots = op.sample_few_shots(examples, target, 1, seed=3)
    shot = op.build_ti_list("PROMPT", target, shots, with_cot=True)
    assert shot[1] == ["text", "Example 0 input picture:"]
    assert shot[8] == ["text", json.dumps(shots[0].reference_payload)]  # whole stored JSON, as upstream prints it
    assert shot[-1][1].startswith("Please analyze the problem step by step")
    assert not any("select_fruit" in str(value) or "ketchup" in str(value) for _, value in plain if _ == "text")


def test_few_shot_sampling_is_reproducible_and_excludes_the_target(dataset):
    examples = op.load_official_examples(dataset)
    target = examples[0]
    first = op.sample_few_shots(examples, target, 4, seed=7)
    assert [s.episode_id for s in first] == [s.episode_id for s in op.sample_few_shots(examples, target, 4, seed=7)]
    assert all(s.episode_id != target.episode_id for s in first)
    assert op.sample_few_shots(examples, target, 0, seed=7) == []


def test_answer_parsing_requires_a_json_fence():
    assert op.parse_official_output('text ```json\n[{"name": "press", "params": {}}]\n``` more') == (True, [{"name": "press", "params": {}}])
    assert op.parse_official_output('[{"name": "press", "params": {}}]') == (False, None)
    assert op.parse_official_output("```json\nnot json\n```") == (False, None)


def test_name_spelling_is_not_repaired():
    reference = pick_place(1, 2)
    exact = op.score_official_answer("select_fruit", reference, True, reference)
    assert exact["legacy_score"] == 80  # upstream total_score omits SPR and PM
    wrong_case = [{"name": "Pick", "params": {"target_entity_name": 1}}, {"name": "place", "params": {"target_container_name": 2}}]
    assert op.score_official_answer("select_fruit", reference, True, wrong_case)["SR"] == 50
    assert op.score_official_answer("select_fruit", reference, False, None) == op.ZERO_SCORES
    assert op.score_official_answer("select_fruit", reference, True, [{"name": "pick"}]) == op.ZERO_SCORES
    assert op.score_official_answer("select_fruit", reference, True, "pick") == op.ZERO_SCORES


def test_scoring_matches_the_upstream_function_on_random_plans():
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")
    if not UPSTREAM_SCORER.exists():
        pytest.skip("pinned upstream scoring source is not checked out here")
    import matplotlib
    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location("upstream_scoring", UPSTREAM_SCORER)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    rng = random.Random(5)

    def operation(name):
        params = {}
        if name != "press" or rng.random() < .5:
            params["target_entity_name"] = rng.randint(0, 3)
        if name in ("place", "insert", "pour") and rng.random() < .8:
            params["target_container_name"] = rng.randint(0, 2)
        return {"name": name, "params": params}

    def plan(count, noise):
        result = []
        for _ in range(count):
            result += [operation(name) for name in rng.choice(BENCHMARK_PATTERNS)]
            if rng.random() < noise:
                result.append(operation(rng.choice(["pick", "place", "push", "Pick"])))
        return result

    for _ in range(400):
        reference = plan(rng.randint(1, 3), 0)
        prediction = plan(rng.randint(1, 3), .3)
        task = rng.choice(["select_fruit", "cook_dishes", "texas_holdem"])
        expected = upstream.get_final_score(reference, prediction, op.official_dependency(task))
        got = op.score_official_answer(task, reference, True, prediction)
        assert got["SR"] == pytest.approx(expected["skill_match_score"])
        assert got["PR"] == pytest.approx(expected["entity_match_score"])
        assert got["SPR"] == pytest.approx(expected["skill_with_entity_match_score"])
        assert got["PM"] == pytest.approx(expected["exact_match_score"])
        assert got["legacy_score"] == pytest.approx(expected["total_score"])


def test_run_resumes_scores_and_separates_unmapped_tasks(dataset, tmp_path):
    examples = op.load_official_examples(dataset)
    calls = []

    def generate(items):
        calls.append(items)
        instruction = items[-2][1]
        if instruction.startswith("fruit"):
            number = int(instruction.split()[1])
            return "```json\n" + json.dumps(pick_place(number, 9)) + "\n```"
        return "no fence at all"

    out = tmp_path / "run"
    summary = op.run_official_protocol(examples, generate, output_dir=out, model_name="fake", seed=1)
    assert summary["protocol"] == op.PROTOCOL and summary["standard"] is True
    assert summary["examples"] == 5 and summary["format_errors"] == 2
    assert summary["official_dimension_examples"]["M&T"] == 3
    assert summary["official_total_score_by_dimension"]["M&T"] == 80
    assert summary["official_total_score_by_dimension"]["Complex"] == 0
    assert summary["unmapped_examples"] == {"select_billiards": 1}
    assert summary["official_mapped_examples"] == 4
    assert summary["extended_mapping"]["examples"] == 5
    assert summary["dimension_mean_of_total_score"] is None  # four dimensions are empty
    assert len(calls) == 5
    op.run_official_protocol(examples, generate, output_dir=out, model_name="fake", seed=1)
    assert len(calls) == 5  # nothing regenerated
    with pytest.raises(ValueError, match="different configuration"):
        op.run_official_protocol(examples, generate, output_dir=out, model_name="fake", seed=2)


def test_internal_metrics_report_the_upstream_mapped_view():
    metrics = PlanningMetrics()
    full = dict.fromkeys(("SR", "PR", "SPR", "PM", "score", "legacy_score"), 100.0)
    half = dict.fromkeys(full, 50.0)
    metrics.update(full, task="select_fruit")
    metrics.update(half, task="select_billiards")  # extension task: not upstream-mapped
    result = metrics.result()
    assert result["official_mapping"]["examples"] == 1
    assert result["official_mapping"]["legacy_total_score_by_dimension"]["M&T"] == 100
    assert result["dimensions"]["M&T"]["examples"] == 2


def test_quoted_ids_fail_upstream_scoring_but_are_reported_supplementally(dataset, tmp_path):
    examples = [e for e in op.load_official_examples(dataset) if e.task == "select_fruit"]
    quoted = {e.episode_id: json.dumps([{"name": o["name"], "params": {k: str(v) for k, v in o["params"].items()}}
                                          for o in e.reference]) for e in examples}
    by_instruction = {e.instruction: quoted[e.episode_id] for e in examples}
    summary = op.run_official_protocol(
        examples, lambda items: "```json\n" + by_instruction[items[-2][1]] + "\n```",
        output_dir=tmp_path / "quoted", model_name="fake")
    assert summary["official_total_score_by_dimension"]["M&T"] == 40  # skills match, quoted ids do not
    assert summary["supplementary_integer_ids"]["official_total_score_by_dimension"]["M&T"] == 80
    assert summary["supplementary_integer_ids"]["answers_with_quoted_ids"] == 3


def test_decoding_is_part_of_the_run_configuration(dataset, tmp_path):
    examples = [e for e in op.load_official_examples(dataset) if e.task == "select_fruit"]

    def generate(items):
        return "```json\n" + json.dumps(pick_place(0, 1)) + "\n```"

    out = tmp_path / "grammar"
    summary = op.run_official_protocol(examples, generate, output_dir=out, model_name="fake", decoding="grammar-dfa:64")
    assert summary["decoding"] == "grammar-dfa:64"
    with pytest.raises(ValueError, match="different configuration"):
        op.run_official_protocol(examples, generate, output_dir=out, model_name="fake")
    plain = op.run_official_protocol(examples, generate, output_dir=tmp_path / "plain", model_name="fake")
    assert "decoding" not in plain  # runs without the option keep their earlier configuration


def test_decoding_labels_identify_the_hmm_and_weight():
    from test_regr.VLABenchAgentInterface.official_hmm import decoding_label, plan_tokens

    assert decoding_label(None, 1.0) == "grammar-dfa:64"
    assert decoding_label("abcdef0123456789", 0.0) == "grammar-dfa:64"
    assert decoding_label("abcdef0123456789", 0.5) == "grammar-dfa:64+hmm:abcdef012345:w0.5"

    class Tokenizer:
        eos_token_id = 9

        def __call__(self, text, add_special_tokens=False):
            return type("Encoded", (), {"input_ids": [ord(c) % 7 for c in text]})()

    tokens, error = plan_tokens('[{"name": "press", "params": {}}]\n```\n', Tokenizer())
    assert error is None and tokens[-1] == 9
    assert plan_tokens("I would pick the apple", Tokenizer())[0] is None
    assert plan_tokens("[]", Tokenizer())[0] is None


def test_hmm_weight_selection_prefers_the_smaller_weight_on_ties():
    candidates = [{"weight": 0.5, "validation_mean_total_score": 40.0}, {"weight": 0.1, "validation_mean_total_score": 40.0},
                  {"weight": 1.0, "validation_mean_total_score": 39.0}]
    best = max(candidates, key=lambda item: (item["validation_mean_total_score"], -item["weight"]))
    assert best["weight"] == 0.1


def test_teacher_samples_are_rewritten_to_integer_ids_before_hmm_fitting():
    from test_regr.VLABenchAgentInterface.official_hmm import canonical_plan_text

    text, error = canonical_plan_text('[{"name": "pick", "params": {"target_entity_name": "6"}}]\n```')
    assert error is None
    assert json.loads(text) == [{"name": "pick", "params": {"target_entity_name": 6}}]
    assert '"6"' not in text and text.startswith("[\n    {")
    assert canonical_plan_text("no plan here")[0] is None
