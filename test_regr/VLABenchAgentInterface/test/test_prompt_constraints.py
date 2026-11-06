"""--prompt-constraints: the rules are told to a frozen VLM in English, with no DFA or perception."""
import json

import pytest
import torch

from .. import dataset, main, native_planning, training
from ..dataset import PlanningExample
from ..official import official_protocol as op
from ..official.causal_graph import build_causal_graph
from ..official.decoding_graph import build_decoding_graph
from ..official.prompt_constraints import (
    ANSWER_GATED_LCS, CAUSAL_RULES, DEFAULT_LEVEL, GRAMMAR_RULES, OFFICIAL_RULES, prompt_constraints_record,
    prompt_constraints_text, prompt_rules,
)
from ..official.selection_graph import build_selection_graph
from .test_native_grammar_lcs import BYTES, EOS

PLAN = [{"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}}]


def named_lcs(graph):
    return {lc.name for lc in graph.logicalConstrains.values() if not lc.name.startswith("LC")}


def worded(rules):
    return {name for rule in rules for name in rule.lcs}


def test_every_rule_matches_a_declared_logical_constraint():
    grammar = native_planning.NativePlanDFA(BYTES, EOS, entity_count=4, max_operations=4).language.bundle
    assert worded(GRAMMAR_RULES) == {rule.name for rule in grammar.planner_grammar_rules}
    assert worded(CAUSAL_RULES) == named_lcs(build_causal_graph("prompt_test_causal").graph)
    declared = (named_lcs(build_decoding_graph("prompt_test_decoding").graph)
                | named_lcs(build_selection_graph("prompt_test_selection").graph))
    assert worded(OFFICIAL_RULES) | set(ANSWER_GATED_LCS) == declared
    assert not worded(OFFICIAL_RULES) & set(ANSWER_GATED_LCS)


def test_levels_are_cumulative_and_official_is_the_default():
    assert DEFAULT_LEVEL == "official"
    grammar, causal, official = (prompt_rules(level) for level in ("grammar", "causal", "official"))
    assert grammar == GRAMMAR_RULES
    assert causal == GRAMMAR_RULES + CAUSAL_RULES
    assert official == GRAMMAR_RULES + CAUSAL_RULES + OFFICIAL_RULES
    with pytest.raises(ValueError, match="unknown prompt constraint level"):
        prompt_rules("task")


def test_text_is_worded_from_the_language_of_the_path():
    benchmark = prompt_constraints_text("official", grammar_profile="benchmark", max_operations=7)
    world = prompt_constraints_text("official", grammar_profile="world", max_operations=7)
    assert "at most 7 skill calls" in benchmark
    assert "open_door takes target_entity_name" in benchmark and "pull takes no parameters" in benchmark
    assert '["pick", "push", "place"]' in benchmark and '["press"]' in benchmark
    assert "open_laptop" not in benchmark and "open_laptop takes target_entity_name" in world
    assert all(mark not in benchmark for mark in ("{", "}", "ifL", "DFA", "LC"))
    assert [line.split(".")[0] for line in benchmark.splitlines() if line[:1].isdigit()] == [
        str(number) for number in range(1, len(prompt_rules("official")) + 1)]
    with pytest.raises(ValueError, match="max-operations"):
        prompt_constraints_text("grammar", grammar_profile="benchmark", max_operations=0)


def test_lower_levels_leave_out_the_higher_rules():
    grammar = prompt_constraints_text("grammar", grammar_profile="benchmark", max_operations=12)
    causal = prompt_constraints_text("causal", grammar_profile="benchmark", max_operations=12)
    official = prompt_constraints_text("official", grammar_profile="benchmark", max_operations=12)
    assert "Causal rules" not in grammar and "Causal rules" in causal
    assert "labeled image" not in causal and "labeled image" in official


def parse(*flags):
    return main.build_parser().parse_args(["evaluate-planner", "--planning-dir", "unused", *flags])


def test_flag_defaults_to_official_and_is_off_when_omitted():
    assert parse().prompt_constraints is None
    assert parse("--prompt-constraints").prompt_constraints == "official"
    assert parse("--prompt-constraints", "causal").prompt_constraints == "causal"


@pytest.mark.parametrize("flags,named", [
    (["--generation-mode", "graph"], "--generation-mode graph"),
    (["--knowledge", "official"], "--knowledge"),
    (["--constraints", "causal"], "--constraints"),
    (["--press-question"], "--press-question"),
    (["--scene-ids"], "--scene-ids"),
    (["--pattern-first"], "--pattern-first"),
    (["--hmm", "best.pt"], "--hmm"),
    (["--official-disable-formula", "ordinal_rank_2"], "--official-disable-formula"),
    (["--perception-cache", "cache"], "--perception-cache"),
])
@pytest.mark.parametrize("protocol", ["internal", "official"])
def test_cli_refuses_flags_that_enforce_or_observe(flags, named, protocol):
    args = parse("--prompt-constraints", "--protocol", protocol, *flags)
    with pytest.raises(ValueError, match=f"omit .*{named}"):
        main.command_evaluate_planner(args)


class Native:
    def __init__(self, outputs):
        self.model = torch.nn.Linear(1, 1)
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, item, **kwargs):
        self.calls.append(kwargs)
        return next(self.outputs), []


def test_prompt_constraints_only_reach_frozen_generation():
    with pytest.raises(ValueError, match="frozen-vlm baseline"):
        training.evaluate_planner(Native([]), [], object(), prompt_constraints="rules")
    with pytest.raises(ValueError, match="requires raw generation"):
        native_planning.FrozenNativePlanner.generate(object(), object(), mode="dfa", prompt_constraints="rules")


def test_native_prompt_places_the_rules_before_the_episode_and_is_unchanged_without_them():
    plain = native_planning.native_prompt("Do it", ("apple",))
    assert plain == native_planning.native_prompt("Do it", ("apple",), constraints=None)
    with_rules = native_planning.native_prompt("Do it", ("apple",), constraints="RULES")
    assert with_rules.replace("RULES\n", "") == plain
    assert with_rules.index("RULES") < with_rules.index("Objects:")


def test_internal_cli_implies_frozen_vlm_and_records_the_text(monkeypatch, capsys):
    args = parse("--prompt-constraints", "--split", "all", "--max-operations", "9", "--no-load-in-4bit")
    def forbidden(*args, **kwargs):
        pytest.fail("prompt constraints entered the graph generation path")
    for name in ("load_planning_examples", "build_constraint_runtime", "build_vlabench_world_graph", "_load_planner"):
        monkeypatch.setattr(main, name, forbidden)
    example = PlanningExample("select_fruit/example0", "Put apple in bowl", tuple(PLAN), (), (), None,
                              entities=("apple", "bowl"))
    monkeypatch.setattr(dataset, "load_scene_planning_examples", lambda *a, **kw: [example])
    planner = Native([json.dumps(PLAN)])
    monkeypatch.setattr(native_planning.FrozenNativePlanner, "load", lambda *a, **kw: planner)
    main.command_evaluate_planner(args)
    result = json.loads(capsys.readouterr().out)
    expected = prompt_constraints_text("official", grammar_profile="world", max_operations=9)
    assert result["generation_mode"] == "frozen-vlm" and result["dfa_constrained"] is False
    assert result["knowledge"] is None and result["task_identity"] is None
    assert result["prompt_constraints"] == prompt_constraints_record("official", expected)
    assert planner.calls == [{"mode": "raw", "sample": False, "max_tokens": 768, "prompt_constraints": expected}]


def official_episode(root):
    episode = root / "select_fruit" / "example0"
    (episode / "input").mkdir(parents=True)
    (episode / "output").mkdir()
    (episode / "input" / "instruction.txt").write_text("Put the apple in the bowl", encoding="utf-8")
    (episode / "input" / "input.png").write_bytes(b"png")
    (episode / "input" / "input_mask.png").write_bytes(b"png")
    (episode / "output" / "plan.json").write_text(json.dumps(PLAN), encoding="utf-8")


def test_official_cli_adds_the_text_after_the_pinned_prompt_without_dfa(tmp_path, monkeypatch, capsys):
    data, out = tmp_path / "data", tmp_path / "out"
    official_episode(data)
    args = main.build_parser().parse_args([
        "evaluate-planner", "--planning-dir", str(data), "--protocol", "official", "--prompt-constraints",
        "--official-output-dir", str(out), "--no-load-in-4bit"])
    # The official level words the perception rules, but only --knowledge official would run perception.
    monkeypatch.setattr(main, "official_perception_options", lambda *_: pytest.fail("prompt constraints use no perception"))
    monkeypatch.setattr(op, "make_grammar_dfa_generator", lambda *a, **kw: pytest.fail("prompt constraints use no DFA"))
    monkeypatch.setattr(native_planning.FrozenNativePlanner, "load", lambda *a, **kw: object())
    seen = []
    reply = "```json\n" + json.dumps(PLAN) + "\n```"
    monkeypatch.setattr(op, "make_vlm_generator", lambda *a, **kw: lambda items: seen.append(items) or reply)
    main.command_evaluate_planner(args)
    output = capsys.readouterr().out
    summary = json.loads(output[output.index("{"):])
    expected = prompt_constraints_text("official", grammar_profile="benchmark", max_operations=op.DFA_MAX_OPERATIONS)
    assert "at most 16 skill calls" in expected
    assert seen[0][0] == ["text", f"{op.load_official_prompt()}\n\n{expected}"]
    assert summary["standard"] is False and summary["dfa_constrained"] is False
    assert summary["decoding"] == "greedy+prompt-constraints-official"
    assert summary["prompt_sha256"] == op.PROMPT_SHA256
    assert summary["prompt_constraints"] == prompt_constraints_record("official", expected)
    assert summary["examples"] == 1 and summary["format_errors"] == 0
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["prompt_constraints"]["sha256"] == prompt_constraints_record("official", expected)["sha256"]


def test_a_resumed_run_refuses_different_prompt_constraints(tmp_path):
    root = tmp_path / "data"
    official_episode(root)
    examples = op.load_official_examples(root)
    reply = "```json\n" + json.dumps(PLAN) + "\n```"
    common = dict(output_dir=tmp_path / "out", model_name="fake")
    op.run_official_protocol(examples, lambda items: reply, **common,
                             prompt_constraints=prompt_constraints_record("causal", "A"))
    with pytest.raises(ValueError, match="different configuration"):
        op.run_official_protocol(examples, lambda items: reply, **common,
                                 prompt_constraints=prompt_constraints_record("causal", "B"))
    with pytest.raises(ValueError, match="different configuration"):
        op.run_official_protocol(examples, lambda items: reply, **common)
