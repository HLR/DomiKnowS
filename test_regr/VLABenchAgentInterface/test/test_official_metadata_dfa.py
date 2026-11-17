"""--protocol official --knowledge metadata: each episode decodes under its bound world LC policy (privileged)."""
import json
from types import SimpleNamespace

import pytest
import torch

from .. import main
from ..dataset import PlanningExample
from ..native_planning import FrozenNativePlanner
from ..official import official_protocol as op
from ..training import build_constraint_runtime
from .test_action_reachability import EOS, TOKENS, context, labels, plan
from .test_official_protocol import make_episode, pick_place


@pytest.fixture(scope="module")
def runtime():
    return build_constraint_runtime(max_entities=3, max_operations=4, name_prefix="official_metadata_dfa")


def stub_planner(reply):
    """A frozen planner whose model writes ``reply`` and whose logits processor still checks every token."""
    def generate(**kwargs):
        processor = kwargs["logits_processor"][0]
        written = [0]
        for token in labels(reply)[:-1] + [EOS]:
            scores = torch.zeros(1, 256)
            allowed = processor(torch.tensor([written]), scores)
            if allowed[0, token].isneginf():
                break
            written.append(token)
        return torch.tensor([written])

    return SimpleNamespace(model=SimpleNamespace(generate=generate), token_bytes=TOKENS, dfas={}, tables={},
                           tokenizer=SimpleNamespace(eos_token_id=EOS,
                                                     decode=lambda ids, **kw: bytes(i for i in ids if i < 128).decode()))


class Batch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


def prepared(planner, items, prefix=""):
    return Batch(input_ids=torch.tensor([[0]])), []


def items():
    return [["text", "prompt"], ["image", "input.png"], ["image", "input_mask.png"], ["text", "put apple in bowl"]]


@pytest.mark.parametrize("destination,accepted", [(1, True), (2, False)])
def test_the_bound_policy_decides_the_plan(runtime, monkeypatch, destination, accepted):
    monkeypatch.setattr(op, "prepare_inputs", prepared)
    planner = stub_planner(plan(destination))
    generate = op.make_grammar_dfa_generator(planner, max_new_tokens=200, causal=True,
                                             episode_policy=lambda items: runtime.dfa_for("vlabench", context()))
    reply = generate(items())
    assert reply.startswith("<!-- metadata-dfa: bound -->")
    assert ("```json" in reply) is accepted


def test_an_unreachable_goal_falls_back_to_the_causal_dfa(runtime, monkeypatch):
    monkeypatch.setattr(op, "prepare_inputs", prepared)
    planner = stub_planner(plan(2))
    unreachable = context(goals={"positive": [["at", "cup", "bowl"]]})
    generate = op.make_grammar_dfa_generator(planner, max_new_tokens=200, causal=True, entity_cap=3,
                                             episode_policy=lambda items: runtime.dfa_for("vlabench", unreachable))
    reply = generate(items())
    assert reply.startswith("<!-- metadata-dfa: fallback-unreachable -->") and "```json" in reply


def test_the_metadata_dfa_refuses_questions_hmm_and_perception():
    with pytest.raises(ValueError, match="metadata DFA"):
        op.make_grammar_dfa_generator(object(), press_question=True, episode_policy=lambda items: None)


def test_cli_binds_each_episode_by_its_labeled_image(tmp_path, monkeypatch):
    make_episode(tmp_path / "data", "select_fruit", 0, "Put the apple in the bowl", pick_place(0, 1))
    example = PlanningExample("M&T/select_fruit/example0", "Put the apple in the bowl", tuple(pick_place(0, 1)), (), (),
                              None, entities=("apple", "bowl"), task="select_fruit")
    monkeypatch.setattr(main, "load_planning_examples", lambda *a, **kw: [example])
    monkeypatch.setattr(FrozenNativePlanner, "load", lambda *a, **kw: SimpleNamespace(model=torch.nn.Linear(1, 1)))
    captured = {}
    monkeypatch.setattr(op, "make_grammar_dfa_generator", lambda *a, **kw: captured.update(decoder=kw) or (lambda _: ""))
    monkeypatch.setattr(op, "run_official_protocol", lambda examples, generate, **kw: captured.update(run=kw, examples=examples) or {})
    args = main.build_parser().parse_args(["evaluate-planner", "--protocol", "official", "--knowledge", "metadata",
                                           "--planning-dir", str(tmp_path / "data"), "--planner-model", "frozen",
                                           "--official-output-dir", str(tmp_path / "run")])
    main.command_evaluate_official(args)
    decoder, run = captured["decoder"], captured["run"]
    assert decoder["causal"] is True and decoder["perception"] is None
    assert run["standard"] is False and run["decoding"].endswith("+causal+metadata-world-policy-nonstrict")
    episode = captured["examples"][0]
    policy = decoder["episode_policy"](op.build_ti_list("prompt", episode))
    assert policy.adapter.entities == {0: "apple", 1: "bowl"}
