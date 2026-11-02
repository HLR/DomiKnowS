import io
import json
from dataclasses import dataclass, replace

import pytest
import torch

from ..utils.paired_references import PairedReferenceAudit, input_fingerprint
from ..eval_planning import build_eval_parser, run_planning_evaluation
from domiknows.generation.decision_trace import decision_record
from domiknows.generation.dfa.core import DFA
from domiknows.generation.dfa.predicate_monitor import (
    CompiledPredicateConstraint, PredicateConstraintProgram, PredicateEvent,
    PredicateMonitorDFA, SemanticTransition,
)


def press(index):
    return ({"name": "press", "params": {"target_entity_name": index}},)


@dataclass(frozen=True)
class Example:
    episode_id: str
    task: str = "density_qa"
    dimension: str = "PhysLaw"
    instruction: str = "Choose the densest object."
    entities: tuple = ("button0", "button1")
    dependency: str = "Sequential"
    operation_sequence: tuple = press(0)


def test_same_prediction_is_rescored_without_changing_inputs_or_original_reference():
    stream = io.StringIO()
    example = Example("PhysicsLaw/density_qa/example1")
    audit = PairedReferenceAudit(stream, corrector=lambda xs: [replace(xs[0], operation_sequence=press(1))])
    before = input_fingerprint(example)
    audit.observe(example, press(1))
    result = audit.report();record = json.loads(stream.getvalue())
    assert example.operation_sequence == press(0) and input_fingerprint(example) == before
    assert result["original"]["strict_plan_match"] == 0
    assert result["corrected"]["strict_plan_match"] == 1
    assert result["changed_reference_ids"] == [example.episode_id]
    assert record["prediction"] == list(press(1))
    assert record["original"]["scores"]["score"] < record["corrected"]["scores"]["score"]


def test_reference_corrector_cannot_change_inference_inputs():
    audit = PairedReferenceAudit(io.StringIO(), corrector=lambda xs: [replace(xs[0], instruction="leaked answer")])
    with pytest.raises(ValueError, match="inference inputs"):
        audit.observe(Example("example1"), press(0))


def test_paired_reference_protocol_rejects_prealtered_labels_before_loading_models():
    args = build_eval_parser().parse_args(["--checkpoint", "unused", "--output", "out.json",
                                          "--paired-density-references", "--updated-data"])
    with pytest.raises(ValueError, match="omit --updated-data"):
        run_planning_evaluation(args)


def monitor(allow_bad=False, grammar_blocks_bad=False):
    base = DFA(frozenset({0, 1}), frozenset({0, 1}),
               {(0, 1): 1, **({} if grammar_blocks_bad else {(0, 0): 1})}, 0, frozenset({1}))
    rule = CompiledPredicateConstraint("eligible_target", "operation",
                                       ("predicate", "eligible", ()), frozenset({"eligible"}))
    class Adapter:
        start_state = 0
        def step(self, state, symbol):
            return SemanticTransition(1, (PredicateEvent("operation", {"eligible": allow_bad or symbol == 1}),))
        def is_accepting(self, state): return state == 1
        def acceptance_events(self, state): return ()
    return PredicateMonitorDFA(base, PredicateConstraintProgram((rule,)), Adapter())


class Vocabulary:
    def token_for_label(self, index): return ("bad", "good")[index]


def test_trace_names_the_violated_rule_without_changing_the_policy():
    dfa = monitor();state = dfa.start_state;allowed = dfa.allowed_tokens(state)
    logits = torch.tensor([10., 0.], requires_grad=True)
    record = decision_record(logits, allowed, 1, Vocabulary(), dfa=dfa, state=state)
    assert record["raw_argmax"] == "bad" and not record["raw_argmax_allowed"]
    assert record["rejection"]["kind"] == "predicate_rules"
    assert [x["rule"] for x in record["rejection"]["violated_rules"]] == ["eligible_target"]
    assert dfa.allowed_tokens(state) == allowed and dfa.step(state, 0) is None
    assert dfa.step(state, 1) is not None and logits.grad is None


@pytest.mark.parametrize("grammar", [False, True])
def test_trace_does_not_invent_rule_attributions(grammar):
    dfa = monitor(allow_bad=True, grammar_blocks_bad=grammar)
    record = decision_record(torch.tensor([10., 0.]), {1}, 1, Vocabulary(), dfa=dfa, state=dfa.start_state)
    assert record["rejection"]["kind"] == ("grammar" if grammar else "completion_pruning")
    assert record["rejection"]["violated_rules"] == []
