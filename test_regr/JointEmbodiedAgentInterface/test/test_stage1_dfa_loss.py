"""Stage 1 trains the planner on the DFA: allowed-mass on the DFA path plus the primal-dual constraint loss."""
import itertools

import pytest
import torch

from domiknows.generation.dfa.vocabulary import TokenVocabulary
from test_regr.EmbodiedAgentInterface.dataset import dummy_dataset

from ..program import DFALossConfig, JointSolverPOIProgram
from ..world_graph import build_joint_runtime, build_joint_world_graph
from .test_joint_pipeline import make_planner

_counter = itertools.count()


@pytest.fixture
def joint_fixture():
    """A runtime of its own: the planner sensors attach once per graph, so tests must not share one."""
    examples = dummy_dataset(max_steps=4)
    world = build_joint_world_graph(f"stage1_dfa_world_{next(_counter)}")
    runtime = build_joint_runtime(
        world, TokenVocabulary(examples[0]["generation_vocab"], eos_token="<eos>"), max_eai_steps=4,
        eai_object_tokens=examples[0]["object_tokens"], eai_action_tokens=examples[0]["action_tokens"],
        eai_action_sequence_tokens=examples[0]["action_tokens"],
        eai_openable_object_tokens=examples[0]["openable_object_tokens"],
        max_vlabench_entities=2, max_vlabench_operations=1)
    return examples, runtime


def vlabench_item(runtime):
    vocabulary = runtime.vlabench_vocabulary
    length = runtime.max_vlabench_operations * 5 + 1
    labels = [vocabulary.label_for_token("skill:press"), vocabulary.label_for_token("arg:target_entity_name"),
              vocabulary.label_for_token("obj:0"), vocabulary.eos_label]
    labels += [vocabulary.eos_label] * (length - len(labels))
    return {"planner_context": {"instruction": "press it", "images": (), "entity_table": ("button0", "bowl")},
            "token_positions": torch.arange(length), "target_plan_labels": torch.tensor(labels),
            "entities": ("button0", "bowl")}


def program_with(runtime, planner, **config):
    config.setdefault("domains", ("eai", "vlabench"))
    return JointSolverPOIProgram(
        runtime, planner, planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.05),
        dfa_loss=DFALossConfig(**config))


def test_dfa_loss_adds_terms_and_updates_the_vlabench_head(joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    program = program_with(runtime, planner)
    before = planner.label_heads["vlabench"].weight.detach().clone()
    loss = program._planner_step("vlabench", vlabench_item(runtime))
    assert loss > 0 and not torch.equal(before, planner.label_heads["vlabench"].weight)
    assert runtime.active_domain is None


def test_total_is_cross_entropy_plus_weighted_allowed_mass(joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    item = vlabench_item(runtime)
    plain = JointSolverPOIProgram(runtime, planner, planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.0))
    with runtime.domain_scope("vlabench", parameter_policy="freeze_inactive"):
        baseline, _ = plain._domain_loss("vlabench", item)
    program = program_with(runtime, planner, masked_ce_domains=(), mass_weight=2.0, pmd_weight=0.0)
    with runtime.domain_scope("vlabench", parameter_policy="freeze_inactive"):
        total, terms = program._domain_loss("vlabench", item)
    assert terms["allowed_mass_loss"] > 0
    assert abs(float(total) - (float(baseline) + 2.0 * terms["allowed_mass_loss"])) < 1e-4


def test_dual_step_follows_the_schedule(joint_fixture):
    _examples, runtime = joint_fixture
    program = program_with(runtime, make_planner(runtime), dual_warmup=1, dual_every=2)
    calls = []
    for optimizer in program.dual_optimizers.values():
        optimizer.step = lambda *a, **k: calls.append(1)
    for _ in range(6):
        program._dual_step()
    assert len(calls) == len(program.dual_optimizers) * 3  # rounds 2, 4 and 6 fire; round 1 is warmup


def violating_logits(runtime, planner, length):
    """End label at step 0 only: the ordered EOS constraint (EOS stays EOS) is violated by every pair (0, j)."""
    vocabulary = runtime.vlabench_vocabulary

    def logits(_domain, _context, _prefix):
        value = torch.zeros(1, length, vocabulary.label_count)
        value[0, 0, vocabulary.eos_label] = 10.0
        value[0, 1:, vocabulary.eos_label] = -10.0
        return value.requires_grad_()

    planner.sequence_logits = logits


def test_constraint_violation_gives_the_loss_and_the_multipliers_ascend(joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    item = vlabench_item(runtime)
    violating_logits(runtime, planner, len(item["target_plan_labels"]))
    program = program_with(runtime, planner, mass_weight=0.0, pmd_weight=1.0, dual_warmup=0, dual_every=1)
    params = [p for g in program.dual_optimizers["vlabench"].param_groups for p in g["params"]]
    before = [p.detach().clone() for p in params]
    with runtime.domain_scope("vlabench", parameter_policy="freeze_inactive"):
        _total, terms = program._domain_loss("vlabench", item)
    assert terms["constraint_loss"] > 1.0  # five violated pairs, multiplier 1
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params)
    program._dual_step()
    assert any(float(after.detach().sum()) > float(old.sum()) for after, old in zip(params, before))


def test_eai_domain_uses_the_same_loss(joint_fixture):
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    program = program_with(runtime, planner)
    before = planner.label_heads["eai"].weight.detach().clone()
    loss = program._planner_step("eai", examples[0])
    assert loss > 0 and not torch.equal(before, planner.label_heads["eai"].weight)


def test_domains_outside_the_dfa_loss_keep_plain_cross_entropy(joint_fixture):
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    program = program_with(runtime, planner, domains=("vlabench",))
    assert set(program.pmd_programs) == {"vlabench"}
    with runtime.domain_scope("eai", parameter_policy="freeze_inactive"):
        _loss, terms = program._domain_loss("eai", examples[0])
    assert terms == {}


def test_ce_only_command_trains_both_heads_without_dfa_or_rl(joint_fixture, monkeypatch, tmp_path):
    from .. import main as cli
    from ..models import FreeDecoding

    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = torch.nn.Linear(1, 1)
    item = vlabench_item(runtime)
    before = {d: planner.label_heads[d].weight.detach().clone() for d in ("eai", "vlabench")}

    def forbidden(*args, **kwargs):
        pytest.fail("CE-only training entered a DFA loss/report or RL path")

    monkeypatch.setattr(cli, "apply_knowledge_argument", lambda args: "test")
    monkeypatch.setattr(cli, "apply_holdout_arguments", lambda args: ())
    monkeypatch.setattr(cli, "_prepare", lambda *args: (
        runtime, examples, examples, {"validation": [item]}, {"train": [item], "validation": [item]}))
    monkeypatch.setattr(cli.JointQwenVLPlanner, "from_pretrained", lambda **kwargs: planner)
    monkeypatch.setattr(cli, "_controller", lambda *args: controller)
    monkeypatch.setattr(cli, "_control_loaders", lambda args: {"train": [None]})
    monkeypatch.setattr(cli, "load_control_task_instructions", lambda *args: {})
    monkeypatch.setattr(JointSolverPOIProgram, "dfa_mass_report", forbidden)
    monkeypatch.setattr(runtime, "dfa_for", forbidden)
    monkeypatch.setattr(cli, "JointReinforcementProgram", forbidden)
    monkeypatch.setattr(cli, "_enter_controller_reinforcement", forbidden)
    monkeypatch.setattr(JointSolverPOIProgram, "train_controller_warmup", forbidden)

    def evaluate_eai(*args, **kwargs):
        assert isinstance(planner._planning_dfa("eai", {}, None), FreeDecoding)
        # Poor baseline performance must still be saved, without an RL exploration gate.
        return {"goal_success": 0.0, "goal_recall": 0.0, "positive_reward_rate": 0.0}

    def evaluate_vla(*args, **kwargs):
        assert isinstance(planner._planning_dfa("vlabench", {}, None), FreeDecoding)
        return {"exact_graph_match": 0.0, "valid": 0.0}

    monkeypatch.setattr(cli, "_evaluate_eai", evaluate_eai)
    monkeypatch.setattr(cli, "evaluate_planner", evaluate_vla)
    saved = {}

    def save(path, **kwargs):
        saved.update(kwargs)
        return path

    monkeypatch.setattr(cli, "save_joint_checkpoint", save)
    monkeypatch.setattr(cli, "load_joint_checkpoint", lambda *a, **kw: saved)
    args = cli.build_parser().parse_args([
        "train-agent", "--ce-only", "--device", "cpu", "--output", str(tmp_path),
        "--stage1-epochs", "1", "--stage1-rounds-per-epoch", "1", "--stage1-eai-warmup-steps", "0",
    ])
    cli.command_train_agent(args)
    assert cli._dfa_loss_config(args) is None
    assert saved["stage"] == "stage1"
    assert saved["metrics"]["training_mode"] == "ce-only"
    assert saved["metrics"]["decoding"] == "free"
    assert all(not torch.equal(before[d], planner.label_heads[d].weight) for d in before)


def test_ce_only_and_two_stage_are_mutually_exclusive():
    from ..main import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-agent", "--ce-only", "--two-stage"])


def test_dfa_mass_report_is_a_probability_for_both_domains_and_restores_training_mode(joint_fixture):
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    program = program_with(runtime, planner)
    planner.train()
    report = program.dfa_mass_report([examples[0]], [vlabench_item(runtime)], limit=1)
    assert set(report) == {"eai", "vlabench"} and all(0.0 < value <= 1.0 for value in report.values())
    assert planner.training
    assert program.dfa_mass_report([], [], limit=1) == {"eai": None, "vlabench": None}


def test_masked_cross_entropy_is_never_above_the_full_one_and_reports_disagreements(joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    item = vlabench_item(runtime)
    full = program_with(runtime, planner, masked_ce_domains=(), mass_weight=0.0, pmd_weight=0.0)
    masked = program_with(runtime, planner, masked_ce_domains=("eai", "vlabench"), mass_weight=0.0, pmd_weight=0.0)
    with runtime.domain_scope("vlabench", parameter_policy="freeze_inactive"):
        full_total, _ = full._domain_loss("vlabench", item)
        masked_total, terms = masked._domain_loss("vlabench", item)
    assert float(masked_total) <= float(full_total) + 1e-6
    assert terms["masked_ce_misses"] == 0.0  # the gold plan is inside the DFA
    assert masked_total.requires_grad


def test_masked_cross_entropy_applies_only_to_the_listed_domains(joint_fixture):
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    only_vlabench = program_with(runtime, planner, masked_ce_domains=("vlabench",), mass_weight=0.0, pmd_weight=0.0)
    both = program_with(runtime, planner, masked_ce_domains=("eai", "vlabench"), mass_weight=0.0, pmd_weight=0.0)
    with runtime.domain_scope("eai", parameter_policy="freeze_inactive"):
        plain_total, plain_terms = only_vlabench._domain_loss("eai", examples[0])
        masked_total, masked_terms = both._domain_loss("eai", examples[0])
    assert plain_terms["masked_ce_misses"] == 0.0  # not masked: nothing is counted
    assert float(masked_total) <= float(plain_total) + 1e-6
