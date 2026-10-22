"""Knowledge inputs, provenance, and compatibility survive every joint entry point."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from domiknows.graph.logicalConstrain import ifL, notL
from ..world_graph import build_joint_world_graph, build_joint_runtime
from ..utils.planner_context import planner_context
from ..utils.knowledge_manifest import check_knowledge_compatibility
from ..models import JointPlannerDomainView
from ..checkpoint import save_joint_checkpoint, load_joint_checkpoint
from .test_joint_pipeline import joint_fixture, make_planner, TinyController


def test_context_rows_and_nested_overrides_preserve_knowledge_without_targets():
    facts = {"goals": {"positive": [["clean", "cup"]]}}
    row = {
        "instruction_text": "Clean the cup",
        "planning_facts": facts,
        "generation_entity_types": ("cup",),
        "semantic_action_permissions": ("clean",),
        "target_action_labels": [1, 2, 0],
        "operation_sequence": ["gold"],
        "planner_context": {"instruction": "nested", "semantic_action_permissions": ()},
    }
    result = planner_context("eai", row)
    assert result["planning_facts"] is facts
    assert result["generation_entity_types"] == ("cup",)
    assert result["semantic_action_permissions"] == ()
    assert result["instruction"] == "nested"
    assert not {"target_action_labels", "operation_sequence"} & result.keys()
    assert planner_context("eai", row, context={"planning_facts": None})["planning_facts"] is None
    assert row["planner_context"] == {"instruction": "nested", "semantic_action_permissions": ()}
    example = SimpleNamespace(
        instruction="Use the button",
        entities=("button",),
        planning_facts=facts,
        predicted_task="press_button",
        planning_interface="compact",
    )
    assert planner_context("vlabench", example)["predicted_task"] == "press_button"
    assert planner_context("vlabench", example)["entity_table"] == ("button",)


@pytest.mark.parametrize("method", ["supervised_loss", "sample_with_logprob", "generate_plan"])
def test_keyword_facade_preserves_all_explicit_knowledge(method):
    class Probe:
        def vocabulary(self, domain):
            return SimpleNamespace(label_count=1, eos_label=0)

        def supervised_loss(self, domain, *, context, target_labels):
            self.context = context
            return torch.tensor(0.0)

        def sample_labels(self, domain, context, dfa, **kwargs):
            self.context = context
            return [0], torch.tensor(0.0)

    probe = Probe()
    view = JointPlannerDomainView(probe, "eai")
    facts = {"initial_state": {"held": {"left": "cup"}}}
    getattr(view, method)(
        instruction="Clean the cup",
        planning_facts=facts,
        semantic_action_permissions=("clean",),
        generation_entity_types=("cup",),
        target_labels=[0],
        dfa=None,
    )
    assert probe.context["planning_facts"] is facts
    assert probe.context["generation_entity_types"] == ("cup",)
    assert probe.context["semantic_action_permissions"] == ("clean",)


def _runtime(examples, vocabulary, name):
    return build_joint_runtime(
        build_joint_world_graph(name),
        vocabulary,
        max_eai_steps=4,
        eai_object_tokens=examples[0]["object_tokens"],
        eai_action_tokens=examples[0]["action_tokens"],
        eai_action_sequence_tokens=examples[0]["action_tokens"],
        eai_openable_object_tokens=examples[0]["openable_object_tokens"],
        max_vlabench_entities=2,
        max_vlabench_operations=1,
    )


def test_manifest_is_stable_across_build_names_and_role_ids(joint_fixture):
    examples, runtime = joint_fixture
    other = _runtime(examples, runtime.eai_vocabulary, "different_build_name")
    assert (
        runtime.knowledge_manifest["definition_sha256"]
        == other.knowledge_manifest["definition_sha256"]
    )
    assert (
        runtime.knowledge_manifest["activation_sha256"]
        == other.knowledge_manifest["activation_sha256"]
    )
    import json
    json.dumps(runtime.knowledge_manifest)  # the inspectable manifest also supports JSON reports
    assert runtime.knowledge_manifest["components"]["joint"]["constraint_count"] == 0
    assert runtime.knowledge_manifest["components"]["eai_world"]["constraint_count"] == 62
    assert runtime.knowledge_manifest["components"]["vlabench_world"]["constraint_count"] == 86


def test_manifest_distinguishes_definitions_from_activation_and_labels(joint_fixture):
    _, runtime = joint_fixture
    world = runtime.world.eai
    rule = world.default_constraints[0]
    baseline = runtime.knowledge_manifest
    old_name, old_active, old_e = rule.name, rule.active, rule.e
    try:
        rule.name = "a_report_label_only"
        assert runtime.knowledge_manifest["definition_sha256"] == baseline["definition_sha256"]
        rule.active = not old_active
        ablation = runtime.knowledge_manifest
        assert ablation["definition_sha256"] == baseline["definition_sha256"]
        assert check_knowledge_compatibility(baseline, ablation)["activation_changed"]
        with world.graph:
            edited = ifL(rule.e[0], notL(world.facts["first_argument_in_task"]("operation")))
        edited.headLC = False
        rule.e = edited.e
        changed = runtime.knowledge_manifest
        assert changed["definition_sha256"] != baseline["definition_sha256"]
        with pytest.raises(ValueError, match="eai_world"):
            check_knowledge_compatibility(baseline, changed)
    finally:
        rule.e, rule.name, rule.active = old_e, old_name, old_active


def test_contextual_binding_changes_are_part_of_knowledge(joint_fixture):
    _, runtime = joint_fixture
    graph = runtime.eai_generation_graph
    rule = next(
        lc
        for lc in graph.logicalConstrains.values()
        if getattr(lc, "_generation_contextual_dfa_specs", ())
    )
    before = runtime.knowledge_manifest
    old = rule._generation_contextual_dfa_specs
    try:
        rule._generation_contextual_dfa_specs = (
            replace(old[0], context_key="different_fact_source"),
            *old[1:],
        )
        assert runtime.knowledge_manifest["definition_sha256"] != before["definition_sha256"]
    finally:
        rule._generation_contextual_dfa_specs = old


def test_new_checkpoints_reject_rule_drift_and_identify_legacy_provenance(tmp_path, joint_fixture):
    _, runtime = joint_fixture
    planner, controller = make_planner(runtime), TinyController()
    path = save_joint_checkpoint(
        tmp_path / "knowledge.pt",
        runtime=runtime,
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=None,
        stage="stage1",
        epoch=0,
        round_robin_cursor=0,
    )
    original = runtime.knowledge_manifest
    graph = runtime.world.eai.graph
    with graph:
        added = ifL(
            runtime.world.eai.actions["open"]("operation"),
            notL(runtime.world.eai.facts["first_argument_supplied"]("operation")),
            name="new_knowledge",
        )
    try:
        assert runtime.knowledge_manifest["definition_sha256"] != original["definition_sha256"]
        with pytest.raises(ValueError, match="knowledge definitions differ"):
            load_joint_checkpoint(path, runtime=runtime, planner=planner, controller=controller)
    finally:
        del graph.logicalConstrains[
            next(key for key, lc in graph.logicalConstrains.items() if lc is added)
        ]
    payload = load_joint_checkpoint(path, runtime=runtime, planner=planner, controller=controller)
    assert payload["knowledge_provenance"]["status"] == "verified"
    payload["compatibility"].pop("knowledge_manifest")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="missing its declared knowledge manifest"):
        load_joint_checkpoint(path, runtime=runtime, planner=planner, controller=controller)
    payload.pop("knowledge_manifest_version")
    torch.save(payload, path)
    payload = load_joint_checkpoint(path, runtime=runtime, planner=planner, controller=controller)
    assert payload["knowledge_provenance"]["status"] == "unrecorded_legacy"


def test_manifest_detects_changed_compiled_program_without_graph_edits(joint_fixture):
    _, runtime = joint_fixture
    before = runtime.knowledge_manifest
    changed = deepcopy(before)
    changed["components"]["eai_world"]["compiled_sha256"] = "different_program"
    with pytest.raises(ValueError, match="compiled knowledge differs"):
        check_knowledge_compatibility(before, changed)


def test_learned_vs_task_fact_sources_are_visible_even_with_identical_active_rules(joint_fixture):
    _, runtime = joint_fixture
    state = runtime.world.vlabench.knowledge_state
    old = state.level
    before = runtime.knowledge_manifest
    try:
        state.level = "learned" if old != "learned" else "task"
        after = runtime.knowledge_manifest
        assert after["definition_sha256"] == before["definition_sha256"]
        assert after["activation_sha256"] != before["activation_sha256"]
        assert check_knowledge_compatibility(before, after)["activation_changed"]
    finally:
        state.level = old


def test_explicit_vlabench_knowledge_ablation_does_not_disable_eai_checks(joint_fixture):
    _, runtime = joint_fixture
    before = runtime.knowledge_manifest
    ablation = deepcopy(before)
    ablation["components"]["vlabench_world"]["definition_sha256"] = "withheld_task_rules"
    with pytest.raises(ValueError, match="vlabench_world"):
        check_knowledge_compatibility(before, ablation)
    assert (
        check_knowledge_compatibility(before, ablation, knowledge_ablation=True)["status"]
        == "knowledge_ablation"
    )
    ablation["components"]["eai_world"]["definition_sha256"] = "changed_eai_rule"
    with pytest.raises(ValueError, match="eai_world"):
        check_knowledge_compatibility(before, ablation, knowledge_ablation=True)


def test_encoded_context_facade_binds_keyword_facts_before_sampling(monkeypatch):
    from test_regr.EmbodiedAgentInterface.utils import constraint_compiler

    bound = object()
    facts = {"goals": {"positive": [["clean", "cup"]]}}

    def bind(dfa, vocabulary, context):
        assert context["planning_facts"] is facts
        assert context["semantic_action_permissions"] == ("clean",)
        assert context["generation_entity_types"] == ("cup",)
        return bound

    monkeypatch.setattr(constraint_compiler, "bind_eai_policy", bind)

    class Probe:
        def vocabulary(self, domain):
            return SimpleNamespace(label_count=1, eos_label=0)

        def sample_labels_from_context(self, domain, encoded, dfa, **kwargs):
            assert dfa is bound
            return [0], torch.tensor(0.0)

    view = JointPlannerDomainView(Probe(), "eai")
    labels, _ = view.sample_with_logprob(
        encoded_context=object(),
        dfa=object(),
        planning_facts=facts,
        semantic_action_permissions=("clean",),
        generation_entity_types=("cup",),
    )
    assert labels == [0]
