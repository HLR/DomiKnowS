import pytest
import torch
import torch.nn as nn

from domiknows.graph import Concept, Graph
from domiknows.graph.dataNode import DataNode, DataNodeBuilder
from domiknows.program.model.pytorch import SolverModel, TorchModel
from domiknows.program.gradient_manager import GradientConflictManager
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor
from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory


def _clear_state():
    Graph.clear()
    Concept.clear()
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()


@pytest.fixture(autouse=True)
def clear_domiknows_state():
    _clear_state()
    yield
    _clear_state()


class _TwoBranchModel(nn.Module):
    def __init__(self, shared_backbone, head):
        super().__init__()
        self.backbone = shared_backbone
        self.head = head

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)


def _build_shared_backbone_graph():
    with Graph("multi_domain_world") as graph:
        item = Concept(name="item")
        cat = item(name="cat")
        dog = item(name="dog")

    item["features"] = FunctionalReaderSensor(
        keyword="features",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32),
    )

    backbone = nn.Linear(4, 4)
    head_cat = nn.Linear(4, 2)
    head_dog = nn.Linear(4, 2)

    model_cat = _TwoBranchModel(backbone, head_cat)
    model_dog = _TwoBranchModel(backbone, head_dog)

    item[cat] = ModuleLearner("features", module=model_cat)
    item[dog] = ModuleLearner("features", module=model_dog)

    torch_model = SolverModel(
        graph,
        poi=[item, graph.constraint],
        inferTypes=[],
        device="cpu",
    )
    return graph, item, cat, dog, backbone, head_cat, head_dog, torch_model, model_cat, model_dog


def test_classify_parameters_identifies_shared_and_domain_private():
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    classification = model.classify_parameters([cat])

    cat_pids = {id(p) for p in head_cat.parameters()}
    dog_pids = {id(p) for p in head_dog.parameters()}
    backbone_pids = {id(p) for p in backbone.parameters()}

    active_private_pids = {id(p) for p in classification["active_private"]}
    inactive_private_pids = {id(p) for p in classification["inactive_private"]}
    shared_pids = {id(p) for p in classification["shared"]}

    assert cat_pids.issubset(active_private_pids)
    assert dog_pids.issubset(inactive_private_pids)
    assert backbone_pids.issubset(shared_pids)


def test_freeze_inactive_policy_protects_inactive_heads():
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    # Initially all require grad
    assert all(p.requires_grad for p in model.parameters())

    graph.set_active_concepts([cat], parameter_policy="freeze_inactive")

    # cat head and backbone still require grad
    assert all(p.requires_grad for p in head_cat.parameters())
    assert all(p.requires_grad for p in backbone.parameters())
    # dog head is frozen
    assert all(not p.requires_grad for p in head_dog.parameters())

    # Switch to dog
    graph.set_active_concepts([dog], parameter_policy="freeze_inactive")
    assert all(not p.requires_grad for p in head_cat.parameters())
    assert all(p.requires_grad for p in head_dog.parameters())
    assert all(p.requires_grad for p in backbone.parameters())

    # Reset
    graph.set_active_concepts(None)
    assert all(p.requires_grad for p in head_cat.parameters())
    assert all(p.requires_grad for p in head_dog.parameters())
    assert all(p.requires_grad for p in backbone.parameters())


def test_freeze_shared_policy_stops_backbone_gradients_in_backward():
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    with graph.active_scope([cat], parameter_policy="freeze_shared"):
        # cat head requires grad
        assert all(p.requires_grad for p in head_cat.parameters())
        # backbone and dog head are both frozen
        assert all(not p.requires_grad for p in backbone.parameters())
        assert all(not p.requires_grad for p in head_dog.parameters())

        # Forward through cat
        inputs = torch.randn(2, 4)
        output = model_cat(inputs)
        loss = output.sum()
        loss.backward()

        # cat head received gradients
        assert head_cat.weight.grad is not None
        # backbone and dog received NO gradients
        assert backbone.weight.grad is None
        assert head_dog.weight.grad is None

    # After exiting active_scope, original requires_grad is restored
    assert all(p.requires_grad for p in backbone.parameters())
    assert all(p.requires_grad for p in head_cat.parameters())
    assert all(p.requires_grad for p in head_dog.parameters())


def test_gradient_conflict_manager_pcgrad_projection():
    shared = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
    head_a = nn.Parameter(torch.tensor([3.0], requires_grad=True))
    head_b = nn.Parameter(torch.tensor([4.0], requires_grad=True))

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = shared
            self.head_a = head_a
            self.head_b = head_b

    model = DummyModel()
    optimizer = torch.optim.SGD([shared, head_a, head_b], lr=0.1)
    manager = GradientConflictManager(model, optimizer)

    # Loss A pushes shared towards positive
    loss_a = (shared * torch.tensor([1.0, 1.0])).sum() + head_a.sum()
    manager.record_scope_gradients("scope_a", loss_a)

    # Loss B pushes shared towards negative (opposing gradient -> conflict!)
    loss_b = (shared * torch.tensor([-2.0, -1.0])).sum() + head_b.sum()
    manager.record_scope_gradients("scope_b", loss_b)

    resolved = manager.resolve_gradients(method="pcgrad")
    assert len(resolved) == 3

    # Applying resolved gradients updates shared parameter with projected gradient
    manager.apply_resolved_gradients(resolved)

    assert shared.grad is not None
    assert head_a.grad is not None
    assert head_b.grad is not None
    # No NaN or Inf
    assert torch.isfinite(shared.grad).all()
    assert torch.isfinite(head_a.grad).all()
    assert torch.isfinite(head_b.grad).all()


def test_nested_active_scope_restores_stack_states():
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    assert all(p.requires_grad for p in model.parameters())

    with graph.active_scope([cat], parameter_policy="freeze_inactive"):
        # cat head & backbone trainable; dog head frozen
        assert all(p.requires_grad for p in head_cat.parameters())
        assert all(p.requires_grad for p in backbone.parameters())
        assert all(not p.requires_grad for p in head_dog.parameters())

        # Nested scope freezes shared backbone as well
        with graph.active_scope([cat], parameter_policy="freeze_shared"):
            assert all(p.requires_grad for p in head_cat.parameters())
            assert all(not p.requires_grad for p in backbone.parameters())
            assert all(not p.requires_grad for p in head_dog.parameters())

        # Exiting inner scope restores outer scope state: backbone is trainable again!
        assert all(p.requires_grad for p in head_cat.parameters())
        assert all(p.requires_grad for p in backbone.parameters())
        assert all(not p.requires_grad for p in head_dog.parameters())

    # Exiting outer scope restores baseline: all trainable!
    assert all(p.requires_grad for p in model.parameters())


def test_newly_frozen_parameters_clear_grad():
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    # Simulate stale gradient on dog head
    for p in head_dog.parameters():
        p.grad = torch.ones_like(p.data)

    # Activating cat with freeze_inactive must clear dog.grad to None
    graph.set_active_concepts([cat], parameter_policy="freeze_inactive")
    for p in head_dog.parameters():
        assert p.grad is None
        assert not p.requires_grad


def test_automatically_activated_is_a_ancestors_do_not_classify_private_as_shared():
    # item is an ancestor of cat and dog. Activating dog activates item automatically,
    # but must NOT cause cat-private parameters to be classified as shared.
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    classification = model.classify_parameters([dog])
    cat_pids = {id(p) for p in head_cat.parameters()}
    shared_pids = {id(p) for p in classification["shared"]}
    inactive_private_pids = {id(p) for p in classification["inactive_private"]}

    # cat parameters must be inactive_private, not shared!
    assert cat_pids.issubset(inactive_private_pids)
    assert not cat_pids.intersection(shared_pids)


def test_unmapped_parameters_conservative_policy():
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    # Add an unmapped parameter directly to the model
    model.unmapped_linear = nn.Linear(3, 3)
    classification = model.classify_parameters([cat])
    unmapped_pids = {id(p) for p in model.unmapped_linear.parameters()}
    assert unmapped_pids.issubset({id(p) for p in classification["unmapped"]})

    # Under freeze_shared, unmapped parameters must freeze to protect unmapped shared backbones
    with graph.active_scope([cat], parameter_policy="freeze_shared"):
        for p in model.unmapped_linear.parameters():
            assert not p.requires_grad
            assert p.grad is None

    # After exiting, unmapped parameters are restored to trainable
    for p in model.unmapped_linear.parameters():
        assert p.requires_grad


def test_gradient_conflict_manager_recommended_workflow():
    shared = nn.Parameter(torch.tensor([2.0, 3.0], requires_grad=True))
    head_a = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    head_b = nn.Parameter(torch.tensor([5.0], requires_grad=True))

    class TwoDomainNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = shared
            self.head_a = head_a
            self.head_b = head_b

    net = TwoDomainNet()
    optimizer = torch.optim.SGD([shared, head_a, head_b], lr=0.1)
    gm = GradientConflictManager(net, optimizer)

    gm.begin_step()

    # Domain A pass
    loss_a = (shared * torch.tensor([1.0, 1.0])).sum() + head_a * 2.0
    gm.capture("domain_a", loss_a)

    # Domain B pass (opposing direction on shared -> PCGrad projection required)
    loss_b = (shared * torch.tensor([-2.0, -1.0])).sum() + head_b * 3.0
    gm.capture("domain_b", loss_b)

    init_shared = shared.clone().detach()
    init_head_a = head_a.clone().detach()
    init_head_b = head_b.clone().detach()

    # Single synchronized step
    gm.step(optimizer)

    assert not torch.equal(shared, init_shared)
    assert not torch.equal(head_a, init_head_a)
    assert not torch.equal(head_b, init_head_b)


def test_checkpoint_retains_learned_parameters_during_freeze_shared():
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    # Outside scope, all are learnable
    base_names = model.get_learnable_parameter_names()
    assert len(base_names) > 0

    with graph.active_scope([cat], parameter_policy="freeze_shared"):
        # Inside freeze_shared, backbone and dog are requires_grad=False
        assert all(not p.requires_grad for p in backbone.parameters())
        # But trainable_state_dict and get_learnable_parameter_names STILL retain them!
        learnable_names = model.get_learnable_parameter_names()
        assert learnable_names == base_names
        saved_state = model.trainable_state_dict()
        assert len(saved_state) == len(base_names)


def test_checkpoint_parameter_ownership_checksum():
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    checksum1 = model.compute_parameter_ownership_checksum()
    assert isinstance(checksum1, str)
    assert len(checksum1) == 64  # SHA-256

    # Adding a parameter changes checksum
    model.extra_param = nn.Parameter(torch.zeros(5, 5))
    checksum2 = model.compute_parameter_ownership_checksum()
    assert checksum1 != checksum2


def test_gradient_conflict_manager_capture_with_freeze_inactive():
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    gm = GradientConflictManager(model, optimizer)

    gm.begin_step()

    # Domain 1: cat
    with graph.active_scope([cat], parameter_policy="freeze_inactive"):
        assert all(p.requires_grad for p in head_cat.parameters())
        assert all(not p.requires_grad for p in head_dog.parameters())
        x = torch.randn(2, 4)
        cat_loss = model_cat(x).sum()
        # This capture must succeed despite head_dog having requires_grad=False!
        gm.capture("cat", cat_loss)

    # Domain 2: dog
    with graph.active_scope([dog], parameter_policy="freeze_inactive"):
        assert all(not p.requires_grad for p in head_cat.parameters())
        assert all(p.requires_grad for p in head_dog.parameters())
        x = torch.randn(2, 4)
        dog_loss = model_dog(x).sum()
        # This capture must also succeed despite head_cat having requires_grad=False!
        gm.capture("dog", dog_loss)

    init_backbone = backbone.weight.clone().detach()
    init_head_cat = head_cat.weight.clone().detach()
    init_head_dog = head_dog.weight.clone().detach()

    gm.step(optimizer, method="pcgrad")

    assert not torch.equal(backbone.weight, init_backbone)
    assert not torch.equal(head_cat.weight, init_head_cat)
    assert not torch.equal(head_dog.weight, init_head_dog)


def test_transaction_lock_enforces_single_thread_ownership():
    import threading
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    gm = GradientConflictManager(model, optimizer)

    gm.begin_step()
    error_raised = []

    def other_thread():
        try:
            gm.begin_step()
        except RuntimeError as e:
            error_raised.append(e)

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()

    assert len(error_raised) == 1
    assert "already in progress" in str(error_raised[0])

    # End the step
    gm.abort_step()

    # Now other thread can succeed
    success = []
    def other_thread_2():
        try:
            gm.begin_step()
            gm.abort_step()
            success.append(True)
        except Exception:
            pass

    t2 = threading.Thread(target=other_thread_2)
    t2.start()
    t2.join()
    assert success == [True]


def test_ownership_repartitioning_alters_checksum():
    import hashlib
    records_baseline = [
        ("backbone.weight", (4, 4), "shared"),
        ("head_cat.weight", (2, 4), "eai_private"),
        ("head_dog.weight", (2, 4), "vlabench_private"),
    ]
    records_repartitioned = [
        ("backbone.weight", (4, 4), "shared"),
        ("head_cat.weight", (2, 4), "shared"),
        ("head_dog.weight", (2, 4), "vlabench_private"),
    ]
    hash1 = hashlib.sha256(repr(records_baseline).encode('utf-8')).hexdigest()
    hash2 = hashlib.sha256(repr(records_repartitioned).encode('utf-8')).hexdigest()
    assert hash1 != hash2

    # In TorchModel, static bindings are hashed
    graph, item, cat, dog, backbone, head_cat, head_dog, model, model_cat, model_dog = _build_shared_backbone_graph()
    c1 = model.compute_parameter_ownership_checksum()

    # Check that checksum is invariant to runtime dynamic activation
    with graph.active_scope([cat], parameter_policy="freeze_inactive"):
        c2 = model.compute_parameter_ownership_checksum()
    assert c1 == c2

    with graph.active_scope([dog], parameter_policy="freeze_shared"):
        c3 = model.compute_parameter_ownership_checksum()
    assert c1 == c3


def test_same_thread_nested_begin_step_is_rejected():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    gm = GradientConflictManager(model, optimizer)

    gm.begin_step()
    # A second begin_step on the same thread MUST raise RuntimeError
    with pytest.raises(RuntimeError, match="already in progress"):
        gm.begin_step()

    gm.abort_step()
    # After abort, begin_step succeeds again
    gm.begin_step()
    gm.abort_step()


def test_abort_step_enforces_thread_ownership():
    import threading
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    gm = GradientConflictManager(model, optimizer)

    gm.begin_step()
    error_raised = []

    def other_thread_abort():
        try:
            gm.abort_step()
        except RuntimeError as e:
            error_raised.append(e)

    t = threading.Thread(target=other_thread_abort)
    t.start()
    t.join()

    assert len(error_raised) == 1
    assert "Cannot abort step; transaction owned by thread" in str(error_raised[0])
    gm.abort_step()


def test_gradient_conflict_manager_parameter_subset_alignment():
    p1 = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    p2 = nn.Parameter(torch.tensor([2.0], requires_grad=True))
    p3 = nn.Parameter(torch.tensor([3.0], requires_grad=True))

    class MultiParamNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3

    net = MultiParamNet()
    optimizer = torch.optim.SGD([p1, p2, p3], lr=0.1)
    gm = GradientConflictManager(net, optimizer)

    gm.begin_step()

    # Scope 1 updates only subset [p1, p3]
    loss1 = p1 * 2.0 + p3 * 4.0
    grads1 = gm.capture("scope1", loss1, parameters=[p1, p3])
    # Must be aligned to [p1, p2, p3] with p2 having None
    assert len(grads1) == 3
    assert grads1[0] is not None
    assert grads1[1] is None
    assert grads1[2] is not None

    # Scope 2 updates only subset [p2, p3]
    loss2 = p2 * 3.0 + p3 * 5.0
    grads2 = gm.capture("scope2", loss2, parameters=[p2, p3])
    assert len(grads2) == 3
    assert grads2[0] is None
    assert grads2[1] is not None
    assert grads2[2] is not None

    # Resolution should align each parameter correctly
    resolved = gm.resolve_gradients(method="average")
    assert len(resolved) == 3
    assert torch.isclose(resolved[0], torch.tensor([2.0]))
    assert torch.isclose(resolved[1], torch.tensor([3.0]))
    assert torch.isclose(resolved[2], torch.tensor([4.5]))  # (4 + 5) / 2

    gm.step(optimizer)


def test_qwenvlplanner_compute_parameter_ownership_checksum():
    from types import SimpleNamespace
    from test_regr.VLABenchAgentInterface.graph import PlanVocabulary
    from test_regr.VLABenchAgentInterface.models import QwenVLPlanner

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(4, 4)
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, *args, **kwargs):
            return None

    vocab = PlanVocabulary(
        skills=("pick",),
        argument_keys=("target_entity_name",),
        skill_arguments=(("pick", ("target_entity_name",)),),
        max_entities=2,
    )
    planner1 = QwenVLPlanner(DummyBackbone(), None, vocab, hidden_size=4, decoder_hidden_size=4)
    c1 = planner1.compute_parameter_ownership_checksum()
    assert isinstance(c1, str)
    assert len(c1) == 64

    planner2 = QwenVLPlanner(DummyBackbone(), None, vocab, hidden_size=4, decoder_hidden_size=4)
    c2 = planner2.compute_parameter_ownership_checksum()
    assert c1 == c2

    # Changing shape changes checksum
    planner3 = QwenVLPlanner(DummyBackbone(), None, vocab, hidden_size=4, decoder_hidden_size=2)
    c3 = planner3.compute_parameter_ownership_checksum()
    assert c1 != c3


def test_cannot_bypass_checkpoint_checksum_with_foreign_planner(tmp_path):
    from types import SimpleNamespace
    from test_regr.VLABenchAgentInterface.training import load_joint_checkpoint, save_joint_checkpoint
    from test_regr.VLABenchAgentInterface.graph import PlanVocabulary
    from test_regr.VLABenchAgentInterface.models import QwenVLPlanner, MultiViewController, TinyImageEncoder

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(4, 4)
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, *args, **kwargs):
            return None

    vocab = PlanVocabulary(
        skills=("pick",),
        argument_keys=("target_entity_name",),
        skill_arguments=(("pick", ("target_entity_name",)),),
        max_entities=2,
    )
    planner = QwenVLPlanner(DummyBackbone(), None, vocab, hidden_size=4, decoder_hidden_size=4)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    runtime = SimpleNamespace(
        world_bundle=SimpleNamespace(domain_checksum="d123"),
        vocabulary=vocab,
    )

    ckpt_path = tmp_path / "vlabench_ckpt.pt"
    save_joint_checkpoint(
        ckpt_path,
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=None,
        runtime=runtime,
        stage="supervised",
        epoch=1,
    )

    # Now load with a planner that does NOT implement compute_parameter_ownership_checksum
    class ForeignPlanner(nn.Module):
        def __init__(self):
            super().__init__()
            self.graph_decoder_version = 1
            self.decoder_hidden_size = 4
            self.w = nn.Linear(4, 4)

    foreign_planner = ForeignPlanner()
    with pytest.raises(ValueError, match="does not implement compute_parameter_ownership_checksum"):
        load_joint_checkpoint(
            ckpt_path,
            planner=foreign_planner,
            controller=controller,
            runtime=runtime,
        )



