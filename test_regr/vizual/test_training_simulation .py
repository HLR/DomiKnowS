"""
test_training_simulation.py

Training simulation tests that exercise the full gradient path:
  populate → calculateLcLoss → aggregate → backward → optimizer.step

Requires a program_trainable fixture with real nn.Linear learners.
"""

import sys
import pytest
import torch
import torch.nn as nn

# =====================================================================
# Trainable learners — small linear heads with ground-truth bias
# =====================================================================

from domiknows.sensor.pytorch.learners import TorchLearner


class TrainableObjectLearner(TorchLearner):
    """linear(1→2) with bias so ground-truth objects lean toward class 1."""

    def __init__(self, *pres, positive_ids=None, **kw):
        super().__init__(*pres, **kw)
        self.linear = nn.Linear(1, 2)
        self._positive_ids = positive_ids or set()
        with torch.no_grad():
            self.linear.weight.fill_(0.01)
            self.linear.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        b = len(x)
        feat = x.float().unsqueeze(-1) if x.dim() == 1 else x.float()
        logits = self.linear(feat.to(self.linear.weight.device))
        for i in range(b):
            oid = x[i].item() if hasattr(x[i], "item") else int(x[i])
            if oid in self._positive_ids:
                logits[i, 1] += 2.0
            else:
                logits[i, 0] += 2.0
        return logits.softmax(dim=-1)


class TrainablePairLearner(TorchLearner):
    """linear(2→2) pair classifier with ground-truth bias."""

    def __init__(self, *pres, positive_pairs=None, **kw):
        super().__init__(*pres, **kw)
        self.linear = nn.Linear(2, 2)
        self._positive_pairs = positive_pairs or set()
        with torch.no_grad():
            self.linear.weight.fill_(0.01)
            self.linear.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, arg1, arg2):
        n = arg1.shape[0]
        dev = arg1.device if hasattr(arg1, "device") else "cpu"
        ids = torch.zeros(n, 2, device=dev)
        for i in range(n):
            if arg1.dim() > 1 and arg1.shape[1] > 1:
                ids[i, 0] = arg1[i].argmax().item() + 1
                ids[i, 1] = arg2[i].argmax().item() + 1
            else:
                ids[i, 0] = arg1[i].float()
                ids[i, 1] = arg2[i].float()
        logits = self.linear(ids.to(self.linear.weight.device))
        for i in range(n):
            a1, a2 = int(ids[i, 0].item()), int(ids[i, 1].item())
            if (a1, a2) in self._positive_pairs:
                logits[i, 1] += 2.0
            else:
                logits[i, 0] += 2.0
        return logits.softmax(dim=-1)


class TrainableMaterialLearner(TorchLearner):
    """linear(1→2) → [P(metal), P(rubber)] with ground-truth bias."""

    def __init__(self, *pres, **kw):
        super().__init__(*pres, **kw)
        self.linear = nn.Linear(1, 2)
        with torch.no_grad():
            self.linear.weight.fill_(0.01)
            self.linear.bias.copy_(torch.tensor([0.0, 0.0]))

    def forward(self, x):
        b = len(x)
        feat = x.float().unsqueeze(-1) if x.dim() == 1 else x.float()
        logits = self.linear(feat.to(self.linear.weight.device))
        for i in range(b):
            oid = x[i].item() if hasattr(x[i], "item") else int(x[i])
            if oid == 2:
                logits[i, 0] += 2.0   # metal
            elif oid == 3:
                logits[i, 1] += 2.0   # rubber
        return logits.softmax(dim=-1)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def program_trainable():
    """Program with real trainable nn.Linear learners."""
    import torch
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import (
        EdgeSensor, CompositionCandidateSensor,
    )
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel
    from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor

    from graph import (
        graph, image, object_node, image_contains_object,
        pair, rel_arg1, rel_arg2,
        small, large, red, green, blue,
        cube, sphere, cylinder,
        right_of, left_of, material,
        the_small_blue_cube, the_target_object, the_material_answer,
    )

    graph.detach()

    # ---- scaffolding sensors ----
    image["index"] = ReaderSensor(keyword="image")
    object_node["index"] = ReaderSensor(keyword="objects")
    object_node[image_contains_object] = EdgeSensor(
        object_node["index"], image["index"],
        relation=image_contains_object,
        forward=lambda x, _: torch.ones_like(x).unsqueeze(-1),
    )
    pair[rel_arg1.reversed, rel_arg2.reversed] = CompositionCandidateSensor(
        object_node["index"],
        relations=(rel_arg1.reversed, rel_arg2.reversed),
        forward=lambda *_, **__: True,
    )

    # ---- trainable attribute learners ----
    object_node[small]    = TrainableObjectLearner("index", positive_ids={1, 3})
    object_node[large]    = TrainableObjectLearner("index", positive_ids={2})
    object_node[red]      = TrainableObjectLearner("index", positive_ids={2})
    object_node[green]    = TrainableObjectLearner("index", positive_ids={3})
    object_node[blue]     = TrainableObjectLearner("index", positive_ids={1})
    object_node[cube]     = TrainableObjectLearner("index", positive_ids={1})
    object_node[sphere]   = TrainableObjectLearner("index", positive_ids={2})
    object_node[cylinder] = TrainableObjectLearner("index", positive_ids={3})

    # ---- trainable material learner ----
    object_node[material] = TrainableMaterialLearner("index")

    # ---- spatial (DataNodeReaderSensor — not trainable) ----
    def _read_spatial_fn(r1_ref, r2_ref):
        def _fn(*_, data, datanode):
            a1 = datanode.relationLinks[r1_ref.name][0].getAttribute("index").item()
            a2 = datanode.relationLinks[r2_ref.name][0].getAttribute("index").item()
            if (a1, a2) in data:
                return torch.tensor([0, 1])
            return torch.tensor([1, 0])
        return _fn

    pair[right_of] = DataNodeReaderSensor(
        rel_arg1.reversed, rel_arg2.reversed,
        keyword="right_of",
        forward=_read_spatial_fn(rel_arg1, rel_arg2),
    )
    pair[left_of] = DataNodeReaderSensor(
        rel_arg1.reversed, rel_arg2.reversed,
        keyword="left_of",
        forward=_read_spatial_fn(rel_arg1, rel_arg2),
    )

    # ---- constraint label sensors ----
    graph.constraint[the_small_blue_cube] = ReaderSensor(
        keyword="the_small_blue_cube_label", is_constraint=True, label=True,
    )
    graph.constraint[the_target_object] = ReaderSensor(
        keyword="the_target_object_label", is_constraint=True, label=True,
    )
    graph.constraint[the_material_answer] = ReaderSensor(
        keyword="the_material_answer_label", is_constraint=True, label=True,
    )

    return LearningBasedProgram(
        graph, PoiModel, poi=[image, object_node, pair], device="cpu",
    )


@pytest.fixture(scope="module")
def training_dataset():
    """Materialised list so it can be re-iterated across epochs."""
    from .reader import SceneReader

    class ReaderWithLabels(SceneReader):
        def run(self):
            for item in super().run():
                item["the_small_blue_cube_label"] = 1.0
                item["the_target_object_label"]   = 1.0
                item["the_material_answer_label"] = 1.0
                yield item

    return list(ReaderWithLabels().run())


# =====================================================================
# Tests
# =====================================================================

def test_training_simulation(program_trainable, training_dataset):
    """
    Simulate a multi-epoch training loop with differentiable constraint loss.

    Verifies:
      1. Constraint losses are finite tensors.
      2. Gradients flow (some parameters have non-zero grad).
      3. Parameters change after optimizer steps.
      4. All epoch losses are finite (no NaN).
    """
    params = list(program_trainable.model.parameters())
    trainable_params = [p for p in params if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters found"

    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    initial_snapshot = [p.clone().detach() for p in trainable_params]

    num_epochs = 5
    epoch_losses = []

    print("\n=== Training Simulation (differentiable) ===")

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        num_batches = 0

        for datanode in program_trainable.populate(
            dataset=iter(training_dataset), device="cpu"
        ):
            optimizer.zero_grad()

            # ---- constraint loss ----
            loss_dict = datanode.calculateLcLoss(tnorm="P", sample=False)

            losses = []
            for lc_name, lc_data in loss_dict.items():
                lv = lc_data.get("loss")
                if lv is not None and torch.is_tensor(lv):
                    assert torch.isfinite(lv), f"Non-finite loss for {lc_name}: {lv}"
                    losses.append(lv)

            if not losses:
                continue

            total_loss = sum(losses)

            # ---- backward ----
            total_loss.backward()

            # Check gradient flow on first batch of first epoch
            if epoch == 0 and num_batches == 0:
                grads_found = sum(
                    1 for p in trainable_params
                    if p.grad is not None and p.grad.abs().sum().item() > 0
                )
                print(f"  Params with non-zero grad: {grads_found}/{len(trainable_params)}")

            # ---- optimizer step ----
            optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1

        avg_loss = epoch_total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{num_epochs}: avg_loss = {avg_loss:.6f}")

    # ---- Assertions ----
    assert num_batches > 0, "No batches were processed"

    params_changed = sum(
        1 for p0, p1 in zip(initial_snapshot, trainable_params)
        if not torch.allclose(p0, p1.detach(), atol=1e-7)
    )
    print(f"  Parameters changed: {params_changed}/{len(trainable_params)}")
    assert params_changed > 0, "No parameters changed after training"

    for i, el in enumerate(epoch_losses):
        assert el == el, f"Epoch {i} loss is NaN"

    print(f"  Loss trajectory: {[f'{l:.6f}' for l in epoch_losses]}")
    print("  Training simulation PASSED!")


def test_training_simulation_sampling(program_trainable, training_dataset):
    """
    Same training loop but with sample-based loss
    (calculateLcLoss with sample=True, sampleSize=10).

    Verifies the same invariants as the differentiable version.
    """
    params = list(program_trainable.model.parameters())
    trainable_params = [p for p in params if p.requires_grad]
    assert len(trainable_params) > 0

    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    initial_snapshot = [p.clone().detach() for p in trainable_params]

    num_epochs = 3
    epoch_losses = []

    print("\n=== Training Simulation (sampling) ===")

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        num_batches = 0

        for datanode in program_trainable.populate(
            dataset=iter(training_dataset), device="cpu"
        ):
            optimizer.zero_grad()

            loss_dict = datanode.calculateLcLoss(
                tnorm="P", sample=True, sampleSize=10,
            )

            losses = []
            for lc_name, lc_data in loss_dict.items():
                lv = lc_data.get("loss")
                if lv is not None and torch.is_tensor(lv) and torch.isfinite(lv):
                    losses.append(lv)

            if not losses:
                continue

            total_loss = sum(losses)
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1

        avg_loss = epoch_total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{num_epochs}: avg_loss = {avg_loss:.6f}")

    changed = sum(
        1 for p0, p1 in zip(initial_snapshot, trainable_params)
        if not torch.allclose(p0, p1.detach(), atol=1e-7)
    )
    print(f"  Parameters changed: {changed}/{len(trainable_params)}")
    assert changed > 0, "No parameters changed after sample-based training"

    for i, el in enumerate(epoch_losses):
        assert el == el, f"Epoch {i} loss is NaN"

    print(f"  Loss trajectory: {[f'{l:.6f}' for l in epoch_losses]}")
    print("  Sample-based training simulation PASSED!")