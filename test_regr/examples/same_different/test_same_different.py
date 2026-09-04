import sys
import pytest
import torch

sys.path.append('.')
sys.path.append('../..')


@pytest.fixture()
def program():
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel

    from .graph import graph, image, object_node, image_contains_object, color

    graph.detach()

    # Image container
    image['index'] = ReaderSensor(keyword='image')

    # Object sensors
    object_node['index'] = ReaderSensor(keyword='objects')
    object_node[image_contains_object] = EdgeSensor(
        object_node['index'], image['index'],
        relation=image_contains_object,
        forward=lambda x, _: torch.ones_like(x).unsqueeze(-1)
    )

    # Color EnumConcept learner on objects
    from .sensors import ColorEnumLearner
    object_node[color] = ColorEnumLearner('index')

    return LearningBasedProgram(graph, PoiModel, poi=[image, object_node])


@pytest.fixture()
def dataset():
    from .reader import SameDifferentReader
    return list(SameDifferentReader().run())


# =====================================================================
# ILP Tests
# =====================================================================

@pytest.mark.gurobi
def test_sameL_ilp_verify(program, dataset):
    """
    Verify sameL and differentL constraints via ILP.

    Both x and y range over the same set of objects, aligned row-by-row.
    Each object is compared to itself, so:
    - sameL should be satisfied (each object has the same color as itself)
    - differentL should be violated
    """
    from .graph import color, same_color, diff_color

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None

        conceptsRelations = (color,)
        datanode.inferILPResults(*conceptsRelations, fun=None, minimizeObjective=False)

        results = datanode.verifyResultsLC(key="/local/argmax")

        print("\n=== verifyResultsLC (sameL/differentL) ===")
        for lc_name, result in results.items():
            print(f"{lc_name}: satisfied={result['satisfied']}")

        assert len(results) > 0, "Expected verification results"


# =====================================================================
# Loss Tests
# =====================================================================

def test_sameL_loss(program, dataset):
    """
    Test sameL constraint loss computation.
    Self-comparison: each object vs itself -> sameL loss should be low.
    """
    from .graph import same_color

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None

        for tnorm in ['L', 'P']:
            lc_results = datanode.calculateLcLoss(tnorm=tnorm)

            lc_name = same_color.lcName
            if lc_name in lc_results:
                loss = lc_results[lc_name]['lossTensor']
                print(f"\nsameL loss (tnorm={tnorm}): shape={loss.shape}, values={loss}")
                assert torch.is_tensor(loss), "Loss should be a tensor"
                assert not torch.isnan(loss).any(), "Loss should not contain NaN"


def test_differentL_loss(program, dataset):
    """
    Test differentL constraint loss computation.
    Self-comparison: each object vs itself -> differentL loss should be high.
    """
    from .graph import diff_color

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None

        for tnorm in ['L', 'P']:
            lc_results = datanode.calculateLcLoss(tnorm=tnorm)

            lc_name = diff_color.lcName
            if lc_name in lc_results:
                loss = lc_results[lc_name]['lossTensor']
                print(f"\ndifferentL loss (tnorm={tnorm}): shape={loss.shape}, values={loss}")
                assert torch.is_tensor(loss), "Loss should be a tensor"
                assert not torch.isnan(loss).any(), "Loss should not contain NaN"


def test_same_and_different_exact_circuit_loss(program, dataset):
    """The circuit path preserves identity when an entity is compared to itself."""
    from .graph import same_color, diff_color

    datanode = next(program.populate(dataset=dataset))
    results = datanode.calculateLcLoss(circuit=True)

    same_result = results[same_color.lcName]
    different_result = results[diff_color.lcName]
    assert same_result["backend"] in {"bdd", "pysdd"}
    assert same_result["probability"].item() == pytest.approx(1.0, abs=1e-6)
    assert same_result["loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert different_result["probability"].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(different_result["loss"])


@pytest.mark.parametrize("tnorm", ["L", "G", "P"])
def test_sameL_loss_uses_every_enum_class(program, dataset, tnorm):
    """sameL must read all K class probabilities, not just the first.

    ``sameL`` is ``OR_j( AND_i( entity_i_has_class_j ) )``. Each object here is
    compared against itself, and the three objects' probability vectors are
    *permutations* of one another (object 1/2 peak on 'red', object 3 peaks on
    'blue'). A correct evaluation therefore has to give all three rows the
    **same** loss — the value depends only on the multiset of class
    probabilities, not on which class happens to be the peak.

    If the gather keeps only class 0, the blue object is scored as
    ``p_red(o)^2`` instead of ``p_blue(o)^2`` and its row diverges sharply.
    """
    datanode = next(program.populate(dataset=dataset))
    results = datanode.calculateLcLoss(tnorm=tnorm)

    from .graph import same_color
    loss = results[same_color.lcName]["lossTensor"]
    assert loss.shape[0] == 3, f"expected one row per object, got {tuple(loss.shape)}"

    red_a, red_b, blue = loss[0].item(), loss[1].item(), loss[2].item()
    assert red_a == pytest.approx(red_b, abs=1e-6)
    assert blue == pytest.approx(red_a, abs=1e-4), (
        f"blue object scored {blue:.4f} vs red {red_a:.4f} — sameL is only "
        "reading the first enum class"
    )


def test_sameL_loss_agrees_with_exact_circuit(program, dataset):
    """The t-norm and circuit backends must not contradict each other.

    The circuit path asserts ``probability == 1.0`` for these self-comparisons
    (see test_same_and_different_exact_circuit_loss). The fuzzy path will not
    reach exactly 0 loss, but it must at least rank every object as more
    satisfied than violated once all classes are read.
    """
    datanode = next(program.populate(dataset=dataset))

    from .graph import same_color
    fuzzy = datanode.calculateLcLoss(tnorm="G")[same_color.lcName]["lossTensor"]
    exact = datanode.calculateLcLoss(circuit=True)[same_color.lcName]["probability"]

    assert exact.item() == pytest.approx(1.0, abs=1e-6)
    # Godel: success = max_j p_j, which for a 3-class softmax peak is ~0.54.
    assert torch.all(fuzzy < 0.55), \
        f"t-norm sameL contradicts the exact backend: losses {fuzzy}"


@pytest.mark.parametrize("tnorm", ["L", "G", "P"])
def test_sameL_differentL_compiled_matches_interpreter(program, dataset, tnorm):
    """R1's compiled path must reproduce sameL/differentL exactly.

    These used to fall back to the interpreter; they are now compiled, so the
    equivalence guarantee has to be asserted explicitly.
    """
    datanode = next(program.populate(dataset=dataset))

    ref = datanode.calculateLcLoss(tnorm=tnorm)
    cmp = datanode.calculateLcLoss(tnorm=tnorm, compiled=True)

    assert set(ref.keys()) == set(cmp.keys())
    for name in ref:
        rt, ct = ref[name]["lossTensor"], cmp[name]["lossTensor"]
        if rt is None or ct is None:
            assert rt is None and ct is None, f"{name}: one path produced None"
            continue
        assert rt.shape == ct.shape, f"{name}: shape {rt.shape} vs {ct.shape}"
        rnan, cnan = torch.isnan(rt), torch.isnan(ct)
        assert torch.equal(rnan, cnan), f"{name}: NaN masks differ"
        assert torch.allclose(rt[~rnan], ct[~cnan], atol=1e-6), \
            f"{name}: values differ\ninterpreter={rt}\ncompiled={ct}"


def test_sameL_differentL_do_not_fall_back(program, dataset, monkeypatch):
    """Guard against a vacuous pass: they must be compiled, not delegated."""
    from domiknows.solver.compiled import formula as formula_mod

    calls = {"fallback": 0}
    orig = formula_mod.LossCalculator.calculate_single_lc_loss

    def spy(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        if isinstance(self, formula_mod.CompiledLossCalculator) and result is not None:
            calls["fallback"] += 1
        return result

    monkeypatch.setattr(formula_mod.LossCalculator, "calculate_single_lc_loss", spy)

    datanode = next(program.populate(dataset=dataset))
    results = datanode.calculateLcLoss(tnorm="P", compiled=True)

    assert results, "no constraints evaluated"
    assert calls["fallback"] == 0, \
        f"{calls['fallback']} constraints fell back despite sameL/differentL support"
