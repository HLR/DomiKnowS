"""Program-level integration of the R3/R4 structure mechanisms.

R1/R2/R5 shape the constraint *loss* and already had a home in ``LossProgram``'s
``CModel`` slot. R3 (:class:`FactorGraphHead`) and R4
(:class:`ConstraintRefinement`) change the *forward pass* and had no seam at all
— correct, tested modules that no Program called. :class:`StructuredModel`
supplies that seam and :class:`StructuredProgram` adds the constraint partition
R3 makes necessary.

The failure mode is a structured model that trains fine while the structure does
nothing, so these tests check the *mechanism*:

* the write-back actually reaches ``poi_loss`` (and does not, under
  ``constraint_only``) — if this silently no-ops, everything downstream is void;
* a structurally-enforced constraint gets **no multiplier**, asserted against
  ``lmbd_index`` directly rather than inferred from a total;
* MAP decoding respects constraints and is never argmax-of-marginals.
"""

import pytest
import torch

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL
from domiknows.program import StructuredProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.structured import StructuredModel
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor


class _Head(torch.nn.Module):
    """Constant-logit head, so beliefs are exactly known before refinement."""

    def __init__(self, logits):
        super().__init__()
        self.lin = torch.nn.Linear(3, len(logits))
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.copy_(torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.lin(x)


def _exclusive_graph(a_logits=(0.0, 3.0), b_logits=(0.0, 3.0)):
    """Two binary siblings under ``nandL`` — both confident, so it is violated.

    Exclusion needs no relation, which keeps the fixture fast and the failure
    mode unambiguous.
    """
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('structured_test') as graph:
        img = Concept(name='img')
        ent = Concept(name='ent')
        (img_ent,) = img.contains(ent)
        a = ent(name='a')
        b = ent(name='b')
        nandL(a('x'), b('x'))

    img['index'] = ReaderSensor(keyword='img')
    ent['index'] = ReaderSensor(keyword='ents')
    ent[img_ent] = EdgeSensor(ent['index'], img['index'], relation=img_ent,
                              forward=lambda x, _: torch.ones_like(x).unsqueeze(-1))
    ent['emb'] = ReaderSensor(keyword='emb')
    ent[a] = ModuleLearner('emb', module=_Head(a_logits))
    ent[b] = ModuleLearner('emb', module=_Head(b_logits))
    ent[a] = ReaderSensor(keyword='a_lbl', label=True)
    ent[b] = ReaderSensor(keyword='b_lbl', label=True)
    return graph, img, ent, a, b


def _typing_graph():
    """conll04-shaped typing rule plus an exclusion, for partition tests."""
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('structured_typing') as graph:
        ent = Concept(name='ent')
        a = ent(name='a'); b = ent(name='b')
        pair = Concept(name='pair')
        (arg1, arg2) = pair.has_a(arg1=ent, arg2=ent)
        link = pair(name='link')
        ifL(link('x', 'y'), andL(a('x'), b('y')))
        nandL(a('x'), b('x'))
    return graph


def _data():
    return {'img': [0], 'ents': [0, 1], 'emb': torch.zeros(2, 3),
            'a_lbl': torch.tensor([1, 0]), 'b_lbl': torch.tensor([0, 1])}


def _program(graph, poi, **kwargs):
    kwargs.setdefault('inferTypes', ['local/softmax'])
    kwargs.setdefault('loss', MacroAverageTracker(NBCrossEntropyLoss()))
    return StructuredProgram(graph, poi=poi, **kwargs)


def _softmax_of(datanode, ent, concept_name):
    return [dn.getAttribute(f'<{concept_name}>/local/softmax').detach()
            for dn in datanode.findDatanodes(select=ent)]


# --------------------------------------------------------------------------- #
# Verification 1 — the seam fires
# --------------------------------------------------------------------------- #

def test_refinement_reaches_the_constraint_side(capsys):
    """Refined beliefs replace the datanode softmax the constraint loss reads.

    Both siblings start confident (p=0.95) and so violate ``nandL``; refinement
    must push both down. ``inferLocal`` computes the softmax key only when
    absent, which is what lets a pre-written value survive.
    """
    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=True,
                       structure_kwargs={'refine_steps': 5, 'refine_step_size': 5.0})
    program.model.eval()
    _loss, _metric, datanode, _builder = program.model(_data(), build=True)

    pa = [float(t[1]) for t in _softmax_of(datanode, ent, 'a')]
    pb = [float(t[1]) for t in _softmax_of(datanode, ent, 'b')]
    print(f'\nrefined P(a)={pa}  P(b)={pb}  (raw was ~0.9526 for both)')

    assert all(p < 0.9 for p in pa), 'refinement did not reach the datanode'
    assert all(p < 0.9 for p in pb)


def test_belief_flow_controls_whether_the_supervised_loss_sees_refinement(capsys):
    """The write-back half of the seam, asserted by its observable effect.

    ``poi_loss`` reads the value cached under the sensor object, so overwriting
    that entry is what makes the supervised loss train through refinement.
    Under ``constraint_only`` it must not.
    """
    losses = {}
    for flow in ('write_back', 'constraint_only'):
        graph, img, ent, a, b = _exclusive_graph()
        program = _program(graph, [img, ent, a, b], refine=True, belief_flow=flow,
                           structure_kwargs={'refine_steps': 5, 'refine_step_size': 5.0})
        program.model.eval()
        loss, _metric, _dn, _builder = program.model(_data(), build=True)
        losses[flow] = float(loss.detach())
    print(f'\nsupervised loss: write_back={losses["write_back"]:.5f}  '
          f'constraint_only={losses["constraint_only"]:.5f}')

    assert losses['write_back'] != pytest.approx(losses['constraint_only'], abs=1e-4), (
        'both flows gave the same supervised loss — the write-back is a no-op')


def test_no_structure_flags_leaves_the_forward_pass_untouched():
    """With refine/factor_graph off, beliefs are exactly the raw head outputs."""
    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=False, factor_graph=False)
    program.model.eval()
    _loss, _metric, datanode, _builder = program.model(_data(), build=True)
    for p in [float(t[1]) for t in _softmax_of(datanode, ent, 'a')]:
        assert p == pytest.approx(0.9525741, abs=1e-5)


def test_structure_warmup_defers_the_structure():
    """``structure_warmup`` keeps raw beliefs until enough training has run."""
    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=True,
                       structure_kwargs={'structure_warmup': 5,
                                         'refine_steps': 5, 'refine_step_size': 5.0})
    program.model.eval()
    _loss, _metric, datanode, _builder = program.model(_data(), build=True)
    assert float(_softmax_of(datanode, ent, 'a')[0][1]) == pytest.approx(0.9525741, abs=1e-5)


def test_warmup_counts_training_steps_not_evaluation_passes():
    """Evaluation must not burn through the warmup budget.

    Counting every forward would make the effective warmup depend on the
    validation schedule — a run that evaluates often would start rewriting
    beliefs having barely trained.
    """
    from domiknows.program.model.base import Mode

    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=True,
                       structure_kwargs={'structure_warmup': 3,
                                         'refine_steps': 5, 'refine_step_size': 5.0})
    program.model.mode(Mode.TEST)
    for _ in range(10):
        program.model(_data(), build=True)
    assert program.model._train_steps == 0, 'eval passes advanced the warmup counter'

    program.model.mode(Mode.TRAIN)
    for _ in range(3):
        program.model(_data(), build=True)
    assert program.model._train_steps == 3


def test_concept_properties_are_matched_by_identity():
    """Write-back targets the Concept-valued property, not a name substring.

    A concept-valued property stores the Concept *object* in ``prop_name``;
    plain properties like ``'emb'`` keep a ``str``. Matching on that is an
    identity check rather than string-munging a fullname.
    """
    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=True)
    mapping = program.model._concept_properties()
    assert set(mapping) == {'a', 'b'}, mapping
    assert 'emb' not in mapping and 'index' not in mapping
    for name, props in mapping.items():
        for prop in props:
            assert prop.prop_name.name == name


# --------------------------------------------------------------------------- #
# Verification 2 — the partition is real
# --------------------------------------------------------------------------- #

def test_structural_candidates_are_syntactic():
    """Detected from constraint type alone, so they are known at construction."""
    graph = _typing_graph()
    names = StructuredProgram.structural_candidates(graph)
    # the ifL typing rule and the nandL exclusion; the nested andL is not a head
    assert len(names) == 2


def test_enforced_constraints_get_no_multiplier():
    """Excluded constraints must not appear in ``lmbd_index`` at all.

    A multiplier pinned at zero would also feed ``al_dual_update_`` an all-zero
    window — a dual that can only ever learn nothing.
    """
    graph = _typing_graph()
    enforced = StructuredProgram.structural_candidates(graph)

    program = StructuredProgram(graph, poi=[], factor_graph=True)
    assert program.structural_partition == enforced
    for name in enforced:
        assert name not in program.cmodel.lmbd_index
        assert name not in program.cmodel.constr
    assert len(program.cmodel.lmbd) == len(program.cmodel.constr)


def test_refinement_alone_does_not_license_exclusion():
    """Refinement moves beliefs but guarantees nothing, so nothing is excluded."""
    graph = _typing_graph()
    program = StructuredProgram(graph, poi=[], refine=True, factor_graph=False)
    assert program.structural_partition == set()
    assert len(program.cmodel.constr) == len(dict(graph.allLogicalConstrainsRecursive))


def test_adaptive_partition_keeps_constraints_and_skips_per_step():
    """'adaptive' trades one unused multiplier for a self-closing coverage gap.

    Every constraint stays in the cmodel, and the penalty is skipped only for
    those the model reports enforced *this step* — so a circuit that falls back
    silently gets its penalty back instead of leaving the constraint held by
    nothing.
    """
    graph = _typing_graph()
    program = StructuredProgram(graph, poi=[], factor_graph=True, partition='adaptive')

    # nothing is dropped at construction
    assert set(program.cmodel.constr) == set(dict(graph.allLogicalConstrainsRecursive))
    assert program.cmodel.skip_provider is not None

    enforced = sorted(program.structural_candidates(graph))
    program.model.enforced_constraints = {enforced[0]}
    assert program.cmodel.skip_provider() == {enforced[0]}

    # a fallback restores the penalty with no further action
    program.model.enforced_constraints = set()
    assert program.cmodel.skip_provider() == set()
    assert 'WARNING' not in program.report_partition()


def test_adaptive_partition_requires_factor_graph():
    """Refinement guarantees nothing, so it must not install a skip provider."""
    graph = _typing_graph()
    program = StructuredProgram(graph, poi=[], refine=True, factor_graph=False,
                                partition='adaptive')
    assert program.cmodel.skip_provider is None


def test_partition_none_keeps_every_constraint():
    graph = _typing_graph()
    program = StructuredProgram(graph, poi=[], factor_graph=True, partition='none')
    assert program.structural_partition == set()
    assert len(program.cmodel.constr) == len(dict(graph.allLogicalConstrainsRecursive))


def test_report_partition_warns_when_exclusion_over_reached():
    """Excluded-but-not-enforced means enforced by *nothing* — must be surfaced."""
    graph = _typing_graph()
    program = StructuredProgram(graph, poi=[], factor_graph=True)
    # Simulate a runtime where only one excluded constraint actually compiled.
    program.model.enforced_constraints = {sorted(program.structural_partition)[0]}
    text = program.report_partition()
    assert 'WARNING' in text
    assert 'unconstrained' in text


def test_report_partition_warns_when_nothing_was_enforced():
    """The worst case must not be the silent one.

    Nothing enforced *and* everything excluded means every excluded constraint
    is held by neither structure nor penalty. Gating the warning on 'something
    was enforced' suppressed exactly this state.
    """
    graph = _typing_graph()
    program = StructuredProgram(graph, poi=[], factor_graph=True)
    program.model.enforced_constraints = set()
    text = program.report_partition()
    assert 'WARNING' in text
    assert 'nothing was enforced structurally' in text


def test_empty_report_does_not_claim_exactness():
    """``exact_fraction`` must be NaN, not 1.0, when the structure never ran."""
    from domiknows.program.model.factorGraphHead import FactorGraphReport
    empty = FactorGraphReport()
    assert empty.exact_fraction != empty.exact_fraction        # NaN
    assert FactorGraphReport(exact=['a']).exact_fraction == 1.0


def test_cmodel_options_still_reach_the_constraint_model():
    """R1/R5 compose with R3/R4 — model-side and cmodel-side kwargs must split."""
    graph = _typing_graph()
    program = StructuredProgram(graph, poi=[], refine=True, factor_graph=True,
                                compile_lc=True, dual_algorithm='augmented')
    assert isinstance(program.model, StructuredModel)
    assert program.model.refine and program.model.factor_graph
    assert program.cmodel.compile_lc is True
    assert program.cmodel.dual_algorithm == 'augmented'


def test_invalid_options_are_rejected():
    graph = _typing_graph()
    with pytest.raises(ValueError):
        StructuredProgram(graph, poi=[], partition='sometimes')
    with pytest.raises(ValueError):
        StructuredProgram(graph, poi=[], belief_flow='whatever')


# --------------------------------------------------------------------------- #
# Verification 3 — not a free-parameter win
# --------------------------------------------------------------------------- #

def test_zeroed_refinement_gate_recovers_the_raw_beliefs():
    """The gate is the layer's only parameter; W=0 must undo the correction."""
    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=True,
                       structure_kwargs={'refine_steps': 5, 'refine_step_size': 5.0})
    program.model.eval()
    program.model(_data(), build=True)          # builds the refiner lazily
    with torch.no_grad():
        for parameter in program.model._refiner.gate.values():
            parameter.zero_()

    graph2, img2, ent2, a2, b2 = _exclusive_graph()
    program2 = _program(graph2, [img2, ent2, a2, b2], refine=True,
                        structure_kwargs={'refine_steps': 5, 'refine_step_size': 5.0})
    program2.model.eval()
    program2.model._refiner = program.model._refiner   # reuse the zeroed gate
    _loss, _metric, datanode, _builder = program2.model(_data(), build=True)
    assert float(_softmax_of(datanode, ent2, 'a')[0][1]) == pytest.approx(
        0.9525741, abs=1e-4)


# --------------------------------------------------------------------------- #
# Verification 4 — MAP decoding (R6)
# --------------------------------------------------------------------------- #

def test_map_decoding_respects_the_constraint():
    """MAP must never leave both exclusive siblings on, unlike the raw argmax."""
    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=False,
                       inferTypes=['local/softmax', 'MAP'])
    program.model.eval()
    _loss, _metric, datanode, _builder = program.model(_data(), build=True)

    for dn in datanode.findDatanodes(select=ent):
        raw_a = dn.getAttribute('<a>/local/softmax')
        raw_b = dn.getAttribute('<b>/local/softmax')
        map_a = dn.getAttribute('<a>/MAP')
        map_b = dn.getAttribute('<b>/MAP')
        assert map_a is not None and map_b is not None, 'MAP results were not written'
        # both raw heads say "true", which violates nandL...
        assert int(raw_a.argmax()) == 1 and int(raw_b.argmax()) == 1
        # ...and MAP refuses to
        assert not (int(map_a.argmax()) == 1 and int(map_b.argmax()) == 1)


def test_map_requires_a_structured_model():
    """``inferTypes=['MAP']`` on a plain SolverModel explains itself."""
    from domiknows.program.model.pytorch import SolverModel
    graph = _typing_graph()
    model = SolverModel(graph, poi=[], inferTypes=['MAP'])
    with pytest.raises(NotImplementedError, match='StructuredModel'):
        model.inferMAPResults(None)


# --------------------------------------------------------------------------- #
# Verification 5 — end to end
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('cmodel_kwargs', [
    {},
    {'compile_lc': True},
    {'dual_algorithm': 'augmented'},
    {'dual_granularity': 'amortized'},
], ids=['plain', 'r1', 'r5a', 'r5b'])
def test_train_step_composes_with_each_cmodel_mechanism(cmodel_kwargs):
    """A training step runs and actually updates parameters, per composition.

    Asserts the *parameters moved*, not that stale ``.grad`` buffers survive:
    the optimizer zeroes gradients at the end of a step (``set_to_none``), so a
    post-hoc grad check only passes for parameters the optimizer never saw —
    which is the very bug it should be catching.
    """
    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=True, **cmodel_kwargs)
    before = {name: p.detach().clone()
              for name, p in program.model.named_parameters()}
    program.train(training_set=[_data()], train_epoch_num=1,
                  Optim=lambda params: torch.optim.SGD(params, lr=1e-1),
                  device='cpu')
    moved = [name for name, p in program.model.named_parameters()
             if not torch.equal(p.detach(), before[name])]
    assert moved, 'training updated no parameter at all'


def test_refinement_gate_is_optimised():
    """The gate must be in ``model.parameters()`` *before* the optimizer is built.

    ``LearningBasedProgram.train`` constructs the optimizer from
    ``model.parameters()`` before any forward runs, so a refiner created lazily
    on the first forward would never be handed to it — the gate would hold its
    initial value for the whole run and R4's only learned parameter would be
    inert. This pins the eager construction that prevents that.
    """
    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=True,
                       structure_kwargs={'refine_steps': 5, 'refine_step_size': 5.0})

    gate_names = [n for n, _ in program.model.named_parameters() if 'gate' in n]
    assert gate_names, 'gate parameters absent before the first forward'

    seen = {}

    def optim(params):
        listed = list(params)
        seen['count'] = len(listed)
        return torch.optim.SGD(listed, lr=0.5)

    before = {n: p.detach().clone()
              for n, p in program.model.named_parameters() if 'gate' in n}
    program.train(training_set=[_data()] * 4, train_epoch_num=3, Optim=optim,
                  device='cpu')

    assert seen['count'] == len(list(program.model.parameters()))
    changed = [n for n, p in program.model.named_parameters()
               if 'gate' in n and not torch.equal(p.detach(), before[n])]
    assert changed, 'the refinement gate never moved — it is not being trained'


def test_gradients_flow_through_refinement_to_the_heads():
    """Write-back means the supervised loss backprops *through* the structure."""
    graph, img, ent, a, b = _exclusive_graph()
    program = _program(graph, [img, ent, a, b], refine=True,
                       structure_kwargs={'refine_steps': 3, 'refine_step_size': 2.0})
    program.model.train()
    loss, _metric, _dn, _builder = program.model(_data(), build=True)
    assert loss.requires_grad
    loss.backward()
    touched = [p for p in program.model.parameters()
               if p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0]
    assert touched, 'gradient did not reach any head through the refinement'


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
