from collections import OrderedDict
import types

import pytest
import torch

from domiknows.graph import (
    Concept,
    Graph,
    Relation,
    andL,
    execute,
    existsL,
    iotaL,
    queryL,
    sumL,
)
from domiknows.graph.concept import EnumConcept
from domiknows.graph.dataNode import DataNode
from domiknows.solver import ilpOntSolverFactory
from domiknows.solver.answerModule import AnswerSolver
from domiknows.utils import setDnSkeletonMode


@pytest.fixture(autouse=True)
def _reset_graph_and_solver_state():
    Graph.clear()
    Concept.clear()
    Relation.clear()
    ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None
    setDnSkeletonMode(False)
    yield
    setDnSkeletonMode(False)
    ilpOntSolverFactory.clear()
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.collectedConceptsAndRelations = None


def _activate(root, *names):
    active = set(names)
    root.getActiveExecutableConstraintNames = types.MethodType(
        lambda self: active,
        root,
    )
    root.setActiveExecutableLCs = types.MethodType(
        lambda self: None,
        root,
    )


def _binary_scene(graph_name, *, executable_factory=None):
    with Graph(graph_name) as graph:
        scene = Concept(name=f'{graph_name}_scene')
        item = Concept(name=f'{graph_name}_item')
        scene.contains(item)
        flag = item(name=f'{graph_name}_flag')
        if executable_factory is not None:
            executable_factory(flag)

    root = DataNode(instanceID=0, ontologyNode=scene)
    return graph, root, item, flag


def _add_binary_items(root, item, flag, logits):
    children = []
    for index, pair in enumerate(logits):
        child = DataNode(instanceID=index, ontologyNode=item)
        child.attributes[f'<{flag.name}>'] = torch.tensor(
            pair,
            dtype=torch.float,
        )
        root.addChildDataNode(child)
        children.append(child)
    return children


def _add_constraint_child(root, *active_names):
    constraint_child = DataNode(
        instanceID=0,
        ontologyNode=root.graph.get_constraint_concept(),
    )
    for name in active_names:
        constraint_child.attributes[f'{name}/label'] = torch.tensor(1.0)
    root.addChildDataNode(constraint_child)
    return constraint_child


def test_active_constraint_names_ignore_non_label_attributes():
    _, root, _, _ = _binary_scene(
        'active_names',
        executable_factory=lambda concept: execute(existsL(concept('x'))),
    )
    constraint_child = _add_constraint_child(root, 'ELC0')
    constraint_child.attributes.update({
        'rootDataNode': object(),
        '<constraint>/local/softmax': torch.tensor([0.0, 1.0]),
        'ELC0/answer': True,
    })

    assert root.getActiveExecutableConstraintNames() == {'ELC0'}
    assert root._getExecutableConstraintDataNode() is constraint_child


def test_constraint_child_lookup_retains_builder_fallback_and_multiple_error():
    _, root, _, _ = _binary_scene('constraint_lookup')
    constraint_concept = root.graph.get_constraint_concept()
    builder_constraint = DataNode(
        instanceID=0,
        ontologyNode=constraint_concept,
    )
    builder_constraint.attributes['ELC0/label'] = torch.tensor(1.0)
    root.myBuilder = types.SimpleNamespace(
        findDataNodesInBuilder=lambda select: [builder_constraint],
    )

    assert root._getExecutableConstraintDataNode() is builder_constraint
    assert root.getExecutableConstraintLabels() == (
        builder_constraint.attributes
    )

    second_constraint = DataNode(
        instanceID=1,
        ontologyNode=constraint_concept,
    )
    root.myBuilder = types.SimpleNamespace(
        findDataNodesInBuilder=lambda select: [
            builder_constraint,
            second_constraint,
        ],
    )
    with pytest.raises(ValueError, match='Multiple constraint datanodes'):
        root._getExecutableConstraintDataNode()

    root.myBuilder = None
    root.addChildDataNode(builder_constraint)
    root.addChildDataNode(second_constraint)

    with pytest.raises(ValueError, match='Multiple constraint datanodes'):
        root._getExecutableConstraintDataNode()


@pytest.mark.gurobi
def test_infer_ilp_without_active_executable_constraint_uses_legacy_path():
    _, root, item, flag = _binary_scene(
        'legacy',
        executable_factory=lambda concept: execute(existsL(concept('x'))),
    )
    children = _add_binary_items(
        root,
        item,
        flag,
        ((0.1, 2.0), (2.0, 0.1)),
    )
    constraint_child = _add_constraint_child(root)
    constraint_child.attributes['ELC0/answer'] = True

    root.inferILPResults(flag)

    assert [child.attributes[f'<{flag.name}>/ILP'].item()
            for child in children] == [1.0, 0.0]
    assert 'ELC0/answer' not in constraint_child.attributes


@pytest.mark.gurobi
def test_active_exists_populates_winning_model_and_repeats_cleanly():
    graph, root, item, flag = _binary_scene(
        'exists_repeat',
        executable_factory=lambda concept: execute(existsL(concept('x'))),
    )
    children = _add_binary_items(
        root,
        item,
        flag,
        ((0.1, 2.0), (2.0, 0.1)),
    )
    constraint_child = _add_constraint_child(root, 'ELC0')

    root.inferILPResults(flag)
    assert [child.attributes[f'<{flag.name}>/ILP'].item()
            for child in children] == [1.0, 0.0]
    assert constraint_child.attributes['ELC0/answer'] is True
    assert isinstance(constraint_child.attributes['ELC0/answer'], bool)

    # Change the preferred entity and force local probabilities to be
    # recomputed. A stale Gurobi variable from the first hypothesis run would
    # fail here with "Variable not in model" or preserve the old assignment.
    children[0].attributes[f'<{flag.name}>'] = torch.tensor([2.0, 0.1])
    children[1].attributes[f'<{flag.name}>'] = torch.tensor([0.1, 2.0])
    for child in children:
        child.attributes.pop(f'<{flag.name}>/local/softmax', None)
    root.collectedConceptsAndRelations = None

    root.inferILPResults(flag)
    assert [child.attributes[f'<{flag.name}>/ILP'].item()
            for child in children] == [0.0, 1.0]
    assert constraint_child.attributes['ELC0/answer'] is True
    assert graph.executableLCs['ELC0'].innerLC.active


@pytest.mark.gurobi
def test_answer_api_returns_hypothesis_without_populating_ilp_results():
    graph, root, item, flag = _binary_scene(
        'answer_api',
        executable_factory=lambda concept: execute(existsL(concept('x'))),
    )
    children = _add_binary_items(
        root,
        item,
        flag,
        ((0.1, 2.0), (2.0, 0.1)),
    )
    constraint_child = _add_constraint_child(root, 'ELC0')
    root.inferLocal()

    answer = AnswerSolver(graph).answer('execute(ELC0)', root)

    assert answer is True
    assert all(
        f'<{flag.name}>/ILP' not in child.attributes
        for child in children
    )
    assert all(
        not any('/ILP/' in key for key in child.attributes)
        for child in children
    )
    assert 'ELC0/answer' not in constraint_child.attributes


@pytest.mark.gurobi
def test_batch_root_dispatches_hypothesis_inference_per_contained_datanode():
    with Graph('batch') as graph:
        batch = Concept(name='batch_root', batch=True)
        scene = Concept(name='batch_scene')
        item = Concept(name='batch_item')
        batch.contains(scene)
        scene.contains(item)
        flag = item(name='batch_flag')
        execute(existsL(flag('x')))

    batch_root = DataNode(instanceID=0, ontologyNode=batch)
    rows = []
    constraint_children = []
    row_logits = (
        ((0.1, 2.0), (2.0, 0.1)),
        ((2.0, 0.1), (0.1, 2.0)),
    )
    for scene_index, logits in enumerate(row_logits):
        scene_root = DataNode(
            instanceID=scene_index,
            ontologyNode=scene,
        )
        batch_root.addChildDataNode(scene_root)
        row = []
        for item_index, pair in enumerate(logits):
            child = DataNode(
                instanceID=scene_index * 10 + item_index,
                ontologyNode=item,
            )
            child.attributes['<batch_flag>'] = torch.tensor(pair)
            scene_root.addChildDataNode(child)
            row.append(child)
        rows.append(row)
        constraint_children.append(
            _add_constraint_child(scene_root, 'ELC0')
        )

    batch_root.inferILPResults(flag)

    assert [
        [child.attributes['<batch_flag>/ILP'].item() for child in row]
        for row in rows
    ] == [[1.0, 0.0], [0.0, 1.0]]
    assert all(
        not any('/ILP/' in key for key in child.attributes)
        for row in rows
        for child in row
    )
    assert [
        constraint_child.attributes['ELC0/answer']
        for constraint_child in constraint_children
    ] == [True, True]
    assert graph.batch is batch


@pytest.mark.gurobi
def test_counting_and_multiclass_query_populate_standard_ilp_outputs():
    graph, root, item, flag = _binary_scene(
        'counting',
        executable_factory=lambda concept: execute(sumL(concept('x'))),
    )
    children = _add_binary_items(
        root,
        item,
        flag,
        ((0.1, 2.0), (2.0, 0.1)),
    )
    constraint_child = _add_constraint_child(root, 'ELC0')

    root.inferILPResults(flag)
    assert sum(
        child.attributes[f'<{flag.name}>/ILP'].item()
        for child in children
    ) == 1.0
    count_answer = constraint_child.attributes['ELC0/answer']
    assert count_answer == 1
    assert isinstance(count_answer, int)
    assert not isinstance(count_answer, bool)

    Graph.clear()
    Concept.clear()
    Relation.clear()
    ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None

    with Graph('query') as query_graph:
        scene = Concept(name='query_scene')
        query_item = Concept(name='query_item')
        scene.contains(query_item)
        target = query_item(name='query_target')
        color = query_item(
            name='query_color',
            ConceptClass=EnumConcept,
            values=['red', 'blue'],
        )
        execute(queryL(color, iotaL(target('x'))))

    query_root = DataNode(instanceID=0, ontologyNode=scene)
    query_child = DataNode(instanceID=0, ontologyNode=query_item)
    query_child.attributes['<query_target>'] = torch.tensor([0.1, 2.0])
    query_child.attributes['<query_color>'] = torch.tensor([2.0, 0.1])
    query_root.addChildDataNode(query_child)
    query_constraint_child = _add_constraint_child(query_root, 'ELC0')

    query_root.inferILPResults(target, color)

    assert torch.equal(
        query_child.attributes['<query_color>/ILP'],
        torch.tensor([1.0, 0.0]),
    )
    assert query_child.attributes['<query_target>/ILP'].item() == 1.0
    assert query_constraint_child.attributes['ELC0/answer'] == 'red'
    assert isinstance(
        query_constraint_child.attributes['ELC0/answer'],
        str,
    )
    assert query_graph.executableLCs['ELC0'].innerLC.active


@pytest.mark.gurobi
def test_multiple_active_constraints_evaluate_joint_cartesian_product(
    monkeypatch,
):
    with Graph('joint') as graph:
        scene = Concept(name='joint_scene')
        item = Concept(name='joint_item')
        scene.contains(item)
        red = item(name='joint_red')
        blue = item(name='joint_blue')
        execute(existsL(red('x')))
        execute(existsL(blue('x')))

    root = DataNode(instanceID=0, ontologyNode=scene)
    child = DataNode(instanceID=0, ontologyNode=item)
    child.attributes['<joint_red>/local/softmax'] = torch.tensor([0.1, 0.9])
    child.attributes['<joint_blue>/local/softmax'] = torch.tensor([0.9, 0.1])
    root.addChildDataNode(child)
    constraint_child = _add_constraint_child(root, 'ELC0', 'ELC1')

    answer_solver = AnswerSolver(graph)
    real_calculate = answer_solver.solver._calculateILPSelection
    calls = []

    def counted_calculate(*args, **kwargs):
        calls.append(tuple(kwargs['extraLogicalConstraints']))
        return real_calculate(*args, **kwargs)

    monkeypatch.setattr(
        answer_solver.solver,
        '_calculateILPSelection',
        counted_calculate,
    )

    result = answer_solver.solve_active_constraints(
        root,
        {'ELC1', 'ELC0'},
        (
            (red, red.name, None, 1),
            (blue, blue.name, None, 1),
        ),
    )

    assert result['hypotheses'] == OrderedDict(
        [('ELC0', True), ('ELC1', False)]
    )
    assert len(calls) == 4
    assert child.attributes['<joint_red>/ILP'].item() == 1.0
    assert child.attributes['<joint_blue>/ILP'].item() == 0.0
    assert constraint_child.attributes['ELC0/answer'] is True
    assert constraint_child.attributes['ELC1/answer'] is False

    constraint_child.attributes.pop('ELC0/label')
    repeated_result = answer_solver.solve_active_constraints(
        root,
        {'ELC1'},
        (
            (red, red.name, None, 1),
            (blue, blue.name, None, 1),
        ),
    )

    assert repeated_result['hypotheses'] == OrderedDict([('ELC1', False)])
    assert 'ELC0/answer' not in constraint_child.attributes
    assert constraint_child.attributes['ELC1/answer'] is False


class _SequencedSolver:
    def __init__(self, objectives):
        self.objectives = iter(objectives)
        self.populated = None

    def _calculateILPSelection(self, *args, **kwargs):
        objective = next(self.objectives)
        if objective is None:
            return None
        return {'objective': objective, 'values': {}}

    def populateILPSelection(self, dn, concepts_relations, values):
        self.populated = values


class _ProbabilityWorldSolver:
    """Score two candidate worlds using the transform supplied by AnswerSolver."""

    worlds = (
        torch.tensor([0.99, 0.52]),
        torch.tensor([0.75, 0.75]),
    )

    def __init__(self):
        self.call_index = 0
        self.populated = None

    def _calculateILPSelection(self, *args, **kwargs):
        world = self.worlds[self.call_index]
        self.call_index += 1
        transformed = kwargs['fun'](world)
        return {
            'objective': float(transformed.sum()),
            'values': {'world': self.call_index - 1},
        }

    def populateILPSelection(self, dn, concepts_relations, values):
        self.populated = values


def test_hypothesis_objective_defaults_to_log_probability_map():
    graph, root, _, _ = _binary_scene(
        'log_probability_objective',
        executable_factory=lambda concept: execute(existsL(concept('x'))),
    )
    _add_constraint_child(root, 'ELC0')

    # Raw sums prefer the first world: 0.99 + 0.52 > 0.75 + 0.75.
    # Joint probability prefers the second: 0.99*0.52 < 0.75*0.75.
    map_solver = _ProbabilityWorldSolver()
    map_result = AnswerSolver(graph, solver=map_solver).solve_active_constraints(
        root,
        {'ELC0'},
        (),
        populate=False,
    )
    assert map_result['hypotheses']['ELC0'] is False
    assert map_result['objective_mode'] == 'log_probability'

    raw_solver = _ProbabilityWorldSolver()
    raw_result = AnswerSolver(graph, solver=raw_solver).solve_active_constraints(
        root,
        {'ELC0'},
        (),
        fun=lambda probabilities: probabilities,
        populate=False,
    )
    assert raw_result['hypotheses']['ELC0'] is True
    assert raw_result['objective_mode'] == 'custom'


def test_objective_direction_ties_and_all_infeasible():
    graph, root, _, _ = _binary_scene(
        'objective',
        executable_factory=lambda concept: execute(existsL(concept('x'))),
    )
    constraint_child = _add_constraint_child(root, 'ELC0')

    tied = AnswerSolver(graph, solver=_SequencedSolver([5.0, 5.0]))
    tied_result = tied.solve_active_constraints(
        root,
        {'ELC0'},
        (),
        populate=False,
    )
    assert tied_result['hypotheses']['ELC0'] is True
    assert 'ELC0/answer' not in constraint_child.attributes

    minimized = AnswerSolver(graph, solver=_SequencedSolver([5.0, 3.0]))
    minimized_result = minimized.solve_active_constraints(
        root,
        {'ELC0'},
        (),
        minimize_objective=True,
    )
    assert minimized_result['hypotheses']['ELC0'] is False
    assert constraint_child.attributes['ELC0/answer'] is False

    constraint_child.attributes['ELC0/answer'] = 'stale'
    infeasible = AnswerSolver(graph, solver=_SequencedSolver([None, None]))
    with pytest.raises(RuntimeError, match='All joint hypotheses were infeasible'):
        infeasible.solve_active_constraints(
            root,
            {'ELC0'},
            (),
        )
    assert 'ELC0/answer' not in constraint_child.attributes


def test_missing_constraint_child_does_not_block_hypothesis_population():
    graph, root, _, _ = _binary_scene(
        'missing_constraint_child',
        executable_factory=lambda concept: execute(existsL(concept('x'))),
    )
    solver = _SequencedSolver([2.0, 1.0])

    result = AnswerSolver(graph, solver=solver).solve_active_constraints(
        root,
        {'ELC0'},
        (),
    )

    assert result['hypotheses']['ELC0'] is True
    assert solver.populated == {}


def test_skeleton_mode_preserves_constraint_answer_population():
    graph, root, _, _ = _binary_scene(
        'skeleton_answer',
        executable_factory=lambda concept: execute(existsL(concept('x'))),
    )
    constraint_child = _add_constraint_child(root, 'ELC0')
    setDnSkeletonMode(True)

    result = AnswerSolver(
        graph,
        solver=_SequencedSolver([2.0, 1.0]),
    ).solve_active_constraints(
        root,
        {'ELC0'},
        (),
    )

    assert result['hypotheses']['ELC0'] is True
    assert constraint_child.attributes['ELC0/answer'] is True


def test_unknown_and_unsupported_active_constraints_fail_clearly():
    graph, root, _, flag = _binary_scene(
        'unsupported',
        executable_factory=lambda concept: execute(
            andL(concept('x'), concept('y'))
        ),
    )
    solver = AnswerSolver(graph, solver=_SequencedSolver([]))

    with pytest.raises(ValueError, match='Unknown active executable'):
        solver.solve_active_constraints(root, {'ELC99'}, (), populate=False)

    with pytest.raises(
        ValueError,
        match="Unsupported executable constraint type 'andL'",
    ):
        solver.solve_active_constraints(
            root,
            {'ELC0'},
            ((flag, flag.name, None, 1),),
            populate=False,
        )


@pytest.mark.gurobi
def test_detached_writer_populates_skeleton_variable_set():
    graph, root, item, flag = _binary_scene('skeleton')
    child = DataNode(instanceID=0, ontologyNode=item)
    root.addChildDataNode(child)
    root.attributes['variableSet'] = {}
    setDnSkeletonMode(True)

    answer_solver = AnswerSolver(graph)
    variable_key = (flag, flag.name, child.getInstanceID(), 0)
    answer_solver.solver.populateILPSelection(
        root,
        ((flag, flag.name, None, 1),),
        {variable_key: 1.0},
    )

    assert torch.equal(
        root.attributes['variableSet']['skeleton_item/<skeleton_flag>/ILP'],
        torch.tensor([[1.0]]),
    )
