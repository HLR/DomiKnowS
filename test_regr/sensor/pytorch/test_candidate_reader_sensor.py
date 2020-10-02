import pytest


@pytest.fixture()
def case():
    from regr.utils import Namespace
    import random

    edge_value = [
        [random.random() > 0.5, random.random() > 0.5],
        [random.random() > 0.5, random.random() > 0.5]]
    case = {
        'container': 'hello world',
        'container_edge': ['hello', 'world'],
        'concept_feature': [
            'hello, {}'.format(random.random()),
            'world, {}'.format(random.random())],
        'constant': random.random(),
        'edge_value': edge_value,
        'edge': {
            'hello':{
                'hello': edge_value[0][0],
                'world': edge_value[0][1]},
            'world':{
                'hello': edge_value[1][0],
                'world': edge_value[1][1]}}}
    case = Namespace(case)
    return case


@pytest.fixture()
def graph(case):
    from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeReaderSensor
    from regr.sensor.pytorch.query_sensor import InstantiateSensor
    from regr.graph import Graph, Concept, Relation, Property

    from .sensors import TestSensor, TestEdgeSensor

    # knowledge
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph() as graph:
        with Graph('sub') as subgraph:
            container = Concept('container')
            concept = Concept('concept')
            (container_contains_concept,) = container.contains(concept)
            edge = Concept('edge')
            (edge_concept1, edge_concept2,) = edge.has_a(arg1=concept, arg2=concept)
            edge['concept1'] = Property('concept1')
            edge['concept2'] = Property('concept2')

    # model
    container['index'] = ReaderSensor(keyword='container_keyword')
    container_contains_concept['forward'] = TestEdgeSensor(
        'index', mode='forward', to='index',
        expected_inputs=[case.container,],
        expected_outputs=case.container_edge)
    concept['index'] = InstantiateSensor(edges=[container_contains_concept['forward']])

    return graph


@pytest.fixture()
def sensor(case, graph):
    from regr.graph import DataNode
    from regr.sensor.pytorch.relation_sensors import CandidateReaderSensor

    concept = graph['sub/concept']
    edge = graph['sub/edge']
    (edge_concept1, edge_concept2,) = edge.has_a()

    collector = []
    def forward(data, datanodes, datanode_concept1, datanode_concept2, constant):
        # update collector
        idx = len(collector)
        collector.append((datanode_concept1, datanode_concept2))
        # current existing datanodes
        assert len(datanodes) == 0
        # test concept 1
        assert isinstance(datanode_concept1, DataNode)
        assert datanode_concept1.getOntologyNode() == concept
        # test concept 2
        assert isinstance(datanode_concept2, DataNode)
        assert datanode_concept2.getOntologyNode() == concept
        # other arguments are like functional sensor
        assert constant == case.constant
        index1 = datanode_concept1.getAttributes().get('index')
        index2 = datanode_concept2.getAttributes().get('index')
        return data[index1][index2]
    sensor = CandidateReaderSensor(case.constant, forward=forward, keyword='edge_keyword')
    edge['index'] = sensor
    return sensor

@pytest.fixture()
def context(case, graph):
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.sensor.pytorch.relation_sensors import CandidateReaderSensor
    from regr.graph import Property, DataNodeBuilder

    context = {
        "graph": graph, 'READER': 1,
        'container_keyword': case.container,
        'edge_keyword': case.edge}

    context = DataNodeBuilder(context)

    def all_properties(node):
        if isinstance(node, Property):
            return node
    for prop in graph.traversal_apply(all_properties):
        for sensor in prop.find(ReaderSensor):
            sensor.fill_data(context)
        for sensor in prop.find(CandidateReaderSensor):
            sensor.fill_data(context)
    return context


def test_functional_sensor(case, sensor, context):
    import torch
    output = sensor(context)
    assert (output == torch.tensor(case.edge_value, device=output.device)).all()


if __name__ == '__main__':
    pytest.main([__file__])
