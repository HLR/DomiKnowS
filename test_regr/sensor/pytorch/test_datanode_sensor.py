import pytest


@pytest.fixture()
def case():
    from regr.utils import Namespace
    import random

    case = {
        'container': 'hello world',
        'container_edge': ['hello', 'world'],
        'reader1': [
            'hello, {}'.format(random.random()),
            'world, {}'.format(random.random())],
        'reader2': [
            'hello, {}'.format(random.random()),
            'world, {}'.format(random.random())],
        'constant': random.random(),
        'output': [
            'output, {}'.format(random.random()),
            'output, {}'.format(random.random()),]}
    case = Namespace(case)
    return case


@pytest.fixture()
def graph(case):
    from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeReaderSensor
    from regr.graph import Graph, Concept

    from .sensors import TestSensor, TestEdgeSensor

    # knowledge
    with Graph() as graph:
        with Graph('sub') as subgraph:
            container = Concept('container')
            concept = Concept('concept')
            (container_contains_concept,) = container.contains(concept)

    # model
    container['raw'] = ReaderSensor(keyword='container_keyword')
    container_contains_concept['forward'] = TestEdgeSensor(
        'raw', mode='forward', keyword='raw',
        expected_inputs=[case.container,],
        expected_outputs=case.container_edge)
    concept['reader1'] = TestSensor(
        'raw', edges=[container_contains_concept['forward']],
        expected_inputs=[case.container_edge,],
        expected_outputs=case.reader1)
    concept['reader2'] = TestSensor(
        'raw', edges=[container_contains_concept['forward']],
        expected_inputs=[case.container_edge,],
        expected_outputs=case.reader2)

    return graph


@pytest.fixture()
def sensor(case, graph):
    from regr.graph import DataNode
    from regr.sensor.pytorch.query_sensor import DataNodeSensor

    concept = graph['sub/concept']  # concept is the only child

    datanodes = []
    def func(datanode, reader1, reader2, constant):
        idx = len(datanodes)
        assert idx < 2
        datanodes.append(datanode)
        # unlike query sensor that takes the list
        # here datanode is a datanode,
        assert isinstance(datanode, DataNode)
        assert datanode.getAttributes().get('raw') == case.container_edge[idx]
        assert datanode.getAttributes().get('reader1') == case.reader1[idx]
        assert datanode.getAttributes().get('reader2') == case.reader2[idx]
        # other arguments are like functional sensor
        assert reader1 == case.reader1
        assert reader2 == case.reader2
        assert constant == case.constant
        return case.output[idx]
    sensor = DataNodeSensor('reader1', concept['reader2'], case.constant, func=func)
    concept['output'] = sensor
    return sensor


@pytest.fixture()
def context(case, graph):
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.graph import Property, DataNodeBuilder

    context = {
        "graph": graph, 'READER': 1,
        'container_keyword': case.container}

    context = DataNodeBuilder(context)

    def all_properties(node):
        if isinstance(node, Property):
            return node
    for prop in graph.traversal_apply(all_properties):
        for _, sensor in prop.find(ReaderSensor):
            sensor.fill_data(context)
    return context


def test_functional_sensor(case, sensor, context):
    output = sensor(context)
    assert output == case.output

if __name__ == '__main__':
    pytest.main([__file__])
