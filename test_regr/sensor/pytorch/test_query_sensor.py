import pytest


@pytest.fixture()
def case():
    from regr.utils import Namespace
    import random

    case = {
        'reader1': 'hello world, {}'.format(random.random()),
        'reader2': 'hello world, {}'.format(random.random()),
        'constant': random.random(),
        'output': 'output, {}'.format(random.random())}
    case = Namespace(case)
    return case


@pytest.fixture()
def graph(case):
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.graph import Graph, Concept

    # knowledge
    Graph.clear()
    Concept.clear()
    with Graph() as graph:
        with Graph('sub') as subgraph:
            concept = Concept('concept')

    # model
    concept['reader1'] = ReaderSensor(keyword='reader1_keyword')
    concept['reader2'] = ReaderSensor(keyword='reader2_keyword')

    return graph


@pytest.fixture()
def sensor(case, graph):
    from regr.graph import DataNode
    from regr.sensor.pytorch.query_sensor import QuerySensor

    concept = graph['sub/concept']

    def forward(datanodes, reader1, reader2, constant):
        # datanodes is a list of datanode
        assert len(datanodes) == 1
        assert isinstance(datanodes[0], DataNode)
        # other arguments are like functional sensor
        assert reader1 == case.reader1
        assert reader2 == case.reader2
        assert constant == case.constant
        return case.output
    sensor = QuerySensor('reader1', concept['reader2'], case.constant, forward=forward)
    concept['query'] = sensor
    return sensor


@pytest.fixture()
def context(case, graph):
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.graph import Property, DataNodeBuilder

    context = {
        "graph": graph, 'READER': 1,
        'reader1_keyword': case.reader1,
        'reader2_keyword': case.reader2,}

    context = DataNodeBuilder(context)

    def all_properties(node):
        if isinstance(node, Property):
            return node
    for prop in graph.traversal_apply(all_properties):
        for sensor in prop.find(ReaderSensor):
            sensor.fill_data(context)
    return context


def test_query_sensor(case, sensor, context):
    output = sensor(context)
    assert output == case.output

if __name__ == '__main__':
    pytest.main([__file__])
