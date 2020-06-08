import pytest


@pytest.fixture()
def case():
    from regr.utils import Namespace
    case = {
        'reader': 'hello world',
        'functional': 'output'
    }
    case = Namespace(case)
    return case


@pytest.fixture()
def sensor(case):
    from regr.sensor.pytorch.query_sensor import FunctionalSensor
    def func(x):
        assert x == case.reader
        return case.functional
    return FunctionalSensor('raw', func=func)


@pytest.fixture()
def concept(case, sensor):
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.graph import Concept

    # knowledge
    concept = Concept()

    # model
    concept['raw'] = ReaderSensor(keyword='raw_input')
    concept['prop'] = sensor

    return concept


@pytest.fixture()
def context(case, concept):
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.graph import Property

    context = {'raw_input': case.reader}

    def all_properties(node):
        if isinstance(node, Property):
            return node
    for prop in concept.traversal_apply(all_properties):
        for _, sensor in prop.find(ReaderSensor):
            sensor.fill_data(context)
    return context


def test_functional_sensor(case, sensor, context):
    output = sensor(context)
    assert output == case.functional

if __name__ == '__main__':
    pytest.main([__file__])
