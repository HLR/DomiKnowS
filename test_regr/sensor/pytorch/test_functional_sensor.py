import pytest


@pytest.fixture()
def case():
    from regr.utils import Namespace
    import random

    case = {
        'reader1': 'hello world, {}'.format(random.random()),
        'reader2': 'hello world, {}'.format(random.random()),
        'constant': random.random(),
        'functional': 'output, {}'.format(random.random())}
    case = Namespace(case)
    return case


@pytest.fixture()
def concept(case):
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.graph import Concept

    # knowledge
    Concept.clear()
    concept = Concept()

    # model
    concept['reader1'] = ReaderSensor(keyword='reader1_keyword')
    concept['reader2'] = ReaderSensor(keyword='reader2_keyword')

    return concept


@pytest.fixture()
def sensor(case, concept):
    from regr.sensor.pytorch.query_sensor import FunctionalSensor
    def forward(reader1, reader2, constant):
        assert reader1 == case.reader1
        assert reader2 == case.reader2
        assert constant == case.constant
        return case.functional
    sensor = FunctionalSensor('reader1', concept['reader2'], case.constant, forward=forward)
    concept['functional'] = sensor
    return sensor


@pytest.fixture()
def context(case, concept):
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.graph import Property

    context = {
        'reader1_keyword': case.reader1,
        'reader2_keyword': case.reader2,}

    def all_properties(node):
        if isinstance(node, Property):
            return node
    for prop in concept.traversal_apply(all_properties):
        for sensor in prop.find(ReaderSensor):
            sensor.fill_data(context)
    return context


def test_functional_sensor(case, sensor, context):
    output = sensor(context)
    assert output == case.functional

if __name__ == '__main__':
    pytest.main([__file__])
