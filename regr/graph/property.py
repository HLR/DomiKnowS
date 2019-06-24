from .base import BaseGraphShallowTree


@BaseGraphShallowTree.localize_namespace
class Property(BaseGraphShallowTree):
    def __init__(self, prop_name):
        cls = type(self)
        context = cls._context[-1]
        self.prop_name = prop_name
        name = '{}-{}'.format(context.name, prop_name)
        BaseGraphShallowTree.__init__(self, name)

    def attach_to_context(self, name=None):
        BaseGraphShallowTree.attach_to_context(self, self.prop_name)

    def attach(self, sub):
        from ..sensor import Sensor
        if isinstance(sub, Sensor):
            BaseGraphShallowTree.attach(self, sub)
        else:
            raise TypeError(
                'Attach Sensor instance to Property, {} instance given.'.format(type(sub)))

    def find(self, *sensor_tests):
        sensor_tests = list(sensor_tests)
        for i in range(len(sensor_tests)):
            if isinstance(sensor_tests[i], type):
                sensor_type = sensor_tests[i] # need a new variable to avoid closure
                sensor_tests[i] = lambda s: isinstance(s, sensor_type)
            if not callable(sensor_tests[i]):
                raise ValueError('Unsupported test {} to find.'.format(sensor_tests[i]))

        for name, sensor in self.items():
            for sensor_test in sensor_tests:
                if not sensor_test(sensor):
                    break
            else:
                from ..sensor import Sensor
                if isinstance(sensor, Sensor):
                    yield name, sensor
