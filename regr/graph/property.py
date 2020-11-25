from .base import BaseGraphShallowTree


@BaseGraphShallowTree.localize_namespace
class Property(BaseGraphShallowTree):
    def __init__(self, prop_name):
        cls = type(self)
        context = cls._context[-1]
        self.prop_name = prop_name
        #name = '{}-{}'.format(context.name, prop_name)
        BaseGraphShallowTree.__init__(self, prop_name)

    def attach_to_context(self, name=None):
        BaseGraphShallowTree.attach_to_context(self, self.prop_name)

    def get_fullname(self, delim='/'):
        if self.sup is None:
            return self.name  # note: when it is not nested in any context, use the auto name
        from .concept import Concept
        if isinstance(self.prop_name, Concept):
            prop_name = '<{}>'.format(self.prop_name.name)
        else:
            prop_name = self.prop_name
        return self.sup.get_fullname(delim) + delim + str(prop_name)

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
                raise ValueError('Unsupported feature.py {} to find.'.format(sensor_tests[i]))

        for name, sensor in self.items():
            for sensor_test in sensor_tests:
                if not sensor_test(sensor):
                    break
            else:
                from ..sensor import Sensor
                if isinstance(sensor, Sensor):
                    yield sensor

    def __call__(self, data_item):
        # make sure every sensor are visited
        for _, sensor in self.items():
            sensor(data_item)
        # and see what is lefted
        return data_item[self]
