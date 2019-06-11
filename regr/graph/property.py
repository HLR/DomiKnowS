if __package__ is None or __package__ == '':
    from base import BaseGraphShallowTree
else:
    from .base import BaseGraphShallowTree
from ..sensor import Sensor


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
        if isinstance(sub, Sensor):
            BaseGraphShallowTree.attach(self, sub)
        else:
            raise TypeError(
                'Attach Sensor instance to Property, {} instance given.'.format(type(sub)))