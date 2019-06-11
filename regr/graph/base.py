from ..base import Scoped, AutoNamed, NamedTreeNode, NamedTree


@NamedTreeNode.localize_context
class BaseGraphTreeNode(AutoNamed, NamedTreeNode):
    def __init__(self, name=None, ontology=None):
        AutoNamed.__init__(self, name)  # name may be update
        NamedTreeNode.__init__(self, self.name)


@BaseGraphTreeNode.share_context
class BaseGraphTree(AutoNamed, NamedTree):
    def __init__(self, name=None, ontology=None):
        AutoNamed.__init__(self, name)  # name may be update
        NamedTree.__init__(self, self.name)


@Scoped.class_scope
@BaseGraphTreeNode.share_context
class BaseGraphShallowTree(BaseGraphTree):
    # disable context
    def __enter__(self):
        raise AttributeError(
            '{} object has no attribute __enter__'.format(type(self).__name__))

    def __exit__(self, exc_type, exc_value, traceback):
        raise AttributeError(
            '{} object has no attribute __exit__'.format(type(self).__name__))

    # disable query
    def query_apply(self, names, func):
        if len(names) > 1:
            raise ValueError(
                'Concept cannot have nested elements. Access properties using property name directly. Query of names {} is not possibly applied.'.format(names))
        # this is only one layer above the leaf layer
        return func(self, names[0])
