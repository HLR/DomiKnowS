if __package__ is None or __package__ == '':
    from regr.base import Scoped, AutoNamed, NamedTreeNode, NamedTree
else:
    from ..base import Scoped, AutoNamed, NamedTreeNode, NamedTree

@NamedTreeNode.localize_context
class BaseGraphTreeNode(AutoNamed, NamedTreeNode):
    def __init__(self, name=None):
        AutoNamed.__init__(self, name)  # name may be update
        NamedTreeNode.__init__(self, self.name)

    @classmethod
    def clear(cls):
        AutoNamed.clear()
        NamedTreeNode.clear()


@BaseGraphTreeNode.share_context
class BaseGraphTree(AutoNamed, NamedTree):
    def __init__(self, name=None):
        AutoNamed.__init__(self, name)  # name may be update
        NamedTree.__init__(self, self.name)

    @classmethod
    def clear(cls):
        AutoNamed.clear()
        NamedTree.clear()


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
                '{} cannot have nested elements. Access properties using property name directly. Query of names {} is not possibly applied.'.format(type(self), names))
        # this is only one layer above the leaf layer
        return func(self, names[0])
