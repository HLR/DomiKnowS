from itertools import chain

if __package__ is None or __package__ == '':
    from domiknows.base import Scoped, AutoNamed, NamedTreeNode, NamedTree
else:
    from ..base import Scoped, AutoNamed, NamedTreeNode, NamedTree


@NamedTreeNode.localize_context
class BaseGraphTreeNode(AutoNamed, NamedTreeNode):
    def __init__(self, name=None):
        super().__init__(name)  # name may be update
        super(AutoNamed, self).__init__(self.name)

    def __repr__(self):
        repr_str = f'{type(self).__name__}(name=\'{self.name}\', fullname=\'{self.fullname}\')'
        return repr_str

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @classmethod
    def clear(cls):
        #AutoNamed.clear() # this call may clear wrong context
        # super() call bring the correct reference to current class `cls`
        super().clear()
        # Sibling class of AutoNamed is NamedTree
        super(AutoNamed, cls).clear()


@BaseGraphTreeNode.share_context
class BaseGraphTree(AutoNamed, NamedTree):
    def __init__(self, name=None):
        super().__init__(name)  # name may be update
        super(AutoNamed, self).__init__(self.name)

    def __repr__(self):
        repr_str = f'{type(self).__name__}(name=\'{self.name}\', fullname=\'{self.fullname}\')'
        return repr_str

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @classmethod
    def clear(cls):
        # see comment above in BaseGraphTreeNode
        super().clear()
        super(AutoNamed, cls).clear()


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
    def parse_query_apply(self, func, *names, delim='/', trim=True):
        name, names = self.extract_name(*names, delim=delim, trim=trim)
        if names:
            raise ValueError(('{} cannot have nested elements. Access properties using property name directly.'
                              'Query of names {} is not possibly applied.'.format(type(self).__name__, names)))
        return func(self, name)
