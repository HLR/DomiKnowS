if __package__ is None or __package__ == '':
    from base import BaseGraphShallowTree
else:
    from .base import BaseGraphShallowTree


@BaseGraphShallowTree.localize_namespace
class Property(BaseGraphShallowTree):
    pass
