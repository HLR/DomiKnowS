from .base import BaseGraphTree


@BaseGraphTree.localize_context
class Trial(BaseGraphTree):
    def __init__(self, data=dict(), obsoleted=set(), name=None):
        BaseGraphTree.__init__(self, name)  # name may be update
        self.data = data
        self.obsoleted = obsoleted

    def __getitem__(self, key):
        try:
            return self.data.__getitem__(key)
        except KeyError:
            if key in self.obsoleted:
                raise
            return self.sup.__getitem__(key)

    def __setitem__(self, key, obj):
        self.data.__setitem__(key, obj)

    def __delitem__(self, key):
        try:
            self.data.__delitem__(key)
        except KeyError:
            pass
        self.obsoleted.add(key)
