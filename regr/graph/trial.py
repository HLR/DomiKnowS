from .base import BaseGraphTree


@BaseGraphTree.localize_context
class TrialTree(BaseGraphTree):
    def __init__(self, trial, name=None):
        super().__init__(name=name)
        self.trial = trial


class Trial():
    @classmethod
    def default(cls):
        default_tree = TrialTree.default()
        if default_tree is not None:
            return default_tree.trial
        return None

    def __init__(self, data=None, obsoleted=None, name=None):
        self.trial_node = TrialTree(self, name=name)
        self.data = data or dict()
        self.obsoleted = obsoleted or set()

    @property
    def sup(self):
        if self.trial_node.sup:
            return self.trial_node.sup.trial
        return None

    def __enter__(self):
        return self.trial_node.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.trial_node.__exit__(exc_type, exc_value, traceback)

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

    def __iter__(self):
        yield from iter(self.data)
        if self.sup:
            yield from filter(lambda x: (x not in self.data) and (x not in self.obsoleted), self.sup)

    def __len__(self):
        return len(list(iter(self)))
