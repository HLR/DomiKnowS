from collections import OrderedDict
if __package__ is None or __package__ == '':
    from base import SubScorable, local_names_and_objs
else:
    from .base import SubScorable, local_names_and_objs

@local_names_and_objs
class Graph(SubScorable):
    # local tree structure
    default = None
    ontology = None
    
    def __init__(self, name=None):
        SubScorable.__init__(self, name)
        self._concepts = OrderedDict()

    def add(self, obj):
        if __package__ is None or __package__ == '':
            import concept
        else:
            from . import concept

        if isinstance(obj, Graph):
            self.subs[obj.name] = obj
        elif isinstance(obj, concept.Concept):
            self.concepts[obj.name] = obj
        else:
            raise TypeError('Add Graph or Concept instance, {} instance given.'.format(type(obj)))

    @property
    def concepts(self):
        return self._concepts

    @property
    def fullname(self):
        if self.sup is None:
            return self.name
        return self.sup.fullname + '/' + self.name

    def what(self):
        wht = SubScorable.what(self)
        wht['concepts'] = dict(self.concepts)
        return wht

    def release(self, prop=None):
        for concept in self.concepts.values():
            concept.release(prop)
        for sub in self.subs.values():
            sub.release(prop)

    def get_multiassign(self):
        for concept in self.concepts.values():
            for _, prop, value in concept.get_multiassign():
                yield self, concept, prop, value
        for sub in self.subs.values():
            for _, concept, prop, value in sub.get_multiassign():
                yield sub, concept, prop, value

    def score(self, *args, **kwargs):
        subscore = SubScorable.score(self, *args, **kwargs)
        return subscore + sum([concept(*args, **kwargs) for concept in self.concepts.values()])

    def __getattr__(self, name):
        if name in self.subs:
            return self.subs[name]
        if name in self.concepts:
            return self.concepts[name]
        raise KeyError('{} is not found in subgraphs or concepts.'.format(name))

    def __getitem__(self, name):
        tokens = name.split('/', 1)
        if len(tokens) > 1:
            return self.subs[tokens[0]][tokens[1]]
        if tokens[0] in self.concepts:
            return self.concepts[tokens[0]]
        raise KeyError('{} is not found in subgraphs or concepts.'.format(tokens[0]))
