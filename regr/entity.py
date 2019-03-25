from collections import defaultdict
from .base import AutoNamed, local_names_and_objs
from .backend import Backend, NumpyBackend
from .graph import Graph


@local_names_and_objs
class Entity(AutoNamed):
    default_backend = NumpyBackend()
    _rel_types = dict()  # relation name (to be call): relation class

    @classmethod
    def update_rel_types(cls, Rel, name=None):
        if name is None:
            name = Rel.suggest_name()
        cls._rel_types[name] = Rel

    @classmethod
    def register_rel_types(cls, name=None):
        return lambda Rel, name=None: cls.update_rel_types(Rel, name)

    '''
    Declare an entity.
    '''

    def __init__(self, name=None, backend=None):
        AutoNamed.__init__(self, name)

        if Graph.default_graph is not None:  # use base class Graph as the global environment
            Graph.default_graph.ent.append(self)

        self._prop = defaultdict(set)  # name : set of binding values
        self._in = defaultdict(set)  # src entity : set of relation instances
        self._out = defaultdict(set)  # dst entity : set of relation instances
        self._b = backend

    def what(self):
        return {'rels': dict(self._out), }

    '''
    Create relation by registered relation types
    '''

    def __getattr__(self, prop):
        cls = type(self)  # bind to the real class
        Rel = cls._rel_types[prop]

        def create_rel(dst, *args, **kwargs):
            rel = Rel(self, dst, *args, **kwargs)
            for _, v in rel.dst:
                self._out[v].add(rel)
                v._in[self].add(rel)
        return create_rel

    '''
    Backend, possible to fallback to class default
    '''
    @property
    def b(self):
        if isinstance(self._b, Backend):
            return self._b
        else:
            return type(self).default_backend

    @property
    def rank(self):
        # remove the first axis for batch
        return len(self.b.shape(self.blf)) - 1

    @property
    def fdim(self):
        # including batch # TODO: this semantic is not consistent with self.rank
        # (batch, dim(s),) -> (batch * dim(s),)
        return self.b.prod(self.b.shape(self.blf))

    '''
    The "distance" used in this entity.
    Lower value indicates better consistency.
    Feature(s) of one instance is always on only the last axis.
    p, q - (nval, batch * len, prop_dim)
    '''

    def distances(self, p, q):
        # inner product
        #q = self.b.reshape(q, (-1, 1))
        #val = self.b.sum(self.b.matmul(p, q))
        # relative entropy
        #q = self.b.reshape(q, (1, -1))
        #val = self.b.sum(p * self.b.log( p / q ))
        # mse
        fdim = 1  # TODO how to get fdim?
        p = self.b.reshape(p, (-1, fdim))
        q = self.b.reshape(q, (fdim, -1))

        vals = p - q
        return vals

    '''
    Properties: get an evaluated property
    '''

    def vals(self, prop):
        prop_vals = []
        # check all self._prop[prop]
        prop_vals.extend(self._prop[prop])

        # if no constant value assigned! the look around
        # check all [entity._prop[prop] * rel for entity, rel in self._in if prop in entity._prop]
        for src, rels in self._in.items():
            for rel in rels:
                vals = rel(self, prop)

        for src, rels in self._in_isa.items():
            for rel in rels:
                # always leave the calculation to the last axis, because matmul do that way
                a = rel.src.blf  # src
                b = rel.dst.blf  # self
                axis = rel.offset
                # the first axis is batch! and axis has +1 already
                # put batch axis and correpsonding axes to the last part
                newaxes = range(1, axis) + range(axis + self.rank,
                                                 len(self.b.shape(a)))  # anything else
                newaxes += [0, ]  # batch
                newaxes += range(axis, axis + self.rank)  # match
                # print(a)
                # print(newaxes)
                p = self.b.reshape(self.b.transpose(
                    a, newaxes), (-1, self.fdim))
                q = self.b.flatten(b)
                print('evaluating {}({},{}):{} ---is-a--> {}:{}'
                      .format(rel.src.name,
                              rel.name,
                              axis,
                              self.b.shape(p),
                              rel.dst.name,
                              self.b.shape(q)))
                distance = self.distance(p, q)
                print('    distance = {}'.format(distance))
                distances.append(distance)
        return prop_vals

    '''
    Properties: get an average value to represent the property
    '''

    def __getitem__(self, prop):
        return self.b.mean(self.vals(prop))

    '''
    Properties: set an value to populate the graph
    '''

    def __setitem__(self, prop, value):
        self._prop[prop].add(value)

    '''
    Evaluate on one property
    '''

    def evaluate(self, prop):
        vals = self.vals(prop)
        return self.b.norm(self.distances(vals, vals))

    '''
    Evaluate all properties
    '''

    def __call__(self):
        # sum up all properties
        return self.b.sum([self.evaluate(prop) for prop in self._prop])
