from collections import defaultdict, Iterable, OrderedDict
from .base import AutoNamed, local_names_and_objs
from .backend import Backend, NumpyBackend
from .graph import Graph
import warnings


def enum(concepts):
    if isinstance(concepts, Concept):
        enum = {0: concepts}.items()
    elif isinstance(concepts, OrderedDict):
        enum = concepts.items()
    elif isinstance(concepts, dict):
        enum = concepts.items()
        warnings.warn('Please use OrderedDict rather than dict to prevent unpredictable order of arguments.' +
                      'For this instance, {} is used.'
                      .format(concepts),
                      UserWarning, stacklevel=3)
    elif isinstance(concepts, Iterable):
        enum = enumerate(concepts)
    else:
        raise TypeError('Unsupported type of concepts. Use Concept, OrderedDict or other Iterable.'
                        .format(type(concepts)))

    # for k, v in enum:
    #    yield (k, v)
    return enum


@local_names_and_objs
class Concept(AutoNamed):
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
    Declare an concept.
    '''

    def __init__(self, name=None, backend=None):
        AutoNamed.__init__(self, name)

        if Graph.default_graph is not None:  # use base class Graph as the global environment
            Graph.default_graph.concept.append(self)

        self._prop = defaultdict(list)  # name : list of binding values
        self._in = defaultdict(set)  # src concepts : set of relation instances
        self._out = defaultdict(set)  # dst concepts : set of relation instances
        self._backend = backend
        # if true, relation value will be include when calculating a property
        self.transparent = False

    def what(self):
        return {'rels': dict(self._out), }

    '''
    Create relation by registered relation types
    '''

    def __getattr__(self, prop):
        cls = type(self)  # bind to the real class
        Rel = cls._rel_types[prop]

        def create_rel(dst, *args, **kwargs):
            dst_name = ','.join(['{}:{}'.format(i, concept.name)
                                 for i, concept in enum(dst)])
            name = '({})-{}-({})'.format(self.name, prop, dst_name)
            rel = Rel(self, dst, name=name, *args, **kwargs)
            for _, v in rel.dst:
                self._out[v].add(rel)
                v._in[self].add(rel)
        return create_rel

    '''
    Backend, possible to fallback to class default
    '''
    @property
    def b(self):
        if isinstance(self._backend, Backend):
            return self._backend
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
    The "distance" used in this concept to measure the consistency.
    Lower value indicates better consistency.
    Feature(s) of one instance is always on only the last axis.
    p, q - (nval, batch * len, prop_dim)
    '''

    def distance(self, p, q):
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
    The aggregation used in this concept to reduce the inconsistent values.
    '''

    def aggregate(sefl, vals, confs):
        # inverse logistic
        def logit(z): return - self.b.log(self.b(1.) / z - self.b(1.))
        logits = logit(confs)

        # thermodynamic softmax
        def t_softmax(z, beta=-1):  # beta = 1/(k_B*T) - Coldness, the lower, the more concentrated
            z = self.b.exp(- beta * z)
            inf = (z == self.b.inf)
            if self.b.any(inf):
                return inf * 1  # just convert type
            else:
                return z / self.b.sum(z)

        return self.b.sum(softmax(logits) * vals)

    '''
    Properties: get all binded values
    '''

    def bvals(self, prop):
        vals = []
        confs = []
        # check all self._prop[prop]
        for val, conf in self._prop[prop]:
            vals.append(prop)
            confs.append(conf)
        return vals, confs

    '''
    Properties: get all values from relations
    '''

    def rvals(self, prop):
        # TODO: not yet clean up this part
        # if no constant value assigned! the look around
        # check all [entity._prop[prop] * rel for entity, rel in self._in if prop in entity._prop]
        for src, rels in self._in.items():
            for rel in rels:
                vals = rel(self, prop)
        for src, rels in self._in.items():
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
                p = self.b.reshape(self.b.transpose(a, newaxes),
                                   (-1, self.fdim))
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
        return vals, probs

    '''
    Properties: get an value for the property
    '''

    def __getitem__(self, prop):
        vals, confs = self.bvals()
        if self.transparent:
            rvals, rconfs = self.rvals()
            vals.extend(rvals)
            confs.extend(rconfs)
        vals = self.b.concatenate(vals, axis=0)
        confs = self.b.concatenate(confs, axis=0)
        return self.aggregate(vals, confs)

    '''
    Properties: set an value to populate the graph
    '''

    def __setitem__(self, prop, value, confidence=None):
        # TODO: revent multiple assignment?
        self._prop[prop].append((value, confidence))

    '''
    Evaluate on property
    '''

    def __call__(self, prop=None):
        if prop is None:
            # sum up all properties
            return self.b.sum([self(prop) for prop in self._prop])

        vals = self.vals(prop)
        return self.b.norm(self.distance(vals, vals))
