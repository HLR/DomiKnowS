from collections import defaultdict, Iterable, OrderedDict
if __package__ is None or __package__ == '':
    from base import Scorable, Propertied, local_names_and_objs
    from backend import Backend, NumpyBackend
else:
    from .base import Scorable, Propertied, local_names_and_objs
    from .backend import Backend, NumpyBackend
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
class Concept(Scorable, Propertied):
    default_backend = NumpyBackend()
    _rel_types = dict()  # relation name (to be call): relation class

    @classmethod
    def update_rel_type(cls, Rel, name=None):
        if name is None:
            name = Rel.suggest_name()
        cls._rel_types[name] = Rel

    @classmethod
    def register_rel_type(cls, name=None):
        return lambda Rel, name=None: cls.update_rel_type(Rel, name)

    def __init__(self, rank=None, name=None, backend=None):
        '''
        Declare an concept.
        '''
        Scorable.__init__(self, name)
        Propertied.__init__(self)

        # register myself
        if __package__ is None or __package__ == '':
            import graph
        else:
            from . import graph
        if graph.Graph.default is not None:  # use base class Graph as the global environment
            graph.Graph.default.add(self)
        self._graph = graph.Graph.default

        # TODO: deal with None here? or when it can be infer? or some other occasion?
        self._rank = rank
        self._in = defaultdict(set)  # src concepts : set of relation instances
        # dst concepts : set of relation instances
        self._out = defaultdict(set)
        self._backend = backend
        # if true, relation value will be include when calculating a property
        self.transparent = False

    def what(self):
        return {'out_rels': dict(self._out), }

    @property
    def fullname(self):
        if self._graph is None:
            return self.name
        return self._graph.fullname + '/' + self.name

    def __getattr__(self, prop):
        '''
        Create relation by registered relation types
        '''
        cls = type(self)  # bind to the real class
        Rel = cls._rel_types[prop]

        def create_rel(dst, *args, **kwargs):
            dst_name = ','.join(['{}:{}'.format(i, concept.name)
                                 for i, concept in enum(dst)])
            name = '{}-{}-({})'.format(self.name, prop, dst_name)
            # TODO: should check the rank of src and dst? or in the constructor? or some other occasion?
            rel = Rel(self, dst, name=name, *args, **kwargs)
            for _, v in rel.dst:
                self._out[v].add(rel)
                v._in[self].add(rel)
        return create_rel

    def __getitem__(self, prop):
        if prop not in self.props:
            return None
        #if len(self.props[prop]) == 1:
        #    return self.props[prop][0]
        return self.props[prop]
        # TODO: shouldn't need above lines. have them to avoid aggr in current dirty version
        return self.aggregate(*self.vals(prop, 0))

    def __setitem__(self, prop, value):
        # TODO: prevent multiple assignment or recursive assignment?
        self.props[prop].append(value)

    def get_multiassign(self):
        for prop, value in self.props.items():
            if len(value) > 1:
                yield self, prop, value

    @property
    def rank(self):
        return self._rank

    @property
    def b(self):
        '''
        Backend shortcut, possible to fallback to class default
        '''
        if isinstance(self._backend, Backend):
            return self._backend
        else:
            return type(self).default_backend

    def distances(self, p, q):
        '''
        The "distance" used in this concept to measure the consistency.
        Lower value indicates better consistency.
        Feature(s) of one instance is always on only the last axis.
        p, q - [(batch, vdim(s...)),...] * nval
        '''
        # inner product
        #q = self.b.reshape(q, (-1, 1))
        #val = self.b.sum(self.b.matmul(p, q))
        # relative entropy
        #q = self.b.reshape(q, (1, -1))
        #val = self.b.sum(p * self.b.log( p / q ))
        # mse

        nval = len(p)
        assert len(p) == len(q)
        p = self.b.reshape(self.b.concatenete(p, axis=0), (nval, -1, 1))
        q = self.b.reshape(self.b.concatenete(q, axis=0), (1, -1, nval))
        vals = self.b.norm(p - q, axis=1)
        return vals

    def aggregate(self, vals, confs):
        '''
        The aggregation used in this concept to reduce the inconsistent values.
        '''
        vals = self.b.concatenate(vals, axis=0)
        confs = self.b.concatenate(confs, axis=0)
        # TODO: deal with None value in confs. The following is not yet a good solution
        confs[confs == None] = self.b.mean(confs[confs != None])

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

        weight = softmax(logits)

        # TODO: should via the same approach as dealing with None values
        vals = self.b.sum(weight * vals)
        confs = self.b.sum(weight * confs)
        return vals, confs

    def bvals(self, prop):
        '''
        Properties: get all binded values

        :param prop: property name
        :type prop: str

        :returns: Return `vals` and `confs` where `vals` is a list of values binded to the property
                  and `confs` is a list of values representing the confidence of each binded value.
                  An element of `vals` should have the shape:
                  ( batch, vdim(s...) )
                  Return `None` is if never binded to this property.
        :rtype: [barray,...], [barray,...]
        '''
        if prop not in self.props or not self.props[prop]:
            return [(None, 0)]
        #vals = []
        #confs = []
        # for val, conf in self.props[prop]:
        #    vals.append(prop)
        #    confs.append(conf)
        vals, confs = zip(*self.props[prop])
        return vals, confs

    def rvals(self, prop, hops=1):
        '''
        Properties: get all values from relations
        '''
        vals = []
        confs = []
        for src, rels in self._in.items():
            for rel in rels:
                rvals, rconfs = rel.T(self, prop, hops - 1)
                vals.extend(rvals)
                confs.extend(rconfs)
        return vals, confs

    def vals(self, prop, hops=1):
        vals, confs = self.bvals(prop)
        if hops > 0:
            rvals, rconfs = self.rvals(prop, hops)
            vals.extend(rvals)
            confs.extend(rconfs)
        return vals, confs

    def score_(self, prop):
        # TODO: some clean up here, focus on only these values
        # Problem: the interface behind this should not need prop?
        vals, confs = self.vals(prop, 1)
        #vals, confs = zip(*values)
        return self.b.norm(self.distances(vals, vals))

    def score(self, prop=None):
        if prop is None:
            return sum([self(prop) for prop, _ in self.props.items()])
        return self.score_(prop)


# TODO: compose concept implement to replace the emum function and complicated details there
class ComposeConcept(Concept):

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

    def __init__(self, concepts):
        name = ','.join(['{}:{}'.format(i, concept.name)
                         for i, concept in enum(dst)])
        Concept.__init__(self, name)

    pass
