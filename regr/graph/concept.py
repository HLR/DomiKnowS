from collections import defaultdict, Iterable, OrderedDict
from itertools import chain
if __package__ is None or __package__ == '':
    from base import BaseGraphTree, Scoped
    from backend import Backend, NumpyBackend
else:
    from .base import BaseGraphTree, Scoped
    from .backend import Backend, NumpyBackend
from ..utils import enum

@Scoped.class_scope
@BaseGraphTree.localize_namespace
class Concept(BaseGraphTree):
    _rels = {}  # catogrory_name : creation callback

    @classmethod
    def relation_type(cls, name=None):
        def update(Rel):
            rel_name = name
            if rel_name is None:
                rel_name = Rel.__name__  # Rel.suggest_name()

            def create(src, *args, **kwargs):
                # add one-by-one
                for rel_src_name, dst in chain(enum(args, cls=Concept, offset=len(src._out)), enum(kwargs, cls=Concept)):
                    rel_inst_name = '{}-{}-{}-{}'.format(
                        src.name, rel_name, rel_src_name, dst.name)
                    # will be added to _in and _out in constructor
                    rel_inst = Rel(src, dst, name=rel_inst_name,
                                   catogrory_name=rel_name)

            cls._rels[rel_name] = create
            return Rel

        return update

    def __init__(self, name=None):
        '''
        Declare an concept.
        '''
        BaseGraphTree.__init__(self, name)

        self._in = OrderedDict()  # relation catogrory_name : list of relation inst
        self._in.setdefault(list)
        self._out = OrderedDict()  # relation catogrory_name : list of relation inst
        self._out.setdefault(list)

        # FIXME: need this? if true, relation value will be include when calculating a property
        self.transparent = False

    # disable context for Concept
    def __enter__(self):
        raise AttributeError(
            '{} object has no attribute __enter__'.format(type(self).__name__))

    def __exit__(self, exc_type, exc_value, traceback):
        raise AttributeError(
            '{} object has no attribute __exit__'.format(type(self).__name__))

    def query_apply(self, names, func):
        if len(names) > 1:
            raise ValueError(
                'Concept cannot have nested elements. Access properties using property name directly. Query of names {} is not possibly applied.'.format(names))
        # this is only one layer above the leaf layer
        return func(names[0])

    def get_apply(self, name):
        return BaseGraphTree.get_apply(self, name)[0]

    def set_apply(self, name, sub):
        if name not in self:
            OrderedDict.__setitem__(self, name, list)
        BaseGraphTree.get_apply(self, name).append(sub)

    def what(self):
        wht = BaseGraphTree.what(self)
        wht['relations'] = dict(self._out)
        return wht

    def __getattr__(self, rel):
        '''
        Create relation by registered relation types
        '''
        cls = type(self)  # bind to the real class

        def handle(*args, **kwargs):
            if not args and not kwargs:
                return cls._out[rel]
            return cls._rels[rel](self, *args, **kwargs)
        return handle

    def get_multiassign(self):
        for prop, value in self.items():
            if len(value) > 1:
                yield self._graph, self, prop, value

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
        if prop not in self or not self[prop]:
            return [(None, 0), ]
        #vals = []
        #confs = []
        # for val, conf in self[prop]:
        #    vals.append(prop)
        #    confs.append(conf)
        vals, confs = zip(*self[prop])
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
