from collections import Iterable, defaultdict, namedtuple
from .base import AutoNamed
from .backend import Backend, NumpyBackend

isa = namedtuple('isa', ('src', 'dst', 'offset', 'name'))


class Entity(AutoNamed):
    default_graph = None
    default_backend = NumpyBackend()

    '''
    Declare an entity.
    '''

    def __init__(self, blf=None, rpt=None, name=None, backend=None):
        AutoNamed.__init__(self, name)
        if Entity.default_graph is not None:
            Entity.default_graph.append(self)

        self._blf = blf
        self._rpt = rpt

        # in
        self._in_isa = defaultdict(list)
        self._in_hasa = defaultdict(list)
        # out
        self._out_isa = defaultdict(list)
        self._out_hasa = defaultdict(list)

        self._b = backend

    @property
    def b(self):
        if isinstance(self._b, Backend):
            return self._b
        else:
            return self.__class__.default_backend

    @property
    def blf(self):
        if self._blf is not None:
            return self._blf
        else:
            # TODO: any fallback?
            return None

    @property
    def rpt(self):
        if self._rpt is not None:
            return self._rpt
        else:
            # TODO: any fallback?
            return None

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
    The 'distance' used in this entity. Lower value indicates better consistency.
    '''

    def distance(self, p, q):
        # inner product
        #q = self.b.reshape(q, (-1, 1))
        #val = self.b.sum(self.b.matmul(p, q))
        # relative entropy
        #q = self.b.reshape(q, (1, -1))
        #val = self.b.sum(p * self.b.log( p / q ))
        # mse
        q = self.b.reshape(q, (1, -1))
        val = self.b.norm(p - q)
        return val

    '''
    Declare 'is-a' relationship.
    '''

    def isa(self, entity, offset=1, rel_name=None):
        if isinstance(entity, Entity):
            # really add it when it is an Entity
            rel = isa(self, entity, offset, rel_name)
            # in
            entity._in_isa[self].append(rel)
            # out
            self._out_isa[entity].append(rel)
        elif isinstance(entity, Iterable):
            # more than one relation, break down and add each
            entities = entity
            if isinstance(entities, dict):
                name_fmt = '{}'
                enum = entities.items()
            else:
                name_fmt = 'arg-{}'
                enum = enumerate(entities)

            offset = 1  # 0 for batch
            for key, entity in enum:
                if rel_name:
                    name = ('{}-' + name_fmt).format(rel_name, key)
                else:
                    name = name_fmt.format(key)
                self.isa(entity, offset, name)
                offset += entity.rank  # TODO: what if the fallback not yet work at that time?
        else:
            raise NotImplementedError(
                'Unsupported type of is-a relation target: {}'.format(entity))

    '''
    Declare 'has-a' relationship.
    '''

    def hasa(self, entity, rel=None):
        pass

    '''
    Evaluate 'is-a' relationship.
    '''

    def eval_isa(self):
        distances = []
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
                print('evaluating {}({},{}):{} ---is-a--> {}:{}'.format(rel.src.name,
                                                                        rel.name,
                                                                        axis,
                                                                        self.b.shape(p),
                                                                        rel.dst.name,
                                                                        self.b.shape(q)))
                distance = self.distance(p, q)
                print('    distance = {}'.format(distance))
                distances.append(distance)
        distance = self.b.sum(distances)
        return distance

    '''
    Evaluate 'has-a' relationship.
    '''

    def eval_hasa(self):
        return 0

    def what(self):
        return {'is-a': dict(self._out_isa),
                'has-a': dict(self._out_hasa), }

    '''
    Evaluate
    '''

    def __call__(self):
        return self.eval_isa() + self.eval_hasa()
