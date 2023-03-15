from collections import OrderedDict
from itertools import chain, product
from typing import Type
from .base import Scoped, BaseGraphTree
from ..utils import enum


@Scoped.class_scope
@BaseGraphTree.localize_namespace
class Concept(BaseGraphTree):
    _rels = {}  # catogrory_name : creation callback

    @classmethod
    def relation_type(cls, name=None):
        def update(Rel):
            if name is not None:
                Rel.name = classmethod(lambda cls: name)

            def create(src, *args, auto_constraint=None, **kwargs):
                # add one-by-one
                rels = []
                for argument_name, dst in chain(enum(args, cls=Concept, offset=len(src._out)), enum(kwargs, cls=Concept)):
                    # will be added to _in and _out in Rel constructor
                    rel_inst = Rel(src, dst, argument_name=argument_name, auto_constraint=auto_constraint)
                    rels.append(rel_inst)
                return rels

            cls._rels[Rel.name()] = create
            return Rel

        return update

    def __init__(self, name=None):
        '''
        Declare an concept.
        '''
        BaseGraphTree.__init__(self, name)

        self._in = OrderedDict()  # relation catogrory_name : list of relation inst
        self._out = OrderedDict()  # relation catogrory_name : list of relation inst

    def __str__(self):
        return self.name
    
    def __rept__(self):
        return type(self) + ":" + self.name
    
    def processLCArgs(self, *args, conceptT=None, **kwargs):
        from domiknows.graph.logicalConstrain import eqL, V
        if len(args) > 1 and isinstance(args[1], eqL):
            nameX = args[0]
            path = (nameX, args[1])
                                    
            return [conceptT, V(name=nameX, v=path)]
        elif len(args) and isinstance(args[0], str):
            name = args[0]
            
            if "path" in kwargs:
                path = kwargs['path']
                
                return [conceptT, V(name=name, v=path)]
            else:
                return [conceptT, V(name=name)]
        elif "path" in kwargs:
            path = kwargs['path']
                                    
            return [conceptT, V(name=None, v=path)]
        else:
            return [conceptT]

    def __call__(self, *args, name=None, ConceptClass=None, auto_constraint=None, **kwargs):
        from .relation import IsA, HasA
        if ConceptClass is None:
            ConceptClass = Concept
            
        if (name is None and len(args) and isinstance(args[0], str)) or ("path" in kwargs):
            if isinstance(self, EnumConcept):
                conceptT = (self, self.name, None, len(self.enum))
            else:
                conceptT = (self, self.name, None, 1)

            return self.processLCArgs(*args, conceptT=conceptT, **kwargs)
            
        if (not args and not kwargs) or name is not None:
            new_concept = ConceptClass(name=name, *args, **kwargs)
            new_concept.is_a(self, auto_constraint=auto_constraint)
            return new_concept
        else:
            return self.has_a(*args, auto_constraint=auto_constraint, **kwargs)

    def relate_to(self, concept, *tests):
        from .relation import Relation

        retval = []
        tests_in = [lambda x: x.src == concept,]
        tests_in.extend(tests)
        for rel in chain(*self._in.values()):
            for test in tests_in:
                if isinstance(test, Type) and issubclass(test, Relation):
                    if not isinstance(rel, test):
                        break
                else:
                    if not test(rel):
                        break
            else:
                retval.append(rel)
        tests_out = [lambda x: x.dst == concept,]
        tests_out.extend(tests)
        for rel in chain(*self._out.values()):
            for test in tests_out:
                if isinstance(test, Type) and issubclass(test, Relation):
                    if not isinstance(rel, test):
                        break
                else:
                    if not test(rel):
                        break
            else:
                retval.append(rel)
        return retval

    def __getitem__(self, name):
        try:
            return self.get_sub(name)
        except KeyError as e:
            raise type(e)(name)

    def __setitem__(self, name, obj):
        if isinstance(name, tuple):
            # for name_, obj_ in zip(name, obj):
            #     self[name_] = obj_
            # return self.__setitem__('joint_'+'_'.join(map(str, name)), obj)
            return self.set_apply(name, obj)
        return super().__setitem__(name, obj)

    def set_apply(self, name, sub):
        from ..sensor import Sensor
        from .property import Property
        if isinstance(sub, Property):
            # call usually come from attach, further from constructor of property
            BaseGraphTree.set_apply(self, name, sub)
        elif isinstance(sub, Sensor):
            if name not in self:
                with self:
                    prop = Property(prop_name=name)
            self.get_apply(name).attach(sub)

    def what(self):
        wht = BaseGraphTree.what(self)
        wht['relations'] = dict(self._out)
        return wht

    def __getattr__(self, rel):
        '''
        Create relation by registered relation types
        '''
        cls = type(self)  # bind to the real class

        try:
            Rel = cls._rels[rel]
        except KeyError as e:
            if  isinstance(self, EnumConcept):
                if rel in self.enum:
                    def ecHandle(*args, **kwargs):
                        conceptT = (self, rel, self.get_index(rel), len(self.enum))
                        return self.processLCArgs(*args, conceptT=conceptT, **kwargs)
                    
                    return ecHandle
            else:
                raise AttributeError(*e.args)
        def handle(*args, **kwargs):
            if not args and not kwargs:
                return self._out.setdefault(rel, [])
            return Rel(self, *args, **kwargs)
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
        confs[confs is None] = self.b.mean(confs[confs is not None])

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

    def candidates(self, root_data, query=None, logger = None):
        def basetype(concept):
            # get inheritance rank
            basetypes = []
            for is_a in concept.is_a():
                basetypes.append(basetype(is_a.dst))

            has_a_basetype = []
            for has_a in concept.has_a():
                has_a_basetype.extend(basetype(has_a.dst))
            basetypes.append(has_a_basetype)

            basetypes = sorted(basetypes, key=lambda x: len(x))
            # TODO: here we assume all possible basetypes are homogenous and get the longest one
            # There should be some other semantic to handle this ambiguousity.
            # if still nothing, the original concept is the base type
            return basetypes[-1] or (concept,)

        base = basetype(self)

        assert len(set(base)) == 1  # homogenous base type

        def get_base_data(root_data, single_base):
            from .dataNode import DataNode
            assert isinstance(root_data, DataNode)
            base_data = root_data.findDatanodes(select = single_base.name)
                
            return base_data
        
            while True:
                if base_data[0].getOntologyNode() == single_base:
                    return base_data
                base_data = list(chain(*(bd.getChildDataNodes() for bd in base_data)))

        base_data = get_base_data(root_data, base[0])
        
        if not base_data:
            if logger:
                logger.info('Found base type - %s - for current concept - %s -'%(base[0],self.name))
                conceptNames, relationNames = root_data.findConceptsNamesInDatanodes()
                logger.warning('Found no candidates for - %s -, existing concepts in DataNode are - %s, relations - %s'%(base[0],conceptNames,relationNames))
            
        if query:
            return filter(query, product(base_data, repeat=len(base)))
        else:
            return product(base_data, repeat=len(base))

    def getOntologyGraph(self):  # lowest sub-graph
        node = self
        while isinstance(node, Concept):  # None is not instance of Concept, If this concept has no graph, it will end up None
            node = node.sup
        return node


class EnumConcept(Concept):
    def __init__(self, name=None, values=[]):
        super().__init__(name=name)
        self.enum = values

    @property
    def enum(self):
        return [e.name for e in self._enum]
    
    @property
    def attributes(self):
        return [(self, e.name, self.get_index(e.name), len(self.enum)) for e in self._enum]

    @enum.setter
    def enum(self, values):
        from enum import Enum
        self._enum = Enum(self.name, values, start=0)

    def get_index(self, value):
        return self._enum[value].value

    def get_value(self, index):
        try:
            t = self._enum(index) 
            return t.name
        except ValueError:
            return None
        
    def get_concept(self, value):
        valueIndex = self.get_index(value)
        return (self, value, valueIndex, len(self.enum))
        