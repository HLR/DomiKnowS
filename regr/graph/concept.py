from collections import OrderedDict
from collections.abc import Iterable
from itertools import chain
from typing import Tuple, Type

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

            def create(src, *args, **kwargs):
                # add one-by-one
                rels = []
                for argument_name, dst in chain(enum(args, cls=Concept, offset=len(src._out)), enum(kwargs, cls=Concept)):
                    # will be added to _in and _out in Rel constructor
                    rel_inst = Rel(src, dst, argument_name=argument_name)
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

    def __call__(self, *args, **kwargs):
        from .relation import IsA, HasA

        if (len(args) + len(kwargs) == 0 or
                ('name' in kwargs) or
                (len(args)==1 and isinstance(args[0], str))):
            new_concept = Concept(*args, **kwargs)
            new_concept.is_a(self)
            return new_concept
        else:
            return self.has_a(*args, **kwargs)

    def parse_query_apply(self, func, *names, delim='/', trim=True):
        if isinstance(names[0], Concept):
            name = names[0]
            names = names[1:]
        else:
            name0s = names[0].split(delim)
            name = name0s[0]
            if trim:
                name = name.strip()
            names = list(chain(name0s[1:], names[1:]))
            if name[0] == '<' and name[-1] == '>':
                for key in self:
                    if key.name == name[1:-1]:
                        name = key
                        break
        if names:
            return self[name].parse_query_apply(func, *names, delim=delim, trim=trim)
        return func(self, name)

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

        def handle(*args, **kwargs):
            if not args and not kwargs:
                return self._out.setdefault(rel, [])
            return cls._rels[rel](self, *args, **kwargs)
        return handle

    def get_multiassign(self):
        for prop, value in self.items():
            if len(value) > 1:
                yield self._graph, self, prop, value
