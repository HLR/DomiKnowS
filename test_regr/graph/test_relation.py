from functools import wraps
from regr.graph import Concept, Property, Relation
from regr.sensor import Sensor, Learner
import pytest


class TestRelation(object):
    def test_disjoint(self):
        from regr.graph.relation import disjoint, NotA

        phrase = Concept()
        people = phrase()
        organization = phrase()
        location = phrase()
        other = phrase()
        o = phrase()

        rels = disjoint(people, organization, location, other, o)

        assert people[organization][0] in rels
        assert isinstance(people[organization][0], NotA)
        assert organization[location][0] in rels
        assert isinstance(organization[location][0], NotA)
        assert location[other][0] in rels
        assert isinstance(location[other][0], NotA)
        assert other[o][0] in rels
        assert isinstance(other[o][0], NotA)
        assert o[people][0] in rels
        assert isinstance(o[people][0], NotA)

    def test_relation(self):
        sentence = Concept('sentence')
        phrase = Concept('phrase')
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)

        rel_sentence_contains_phrase['forward'] = Sensor()
        rel_sentence_contains_phrase['forward'] = Learner()
        rel_sentence_contains_phrase['backward'] = Sensor()
        rel_sentence_contains_phrase['backward'] = Learner()

        def member(self):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(self, *args, **kwargs)
                return wrapper
            return decorator

        @member(rel_sentence_contains_phrase)
        def forward(self, ):
            assert isinstance(self['forward'], Property)
            assert len(list(self['forward'].find())) == 2
            assert len(list(self['forward'].find(Learner))) == 1
            assert isinstance(self['backward'], Property)
            assert len(list(self['backward'].find())) == 2
            assert len(list(self['backward'].find(Sensor))) == 2 # Learner is extended from sensor
            return self # self contains two properties (forward and backward), so it is not falsy

        rel_sentence_contains_phrase.forward = forward

        assert rel_sentence_contains_phrase.forward()
        assert rel_sentence_contains_phrase.backward(0) is None
