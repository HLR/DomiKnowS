from functools import wraps
import random

from domiknows.graph import Concept, Property, Relation
from domiknows.sensor import Sensor, Learner
import pytest


class TestRelation(object):
    def test_disjoint(self):
        from domiknows.graph.relation import disjoint, NotA

        phrase = Concept()
        people = phrase()
        organization = phrase()
        location = phrase()
        other = phrase()
        o = phrase()

        rels = disjoint(people, organization, location, other, o)

        assert people.relate_to(organization)[0] in rels
        assert isinstance(people.relate_to(organization)[0], NotA)
        assert organization.relate_to(location)[0] in rels
        assert isinstance(organization.relate_to(location)[0], NotA)
        assert location.relate_to(other)[0] in rels
        assert isinstance(location.relate_to(other)[0], NotA)
        assert other.relate_to(o)[0] in rels
        assert isinstance(other.relate_to(o)[0], NotA)
        assert o.relate_to(people)[0] in rels
        assert isinstance(o.relate_to(people)[0], NotA)

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

        echo_number = random.random()

        def backward(number):
            return number == echo_number

        rel_sentence_contains_phrase.backward = backward

        # NB: notice the difference of forward and backward usage.
        # Functions assign to member of instance are not bounded to `self` automatically.
        # The `forward` example is wrapped by the member decorator to work with `self`,
        # what make it acts like a member function that can be call without `self`.
        # The `backward` example is the default way. And it acts like a static member.

        assert rel_sentence_contains_phrase.forward()
        assert rel_sentence_contains_phrase.backward(echo_number)
