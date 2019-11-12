import pytest
from regr.graph import Concept, Relation, Property
from regr.graph.relation import Contains, HasA, IsA
from regr.sensor import Sensor


class TestConcept(object):
    def test_concept(self):
        sentence = Concept('sentence')
        phrase = Concept('phrase')
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        assert rel_sentence_contains_phrase.src == sentence
        assert rel_sentence_contains_phrase.dst == phrase
        assert isinstance(rel_sentence_contains_phrase, Contains)

        pair = Concept('pair')
        (rel_pair_has_phrase_phrase, rel_pair_has_phrase_phrase2) = pair(phrase, phrase)
        assert rel_pair_has_phrase_phrase.src == pair
        assert rel_pair_has_phrase_phrase.dst == phrase
        assert isinstance(rel_pair_has_phrase_phrase, HasA)
        assert rel_pair_has_phrase_phrase2.src == pair
        assert rel_pair_has_phrase_phrase2.dst == phrase
        assert isinstance(rel_pair_has_phrase_phrase2, HasA)
        assert rel_pair_has_phrase_phrase is not rel_pair_has_phrase_phrase2

        people = phrase('people')
        (rel_people_is_phrase,) = phrase.relate_to(people)
        assert rel_people_is_phrase.src == people
        assert rel_people_is_phrase.dst == phrase
        assert isinstance(rel_people_is_phrase, IsA)

        organization = phrase('organization')
        workfor = pair('work_for')
        workfor(people, organization)
        (rel_workfor_has_people,) = workfor.relate_to(people)
        (rel_workfor_has_organization,) = workfor.relate_to(organization)
        assert rel_workfor_has_people.src == workfor
        assert rel_workfor_has_people.dst == people
        assert isinstance(rel_workfor_has_people, HasA)
        assert rel_workfor_has_organization.src == workfor
        assert rel_workfor_has_organization.dst == organization
        assert isinstance(rel_workfor_has_organization, HasA)

        sensor = Sensor()
        phrase[people] = sensor
        assert isinstance(phrase[people], Property)
        assert phrase[people] is phrase['<people>']
        assert sensor.fullname == 'phrase/<people>/sensor'
        assert phrase[people, 'sensor'] is sensor
