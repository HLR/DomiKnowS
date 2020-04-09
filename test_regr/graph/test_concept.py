import pytest

@pytest.fixture()
def graph(request):
    from regr.graph import Graph, Concept, Relation


    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global') as graph:
        graph.ontology = ('http://ontology.ihmc.us/ML/EMR.owl', './examples/emr/')

        with Graph('linguistic') as ling_graph:
            word = Concept(name='word')
            phrase = Concept(name='phrase')
            sentence = Concept(name='sentence')
            phrase.has_many(word)
            sentence.has_many(phrase)

            pair = Concept(name='pair')
            pair.has_a(phrase, phrase)

        with Graph('application') as app_graph:
            entity = Concept(name='entity')
            entity.is_a(phrase)

            people = Concept(name='people')
            organization = Concept(name='organization')
            location = Concept(name='location')
            other = Concept(name='other')
            o = Concept(name='O')

            people.is_a(entity)
            organization.is_a(entity)
            location.is_a(entity)
            other.is_a(entity)
            o.is_a(entity)

            people.not_a(organization)
            people.not_a(location)
            people.not_a(other)
            people.not_a(o)
            organization.not_a(people)
            organization.not_a(location)
            organization.not_a(other)
            organization.not_a(o)
            location.not_a(people)
            location.not_a(organization)
            location.not_a(other)
            location.not_a(o)
            other.not_a(people)
            other.not_a(organization)
            other.not_a(location)
            other.not_a(o)
            o.not_a(people)
            o.not_a(organization)
            o.not_a(location)
            o.not_a(other)

            work_for = Concept(name='work_for')
            work_for.is_a(pair)
            work_for.has_a(people, organization)

            located_in = Concept(name='located_in')
            located_in.is_a(pair)
            located_in.has_a(location, location)

            live_in = Concept(name='live_in')
            live_in.is_a(pair)
            live_in.has_a(people, location)

            orgbase_on = Concept(name='orgbase_on')
            orgbase_on.is_a(pair)
            orgbase_on.has_a(organization, location)

            kill = Concept(name='kill')
            kill.is_a(pair)
            kill.has_a(people, people)

    yield graph

    #------------------
    # tear down
    #------------------
    Graph.clear()
    Concept.clear()
    Relation.clear()

@pytest.fixture()
def sentence_data(graph):
    from regr.graph import DataNode

    sentence = graph['linguistic/sentence']
    phrase = graph['linguistic/phrase']
    sentence_data = 'John works for IBM'

    sentence_node = DataNode(instanceID=0, instanceValue=sentence_data, ontologyNode=sentence)
    sentence_node.relationLinks['contains'] = []
    for i, phrase_data in enumerate(sentence_data.split()):
        phrase_node = DataNode(instanceID=i, instanceValue=phrase_data, ontologyNode=phrase)
        sentence_node.relationLinks['contains'].append(phrase_node)
    return sentence_node

@pytest.fixture()
def phrase_data(sentence_data):
    return sentence_data.getChildDataNodes()[0]

@pytest.fixture()
def model_trial(graph, sentence_data):
    from regr.graph import Trial, DataNode

    people = graph['application/people']
    organization = graph['application/organization']

    model_trial = Trial()  # model_trail should come from model run
    phrase_John, phrase_works, phrase_for, phrase_IBM = sentence_data.getChildDataNodes()

    model_trial[people, phrase_John] = 0.80
    model_trial[people, phrase_works] = 0.18
    model_trial[people, phrase_for] = 0.05
    model_trial[people, phrase_IBM] = 0.51

    model_trial[organization, phrase_John] = 0.51
    model_trial[organization, phrase_works] = 0.18
    model_trial[organization, phrase_for] = 0.05
    model_trial[organization, phrase_IBM] = 0.70

    phrase_John.predictions = {
        DataNode.PredictionType["Learned"]: {
            (people): 0.80,
            (organization): 0.51}}
    phrase_works.predictions = {
        DataNode.PredictionType["Learned"]: {
            (people): 0.18,
            (organization): 0.18}}
    phrase_for.predictions = {
        DataNode.PredictionType["Learned"]: {
            (people): 0.05,
            (organization): 0.05}}
    phrase_IBM.predictions = {
        DataNode.PredictionType["Learned"]: {
            (people): 0.51,
            (organization): 0.70}}

    return model_trial

def test_candidates_people_in_phrase(graph, phrase_data):
    people = graph['application/people']
    people_candidates = list(people.candidates(phrase_data))
    assert len(people_candidates) == 1
    assert people_candidates[0] == (phrase_data,)

def test_candidates_people_in_sentence(graph, sentence_data):
    people = graph['application/people']
    people_candidates = list(people.candidates(sentence_data))
    assert len(people_candidates) == 4
    assert people_candidates[0] == (sentence_data.getChildDataNodes()[0],)  # John
    assert people_candidates[1] == (sentence_data.getChildDataNodes()[1],)  # works
    assert people_candidates[2] == (sentence_data.getChildDataNodes()[2],)  # for
    assert people_candidates[3] == (sentence_data.getChildDataNodes()[3],)  # IBM

def test_candidates_workfor_in_phrase(graph, phrase_data):
    work_for = graph['application/work_for']
    candidates = list(work_for.candidates(phrase_data))
    assert len(candidates) == 1
    assert candidates[0] == (phrase_data, phrase_data)

def test_candidates_workfor_in_sentence(graph, sentence_data):
    work_for = graph['application/work_for']
    candidates = list(work_for.candidates(sentence_data))
    assert len(candidates) == 16
    assert candidates[0] == (sentence_data.getChildDataNodes()[0], sentence_data.getChildDataNodes()[0])
    assert candidates[6] == (sentence_data.getChildDataNodes()[1], sentence_data.getChildDataNodes()[2])
    assert candidates[12] == (sentence_data.getChildDataNodes()[3], sentence_data.getChildDataNodes()[0])

def test_candidates_kill_in_phrase(graph, phrase_data):
    kill = graph['application/kill']
    candidates = list(kill.candidates(phrase_data))
    assert len(candidates) == 1
    assert candidates[0] == (phrase_data, phrase_data)

def test_candidates_kill_in_sentence(graph, sentence_data):
    kill = graph['application/kill']
    candidates = list(kill.candidates(sentence_data))
    assert len(candidates) == 16
    assert candidates[0] == (sentence_data.getChildDataNodes()[0], sentence_data.getChildDataNodes()[0])
    assert candidates[6] == (sentence_data.getChildDataNodes()[1], sentence_data.getChildDataNodes()[2])
    assert candidates[12] == (sentence_data.getChildDataNodes()[3], sentence_data.getChildDataNodes()[0])

def test_candidates_query_dummy(graph, sentence_data):
    kill = graph['application/kill']
    def query(candidate):
        return (candidate[0].instanceID % 2) and (candidate[1].instanceID % 2)
    candidates = list(kill.candidates(sentence_data, query=query))
    assert len(candidates) == 4
    assert candidates[0] == (sentence_data.getChildDataNodes()[1], sentence_data.getChildDataNodes()[1])
    assert candidates[1] == (sentence_data.getChildDataNodes()[1], sentence_data.getChildDataNodes()[3])
    assert candidates[2] == (sentence_data.getChildDataNodes()[3], sentence_data.getChildDataNodes()[1])
    assert candidates[3] == (sentence_data.getChildDataNodes()[3], sentence_data.getChildDataNodes()[3])

def test_candidates_query_trial(graph, sentence_data, model_trial):
    work_for = graph['application/work_for']
    people = graph['application/people']
    organization = graph['application/organization']
    def query(candidate):
        with model_trial:
            return (people.predict(sentence_data, candidate[0]) > 0.5 and
                    organization.predict(sentence_data, candidate[1]) > 0.5 and
                    candidate[0] != candidate[1]
                   )
    candidates = list(work_for.candidates(sentence_data, query=query))
    assert len(candidates) == 2
    # John 0.80 people, IBM 0.70 organization
    assert candidates[0] == (sentence_data.getChildDataNodes()[0], sentence_data.getChildDataNodes()[3])
    # IBM 0.51 people, John 0.51 organization, just for testing
    assert candidates[1] == (sentence_data.getChildDataNodes()[3], sentence_data.getChildDataNodes()[0])

def test_getOntologyGraph(graph):
    linguistic = graph['linguistic']
    phrase = graph['linguistic/phrase']
    assert phrase.getOntologyGraph() == linguistic

    application = graph['application']
    people = application['people']
    assert people.getOntologyGraph() == application
    
    
    
    
    
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
