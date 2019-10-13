from regr.graph import Concept
import pytest


@pytest.fixture()
def graph(request):
    from regr.graph import Graph, Relation
    with Graph() as graph:
        with Graph('linguistic'):
            sentence = Concept('sentence')
            phrase = Concept('phrase')
            sentence.contain(phrase)
        with Graph('application'):
            people = Concept('people')
            organization = Concept('organization')
            people.is_a(phrase)
            organization.is_a(phrase)
            student = Concept('student')
            school = Concept('school')
            student.is_a(people)
            school.is_a(organization)
            work_for = Concept('work_for')
            work_for.has_a(people)
            work_for.has_a(organization)
            pair = Concept('pair')
            pair.has_a(phrase, phrase)
            kill = Concept('kill')
            kill.is_a(pair)
            kill.has_a(people, people)
            # NB: the difference between work_for and kill is that
            #     kill has multiple paths to its base type.
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
    sentence_1 = 'John works for IBM'

    sentence_data_1 = DataNode(instanceID=0,
                               instanceValue=sentence_1,
                               ontologyNode=sentence,
                               childInstanceNodes=[
                                   DataNode(instanceID=i,
                                            instanceValue=phrase_data,
                                            ontologyNode=phrase,
                                            childInstanceNodes=[])
                                   for i, phrase_data in enumerate(sentence_1.split())])
    return sentence_data_1

@pytest.fixture()
def phrase_data(sentence_data):
    return sentence_data.getChildInstanceNodes()[0]

@pytest.fixture()
def model_trial(graph, sentence_data):
    from regr.graph import Trial

    people = graph['application/people']
    organization = graph['application/organization']

    model_trial = Trial()  # model_trail should come from model run
    phrase_John, phrase_works, phrase_for, phrase_IBM = sentence_data.getChildInstanceNodes()

    model_trial[people, phrase_John] = 0.80
    model_trial[people, phrase_works] = 0.18
    model_trial[people, phrase_for] = 0.05
    model_trial[people, phrase_IBM] = 0.51

    model_trial[organization, phrase_John] = 0.51
    model_trial[organization, phrase_works] = 0.18
    model_trial[organization, phrase_for] = 0.05
    model_trial[organization, phrase_IBM] = 0.70

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
    assert people_candidates[0] == (sentence_data.getChildInstanceNodes()[0],)  # John
    assert people_candidates[1] == (sentence_data.getChildInstanceNodes()[1],)  # works
    assert people_candidates[2] == (sentence_data.getChildInstanceNodes()[2],)  # for
    assert people_candidates[3] == (sentence_data.getChildInstanceNodes()[3],)  # IBM

def test_candidates_workfor_in_phrase(graph, phrase_data):
    work_for = graph['application/work_for']
    candidates = list(work_for.candidates(phrase_data))
    assert len(candidates) == 1
    assert candidates[0] == (phrase_data, phrase_data)

def test_candidates_workfor_in_sentence(graph, sentence_data):
    work_for = graph['application/work_for']
    candidates = list(work_for.candidates(sentence_data))
    assert len(candidates) == 16
    assert candidates[0] == (sentence_data.getChildInstanceNodes()[0], sentence_data.getChildInstanceNodes()[0])
    assert candidates[6] == (sentence_data.getChildInstanceNodes()[1], sentence_data.getChildInstanceNodes()[2])
    assert candidates[12] == (sentence_data.getChildInstanceNodes()[3], sentence_data.getChildInstanceNodes()[0])

def test_candidates_kill_in_phrase(graph, phrase_data):
    kill = graph['application/kill']
    candidates = list(kill.candidates(phrase_data))
    assert len(candidates) == 1
    assert candidates[0] == (phrase_data, phrase_data)

def test_candidates_kill_in_sentence(graph, sentence_data):
    kill = graph['application/kill']
    candidates = list(kill.candidates(sentence_data))
    assert len(candidates) == 16
    assert candidates[0] == (sentence_data.getChildInstanceNodes()[0], sentence_data.getChildInstanceNodes()[0])
    assert candidates[6] == (sentence_data.getChildInstanceNodes()[1], sentence_data.getChildInstanceNodes()[2])
    assert candidates[12] == (sentence_data.getChildInstanceNodes()[3], sentence_data.getChildInstanceNodes()[0])

def test_candidates_query_dummy(graph, sentence_data):
    kill = graph['application/kill']
    def query(candidate):
        return (candidate[0].instanceID % 2) and (candidate[1].instanceID % 2)
    candidates = list(kill.candidates(sentence_data, query=query))
    assert len(candidates) == 4
    assert candidates[0] == (sentence_data.getChildInstanceNodes()[1], sentence_data.getChildInstanceNodes()[1])
    assert candidates[1] == (sentence_data.getChildInstanceNodes()[1], sentence_data.getChildInstanceNodes()[3])
    assert candidates[2] == (sentence_data.getChildInstanceNodes()[3], sentence_data.getChildInstanceNodes()[1])
    assert candidates[3] == (sentence_data.getChildInstanceNodes()[3], sentence_data.getChildInstanceNodes()[3])

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
    assert candidates[0] == (sentence_data.getChildInstanceNodes()[0], sentence_data.getChildInstanceNodes()[3])
    # IBM 0.51 people, John 0.51 organization, just for testing
    assert candidates[1] == (sentence_data.getChildInstanceNodes()[3], sentence_data.getChildInstanceNodes()[0])

def test_getOntologyGraph(graph):
    linguistic = graph['linguistic']
    phrase = graph['linguistic/phrase']
    assert phrase.getOntologyGraph() == linguistic

    application = graph['application']
    people = application['people']
    assert people.getOntologyGraph() == application
