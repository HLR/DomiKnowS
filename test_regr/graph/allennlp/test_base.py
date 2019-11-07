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
            phrase.contains(word)
            sentence.contains(phrase)

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
def allennlp_graph(graph):
    from regr.graph.allennlp.base import AllenNlpGraph
    from regr.sensor import Sensor, Learner


    sentence = graph['linguistic/sentence']
    phrase = graph['linguistic/phrase']
    people = graph['application/people']
    #import pdb; pdb.set_trace()
    sentence['raw'] = Sensor()
    phrase['raw'] = Sensor()
    phrase['encode'] = Learner()
    people['label'] = Learner()
    people['label'] = Sensor()
    return AllenNlpGraph(graph)


@pytest.fixture()
def context(allennlp_graph):
    from random import choices
    import numpy as np


    sentence = allennlp_graph['linguistic/sentence']
    phrase = allennlp_graph['linguistic/phrase']
    people = allennlp_graph['application/people']

    context = {}
    batch = 8
    length = 12
    feature = 24
    vocab = ['John', 'works', 'for', 'IBM']

    sentences = [' '.join(choices(vocab, k=length)) for _ in range(batch)]
    context[sentence['raw'].fullname] = sentences
    context[phrase['raw'].fullname] = [sentence.split() for sentence in sentences]
    context[phrase['encode'].fullname] = np.random.rand(batch, length, feature)
    context[people['label'].fullname] = np.random.rand(batch, length, 2)

    return context

def test_populate(allennlp_graph, context):
    sentence = allennlp_graph['linguistic/sentence']
    phrase = allennlp_graph['linguistic/phrase']
    people = allennlp_graph['application/people']

    #import pdb; pdb.set_trace()
    trial, data_dict = allennlp_graph.populate(context, sentence)

    # sentence
    for node, context_value in zip(data_dict[sentence], context[sentence['raw'].fullname]):
        assert node.raw == context_value
