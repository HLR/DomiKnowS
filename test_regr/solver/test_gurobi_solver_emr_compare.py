import pytest
from itertools import chain
import numpy as np

@pytest.fixture()
def emr_graph(request):
    from regr.graph import Graph, Concept, Relation

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global') as graph:
        graph.ontology = ('http://ontology.ihmc.us/ML/EMR.owl', './examples/emr')

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
def emr_input(emr_graph):
    import numpy as np

    test_phrase = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]
    app_graph = emr_graph["application"]
    conceptNamesList = [app_graph["people"],
                        app_graph["organization"],
                        app_graph["other"],
                        app_graph["location"],
                        app_graph["O"]]
    relationNamesList = [app_graph["work_for"],
                         app_graph["live_in"],
                         app_graph["located_in"]]

    # tokens
    test_graphResultsForPhraseToken = {}
    test_graphResultsForPhraseToken[app_graph["people"]] =       np.random.rand(4)
    test_graphResultsForPhraseToken[app_graph["organization"]] = np.random.rand(4)
    test_graphResultsForPhraseToken[app_graph["other"]] =        np.random.rand(4)
    test_graphResultsForPhraseToken[app_graph["location"]] =     np.random.rand(4)
    test_graphResultsForPhraseToken[app_graph["O"]] =            np.random.rand(4)
    
    test_graphResultsForPhraseRelation = dict()
    eye_cut = 0
    work_for_relation_table = np.random.rand(4, 4) * (1 - np.eye(4,4)*eye_cut)
    test_graphResultsForPhraseRelation[app_graph["work_for"]] = work_for_relation_table
    live_in_relation_table = np.random.rand(4, 4) * (1 - np.eye(4,4)*eye_cut)
    test_graphResultsForPhraseRelation[app_graph["live_in"]] = live_in_relation_table
    located_in_relation_table = np.random.rand(4, 4) * (1 - np.eye(4,4)*eye_cut)
    test_graphResultsForPhraseRelation[app_graph["located_in"]] = located_in_relation_table

    return test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation


def get_graph_result(emr_graph, emr_input):
    from regr.solver.gurobi_solver import GurobiSolver

    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input
    
    test_graphResultsForPhraseToken_ = {k.name: v for k, v in test_graphResultsForPhraseToken.items()}
    test_graphResultsForPhraseRelation_ = {k.name: v for k, v in test_graphResultsForPhraseRelation.items()}

    # ------Call solver -------
    test_graph = emr_graph

    myilpOntSolver = GurobiSolver(lazy_not=True, self_relation=False)
    tokenResult, relationsResult = myilpOntSolver.solve_legacy(test_phrase,
                                                               test_graphResultsForPhraseToken,
                                                               test_graphResultsForPhraseRelation
                                                              )
    return tokenResult, relationsResult


@pytest.fixture()
def graph_result(emr_graph, emr_input):
    return get_graph_result(emr_graph, emr_input)


def get_owl_result(emr_graph, emr_input):
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from regr.solver.ilpOntSolver import ilpOntSolver

    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input

    test_graphResultsForPhraseToken = {k.name: v for k, v in test_graphResultsForPhraseToken.items()}
    test_graphResultsForPhraseRelation = {k.name: v for k, v in test_graphResultsForPhraseRelation.items()}

    # ------Call solver -------
    test_graph = emr_graph

    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(test_graph)
    tokenResult, relationsResult, _ = myilpOntSolver.calculateILPSelection(test_phrase,
                                                                           test_graphResultsForPhraseToken,
                                                                           test_graphResultsForPhraseRelation
                                                                          )
    return tokenResult, relationsResult


@pytest.fixture()
def owl_result(emr_graph, emr_input):
    return get_owl_result(emr_graph, emr_input)


#@pytest.mark.skip
def test_compare_emr(emr_graph, emr_input, owl_result, graph_result):
    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input
    test_graphResultsForPhraseToken = {k.name: v for k, v in test_graphResultsForPhraseToken.items()}
    test_graphResultsForPhraseRelation = {k.name: v for k, v in test_graphResultsForPhraseRelation.items()}

    tokenResult, relationsResult = owl_result
    graph_tokenResult, graph_relationsResult = graph_result
    graph_tokenResult = {k.name: v for k, v in graph_tokenResult.items()}
    graph_relationsResult = {k.name: v for k, v in graph_relationsResult.items()}

    app_graph = emr_graph["application"]

    print('input', '-'*10)
    for k, v in test_graphResultsForPhraseToken.items():
        print(k)
        print(v)
    for k, v in test_graphResultsForPhraseRelation.items():
        print(k)
        print(v)
    print('owl', '-'*10)
    for k, v in tokenResult.items():
        print(k)
        print(v)
    for k, v in relationsResult.items():
        print(k)
        print(v)
    print('graph', '-'*10)
    for k, v in graph_tokenResult.items():
        print(k)
        print(v)
    for k, v in graph_relationsResult.items():
        print(k)
        print(v)

    # -------Compare objective
    obj = 0
    for key in set(chain(tokenResult, graph_tokenResult)):
        obj += (tokenResult[key] * test_graphResultsForPhraseToken[key]).sum()
        obj += ((1 - tokenResult[key]) * (1 - test_graphResultsForPhraseToken[key])).sum()
    for key in set(chain(relationsResult, graph_relationsResult)):
        obj += (relationsResult[key] * test_graphResultsForPhraseRelation[key]).sum()
        obj += ((1 - relationsResult[key]) * (1 - test_graphResultsForPhraseRelation[key])).sum()
    print('objective OWL:', obj)
    obj = 0
    for key in set(chain(tokenResult, graph_tokenResult)):
        obj += (graph_tokenResult[key] * test_graphResultsForPhraseToken[key]).sum()
        obj += ((1 - graph_tokenResult[key]) * (1 - test_graphResultsForPhraseToken[key])).sum()
    for key in set(chain(relationsResult, graph_relationsResult)):
        obj += (graph_relationsResult[key] * test_graphResultsForPhraseRelation[key] ).sum()
        obj += ((1 - graph_relationsResult[key]) * (1 - test_graphResultsForPhraseRelation[key])).sum()
    print('objective graph:', obj)

    # -------Evaluate results
    for key in set(chain(tokenResult, graph_tokenResult)):
        assert (tokenResult[key] == graph_tokenResult[key]).all()
    for key in set(chain(relationsResult, graph_relationsResult)):
        assert (relationsResult[key] == graph_relationsResult[key]).all()


def test_benchmark_graph_result(emr_graph, emr_input, benchmark):
    benchmark(get_graph_result, emr_graph, emr_input)


def test_benchmark_owl_result(emr_graph, emr_input, benchmark):
    benchmark(get_owl_result, emr_graph, emr_input)