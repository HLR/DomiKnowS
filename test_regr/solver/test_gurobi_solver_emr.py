import pytest

@pytest.fixture()
def emr_graph(request):
    from regr.graph import Graph, Concept, Relation

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global') as graph:
        graph.ontology = ('http://ontology.ihmc.us/ML/EMR.owl', './')

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
    #                                                                      John  works for   IBM
    test_graphResultsForPhraseToken[app_graph["people"]] =       np.array([0.70, 0.10, 0.02, 0.60])
    test_graphResultsForPhraseToken[app_graph["organization"]] = np.array([0.50, 0.20, 0.03, 0.91])
    test_graphResultsForPhraseToken[app_graph["other"]] =        np.array([0.30, 0.60, 0.05, 0.50])
    test_graphResultsForPhraseToken[app_graph["location"]] =     np.array([0.30, 0.40, 0.10, 0.30])
    test_graphResultsForPhraseToken[app_graph["O"]] =            np.array([0.10, 0.90, 0.90, 0.10])
    
    test_graphResultsForPhraseRelation = dict()
    # work_for
    #                                    John  works for   IBM
    work_for_relation_table = np.array([[0.40, 0.20, 0.20, 0.63],  # John
                                        [0.00, 0.00, 0.40, 0.30],  # works
                                        [0.02, 0.03, 0.05, 0.10],  # for
                                        [0.65, 0.20, 0.10, 0.30],  # IBM
                                        ])
    test_graphResultsForPhraseRelation[app_graph["work_for"]] = work_for_relation_table

    # live_in
    #                                   John  works for   IBM
    live_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                       [0.00, 0.00, 0.20, 0.10],  # works
                                       [0.02, 0.03, 0.05, 0.10],  # for
                                       [0.10, 0.20, 0.10, 0.00],  # IBM
                                       ])
    test_graphResultsForPhraseRelation[app_graph["live_in"]] = live_in_relation_table

    # located_in
    #                                      John  works for   IBM
    located_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                          [0.00, 0.00, 0.00, 0.00],  # works
                                          [0.02, 0.03, 0.05, 0.10],  # for
                                          [0.03, 0.20, 0.10, 0.00],  # IBM
                                          ])
    test_graphResultsForPhraseRelation[app_graph["located_in"]] = located_in_relation_table

    yield test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation


@pytest.mark.gurobi
def test_main_emr(emr_graph, emr_input):
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    import logging

    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input

    # ------Call solver -------
    test_graph = emr_graph
    # prepare solver
    ilpConfig = {
        'ilpSolver' : 'mini',
        'log_level' : logging.DEBUG,
        'log_filename' : 'ilpOntSolver.log',
        'log_filesize' : 5*1024*1024*1024,
        'log_backupCount' : 5,
        'log_fileMode' : 'a'
    }
    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(emr_graph, _ilpConfig=ilpConfig, lazy_not=True, self_relation=False)
    tokenResult, relationsResult = myilpOntSolver.solve_legacy(test_phrase,
                                                               test_graphResultsForPhraseToken,
                                                               test_graphResultsForPhraseRelation
                                                              )

    # -------Evaluate results
    # token results:
    # expect John[people] works[o] for[o] IBM[organization]
    app_graph = emr_graph["application"]

    assert tokenResult[app_graph['people']].sum() == 1
    assert tokenResult[app_graph['organization']].sum() == 1
    assert tokenResult[app_graph['other']].sum() == 0
    assert tokenResult[app_graph['location']].sum() == 0
    assert tokenResult[app_graph['O']].sum() == 2

    assert tokenResult[app_graph['people']][0] == 1  # John
    assert tokenResult[app_graph['O']][1] == 1  # works
    assert tokenResult[app_graph['O']][2] == 1  # for
    assert tokenResult[app_graph['organization']][3] == 1  # IBM

    # relation results:
    # (John, IBM)[work_for]
    assert relationsResult[app_graph['work_for']].sum().sum() == 1
    assert relationsResult[app_graph['work_for']][0][3] == 1 # John - IBM
    assert relationsResult[app_graph['live_in']].sum().sum() == 0
    assert relationsResult[app_graph['located_in']].sum().sum() == 0
