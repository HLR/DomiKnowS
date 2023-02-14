import pytest

# Testing ILP solver based on constrains specified using logical constrains (not in ontology)
@pytest.fixture()
def emr_input(request):
    import numpy as np
    from domiknows.graph import Graph, Concept, andL, nandL, notL, ifL, existsL

    with Graph('global') as graph:
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
            #people.is_a(entity)
            ifL(people, entity)
            #organization.is_a(entity)
            ifL(organization, entity)
            #location.is_a(entity)
            ifL(location, entity)
            #other.is_a(entity)
            ifL(other, entity)
            #o.is_a(entity) 
            ifL(o, entity)
            
            #people.not_a(organization)
            nandL(people, organization)
            #people.not_a(location)
            nandL(people,location)
            #people.not_a(other)
            nandL(people, other)
            #people.not_a(o)
            nandL(people, o)
            
            #organization.not_a(people)
            nandL(organization, people)
            #organization.not_a(location)
            nandL(organization, location)
            #organization.not_a(other)
            nandL(organization, other)
            #organization.not_a(o)
            nandL(organization, o)
            
            #location.not_a(people)
            nandL(location, people)
            #location.not_a(organization)
            nandL(location, organization)
            #location.not_a(other)
            nandL(location, other)
            #location.not_a(o)
            nandL(location, o)
            
            #other.not_a(people)
            nandL(other, people)
            #other.not_a(organization)
            nandL(other, organization)
            #other.not_a(location)
            nandL(other, location)
            #other.not_a(o)
            nandL(other, o)
            
            #o.not_a(people)
            nandL(o, people)
            #o.not_a(organization)
            nandL(o, organization)
            #o.not_a(location)
            nandL(o, location)
            #o.not_a(other)
            nandL(o, other)

            #existsL(people, ('p',))
            #notL(other, ('o',))
            #notL(existsL(location, ('l',)), ('e', ))
            #existsL(notL(people))
            
            work_for = Concept(name='work_for')
            work_for.is_a(pair)
            #work_for.has_a(people, organization)
            ifL(work_for, ('x','y'), people, ('x',))
            ifL(work_for, ('x','y'), organization, ('y',))

            located_in = Concept(name='located_in')
            located_in.is_a(pair)
            #located_in.has_a(location, location)
            ifL(located_in, ('x','y'), location, ('x',))
            ifL(located_in, ('x','y'), location, ('y',))

            live_in = Concept(name='live_in')
            live_in.is_a(pair)
            #live_in.has_a(people, location)
            ifL(live_in, ('x','y'), people, ('x',))
            ifL(live_in, ('x','y'), location, ('y',))
            
    test_phrase = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]
    conceptNamesList = ['people', 'organization', 'other', 'location', 'O']
    relationNamesList = ['work_for', 'live_in', 'located_in']

    # tokens
    test_graphResultsForPhraseToken = {}
    #                                                           John  works for  IBM
    test_graphResultsForPhraseToken['people'] =       np.array([0.7, 0.1, 0.02, 0.6])
    test_graphResultsForPhraseToken['organization'] = np.array([0.5, 0.2, 0.03, 0.91])
    test_graphResultsForPhraseToken['other'] =        np.array([0.3, 0.6, 0.05, 0.5])
    test_graphResultsForPhraseToken['location'] =     np.array([0.3, 0.4, 0.1 , 0.3])
    test_graphResultsForPhraseToken['O'] =            np.array([0.1, 0.9, 0.9 , 0.1])
    
    test_graphResultsForPhraseRelation = dict()
    # work_for
    #                                    John  works for   IBM
    work_for_relation_table = np.array([[0.40, 0.20, 0.20, 0.63],  # John
                                        [0.00, 0.00, 0.40, 0.30],  # works
                                        [0.02, 0.03, 0.05, 0.10],  # for
                                        [0.65, 0.20, 0.10, 0.30],  # IBM
                                        ])
    test_graphResultsForPhraseRelation['work_for'] = work_for_relation_table

    # live_in
    #                                   John  works for   IBM
    live_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                       [0.00, 0.00, 0.20, 0.10],  # works
                                       [0.02, 0.03, 0.05, 0.10],  # for
                                       [0.10, 0.20, 0.10, 0.00],  # IBM
                                       ])
    test_graphResultsForPhraseRelation['live_in'] = live_in_relation_table

    # located_in
    #                                      John  works for   IBM
    located_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                          [0.00, 0.00, 0.00, 0.00],  # works
                                          [0.02, 0.03, 0.05, 0.10],  # for
                                          [0.03, 0.20, 0.10, 0.00],  # IBM
                                          ])
    test_graphResultsForPhraseRelation['located_in'] = located_in_relation_table

    yield app_graph, test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation

@pytest.mark.skip(reason="define model building datanode")
@pytest.mark.gurobi
def test_main_emr_owl(emr_input):
    from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from domiknows.solver.ilpOntSolver import ilpOntSolver

    app_graph, test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input

    # ------Call solver -------
    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(app_graph)
    tokenResult, relationsResult, _ = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, None)

    # -------Evaluate results
    # token results:
    # expect John[people] works[o] for[o] IBM[organization]
    assert tokenResult['people'].sum() == 1
    assert tokenResult['organization'].sum() == 1
    assert tokenResult['other'].sum() == 0
    assert tokenResult['location'].sum() == 0
    assert tokenResult['O'].sum() == 2

    assert tokenResult['people'][0] == 1  # John
    assert tokenResult['O'][1] == 1  # works
    assert tokenResult['O'][2] == 1  # for
    assert tokenResult['organization'][3] == 1  # IBM

    # relation results:
    # (John, IBM)[work_for]
    assert relationsResult['work_for'].sum().sum() == 1
    assert relationsResult['work_for'][0][3] == 1 # John - IBM
    assert relationsResult['live_in'].sum().sum() == 0
    assert relationsResult['located_in'].sum().sum() == 0
