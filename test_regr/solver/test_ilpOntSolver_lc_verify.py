import pytest

# Testing ILP solver verify method based on constrains specified using logical constrains (not in ontology)
def emr_graph():
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
            
            people.is_a(entity)
            organization.is_a(entity)
            location.is_a(entity)
            other.is_a(entity)
            o.is_a(entity) 
            
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

            work_for = Concept(name='work_for')
            work_for.is_a(pair)
            work_for.has_a(people, organization)
           
            located_in = Concept(name='located_in')
            located_in.is_a(pair)
            located_in.has_a(location, location)
            
            live_in = Concept(name='live_in')
            live_in.is_a(pair)
            live_in.has_a(people, location)
            
    return app_graph

@pytest.fixture()
def emr_input(request):
    import numpy as np

    test_phrase = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]
    conceptNamesList = ['people', 'organization', 'other', 'location', 'O']
    relationNamesList = ['work_for', 'live_in', 'located_in']

    # tokens
    test_graphResultsForPhraseToken = {}
    #                                                         John  works for  IBM
    test_graphResultsForPhraseToken['people'] =       np.array([1,   0,    0,   0])
    test_graphResultsForPhraseToken['organization'] = np.array([0,   0,    0,   1])
    test_graphResultsForPhraseToken['other'] =        np.array([0,   0,    0,   0])
    test_graphResultsForPhraseToken['location'] =     np.array([0,   0,    0,   0])
    test_graphResultsForPhraseToken['O'] =            np.array([0,   1,    1,   0])
    
    test_graphResultsForPhraseRelation = dict()
    # work_for
    #                                   John  works for   IBM
    work_for_relation_table = np.array([[0,    0,    0,    1],  # John
                                        [0,    0,    0,    0],  # works
                                        [0,    0,    0,    0],  # for
                                        [0,    0,    0,    0],  # IBM
                                        ])
    test_graphResultsForPhraseRelation['work_for'] = work_for_relation_table

    # live_in
    #                                  John  works for   IBM
    live_in_relation_table = np.array([[0,    0,    0,    0],  # John
                                       [0,    0,    0,    0],  # works
                                       [0,    0,    0,    0],  # for
                                       [0,    0,    0,    0],  # IBM
                                       ])
    test_graphResultsForPhraseRelation['live_in'] = live_in_relation_table

    # located_in
    #                                     John  works for   IBM
    located_in_relation_table = np.array([[0,    0,    0,    0],  # John
                                          [0,    0,    0,    0],  # works
                                          [0,    0,    0,    0],  # for
                                          [0,    0,    0,    0],  # IBM
                                          ])
    test_graphResultsForPhraseRelation['located_in'] = located_in_relation_table

    yield test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation

@pytest.mark.skip(reason="define model building datanode")
@pytest.mark.gurobi
def test_main_emr_verify(emr_input):
    import numpy as np

    from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from domiknows.solver.ilpOntSolver import ilpOntSolver

    app_graph = emr_graph()

    myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(app_graph)

    test_phrase = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]
    conceptNamesList = ['people', 'organization', 'other', 'location', 'O']
    relationNamesList = ['work_for', 'live_in', 'located_in']
    
    # Call verify on explicite data -------
    test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation = emr_input
    
    verifyResult = myilpOntSolver.verifySelectionLC(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, None)
    
    assert verifyResult # - correct
    
    # Test corrected model
    test_graphResultsForPhraseToken = {}
    for c in conceptNamesList:
        test_graphResultsForPhraseToken[c] = np.zeros((len(test_phrase), ))
        
    test_graphResultsForPhraseRelation = {}
    for r in relationNamesList:
        test_graphResultsForPhraseRelation[r] = np.zeros((len(test_phrase), len(test_phrase)))
        
    test_graphResultsForPhraseToken['people'][0] = 1  # John
    test_graphResultsForPhraseToken['O'][1] = 1  # works
    test_graphResultsForPhraseToken['O'][2] = 1  # for
    test_graphResultsForPhraseToken['organization'][3] = 1  # IBM

    test_graphResultsForPhraseRelation['work_for'][0][3] = 1 # John work_for IBM

    # ------Call verify -------
    verifyResult = myilpOntSolver.verifySelectionLC(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, None)
    
    # -------Evaluate results
    assert verifyResult # - correct
    
    # Test uncorrected  model - conflict with nandL(people, organization)
    test_graphResultsForPhraseToken = {}
    for c in conceptNamesList:
        test_graphResultsForPhraseToken[c] = np.zeros((len(test_phrase), ))
        
    test_graphResultsForPhraseRelation = {}
    for r in relationNamesList:
        test_graphResultsForPhraseRelation[r] = np.zeros((len(test_phrase), len(test_phrase)))
        
    test_graphResultsForPhraseToken['people'][0] = 1  # John
    test_graphResultsForPhraseToken['organization'][0] = 1 # John - conflict
    test_graphResultsForPhraseToken['O'][1] = 1  # works
    test_graphResultsForPhraseToken['O'][2] = 1  # for
    test_graphResultsForPhraseToken['organization'][3] = 1  # IBM

    test_graphResultsForPhraseRelation['work_for'][0][3] = 1 # John work_for IBM 
    
     # ------Call verify -------
    verifyResult = myilpOntSolver.verifySelectionLC(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, None)
    
    # -------Evaluate results
    assert not verifyResult # - uncorrected
    
    # Test uncorrected  model - conflict with  ifL(work_for, ('x','y'), organization, ('y',))
    test_graphResultsForPhraseToken = {}
    for c in conceptNamesList:
        test_graphResultsForPhraseToken[c] = np.zeros((len(test_phrase), ))
        
    test_graphResultsForPhraseRelation = {}
    for r in relationNamesList:
        test_graphResultsForPhraseRelation[r] = np.zeros((len(test_phrase), len(test_phrase)))
        
    test_graphResultsForPhraseToken['people'][0] = 1  # John
    test_graphResultsForPhraseToken['O'][1] = 1  # works
    test_graphResultsForPhraseToken['O'][2] = 1  # for
    test_graphResultsForPhraseToken['people'][3] = 1  # IBM - conflict


    test_graphResultsForPhraseRelation['work_for'][0][3] = 1 # John work_for IBM 
    
     # ------Call verify -------
    verifyResult = myilpOntSolver.verifySelectionLC(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, None)
    
    # -------Evaluate results
    assert not verifyResult # - uncorrected
    