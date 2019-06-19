import numpy as np
import pandas as pd
from regr.graph import Graph, Concept
from regr.solver.ilpSelectClassification import ilpOntSolver
import pytest


@pytest.mark.gurobi
def test_main() :
    with Graph('global') as graph:
        graph.ontology='http://ontology.ihmc.us/ML/EMR.owl'
            
        with Graph('linguistic') as ling_graph:
            ling_graph.ontology='http://trips.ihmc.us/ont'
            phrase = Concept(name='phrase')
                
        with Graph('application') as app_graph:
            #app_graph.ontology='http://trips.ihmc.us/ont'
            app_graph.ontology='http://ontology.ihmc.us/ML/EMR.owl'
        
    test_graph = app_graph
        
    test_phrase = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]

    tokenList = ["John", "works", "for", "IBM"]
    conceptNamesList = ["people", "organization", "other", "location", "O"]
    relationNamesList = ["work_for", "live_in", "located_in"]
    
    #                         peop  org   other loc   O
    phrase_table = np.array([[0.70, 0.98, 0.95, 0.02, 0.00], # John
                             [0.00, 0.50, 0.40, 0.60, 0.90], # works
                             [0.02, 0.03, 0.05, 0.10, 0.90], # for
                             [0.92, 0.93, 0.93, 0.90, 0.00], # IBM
                            ])
    test_graphResultsForPhraseToken = pd.DataFrame(phrase_table, index=tokenList, columns=conceptNamesList)
    
    test_graphResultsForPhraseRelation = dict()
    
    # work_for
    #                                    John  works for   IBM
    work_for_relation_table = np.array([[0.50, 0.20, 0.20, 0.26], # John
                                        [0.00, 0.00, 0.40, 0.30], # works
                                        [0.02, 0.03, 0.05, 0.10], # for
                                        [0.63, 0.20, 0.10, 0.90], # IBM
                                       ])
    work_for_current_graphResultsForPhraseRelation = pd.DataFrame(work_for_relation_table, index=tokenList, columns=tokenList)
    test_graphResultsForPhraseRelation["work_for"] = work_for_current_graphResultsForPhraseRelation
    
    # live_in
    #                                   John  works for   IBM
    live_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06], # John
                                       [0.00, 0.00, 0.20, 0.10], # works
                                       [0.02, 0.03, 0.05, 0.10], # for
                                       [0.10, 0.20, 0.10, 0.00], # IBM
                                       ])
    live_in_current_graphResultsForPhraseRelation = pd.DataFrame(live_in_relation_table, index=tokenList, columns=tokenList)
    test_graphResultsForPhraseRelation["live_in"] = live_in_current_graphResultsForPhraseRelation
        
    # located_in
    #                                      John  works for   IBM
    located_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06], # John
                                          [0.00, 0.00, 0.00, 0.00], # works
                                          [0.02, 0.03, 0.05, 0.10], # for
                                          [0.03, 0.20, 0.10, 0.00], # IBM
                                         ])
    located_in_current_graphResultsForPhraseRelation = pd.DataFrame(located_in_relation_table, index=tokenList, columns=tokenList)
    test_graphResultsForPhraseRelation["located_in"] = located_in_current_graphResultsForPhraseRelation
        
    # ------Call solver -------
    
    myilpOntSolver = ilpOntSolver.getInstance(test_graph, ontologyPathname="./examples/emr/")
    tokenResult, relationsResult = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation)

    myilpOntSolver2 = ilpOntSolver.getInstance(ling_graph, ontologyPathname="./examples/emr/")
    tokenResult, relationsResult = myilpOntSolver2.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation)
    
    print("\nResults - ")
    print(tokenResult)
    
    if relationsResult != None :
        for name, result in relationsResult.items():
            print("\n")
            print(name)
            print(result)