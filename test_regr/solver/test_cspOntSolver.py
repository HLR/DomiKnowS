#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:21:25 2019

@author: amartinh
"""




import numpy as np
import owlready2 
from owlready2 import *
from ilpOntSolverFactory import ilpOntSolverFactory
from ilpOntSolver import ilpOntSolver
from graph import Graph

from constraint import *


test_phrase = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]
conceptNamesList = ["people", "organization", "other", "location", "O"]
relationNamesList = ["work_for", "live_in", "located_in"]

# tokens
test_graphResultsForPhraseToken = {}
#                                                           John  works for  IBM
test_graphResultsForPhraseToken["people"] =       np.array([0.7, 0.1, 0.02, 0.6])
test_graphResultsForPhraseToken["organization"] = np.array([0.5, 0.2, 0.03, 0.91])
test_graphResultsForPhraseToken["other"] =        np.array([0.3, 0.6, 0.05, 0.5])
test_graphResultsForPhraseToken["location"] =     np.array([0.3, 0.4, 0.1 , 0.3])
test_graphResultsForPhraseToken["O"] =            np.array([0.1, 0.9, 0.9 , 0.1])

test_graphResultsForPhraseRelation = dict()
# work_for
#                                    John  works for   IBM
work_for_relation_table = np.array([[0.40, 0.20, 0.20, 0.63],  # John
                                    [0.00, 0.00, 0.40, 0.30],  # works
                                    [0.02, 0.03, 0.05, 0.10],  # for
                                    [0.65, 0.20, 0.10, 0.30],  # IBM
                                    ])
test_graphResultsForPhraseRelation["work_for"] = work_for_relation_table

# live_in
#                                   John  works for   IBM
live_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                   [0.00, 0.00, 0.20, 0.10],  # works
                                   [0.02, 0.03, 0.05, 0.10],  # for
                                   [0.10, 0.20, 0.10, 0.00],  # IBM
                                   ])
test_graphResultsForPhraseRelation["live_in"] = live_in_relation_table

# located_in
#                                      John  works for   IBM
located_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06],  # John
                                      [0.00, 0.00, 0.00, 0.00],  # works
                                      [0.02, 0.03, 0.05, 0.10],  # for
                                      [0.03, 0.20, 0.10, 0.00],  # IBM
                                      ])
test_graphResultsForPhraseRelation["located_in"] = located_in_relation_table


# ------Call solver -------
test_graph = Graph(iri='http://ontology.ihmc.us/ML/EMR.owl', local='./examples/emr/')

            


       
conceptNames = list(test_graphResultsForPhraseToken)
#
tokens = None
if all(isinstance(item, tuple) for item in test_phrase):
    tokens = [x for x, _ in test_phrase]
elif all(isinstance(item, string_types) for item in test_phrase):
    tokens = test_phrase
#domains =[]
#variables = []




myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(test_graph)
tokenResult, relationResult,_ = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, None)



print(tokenResult)
print(relationResult)




