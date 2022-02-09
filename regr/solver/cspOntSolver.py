from six import string_types

# numpy
import numpy as np

# ontology
from owlready2 import *

# Gurobi
from gurobipy import *
#from examples.emr.emr.graph import phrase

from constraint import *
#contraint problem

if __package__ is None or __package__ == '': 
    from ilpConfig import ilpConfig
    from ilpOntSolver import ilpOntSolver
 
    from gurobiILPBooleanMethods import gurobiILPBooleanProcessor
else:
    from ilpConfig import ilpConfig 
    from ilpOntSolver import ilpOntSolver

    from gurobiILPBooleanMethods import gurobiILPBooleanProcessor
    
import logging
import datetime

class CSPOntSolver(ilpOntSolver):
    
    def __init__(self) -> None:
        super().__init__()
        self.myIlpBooleanProcessor = gurobiILPBooleanProcessor()
               
    def addTokenConstrains(self, tokens, conceptNames,graphResultsForPhraseToken, graphResultsForPhraseRelation):
        
        problem = Problem()
        if graphResultsForPhraseToken is None:
            return None
        variables = []
        notVariables = []
    
        variablesDictionary = {}
        # Create variables and constraints for token - concept and negative variables
        for tokenIndex, token in enumerate(tokens): 
        
            for conceptName in conceptNames: 
                #x[token, conceptName]= "x_%s_is_%s"%(token, conceptName)
                variable = "%s_is_%s"%(token, conceptName)
                notVariable = "%s_is_not_%s"%(token, conceptName)
                variables.append(variable)
                notVariables.append(notVariable)
                
                #get probabilities to maximize later
                variablesDictionary[variable] = graphResultsForPhraseToken[conceptName][tokenIndex]
                variablesDictionary[notVariable] = 1 - (graphResultsForPhraseToken[conceptName][tokenIndex])
                
                problem.addVariable(variable , [0,1])
                problem.addVariable(notVariable, [0,1])
                problem.addConstraint(lambda a, b: a != b, (variable , notVariable))
        
        
        ######################################
        ######### find disjoint constraints (a nand b)
        ######################################
        disjointsDictionary = {}
        for i in range(len(conceptNames)):
             currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptNames[i]))   
             for d in currentConcept.disjoints():
                 
                 if d.entities[0]._name not in disjointsDictionary.keys():
                     disjointsDictionary[d.entities[0]._name] = [d.entities[1]._name]
                 else:
                     if d.entities[1]._name not in disjointsDictionary[d.entities[0]._name]:
                         disjointsDictionary[d.entities[0]._name].append(d.entities[1]._name)

        
        
        ######################################
        ######### find equivalent constraints (a and b)
        ######################################
        equivalentToDictionary = {}
        for i in range(len(conceptNames)):
             currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptNames[i])) 
             equivalentToDictionary[conceptNames[i]] = currentConcept.equivalent_to
        
        
        
        ######################################
        ######### find subclassOfconstraints (a -> b)
        ######################################
        subClassOfDictionary = {}
        for i in range(len(conceptNames)):
             currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptNames[i])) 
             subClassOfDictionary[currentConcept.name] = []
             for ancestor in currentConcept.ancestors(include_self = False):
                 if str(ancestor) in conceptNames:
                     subClassOfDictionary[currentConcept.name].append(str(ancestor))
        
        
        ######################################
        ######### find intersection and(a ,b,c)
        ######################################
        intersectionDictionary  = {}
        for i in range(len(conceptNames)):
              currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptNames[i])) 
              #print("currentconcept:", currentConcept.name)
              #print("intersection:")
              for intersect in currentConcept.constructs(Prop = None):
                  pass
                  #print(type(intersect))
                  
             
        
           
        
        for i in range(len(variables)):
            for j in range(i+1,len(variables)):
                if variables[i].split('_')[0] == variables[j].split('_')[0]:
                    key = variables[i].split('_')[2] 
                    value = variables[j].split('_')[2]

                    #disjoint constraints
                    if  value in disjointsDictionary[key] :
                        problem.addConstraint(lambda a, b: not(a and b), (variables[i] , variables[j]))
                    
                    # equivalent to constraints
                    if  value in equivalentToDictionary[key] :
                        problem.addConstraint(lambda a, b: (a and b), (variables[i] , variables[j]))
            
        # solutions = problem.getSolutions()

        # allSums = []
        # for solution in solutions:
        #     sums = 0
            
        #     for i in solution.keys():
        #         sums += (solution[i]*variablesDictionary[i])
        #     allSums.append(sums)
        # index = (allSums.index(max(allSums)))
        
        # print("\n")
        # print("Best Solution for Tokens: \n")
        # count = 0
        # for i, j in solutions[index].items():
        #     if count == len(solutions[index].keys())/2:
        #         break
        #     if j == 1:
        #         print(i)
        #     count += 1 
        # print("\n")
        
        #############################################################################################################################
        ################# 2 Relation Constrains
        #############################################################################################################################
        
        relationVariables = []
        relationNotVariables = []
        
        
        relationNames = list(graphResultsForPhraseRelation)
            
        # Create variables for relation - token - token and negative variables
        for relationName in relationNames:            
            for token1Index, token1 in enumerate(tokens): 
                for token2Index, token2 in enumerate(tokens):
                    if token1 == token2:
                        continue
    
                    relationVariable = "%s_%s_%s"%(token1, relationName, token2)
                    relationNotVariable = "%s_not_%s_%s"%(token1, relationName, token2)
                    
                    relationVariables.append(relationVariable)
                    relationNotVariables.append(relationNotVariable)
                    

                    variablesDictionary[relationVariable] = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                    variablesDictionary[relationNotVariable] = 1 - (graphResultsForPhraseRelation[relationName][token1Index][token2Index])
                    
                    problem.addVariable(relationVariable  , [0,1])
                    problem.addVariable(relationNotVariable, [0,1])
                    problem.addConstraint(lambda a, b: a != b, (relationVariable , relationNotVariable))
                    
        
        
        ######################################
        ######### Add constraints based on property domain and range statements in ontology - P(x,y) -> D(x), R(y)
        ######################################
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue
    
            self.myLogger.debug("Relation \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentRelation.name, relationName))
    
            currentRelationDomain = currentRelation.get_domain() # domains_indirect()
            currentRelationRange = currentRelation.get_range()
            
            
            
            for domain in currentRelationDomain:
                if domain._name not in conceptNames:
                    continue
                
                      
                for rang in currentRelationRange:
                    if rang.name not in conceptNames:
                        continue
                    
                    
                    for token in tokens:
                        for token1 in tokens:
                            if token == token1:
                                continue
                            
                           
                            variable1 = (token + "_is_" + domain._name)
                            variable2 =  token +"_" + currentRelation._name +"_" + token1
                        
                            problem.addConstraint(lambda a, b: a or (not b), (variable1 , variable2))
                            
                            variable3 = token1 + "_is_" + rang._name  
                            problem.addConstraint(lambda a, b: (a) or (not b) , (variable3 , variable2))
     
        
        #################
        ### Best Solution
        #################                    
                        
        solutions = problem.getSolutions()
        allSums = []
        
        
        for solution in solutions:
            sums = 0
            
            
            for i in solution.keys():
                sums += (solution[i]*variablesDictionary[i])
            allSums.append(sums)
        
        index = (allSums.index(max(allSums)))
        
        

        tokenResult = {}  
        for concept in conceptNames:
            tokenResult[concept] = np.zeros(len(tokens))
            

        relationResult = {}
        for relation in relationNames:
            relationResult[relation] = np.zeros((len(tokens) ,len(tokens)))
        
        count = 0
        for i, j in solutions[index].items():
            if j == 1:

                items = i.split('_')
               
                conceptName = items[2]
                tokenIndex = tokens.index(items[0])
                
                
                array = np.zeros(len(tokens))
                array[tokenIndex] = 1
                
                
                if count < len(tokens):
                    tokenResult[conceptName] = tokenResult[conceptName] + array
                
                
                
                count += 1
                if count == len(tokens) + 1:

                    items = i.split('_')
             
                    array = np.zeros((len(tokens) ,len(tokens)))
                
                    token1Index = tokens.index(items[0])
                    token2Index = tokens.index(items[3])
                    
                    array[token1Index][token2Index] =  1
                    key = items[1] + "_" + items[2]
                    relationResult[key] = array
                    
                    
                    break
         
        return tokenResult, relationResult

            
        
        #############################################################################################################################
        ################# 3 Relation Constrains
        #############################################################################################################################


        
    def calculateILPSelection(self, phrase, fun=None, epsilon = 0.00001, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None):

        if self.ilpSolver == None:
            self.myLogger.info('ILP solver not provided - returning unchanged results')
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation
        
        start = datetime.datetime.now()
        
        self.myLogger.info('Start for phrase %s'%(phrase))

        if graphResultsForPhraseToken is None:
            self.myLogger.info('graphResultsForPhraseToken is None - returning unchanged results')
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation
        
        conceptNames = list(graphResultsForPhraseToken)
        tokens = None
        if all(isinstance(item, tuple) for item in phrase):
            tokens = [x for x, _ in phrase]
        elif all(isinstance(item, string_types) for item in phrase):
            tokens = phrase
        else:
            self.myLogger.info('Phrase type is not supported %s - returning unchanged results'%(phrase))
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation
        
 

        tokenResult, relationResult = self.addTokenConstrains( tokens, conceptNames, graphResultsForPhraseToken, graphResultsForPhraseRelation)
        tripleRelationResult = {}
        
        end = datetime.datetime.now()
        elapsed = end - start
        self.myLogger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # return results of ILP optimization
        return tokenResult, relationResult, tripleRelationResult