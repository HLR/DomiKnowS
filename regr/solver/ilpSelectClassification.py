# numpy
import numpy as np

# pandas
import pandas as pd

# ontology
from owlready2 import *

import os
from pkg_resources import resource_filename
from _operator import sub

# path to Meta Graph ontology
graphMetaOntologyPathname = resource_filename('regr', 'ontology/ML')

# path
from pathlib import Path

#
from typing import Dict
from torch import Tensor
import torch
import pandas as pd

if __package__ is None or __package__ == '':
    from regr.graph.concept import Concept, enum
    from regr.graph.graph import Graph
    from regr.utils import printablesize, get_prop_result

    from regr.sensor.allennlp.sensor import SentenceEmbedderSensor
    from regr.graph.allennlp import *
    
    from regr.solver.ilpConfig import ilpConfig 
    from regr.solver.ilpBooleanMethods import *
else:
    from ..graph.concept import Concept, enum
    from ..graph.graph import Graph
    from ..utils import printablesize, get_prop_result
    
    from ..sensor.allennlp.sensor import SentenceEmbedderSensor
    from ..graph.allennlp.base import *
    
    from .ilpConfig import ilpConfig 
    from .ilpBooleanMethods import *

import logging
import datetime

class ilpOntSolver:
    __instances = {}
    
    __negVarTrashhold = 1.0
    
    myLogger = None

    ilpSolver = None
    myIlpBooleanProcessor = None
    
    def __init__(self) -> None:
        super().__init__()
               
    @staticmethod
    def getInstance(graph, _iplConfig = ilpConfig):
        if (graph is not None) and (graph.ontology is not None):
            if graph.ontology.iri not in ilpOntSolver.__instances:
                ilpOntSolver.__instances[graph.ontology.iri] = ilpOntSolver()
                ilpOntSolver.__instances[graph.ontology.iri].setup_solver_logger() 

                ilpOntSolver.__instances[graph.ontology.iri].myGraph = graph
                    
                ilpOntSolver.__instances[graph.ontology.iri].loadOntology(graph.ontology.iri, graph.ontology.local)
            
                if _iplConfig is not None:
                    ilpOntSolver.__instances[graph.ontology.iri].ilpSolver = _iplConfig['ilpSolver']
            
                ilpOntSolver.__instances[graph.ontology.iri].myIlpBooleanProcessor = ilpBooleanProcessor(_iplConfig)
                ilpOntSolver.__instances[graph.ontology.iri].myLogger.info("Returning existing new ilpOntSolver for %s"%(graph.ontology.iri))
            else:
                ilpOntSolver.__instances[graph.ontology.iri].myLogger.info("Returning existing ilpOntSolver for %s"%(graph.ontology.iri))
            
        return ilpOntSolver.__instances[graph.ontology.iri]
    
    def setup_solver_logger(self, log_filename='ilpOntSolver.log'):
        logger = logging.getLogger(__name__)
    
        # create file handler and set level to info
        ch = logging.FileHandler(log_filename)
        logger.setLevel(logging.DEBUG)
    
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(funcName)s - %(message)s')
    
        # add formatter to ch
        ch.setFormatter(formatter)
    
        # add ch to logger
        logger.addHandler(ch)
        
        print("Log file is in: ", ch.baseFilename)
        self.myLogger = logger
            
    def loadOntology(self, ontologyURL, ontologyPathname=None):
        start = datetime.datetime.now()
        self.myLogger.info('Start')
        
        currentPath = Path(os.path.normpath("./")).resolve()
        
        # Check if Graph Meta ontology path is correct
        graphMetaOntologyPath = Path(os.path.normpath(graphMetaOntologyPathname))
        graphMetaOntologyPath = graphMetaOntologyPath.resolve()
        if not os.path.isdir(graphMetaOntologyPath):
            self.myLogger.error("Path to load Graph ontology: %s does not exists in current directory %s"%(graphMetaOntologyPath,currentPath))
            exit()
            
        if ontologyPathname is not None:
            # Check if specific ontology path is correct
            ontologyPath = Path(os.path.normpath(ontologyPathname))
            ontologyPath = ontologyPath.resolve()
            if not os.path.isdir(ontologyPath):
                self.myLogger.error("Path to load ontology: %s does not exists in current directory %s"%(ontologyURL,currentPath))
                exit()

            onto_path.append(graphMetaOntologyPath)  # the folder with the Graph Meta ontology
            onto_path.append(ontologyPath) # the folder with the ontology for the specific  graph
    
        # Load specific ontology
        try :
            self.myOnto = get_ontology(ontologyURL)
            self.myOnto.load(only_local = True, fileobj = None, reload = False, reload_if_newer = False)
        except FileNotFoundError as e:
            self.myLogger.warning("Error when loading - %s from: %s"%(ontologyURL, ontologyPathname))
    
        end = datetime.datetime.now()
        elapsed = end - start
        self.myLogger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        return self.myOnto
    
    def addTokenConstrains(self, m, tokens, conceptNames, x, graphResultsForPhraseToken):
        if graphResultsForPhraseToken is None:
            return None
        
        # Create variables for token - concept and negative variables
        for tokenIndex, token in enumerate(tokens):            
            for conceptName in conceptNames: 
                if self.ilpSolver == "Gurobi":
                    x[token, conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName))
                elif self.ilpSolver == "GEKKO":
                    x[token, conceptName]=m.Var(0, lb=0, ub=1, integer=True, name="x_%s_%s"%(token, conceptName))
                
                #it = np.nditer(graphResultsForPhraseToken[conceptName], flags=['c_index', 'multi_index'])
                
                currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                if currentProbability < ilpOntSolver.__negVarTrashhold:
                    if self.ilpSolver == "Gurobi":
                        x[token, conceptName+'-neg']=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName+'-neg'))
                    elif self.ilpSolver == "GEKKO":
                        x[token, conceptName]=m.Var(0, lb=0, ub=1, integer=True, name="x_%s_%s"%(token, conceptName+'-neg'))
    
        # Add constraints forcing decision between variable and negative variables 
        for conceptName in conceptNames:
            for token in tokens:
                if (token, conceptName+'-neg') in x:
                    if self.ilpSolver == "Gurobi":
                        constrainName = 'c_%s_%sselfDisjoint'%(token, conceptName)
                        
                        m.addConstr(x[token, conceptName] + x[token, conceptName+'-neg'], GRB.LESS_EQUAL, 1, name=constrainName)
                    elif self.ilpSolver == "GEKKO":
                        m.Equation(x[token, conceptName] + x[token, conceptName+'-neg'] <= 1)
                    
        if self.ilpSolver == "Gurobi":       
            m.update()
        elif self.ilpSolver == "GEKKO":
            pass
         
        self.myLogger.info("Created %i ilp variables for tokens"%(len(x)))
        
        # -- Add constraints based on concept disjoint statements in ontology - not(and(var1, var2)) = nand(var1, var2)
        foundDisjoint = dict() # too eliminate duplicates
        for conceptName in conceptNames:
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
            
            if currentConcept is None :
                continue
            
            self.myLogger.debug("Concept \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentConcept.name, conceptName))
                
            for d in currentConcept.disjoints():
                disjointConcept = d.entities[1]._name
                    
                if currentConcept._name == disjointConcept:
                    disjointConcept = d.entities[0]._name
                        
                    if currentConcept._name == disjointConcept:
                        continue
                        
                if disjointConcept not in conceptNames:
                     continue
                        
                if conceptName in foundDisjoint:
                    if disjointConcept in foundDisjoint[conceptName]:
                        continue
                
                if disjointConcept in foundDisjoint:
                    if conceptName in foundDisjoint[disjointConcept]:
                        continue
                            
                for token in tokens:
                    if self.ilpSolver == "Gurobi":       
                        constrainName = 'c_%s_%s_Disjoint_%s'%(token, conceptName, disjointConcept)
                        
                        #m.addConstr(self.myIlpBooleanProcessor.nandVar(m, x[token, conceptName], x[token, disjointConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)

                        # short version ensuring that logical expression is SATISFY - no generating variable holding the result of evaluating the expression
                        m.addConstr(x[token, conceptName] + x[token, disjointConcept], GRB.LESS_EQUAL, 1, name=constrainName)                 
                       
                    elif self.ilpSolver == "GEKKO":
                        
                        #m.Equation(self.myIlpBooleanProcessor.nandVar(m, x[token, conceptName], x[token, disjointConcept]) >= 1)

                        # short version ensuring that logical expression is SATISFY - no generating variable holding the result of evaluating the expression
                        m.Equation(x[token, conceptName] + x[token, disjointConcept] <= 1) 
                               
                if not (conceptName in foundDisjoint):
                    foundDisjoint[conceptName] = {disjointConcept}
                else:
                    foundDisjoint[conceptName].add(disjointConcept)
                           
            if conceptName in foundDisjoint:
                self.myLogger.info("Created - disjoint - constrains between concept \"%s\" and concepts %s"%(conceptName,foundDisjoint[conceptName]))

        # -- Add constraints based on concept equivalent statements in ontology - and(var1, av2)
        foundEquivalent = dict() # too eliminate duplicates
        for conceptName in conceptNames:
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for equivalentConcept in currentConcept.equivalent_to:
                if equivalentConcept.name not in conceptNames:
                     continue
                        
                if conceptName in foundEquivalent:
                    if equivalentConcept.name in foundEquivalent[conceptName]:
                        continue
                
                if equivalentConcept.name in foundEquivalent:
                    if conceptName in foundEquivalent[equivalentConcept.name]:
                        continue
                            
                for token in tokens:
                    if self.ilpSolver == "Gurobi":       
                        constrainName = 'c_%s_%s_Equivalent_%s'%(token, conceptName, equivalentConcept.name)
                        
                        m.addConstr(self.myIlpBooleanProcessor.andVar(m, x[token, conceptName], x[token, equivalentConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)
                    elif self.ilpSolver == "GEKKO":
                        m.Equation(self.myIlpBooleanProcessor.andVar(m, x[token, conceptName], x[token, equivalentConcept]) >= 1)

                if not (conceptName in foundEquivalent):
                    foundEquivalent[conceptName] = {equivalentConcept.name}
                else:
                    foundEquivalent[conceptName].add(equivalentConcept.name)
           
            if conceptName in foundEquivalent:
                self.myLogger.info("Created - equivalent - constrains between concept \"%s\" and concepts %s"%(conceptName,foundEquivalent[conceptName]))
    
        # -- Add constraints based on concept subClassOf statements in ontology - var1 -> var2
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for ancestorConcept in currentConcept.ancestors(include_self = False) :
                if ancestorConcept.name not in conceptNames :
                     continue
                            
                for token in tokens:
                    if self.ilpSolver == "Gurobi":       
                        constrainName = 'c_%s_%s_Ancestor_%s'%(token, currentConcept, ancestorConcept.name)
                        
                        m.addConstr(self.myIlpBooleanProcessor.ifVar(m, x[token, conceptName], x[token, ancestorConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)
                    elif self.ilpSolver == "GEKKO":
                        m.Equation(self.myIlpBooleanProcessor.ifVar(m, x[token, conceptName], x[token, ancestorConcept]) >= 1)

                self.myLogger.info("Created - subClassOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,ancestorConcept.name))

        # -- Add constraints based on concept intersection statements in ontology - and(var1, var2, var3, ..)
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is And :
                    
                    for token in tokens:
                        _varAnd = m.addVar(name="andVar_%s"%(constrainName))
        
                        andList = []
                    
                        for currentClass in conceptConstruct.Classes :
                            andList.append(x[token, currentClass.name])
    
                        andList.append(x[token, conceptName])
                        
                        if self.ilpSolver == "Gurobi":
                            constrainName = 'c_%s_%s_Intersection'%(token, conceptName)
    
                            m.addConstr(self.myIlpBooleanProcessor.andVar(m, andList), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif self.ilpSolver == "GEKKO":
                            m.Equation(self.myIlpBooleanProcessor.andVar(m, andList) >= 1)
        
        # -- Add constraints based on concept union statements in ontology -  or(var1, var2, var3, ..)
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is Or :
                    
                    for token in tokens:
                        _varOr = m.addVar(name="orVar_%s"%(constrainName))
        
                        orList = []
                    
                        for currentClass in conceptConstruct.Classes :
                            orList.append(x[token, currentClass.name])
    
                        orList.append(x[token, conceptName])
                        
                        if self.ilpSolver == "Gurobi": 
                            constrainName = 'c_%s_%s_Union'%(token, conceptName)
   
                            m.addConstr(self.myIlpBooleanProcessor.orVar(m, orList), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif self.ilpSolver == "GEKKO":
                            m.Equation(self.myIlpBooleanProcessor.orVar(m, orList) >= 1)
        
        # -- Add constraints based on concept objectComplementOf statements in ontology - xor(var1, var2)
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is Not :
                    
                    complementClass = conceptConstruct.Class

                    for token in tokens:       
                        if self.ilpSolver == "Gurobi":    
                            constrainName = 'c_%s_%s_ComplementOf_%s'%(token, conceptName, complementClass.name)
                            
                            m.addConstr(self.myIlpBooleanProcessor.xorVar(m, x[token, conceptName], x[token, complementClass.name]), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif self.ilpSolver == "GEKKO":
                            m.Equation(self.myIlpBooleanProcessor.xorVar(m, x[token, conceptName], x[token, complementClass.name]) >= 1)

                    self.myLogger.info("Created - objectComplementOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,complementClass.name))
                        
        # ---- No supported yet
    
            # -- Add constraints based on concept disjonitUnion statements in ontology -  Not supported by owlready2 yet
        
            # -- Add constraints based on concept oneOf statements in ontology - ?
    
        if self.ilpSolver == "Gurobi":  
            m.update()
        elif self.ilpSolver == "GEKKO":
            pass
        
        # Add objectives
        X_Q = None
        for tokenIndex, token in enumerate(tokens):
            for conceptName in conceptNames:
                if self.ilpSolver == "Gurobi":
                    X_Q += graphResultsForPhraseToken[conceptName][tokenIndex]*x[token, conceptName]
                elif self.ilpSolver == "GEKKO":
                    if X_Q is None:
                        X_Q = graphResultsForPhraseToken[conceptName][tokenIndex]*x[token, conceptName]
                    else:
                        X_Q += graphResultsForPhraseToken[conceptName][tokenIndex]*x[token, conceptName]

                if (token, conceptName+'-neg') in x: 
                    if self.ilpSolver == "Gurobi":
                        X_Q += (1-graphResultsForPhraseToken[conceptName][tokenIndex])*x[token, conceptName+'-neg']
                    elif self.ilpSolver == "GEKKO":
                        if X_Q is None:
                            X_Q = (1-graphResultsForPhraseToken[conceptName][tokenIndex])*x[token, conceptName+'-neg']
                        else:
                            X_Q += (1-graphResultsForPhraseToken[conceptName][tokenIndex])*x[token, conceptName+'-neg']

        return X_Q
     
    def addRelationsConstrains(self, m, tokens, conceptNames, x, y, graphResultsForPhraseRelation):
        if graphResultsForPhraseRelation is None:
            return None
        
        relationNames = graphResultsForPhraseRelation.keys()
            
        # Create variables for relation - token - token and negative variables
        for relationName in relationNames:            
            for token1Index, token1 in enumerate(tokens): 
                for token2Index, token2 in enumerate(tokens):
                    if token1 == token2:
                        continue
    
                    if self.ilpSolver == "Gurobi":
                        y[relationName, token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(relationName, token1, token2))
                    elif self.ilpSolver == "GEKKO":
                        y[relationName, token1, token2]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s_%s_%s"%(relationName, token1, token2))

                    if graphResultsForPhraseRelation[relationName][token2Index][token1Index] < ilpOntSolver.__negVarTrashhold:
                        if self.ilpSolver == "Gurobi":
                            y[relationName+'-neg', token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s-neg_%s_%s"%(relationName, token1, token2))
                        elif self.ilpSolver == "GEKKO":
                            y[relationName+'-neg', token1, token2]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s-neg_%s_%s"%(relationName, token1, token2))
                        
        # Add constraints forcing decision between variable and negative variables 
        for relationName in relationNames:
            for token in tokens: 
                for token1 in tokens:
                    if token == token1:
                        continue
                    
                    if (relationName+'-neg', token, token1) in y: 
                        if self.ilpSolver == "Gurobi":
                            constrainName = 'c_%s_%s_%sselfDisjoint'%(token, token1, relationName)
                            m.addConstr(y[relationName, token, token1] + y[relationName+'-neg', token, token1], GRB.LESS_EQUAL, 1, name=constrainName)
                        elif self.ilpSolver == "GEKKO":
                            m.Equation(y[relationName, token, token1] + y[relationName+'-neg', token, token1] <= 1)
                                
        if self.ilpSolver == "Gurobi":
            m.update()
        elif self.ilpSolver == "GEKKO":
            pass
    
        self.myLogger.info("Created %i ilp variables for relations"%(len(y)))

        # -- Add constraints based on property domain and range statements in ontology - P(x,y) -> D(x), R(y)
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
                        
                for range in currentRelationRange:
                    if range.name not in conceptNames:
                        continue
                            
                    for token in tokens:
                        for token1 in tokens:
                            if token == token1:
                                continue
                             
                            if self.ilpSolver == "Gurobi":
                                constrainNameDomain = 'c_domain_%s_%s_%s'%(currentRelation, token, token1)
                                constrainNameRange = 'c_range_%s_%s_%s'%(currentRelation, token, token1)
                                
                                #m.addConstr(self.myIlpBooleanProcessor.ifVar(m, y[currentRelation._name, token, token1], x[token, domain._name]), GRB.GREATER_EQUAL, 1, name=constrainNameDomain)
                                #m.addConstr(self.myIlpBooleanProcessor.ifVar(m, y[currentRelation._name, token, token1], x[token1, range._name]), GRB.GREATER_EQUAL, 1, name=constrainNameRange)
                                
                                # short version ensuring that logical expression is SATISFY - no generating variable holding the result of evaluating the expression
                                m.addConstr(x[token, domain._name] - y[currentRelation._name, token, token1], GRB.GREATER_EQUAL, 0, name=constrainNameDomain)
                                m.addConstr(x[token1, range._name] - y[currentRelation._name, token, token1], GRB.GREATER_EQUAL, 0, name=constrainNameRange)
                            elif self.ilpSolver == "GEKKO":
                                #m.Equation(self.myIlpBooleanProcessor.ifVar(m, y[currentRelation._name, token, token1], x[token, domain._name]) >= 1)
                                #m.Equation(self.myIlpBooleanProcessor.ifVar(m, y[currentRelation._name, token, token1], x[token, range._name]) >= 1)
                                
                                # short version ensuring that logical expression is SATISFY - no generating variable holding the result of evaluating the expression
                                m.Equation(x[token, domain._name] - y[currentRelation._name, token, token1] >= 0)
                                m.Equation(x[token1, range._name] - y[currentRelation._name, token, token1] >= 0)
                                
                    self.myLogger.info("Created - domain-range - constrains for relation \"%s\" for domain \"%s\" and range \"%s\""%(relationName,domain._name,range._name))

          # -- Add constraints based on property subProperty statements in ontology R subproperty of S - R(x, y) -> S(x, y)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            for superProperty in currentRelation.is_a:
                if superProperty.name not in graphResultsForPhraseRelation:
                    continue
                
                for token in tokens: 
                    for token1 in tokens:
                        if token == token1:
                            continue
                        
                        if self.ilpSolver == "Gurobi":
                            constrainName = 'c_%s_%s_%s_SuperProperty_%s'%(token, token1, relationName, superProperty.name)
                            
                            #m.addConstr(self.myIlpBooleanProcessor.ifVar(m, y[relationName, token, token1], y[superProperty.name, token, token1]), GRB.GREATER_EQUAL, 1, name=constrainName)
                            m.addConstr(y[superProperty.name, token, token1] - y[relationName, token, token1], GRB.GREATER_EQUAL, 0, name=constrainName)
                        elif self.ilpSolver == "GEKKO":
                            #m.Equation(self.myIlpBooleanProcessor.ifVar(m, y[relationName, token, token1], y[superProperty.name, token, token1]) >= 1)
                            m.Equation(y[superProperty.name, token, token1] - y[relationName, token, token1] >= 0)
            
        # -- Add constraints based on property equivalentProperty statements in ontology -  and(R, S)
        foundEquivalent = dict() # too eliminate duplicates
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue
                
            for equivalentProperty in currentRelation.equivalent_to:
                if equivalentProperty.name not in graphResultsForPhraseRelation:
                     continue
                        
                if relationName in foundEquivalent:
                    if equivalentProperty.name in foundEquivalent[relationName]:
                        continue
                
                if equivalentProperty.name in foundEquivalent:
                    if relationName in foundEquivalent[equivalentProperty.name]:
                        continue
                            
                for token in tokens: 
                    for token1 in tokens:
                        if token == token1:
                            continue
                        
                        if self.ilpSolver == "Gurobi":
                            constrainName = 'c_%s_%s_%s_EquivalentProperty_%s'%(token, token1, relationName, equivalentProperty.name)
                            
                            m.addConstr(self.myIlpBooleanProcessor.andVar(m, y[relationName, token, token1], y[equivalentProperty.name, token, token1]), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif self.ilpSolver == "GEKKO":
                            m.Equation(self.myIlpBooleanProcessor.andVar(m, y[relationName, token, token1], y[equivalentProperty.name, token, token1]) >= 1)
                            
                    if not (relationName in foundEquivalent):
                        foundEquivalent[relationName] = {equivalentProperty.name}
                    else:
                        foundEquivalent[relationName].add(equivalentProperty.name)
        
        # -- Add constraints based on property inverseProperty statements in ontology - S(x,y) -> R(y,x)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            currentRelationInverse = currentRelation.get_inverse_property()
            
            if not currentRelationInverse:
                continue
            
            if currentRelationInverse.name not in graphResultsForPhraseRelation:
                continue
                 
            if currentRelationInverse is not None:
                for token in tokens: 
                    for token1 in tokens:
                        if token == token1:
                            continue
                        
                        if self.ilpSolver == "Gurobi":
                            constrainName = 'c_%s_%s_%s_InverseProperty'%(token, token1, relationName)
                            
                            m.addGenConstrIndicator(y[relationName, token, token1], True, y[currentRelationInverse.name, token1, token], GRB.EQUAL, 1)
                            
                            m.addConstr(self.myIlpBooleanProcessor.ifVar(m, y[equivalentProperty.name, token, token1], y[relationName, token, token1]), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif self.ilpSolver == "GEKKO":
                            m.Equation(self.myIlpBooleanProcessor.ifVar(m, y[equivalentProperty.name, token, token1], y[relationName, token, token1]) >= 1)
                            
        # -- Add constraints based on property functionalProperty statements in ontology - at most one P(x,y) for x
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if self.ilpSolver == "Gurobi":
                functionalLinExpr =  LinExpr()
            elif self.ilpSolver == "GEKKO":
                functionalLinExpr = None
    
            if FunctionalProperty in currentRelation.is_a:
                for token in tokens: 
                    for token1 in tokens:
                        if token == token1:
                            continue
                        
                        functionalLinExpr += y[relationName, token, token1]
                
                constrainName = 'c_%s_%s_%s_FunctionalProperty'%(token, token1, relationName)
                m.addConstr(newLinExpr, GRB.LESS_EQUAL, 1, name=constrainName)
        
        # -- Add constraints based on property inverseFunctionaProperty statements in ontology - at most one P(x,y) for y
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if InverseFunctionalProperty in currentRelation.is_a:
                for token in tokens: 
                    for token1 in tokens:
                        if token == token1:
                            continue
                        
                        functionalLinExpr += y[relationName, token1, token]
    
                constrainName = 'c_%s_%s_%s_InverseFunctionalProperty'%(token, token1, relationName)
                m.addConstr(newLinExpr, GRB.LESS_EQUAL, 1, name=constrainName)
        
        # -- Add constraints based on property reflexiveProperty statements in ontology - P(x,x)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if ReflexiveProperty in currentRelation.is_a:
                for token in tokens: 
                    constrainName = 'c_%s_%s_ReflexiveProperty'%(token, relationName)
                    m.addConstr(y[relationName, token, token], GRB.EQUAL, 1, name=constrainName)  
                        
        # -- Add constraints based on property irreflexiveProperty statements in ontology - not P(x,x)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if IrreflexiveProperty in currentRelation.is_a:
                for token in tokens: 
                    constrainName = 'c_%s_%s_ReflexiveProperty'%(token, relationName)
                    m.addConstr(y[relationName, token, token], GRB.EQUAL, 0, name=constrainName)  
                    
        # -- Add constraints based on property symetricProperty statements in ontology - R(x, y) -> R(y,x)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if SymmetricProperty in currentRelation.is_a:
                for token in tokens: 
                    for token1 in tokens:
                        if token == token1:
                            continue
                        
                        constrainName = 'c_%s_%s_%s_SymmetricProperty'%(token, token1, relationName)
                        m.addGenConstrIndicator(y[relationName, token, token1], True, y[relationName, token1, token], GRB.EQUAL, 1)
        
        # -- Add constraints based on property asymetricProperty statements in ontology - not R(x, y) -> R(y,x)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if AsymmetricProperty in currentRelation.is_a:
                for token in tokens: 
                    for token1 in tokens:
                        if token == token1:
                            continue
                        
                        constrainName = 'c_%s_%s_%s_AsymmetricProperty'%(token, token1, relationName)
                        m.addGenConstrIndicator(y[relationName, token, token1], True, y[relationName, token1, token], GRB.EQUAL, 0)  
                        
        # -- Add constraints based on property transitiveProperty statements in ontology - P(x,y) and P(y,z) - > P(x,z)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if TransitiveProperty in currentRelation.is_a:
                for token in tokens: 
                    for token1 in tokens:
                        if token == token1:
                            continue
                        
                        constrainName = 'c_%s_%s_%s_TransitiveProperty'%(token, token1, relationName)
                        #m.addGenConstrIndicator(y[relationName, token, token1], True, y[relationName, token1, token], GRB.EQUAL, 1)  
                               
        # -- Add constraints based on property allValueFrom statements in ontology
    
        # -- Add constraints based on property hasValueFrom statements in ontology
    
        # -- Add constraints based on property objectHasSelf statements in ontology
        
        # -- Add constraints based on property disjointProperty statements in ontology
    
        # -- Add constraints based on property key statements in ontology
    
    
        # -- Add constraints based on property exactCardinality statements in ontology
    
        # -- Add constraints based on property minCardinality statements in ontology
        
        # -- Add constraints based on property maxCardinality statements in ontology
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
    
        # ---- Related to DataType properties - not sure yet if we need to support them
        
            # -- Add constraints based on property dataSomeValuesFrom statements in ontology
        
            # -- Add constraints based on property dataHasValue statements in ontology
    
            # -- Add constraints based on property dataAllValuesFrom statements in ontology
    
        if self.ilpSolver == "Gurobi":
            m.update()
        elif self.ilpSolver == "GEKKO":
            pass
    
        # Add objectives
        Y_Q  = None
        for relationName in relationNames:
            for token1 in  enumerate(tokens):
                for token2 in  enumerate(tokens):
                    if token1 == token2 :
                        continue
    
                    if self.ilpSolver == "Gurobi":
                        Y_Q += graphResultsForPhraseRelation[relationName][token2Index][token1Index]*y[relationName, token1, token2]
                    elif self.ilpSolver == "GEKKO":
                        if Y_Q is None:
                            Y_Q = graphResultsForPhraseRelation[relationName][token2Index][token1Index]*y[relationName, token1, token2]
                        else:
                            Y_Q += graphResultsForPhraseRelation[relationName][token2Index][token1Index]*y[relationName, token1, token2]

        
                    if (relationName+'-neg', token, token1) in y: 
                        if self.ilpSolver == "Gurobi":
                            Y_Q += (1-graphResultsForPhraseRelation[relationName][token2Index][token1Index])*y[relationName+'-neg', token1, token2]
                        elif self.ilpSolver == "GEKKO":
                            if Y_Q is None:
                                Y_Q = (1-graphResultsForPhraseRelation[relationName][token2Index][token1Index])*y[relationName+'-neg', token1, token2]
                            else:
                                Y_Q += (1-graphResultsForPhraseRelation[relationName][token2Index][token1Index])*y[relationName+'-neg', token1, token2]
        
        return Y_Q
    
    def addTripleRelationsConstrains(self, m, tokens, conceptNames, x, y, z, graphResultsForPhraseTripleRelation):
        if graphResultsForPhraseTripleRelation is None:
            return None
        
        tripleRelationNames = graphResultsForPhraseTripleRelation.keys()
            
        # Create variables for relation - token - token -token and negative variables
        for tripleRelationName in tripleRelationNames:            
            for tokenIndex, token in enumerate(tokens): 
                for token1Index, token1 in enumerate(tokens):
                    if token1 == token:
                        continue
                        
                    for token2Index, token2 in enumerate(tokens):
                        if token2 == token:
                            continue
                        
                        if token2 == token1:
                            continue
                    
                        if self.ilpSolver == "Gurobi":
                            z[tripleRelationName, token, token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s_%s"%(tripleRelationName, token, token1, token2))
                        elif self.ilpSolver == "GEKKO":
                            z[tripleRelationName, token, token1, token2]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s_%s_%s_%s"%(tripleRelationName, token, token1, token2))
    
                        if graphResultsForPhraseTripleRelation[tripleRelationName][token2Index][token1Index][tokenIndex] < ilpOntSolver.__negVarTrashhold:
                            if self.ilpSolver == "Gurobi":
                                z[tripleRelationName+'-neg', token, token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s-neg_%s_%s_%s"%(tripleRelationName, token, token1, token2))
                            elif self.ilpSolver == "GEKKO":
                                z[tripleRelationName+'-neg', token, token1, token2]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s-neg_%s_%s_%s"%(tripleRelationName, token, token1, token2))
                        
        # Add constraints forcing decision between variable and negative variables 
        for tripleRelationName in tripleRelationNames:
            for token in tokens: 
                for token1 in tokens:
                    if token1 == token:
                        continue
                        
                    for token2 in tokens:
                        if token2 == token:
                            continue
                        
                        if token2 == token1:
                            continue
                
                        if (tripleRelationName+'-neg', token, token1, token2) in z: 
                            if self.ilpSolver == "Gurobi":
                                constrainName = 'c_%s_%s_%s_%sselfDisjoint'%(token, token1, token2, tripleRelationName)
                                m.addConstr(z[tripleRelationName, token, token1, token2] + z[tripleRelationName+'-neg', token, token1, token2], GRB.LESS_EQUAL, 1, name=constrainName)
                        elif self.ilpSolver == "GEKKO":
                            m.Equation(z[tripleRelationName, token, token1, token2] + z[tripleRelationName+'-neg', token, token1, token2] <= 1)
                            
        if self.ilpSolver == "Gurobi":
            m.update()
        elif self.ilpSolver == "GEKKO":
            pass
    
        self.myLogger.info("Created %i ilp variables for relations"%(len(y)))

        # -- Add constraints 
        for tripleRelationName in graphResultsForPhraseTripleRelation:
            currentTripleRelation = self.myOnto.search_one(iri = "*%s"%(tripleRelationName))
                
            if currentTripleRelation is None:
                continue
    
            self.myLogger.debug("Relation \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentTripleRelation.name, currentTripleRelation))
            
            ancestorConcept = None
            for _ancestorConcept in currentTripleRelation.ancestors(include_self = False):
                if _ancestorConcept.name == "Thing":
                     continue
                 
                ancestorConcept = _ancestorConcept
                break
            
            if ancestorConcept is None:
                break
            
            tripleProperties = {}
            triplePropertiesRanges = {}    

            for property in self.myOnto.object_properties():
                _domain = property.domain
                
                if _domain is None:
                    break
                
                domain = _domain[0]._name
                if domain is ancestorConcept.name:                    
                    for superProperty in property.is_a:
                        if superProperty is None:
                            continue
                        
                        if superProperty.name == 'first':
                            tripleProperties['1'] = property
                            _range = property.range
                
                            if _range is None:
                                break
                
                            range = _range[0]._name
                            triplePropertiesRanges['1'] = range
                        elif superProperty.name == 'second':
                            tripleProperties['2'] = property
                            _range = property.range
                
                            if _range is None:
                                break
                
                            range = _range[0]._name
                            triplePropertiesRanges['2'] = range
                        elif superProperty.name == 'third':
                            tripleProperties['3'] = property
                            _range = property.range
                
                            if _range is None:
                                break
                
                            range = _range[0]._name
                            triplePropertiesRanges['3'] = range
                                        
            for toke1n in tokens:
                for token2 in tokens:
                    if token2 == token1:
                        continue
                        
                    for token3 in tokens:
                        if token3 == token1:
                            continue
                        
                        if token3 == token2:
                            continue
                     
                        if self.ilpSolver == "Gurobi":
                            constrainNameTriple = 'c_triple_%s_%s_%s_%s'%(tripleRelationName, token1, token2, token3)
                            
                            m.addConstr(x[token1, triplePropertiesRanges['1']] + x[token2, triplePropertiesRanges['2']] + x[token3, triplePropertiesRanges['3']] - z[tripleRelationName, token1, token2, token3], 
                                        GRB.GREATER_EQUAL, 0, name=constrainNameTriple)
                        elif self.ilpSolver == "GEKKO":
                            #m.Equation(x[token1, range._name] - y[currentRelation._name, token, token1] >= 0)
                            pass
                                    
                        self.myLogger.info("Created - triple - constrains for relation \"%s\" for tokens \"%s\", \"%s\", \"%s\""%(tripleRelationName,token1,token2,token3))
    
                
        if self.ilpSolver == "Gurobi":
            m.update()
        elif self.ilpSolver == "GEKKO":
            pass
    
        # Add objectives
        Z_Q  = None
        for tripleRelationName in tripleRelationNames:
            for token in tokens: 
                for token1 in tokens:
                    if token1 == token:
                        continue
                        
                        for token2 in tokens:
                            if token2 == token:
                                continue
                            
                            if token2 == token1:
                                continue
    
                        if self.ilpSolver == "Gurobi":
                            Z_Q += graphResultsForPhraseTripleRelation[tripleRelationName][token2][token1][token]*y[tripleRelationName, token, token1, token2]
                        elif self.ilpSolver == "GEKKO":
                            if Z_Q is None:
                                Z_Q = graphResultsForPhraseTripleRelation[tripleRelationName][token2][token1][token]*y[tripleRelationName, token, token1, token2]
                            else:
                                Z_Q += graphResultsForPhraseTripleRelation[tripleRelationName][token2][token1][token]*y[tripleRelationName, token, token1, token2]
    
            
                        if (tripleRelationName+'-neg', token, token1, token2) in z: 
                            if self.ilpSolver == "Gurobi":
                                Z_Q += (1-graphResultsForPhraseTripleRelation[tripleRelationName][token2][token1][token])*y[tripleRelationName+'-neg', token, token1, token2]
                            elif self.ilpSolver == "GEKKO":
                                if Z_Q is None:
                                    Z_Q = (1-graphResultsForPhraseTripleRelation[tripleRelationName][token2][token1][token])*y[tripleRelationName+'-neg', token, token1, token2]
                                else:
                                    z_Q += (1-graphResultsForPhraseTripleRelation[tripleRelationName][token2][token1][token])*y[tripleRelationName+'-neg', token, token1, token2]
                
        return Z_Q
        
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation):
    
        if self.ilpSolver == None:
            self.myLogger.info('ILP solver not provided - returning unchanged results')
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation
        
        start = datetime.datetime.now()
        self.myLogger.info('Start for phrase %s'%(phrase))
        self.myLogger.info('graphResultsForPhraseToken \n%s'%(graphResultsForPhraseToken))
        self.myLogger.info('graphResultsForPhraseTripleRelation \n%s'%(graphResultsForPhraseTripleRelation))

        if graphResultsForPhraseRelation is not None:
            for relation in graphResultsForPhraseRelation:
                self.myLogger.info('graphResultsForPhraseRelation for relation \"%s\" \n%s'%(relation, graphResultsForPhraseRelation[relation]))
        
        if graphResultsForPhraseTripleRelation is not None:
            for relation in graphResultsForPhraseTripleRelation:
                self.myLogger.info('graphResultsForPhraseTripleRelation for relation \"%s\" \n%s'%(relation, graphResultsForPhraseTripleRelation[relation]))

        conceptNames = graphResultsForPhraseToken.keys()
        tokens = [x for x, _ in phrase]
                    
        try:
            if self.ilpSolver == "Gurobi":
                # Create a new Gurobi model
                m = Model("decideOnClassificationResult")
                m.params.outputflag = 0
            elif self.ilpSolver == "GEKKO":
                m = GEKKO()
                solverDisp=False # remote solver used
                m.options.LINEAR = 1
                m.options.IMODE = 3 #steady state optimization
                #m.options.CSV_READ = 0
                #m.options.CSV_WRITE = 0 # no result files are written
                m.options.WEB  = 0
                gekkoTresholdForTruth = 0.5
            
            # Variables for concept - token
            x={}
    
            # Variables for relation - token, token
            y={}
            
            # Variables for relation - token, token, token
            z={}
                
            # -- Set objective
            Q = None
            
            X_Q = self.addTokenConstrains(m, tokens, conceptNames, x, graphResultsForPhraseToken)
            if X_Q is not None:
                if Q is None:
                    Q = X_Q
                else:
                    Q += X_Q
            
            Y_Q = self.addRelationsConstrains(m, tokens, conceptNames, x, y, graphResultsForPhraseRelation)
            if Y_Q is not None:
                Q += Y_Q
                
            Z_Q = self.addTripleRelationsConstrains(m, tokens, conceptNames, x, y, z, graphResultsForPhraseTripleRelation)
            if Z_Q is not None:
                Q += Z_Q
            
            if self.ilpSolver  == "Gurobi":
                m.setObjective(Q, GRB.MAXIMIZE)
            elif self.ilpSolver == "GEKKO":
                m.Obj((-1) * Q)
    
            # Token is associated with a single concept
            #for token in tokens:
            #   constrainName = 'c_%s'%(token)
            #    m.addConstr(quicksum(x[token, conceptName] for conceptName in conceptNames), GRB.LESS_EQUAL, 1, name=constrainName)
            
            if self.ilpSolver == "Gurobi":
                m.update()
            elif self.ilpSolver == "GEKKO":
                pass
            
            startOptimize = datetime.datetime.now()
            if self.ilpSolver == "Gurobi":
                self.myLogger.info('Optimizing model with %i variables and %i constrains'%(m.NumVars, m. NumConstrs))
            elif self.ilpSolver == "GEKKO":
                pass

            if self.ilpSolver == "Gurobi":
                m.optimize()
            elif self.ilpSolver == "GEKKO":
                m.solve(disp=solverDisp)
            
            endOptimize = datetime.datetime.now()
            elapsedOptimize = endOptimize - startOptimize

            if self.ilpSolver == "Gurobi":
                if m.status == GRB.Status.OPTIMAL:
                    self.myLogger.info('Optimal solution was found - elapsed time: %ims'%(elapsedOptimize.microseconds/1000))
                elif m.status == GRB.Status.INFEASIBLE:
                     self.myLogger.warning('Model was proven to be infeasible.')
                elif m.status == GRB.Status.INF_OR_UNBD:
                     self.myLogger.warning('Model was proven to be infeasible or unbound.')
                elif m.status == GRB.Status.UNBOUNDED:
                     self.myLogger.warning('Model was proven to be unbound.')
                else:
                     self.myLogger.warning('Optimal solution not was found - error code %i'%(m.status))
            elif self.ilpSolver == "GEKKO":
                self.myLogger.info('Optimal solution was found - elapsed time: %ims'%(elapsedOptimize.microseconds/1000))

            # Collect results for tokens
            tokenResult = None
            if graphResultsForPhraseToken is not None:
                tokenResult = pd.DataFrame(0, index=tokens, columns=conceptNames)
                if self.ilpSolver == "Gurobi":
                    if x or True:
                        if m.status == GRB.Status.OPTIMAL:

                            solution = m.getAttr('x', x)

                            for token in tokens :
                                for conceptName in conceptNames:
                                    if solution[token, conceptName] == 1:                                    
                                        tokenResult[conceptName][token] = 1
                                        ilpOntSolver.__logger.info('Solution \"%s\" is \"%s\"'%(token,conceptName))

                elif self.ilpSolver == "GEKKO":
                    for token in tokens :
                        for conceptName in conceptNames:
                            if x[token, conceptName].value[0] > gekkoTresholdForTruth:
                                tokenResult[conceptName][token] = 1
                                ilpOntSolver.__logger.info('Solution \"%s\" is \"%s\"'%(token,conceptName))

            # Collect results for relations
            relationsResult = None
            if graphResultsForPhraseRelation is not None: 
                relationsResult = {}
                relationNames = graphResultsForPhraseRelation.keys()
                if self.ilpSolver == "Gurobi":
                    if y or True:
                        if m.status == GRB.Status.OPTIMAL:
                            solution = m.getAttr('x', y)
                            
                            for relationName in relationNames:
                                relationResult = pd.DataFrame(0, index=tokens, columns=tokens)
                                
                                for token in tokens :
                                    for token1 in tokens:
                                        if token == token1:
                                            continue
                                        
                                        if solution[relationName, token, token1] == 1:
                                            relationResult[token1][token] = 1
                                            
                                            self.myLogger.info('Solution \"%s\" is in relation \"%s\" with \"%s\"'%(token1,relationName,token))
        
                                relationsResult[relationName] = relationResult
                elif self.ilpSolver == "GEKKO":
                    for relationName in relationNames:
                        relationResult = pd.DataFrame(0, index=tokens, columns=tokens)
                        
                        for token in tokens:
                            for token1 in tokens:
                                if token == token1:
                                    continue
                                
                                if y[relationName, token, token1].value[0] > gekkoTresholdForTruth:
                                    relationResult[token1][token] = 1
                                    
                                    self.myLogger.info('Solution \"%s\" is in relation \"%s\" with \"%s\"'%(token1,relationName,token))
    
                        relationsResult[relationName] = relationResult
                    
            # Collect results for triple relations
            tripleRelationsResult = None
            if graphResultsForPhraseTripleRelation is not None:
                tripleRelationsResult = {}
                tripleRelationNames = graphResultsForPhraseTripleRelation.keys()
                
                for tripleRelationName in tripleRelationNames:
                    tripleRelationsResult[tripleRelationName] = np.zeros((len(tokens), len(tokens), len(tokens)))
                
                if self.ilpSolver == "Gurobi":
                    if z or True:
                        if m.status == GRB.Status.OPTIMAL:
                            solution = m.getAttr('x', z)
                            
                            for tripleRelationName in tripleRelationNames:
                                
                                for tokenIndex, token in enumerate(tokens):
                                    for token1Index, token1 in enumerate(tokens):
                                        if token == token1:
                                            continue
                                        
                                        for token2Index, token2 in enumerate(tokens):
                                            if token2 == token:
                                                continue
                                            
                                            if token2 == token1:
                                                continue
                                        
                                            currentSolutionValue = solution[tripleRelationName, token, token1, token2]
                                            if solution[tripleRelationName, token, token1, token2] == 1:
                                                tripleRelationsResult[tripleRelationName][tokenIndex, token1Index, token2Index]= 1
                                            
                                            self.myLogger.info('Solution \"%s\" is in triple relation \"%s\" with \"%s\" and \"%s\"'%(token1,tripleRelationName,token, token2))
        
        except:
            self.myLogger.error('Error')
            raise
           
        end = datetime.datetime.now()
        elapsed = end - start
        self.myLogger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # return results of ILP optimization
        return tokenResult, relationsResult, tripleRelationsResult

    DataInstance = Dict[str, Tensor]
    
    def inferSelection(self, graph: Graph, data: DataInstance, vocab=None) -> DataInstance:
        # build concept (property) group by rank (# of has-a)
        prop_dict = defaultdict(list)

        for prop in graph.poi:
            concept = prop.sup
            prop_dict[len(concept.has_a()) or 1].append(prop)
        max_rank = max(prop_dict.keys())

        # find base, assume only one base for now
        # FIXME: that means we are only considering problems with only one sentence
        tokens_sensors = graph.get_sensors(SentenceEmbedderSensor)
        name, tokens_sensor = tokens_sensors[0]
        namespace = tokens_sensor.key

        mask = tokens_sensor.get_mask(data)
        mask_len = mask.sum(dim=1).clone().cpu().detach().numpy()  # (b, )

        # (b, l) # FIXME '_index' is tricky
        sentence = data[tokens_sensor.fullname + '_index'][namespace]
        batch_size, length = sentence.shape
        device = sentence.device

        values = [defaultdict(dict) for _ in range(batch_size)]
        for rank, props in prop_dict.items():
            for prop in props:
                #name = '{}_{}'.format(prop.sup.name, prop.name)
                name = prop.name  # current implementation has concept name in prop name
                # pred_logit - (b, l...*r, c) - for dim-c, [0] is neg, [1] is pos
                # mask - (b, l...*r)
                label, pred_logit, mask = get_prop_result(prop, data)
                # pred - (b, l...*r, c)
                pred = torch.nn.functional.softmax(pred_logit, dim=-1)
                pos_index = torch.tensor(1, device=device).long()
                # (b, l...*r)
                batched_value = pred.index_select(-1, pos_index).squeeze(dim=-1)
                # copy and detach, time consuming I/O
                batched_value = batched_value.clone().cpu().detach().numpy()

                for batch_index, batch_index_d in zip(range(batch_size), torch.arange(batch_size, dtype=torch.long, device=device)):
                    # (l...*r)
                    value = batched_value[batch_index_d]
                    # apply mask
                    # (l'...*r)
                    value = value[(slice(0, mask_len[batch_index]),) * rank]
                    values[batch_index][rank][name] = value
        #import pdb; pdb.set_trace()

        results = []
        # inference per instance
        for batch_index, batch_index_d in zip(range(batch_size), torch.arange(batch_size, dtype=torch.long, device=device)):
            # prepare tokens
            if vocab:
                tokens = ['{}_{}'.format(i, vocab.get_token_from_index(int(sentence[batch_index_d, i_d]),
                                                                       namespace=namespace))
                          for i, i_d in zip(range(mask_len[batch_index]), torch.arange(mask_len[batch_index], dtype=torch.long, device=device))]
            else:
                tokens = [str(i) for i in range(mask_len[batch_index])]
            # prepare tables
            table_list = []
            for rank in range(1, max_rank + 1):
                if rank in values[batch_index]:
                    table_list.append(values[batch_index][rank])
                else:
                    table_list.append(None)
            #import pdb; pdb.set_trace()
            # Do inference
            try:
                # following statement should be equivalent to
                # - EMR:
                # result_list = self.calculateILPSelection(tokens, concept_dict, relation_dict)
                # - SPRL:
                # result_list = self.calculateILPSelection(tokens, concept_dict, None, triplet_dict)
                result_table_list = self.calculateILPSelection(tokens, *table_list)

                #import pdb; pdb.set_trace()
                if all([result_table is None for result_table in result_table_list]):
                    raise RuntimeError('No result from solver. Check any log from the solver.')
            except:
                # whatever, raise it
                raise

            # collect result in batch
            result = defaultdict(dict)
            for rank, props in prop_dict.items():
                for prop in props:
                    name = prop.name
                    # (l'...*r)
                    # return structure is a pd?
                    # result_table_list started from 0
                    result[rank][name] = result_table_list[rank - 1][name]
            results.append(result)
        #import pdb; pdb.set_trace()

        # put results back
        for rank, props in prop_dict.items():
            for prop in props:
                name = prop.name
                instance_value_list = []
                for batch_index in range(batch_size):
                    # (l'...*r)
                    instance_value = results[batch_index][rank][name]
                    # (l...*r)
                    instance_value_pad = np.empty([length, ] * rank)
                    instance_value_pad[(slice(0, mask_len[batch_index]),) * rank] = instance_value
                    # (l...*r)
                    instance_value_d = torch.tensor(instance_value_pad, device=device)
                    instance_value_list.append(instance_value_d)
                # (b, l...*r)
                batched_value = torch.stack(instance_value_list, dim=0)
                # (b, l...*r, 2)
                batched_value = torch.stack([1 - batched_value, batched_value], dim=-1)
                # undo softmax
                logits_value = torch.log(batched_value / (1 - batched_value))  # Go to +- inf
                # Put it back finally
                #import pdb; pdb.set_trace()
                data[prop.fullname] = logits_value

        return data


# --------- Testing

def main():
    with Graph('global') as emrGraph:
        emrGraph.ontology = ('http://ontology.ihmc.us/ML/EMR.owl', './')
    
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

    data = None
    myVocab = None
    allenEmrGraph = AllenNlpGraph(emrGraph)
    myilpOntSolver = ilpOntSolver.getInstance(allenEmrGraph)

    myData = myilpOntSolver.inferSelection(allenEmrGraph, data, vocab=myVocab)

def mainOld():
    test_graph = Graph(iri='http://ontology.ihmc.us/ML/EMR.owl', local='./examples/emr/')

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
    
    myilpOntSolver = ilpOntSolver.getInstance(test_graph)
    tokenResult, relationsResult, _ = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, None)

    print("\nResults - ")
    print(tokenResult)
    
    if relationsResult != None :
        for name, result in relationsResult.items():
            print("\n")
            print(name)
            print(result)
            
def sprlMain():
    import numpy as np

    from regr.graph import Graph, Concept, Relation

    Graph.clear()
    Concept.clear()

    with Graph('spLanguage') as splang_Graph:
        splang_Graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './examples/SpRL_new')
    
        with Graph('linguistic') as ling_graph:
            ling_graph.ontology = ('http://ontology.ihmc.us/ML/PhraseGraph.owl', './')
            word = Concept(name ='word')
            phrase = Concept(name = 'phrase')
            sentence = Concept(name = 'sentence')
            phrase.has_many(word)
            sentence.has_many(phrase)
    
        with Graph('application') as app_graph:
            splang_Graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './examples/SpRL_new')
    
            trajector = Concept(name='TRAJECTOR')
            landmark = Concept(name='LANDMARK')
            noneentity = Concept(name='NONE')
            spatialindicator = Concept(name='SPATIALINDICATOR')
            trajector.is_a(phrase)
            landmark.is_a(phrase)
            noneentity.is_a(phrase)
            spatialindicator.is_a(phrase)
    
            trajector.not_a(noneentity)
            trajector.not_a(spatialindicator)
            landmark.not_a(noneentity)
            landmark.not_a(spatialindicator)
            noneentity.not_a(trajector)
            noneentity.not_a(landmark)
            noneentity.not_a(spatialindicator)
            spatialindicator.not_a(trajector)
            spatialindicator.not_a(landmark)
            spatialindicator.not_a(noneentity)
    
            triplet = Concept(name='triplet')
            region = Concept(name='region')
            region.is_a(triplet)
            relation_none= Concept(name='relation_none')
            relation_none.is_a(triplet)
            distance = Concept(name='distance')
            distance.is_a(triplet)
            direction = Concept(name='direction')
            direction.is_a(triplet)

    #------------------
    # sample input
    #------------------
    sentence = "About 20 kids in traditional clothing and hats waiting on stairs ."
    
    # NB
    # Input to the inference in SPRL example is a bit different.
    # After processing the original sentence, only phrases (with feature extracted) remain.
    # They might not come in the original order.
    # There could be repeated token in different phrase.
    # However, those should not influence the design of inference interface.
    phrases = ["stairs",                 # s     - GT: LANDMARK
                "About 20 kids ",        # a/2/k - GT: TRAJECTOR
                "About 20 kids",         # a/2/k - GT: TRAJECTOR
                "on",                    # o     - GT: SPATIALINDICATOR
                "hats",                  # h     - GT: NONE
                "traditional clothing"]  # tc    - GT: NONE
    # SPRL relations are triplet with this combination
    # (landmark, trajector, spatialindicator)
    # Relation GT:
    # ("stairs", "About 20 kids ", "on") : direction (and other? @Joslin please confirm if there is any other)
    # ("stairs", "About 20 kids", "on") : direction
    
    test_phrase = [(phrase, 'NP') for phrase in phrases] # Not feasible to have POS-tag. Usually they are noun phrase.
    
    #------------------
    # sample inference setup
    #------------------
    conceptNamesList = ["TRAJECTOR", "LANDMARK", "SPATIALINDICATOR", "NONE"]
    relationNamesList = ["direction", "distance", "region", "relation_none"]
    
    #------------------
    # sample output from learners
    #------------------
    # phrase

    phrase_table = dict()
    #                                             s    a/2/k a/2/k o     h     t/c
    phrase_table['TRAJECTOR']        = np.array([0.37, 0.72, 0.78, 0.01, 0.42, 0.22])
    phrase_table['LANDMARK']         = np.array([0.68, 0.15, 0.33, 0.03, 0.43, 0.13])
    phrase_table['SPATIALINDICATOR'] = np.array([0.05, 0.03, 0.02, 0.93, 0.03, 0.01])
    phrase_table['NONE']             = np.array([0.2 , 0.61, 0.48, 0.03, 0.5 , 0.52])
 
    relation_triple_tables = dict()
    
    # direction
    # triplet relation is a 3D array
    direction_relation_table = np.random.rand(6, 6, 6) * 0.2
    direction_relation_table[0, 1, 3] = 0.85 # ("stairs", "About 20 kids ", "on") - GT
    direction_relation_table[0, 2, 3] = 0.78 # ("stairs", "About 20 kids", "on") - GT
    direction_relation_table[1, 0, 3] = 0.32 # ("About 20 kids ", "stairs", "on")
    direction_relation_table[1, 2, 3] = 0.30 # ("About 20 kids ", "About 20 kids", "on")
    direction_relation_table[2, 0, 3] = 0.31 # ("About 20 kids", "stairs", "on")
    direction_relation_table[2, 1, 3] = 0.30 # ("About 20 kids", "About 20 kids ", "on")
    direction_relation_table[0, 4, 3] = 0.42 # ("stairs", "hat", "on")
    direction_relation_table[0, 5, 3] = 0.52 # ("stairs", "traditional clothing", "on")
    direction_relation_table[1, 4, 3] = 0.32 # ("About 20 kids ", "hat", "on")
    direction_relation_table[1, 5, 3] = 0.29 # ("About 20 kids ", "traditional clothing", "on")
    direction_relation_table[2, 4, 3] = 0.25 # ("About 20 kids ", "hat", "on")
    direction_relation_table[2, 5, 3] = 0.27 # ("About 20 kids ", "traditional clothing", "on")
    # ... can be more
    relation_triple_tables["direction"] = direction_relation_table
    
    # distance
    # triplet relation is a 3D array
    distance_relation_table = np.random.rand(6, 6, 6) * 0.2
    distance_relation_table[0, 1, 3] = 0.25 # ("stairs", "About 20 kids ", "on")
    distance_relation_table[0, 2, 3] = 0.38 # ("stairs", "About 20 kids", "on")
    distance_relation_table[1, 0, 3] = 0.32 # ("About 20 kids ", "stairs", "on")
    distance_relation_table[1, 2, 3] = 0.30 # ("About 20 kids ", "About 20 kids", "on")
    distance_relation_table[2, 0, 3] = 0.31 # ("About 20 kids", "stairs", "on")
    distance_relation_table[2, 1, 3] = 0.30 # ("About 20 kids", "About 20 kids ", "on")
    distance_relation_table[0, 4, 3] = 0.22 # ("stairs", "hat", "on")
    distance_relation_table[0, 5, 3] = 0.12 # ("stairs", "traditional clothing", "on")
    distance_relation_table[1, 4, 3] = 0.22 # ("About 20 kids ", "hat", "on")
    distance_relation_table[1, 5, 3] = 0.39 # ("About 20 kids ", "traditional clothing", "on")
    distance_relation_table[2, 4, 3] = 0.15 # ("About 20 kids ", "hat", "on")
    distance_relation_table[2, 5, 3] = 0.27 # ("About 20 kids ", "traditional clothing", "on")
    # ... can be more
    relation_triple_tables["distance"] = distance_relation_table
    
    # region
    # triplet relation is a 3D array
    region_relation_table = np.random.rand(6, 6, 6) * 0.2
    region_relation_table[0, 1, 3] = 0.25 # ("stairs", "About 20 kids ", "on")
    region_relation_table[0, 2, 3] = 0.38 # ("stairs", "About 20 kids", "on")
    region_relation_table[1, 0, 3] = 0.32 # ("About 20 kids ", "stairs", "on")
    region_relation_table[1, 2, 3] = 0.30 # ("About 20 kids ", "About 20 kids", "on")
    region_relation_table[2, 0, 3] = 0.31 # ("About 20 kids", "stairs", "on")
    region_relation_table[2, 1, 3] = 0.30 # ("About 20 kids", "About 20 kids ", "on")
    region_relation_table[0, 4, 3] = 0.22 # ("stairs", "hat", "on")
    region_relation_table[0, 5, 3] = 0.12 # ("stairs", "traditional clothing", "on")
    region_relation_table[1, 4, 3] = 0.22 # ("About 20 kids ", "hat", "on")
    region_relation_table[1, 5, 3] = 0.39 # ("About 20 kids ", "traditional clothing", "on")
    region_relation_table[2, 4, 3] = 0.15 # ("About 20 kids ", "hat", "on")
    region_relation_table[2, 5, 3] = 0.27 # ("About 20 kids ", "traditional clothing", "on")
    # ... can be more
    relation_triple_tables["region"] = region_relation_table
    
    # relation_none
    # triplet relation is a 3D array
    relation_none_relation_table = np.random.rand(6, 6, 6) * 0.8
    relation_none_relation_table[0, 1, 3] = 0.15 # ("stairs", "About 20 kids ", "on")
    relation_none_relation_table[0, 2, 3] = 0.08 # ("stairs", "About 20 kids", "on")
    relation_triple_tables["relation_none"] = relation_none_relation_table
    
    solver = ilpOntSolver.getInstance(splang_Graph)
    
    phrase_table, _, relation_triple_tables = solver.calculateILPSelection(
    test_phrase,       # original phrase, for reference purpose
    phrase_table,      # single concept table
    None,              # relation as tuple of concepts, used in EMR
    relation_triple_tables)   # relation as triplet of concetps, used in SPRL

    print("\nResults - ")
    print(phrase_table)
    
    if relation_triple_tables != None :
        for name, result in relation_triple_tables.items():
            print("\n")
            print(name)
            print(result)
            
if __name__ == '__main__' :
    mainOld()
