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
    from regr.utils import printablesize

    from regr.sensor.allennlp.sensor import SentenceEmbedderSensor
    from regr.graph.allennlp import *
    
    from regr.solver.ilpConfig import ilpConfig 
    from regr.solver.ilpBooleanMethods import *
else:
    from ..graph.concept import Concept, enum
    from ..graph.graph import Graph
    from ..utils import printablesize
    
    from ..sensor.allennlp.sensor import SentenceEmbedderSensor
    from ..graph.allennlp.base import *
    
    from .ilpConfig import ilpConfig 
    from .ilpBooleanMethods import *

import logging
import datetime

class ilpOntSolver:
    __instances = {}
    __logger = logging.getLogger(__name__)
    
    __negVarTrashhold = 1.0
    
    ilpSolver = None
    myIlpBooleanProcessor = None
    
    def __init__(self) -> None:
        super().__init__(lazy=False)
       
    def __init__(self, ontologyURL, ontologyPathname=None, _iplConfig = ilpConfig) -> None:

        if ontologyURL in ilpOntSolver.__instances:
             pass
        else:  
            ilpOntSolver.__logger.info("Creating new ilpOntSolver for ontology: %s"%(ontologyURL))

            self.loadOntology(ontologyURL, ontologyPathname)
            ilpOntSolver.__instances[ontologyURL] = self 
            
            if _iplConfig is not None:
                self.ilpSolver = _iplConfig['ilpSolver']
            
            self.myIlpBooleanProcessor = ilpBooleanProcessor(_iplConfig)
               
    @staticmethod
    def getInstance(graph, _iplConfig = ilpConfig):
        if (graph is not None) and (graph.ontology is not None):
            if graph.ontology.iri not in ilpOntSolver.__instances:
                ilpOntSolver(graph.ontology.iri, graph.ontology.local, _iplConfig = ilpConfig)
                ilpOntSolver.__instances[graph.ontology.iri].myGraph = graph
            else:
                ilpOntSolver.__logger.info("Returning existing ilpOntSolver for %s"%(graph.ontology))
            
        return ilpOntSolver.__instances[graph.ontology.iri]
       
    def loadOntology(self, ontologyURL, ontologyPathname=None):
        start = datetime.datetime.now()
        ilpOntSolver.__logger.info('Start')
        
        currentPath = Path(os.path.normpath("./")).resolve()
        
        # Check if Graph Meta ontology path is correct
        graphMetaOntologyPath = Path(os.path.normpath(graphMetaOntologyPathname))
        graphMetaOntologyPath = graphMetaOntologyPath.resolve()
        if not os.path.isdir(graphMetaOntologyPath):
            ilpOntSolver.__logger.error("Path to load Graph ontology: %s does not exists in current directory %s"%(graphMetaOntologyPath,currentPath))
            exit()
            
        if ontologyPathname is not None:
            # Check if specific ontology path is correct
            ontologyPath = Path(os.path.normpath(ontologyPathname))
            ontologyPath = ontologyPath.resolve()
            if not os.path.isdir(ontologyPath):
                ilpOntSolver.__logger.error("Path to load ontology: %s does not exists in current directory %s"%(ontologyURL,currentPath))
                exit()

            onto_path.append(graphMetaOntologyPath)  # the folder with the Graph Meta ontology
            onto_path.append(ontologyPath) # the folder with the ontology for the specific  graph
    
        # Load specific ontology
        try :
            self.myOnto = get_ontology(ontologyURL)
            self.myOnto.load(only_local = True, fileobj = None, reload = False, reload_if_newer = False)
        except FileNotFoundError as e:
            ilpOntSolver.__logger.warning("Error when loading - %s from: %s"%(ontologyURL, ontologyPathname))
    
        end = datetime.datetime.now()
        elapsed = end - start
        ilpOntSolver.__logger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        return self.myOnto
    
    def addTokenConstrains(self, m, tokens, conceptNames, x, graphResultsForPhraseToken):
        # Create variables for token - concept and negative variables
        for token in tokens:            
            for conceptName in conceptNames: 
                if self.ilpSolver == "Gurobi":
                    x[token, conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName))
                elif self.ilpSolver == "GEKKO":
                    x[token, conceptName]=m.Var(0, lb=0, ub=1, integer=True, name="x_%s_%s"%(token, conceptName))
                
                if graphResultsForPhraseToken[conceptName][token] < ilpOntSolver.__negVarTrashhold:
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
         
        ilpOntSolver.__logger.info("Created %i ilp variables for tokens"%(len(x)))
        
        # -- Add constraints based on concept disjoint statements in ontology - not(and(var1, var2)) = nand(var1, var2)
        foundDisjoint = dict() # too eliminate duplicates
        for conceptName in conceptNames:
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
            
            if currentConcept is None :
                continue
            
            ilpOntSolver.__logger.debug("Concept \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentConcept.name, conceptName))
                
            for d in currentConcept.disjoints():
                disjointConcept = d.entities[1]._name
                    
                if currentConcept._name == disjointConcept:
                    disjointConcept = d.entities[0]._name
                        
                    if currentConcept._name == disjointConcept:
                        continue
                        
                if disjointConcept not in graphResultsForPhraseToken.columns:
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
                ilpOntSolver.__logger.info("Created - disjoint - constrains between concept \"%s\" and concepts %s"%(conceptName,foundDisjoint[conceptName]))

        # -- Add constraints based on concept equivalent statements in ontology - and(var1, av2)
        foundEquivalent = dict() # too eliminate duplicates
        for conceptName in conceptNames:
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for equivalentConcept in currentConcept.equivalent_to:
                if equivalentConcept.name not in graphResultsForPhraseToken.columns:
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
                ilpOntSolver.__logger.info("Created - equivalent - constrains between concept \"%s\" and concepts %s"%(conceptName,foundEquivalent[conceptName]))
    
        # -- Add constraints based on concept subClassOf statements in ontology - var1 -> var2
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for ancestorConcept in currentConcept.ancestors(include_self = False) :
                if ancestorConcept.name not in graphResultsForPhraseToken.columns :
                     continue
                            
                for token in tokens:
                    if self.ilpSolver == "Gurobi":       
                        constrainName = 'c_%s_%s_Ancestor_%s'%(token, currentConcept, ancestorConcept.name)
                        
                        m.addConstr(self.myIlpBooleanProcessor.ifVar(m, x[token, conceptName], x[token, ancestorConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)
                    elif self.ilpSolver == "GEKKO":
                        m.Equation(self.myIlpBooleanProcessor.ifVar(m, x[token, conceptName], x[token, ancestorConcept]) >= 1)

                ilpOntSolver.__logger.info("Created - subClassOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,ancestorConcept.name))

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

                    ilpOntSolver.__logger.info("Created - objectComplementOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,complementClass.name))
                        
        # ---- No supported yet
    
            # -- Add constraints based on concept disjonitUnion statements in ontology -  Not supported by owlready2 yet
        
            # -- Add constraints based on concept oneOf statements in ontology - ?
    
        if self.ilpSolver == "Gurobi":  
            m.update()
        elif self.ilpSolver == "GEKKO":
            pass
        
        # Add objectives
        X_Q = None
        for token in tokens:
            for conceptName in conceptNames:
                if self.ilpSolver == "Gurobi":
                    X_Q += graphResultsForPhraseToken[conceptName][token]*x[token, conceptName]
                elif self.ilpSolver == "GEKKO":
                    if X_Q is None:
                        X_Q = graphResultsForPhraseToken[conceptName][token]*x[token, conceptName]
                    else:
                        X_Q += graphResultsForPhraseToken[conceptName][token]*x[token, conceptName]

                if (token, conceptName+'-neg') in x: 
                    if self.ilpSolver == "Gurobi":
                        X_Q += (1-graphResultsForPhraseToken[conceptName][token])*x[token, conceptName+'-neg']
                    elif self.ilpSolver == "GEKKO":
                        if X_Q is None:
                            X_Q = (1-graphResultsForPhraseToken[conceptName][token])*x[token, conceptName+'-neg']
                        else:
                            X_Q += (1-graphResultsForPhraseToken[conceptName][token])*x[token, conceptName+'-neg']

        return X_Q
     
    def addRelationsConstrains(self, m, tokens, conceptNames, x, y, graphResultsForPhraseRelation):
        relationNames = graphResultsForPhraseRelation.keys()
            
        for relationName in relationNames:            
            for token in tokens: 
                for token1 in tokens:
                    if token == token1:
                        continue
    
                    if self.ilpSolver == "Gurobi":
                        y[relationName, token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(relationName, token, token1))
                    elif self.ilpSolver == "GEKKO":
                        y[relationName, token, token1]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s_%s_%s"%(relationName, token, token1))

                    if graphResultsForPhraseRelation[relationName][token1][token] < ilpOntSolver.__negVarTrashhold:
                        if self.ilpSolver == "Gurobi":
                            y[relationName+'-neg', token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s-neg_%s_%s"%(relationName, token, token1))
                        elif self.ilpSolver == "GEKKO":
                            y[relationName+'-neg', token, token1]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s-neg_%s_%s"%(relationName, token, token1))
                        
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
    
        ilpOntSolver.__logger.info("Created %i ilp variables for relations"%(len(y)))

        # -- Add constraints based on property domain and range statements in ontology - P(x,y) -> D(x), R(y)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue
    
            ilpOntSolver.__logger.debug("Relation \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentRelation.name, relationName))
    
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
                                
                    ilpOntSolver.__logger.info("Created - domain-range - constrains for relation \"%s\" for domain \"%s\" and range \"%s\""%(relationName,domain._name,range._name))

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
            for token in tokens:
                for token1 in tokens:
                    if token == token1 :
                        continue
    
                    if self.ilpSolver == "Gurobi":
                        Y_Q += graphResultsForPhraseRelation[relationName][token1][token]*y[relationName, token, token1]
                    elif self.ilpSolver == "GEKKO":
                        if Y_Q is None:
                            Y_Q = graphResultsForPhraseRelation[relationName][token1][token]*y[relationName, token, token1]
                        else:
                            Y_Q += graphResultsForPhraseRelation[relationName][token1][token]*y[relationName, token, token1]

        
                    if (relationName+'-neg', token, token1) in y: 
                        if self.ilpSolver == "Gurobi":
                            Y_Q += (1-graphResultsForPhraseRelation[relationName][token1][token])*y[relationName+'-neg', token, token1]
                        elif self.ilpSolver == "GEKKO":
                            if Y_Q is None:
                                Y_Q = (1-graphResultsForPhraseRelation[relationName][token1][token])*y[relationName+'-neg', token, token1]
                            else:
                                Y_Q += (1-graphResultsForPhraseRelation[relationName][token1][token])*y[relationName+'-neg', token, token1]
        
        return Y_Q
        
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken, graphResultsForPhraseRelation):
    
        if self.ilpSolver == None:
            ilpOntSolver.__logger.info('ILP solver not provided - returning unchanged results')
            return graphResultsForPhraseToken, graphResultsForPhraseRelation
        
        start = datetime.datetime.now()
        ilpOntSolver.__logger.info('Start for phrase %s'%(phrase))
        ilpOntSolver.__logger.info('graphResultsForPhraseToken \n%s'%(graphResultsForPhraseToken))
        
        for relation in graphResultsForPhraseRelation:
            ilpOntSolver.__logger.info('graphResultsForPhraseRelation for relation \"%s\" \n%s'%(relation, graphResultsForPhraseRelation[relation]))

        tokenResult = None
        relationsResult = None
        
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
                
            # Get list of tokens and concepts from panda dataframe graphResultsForPhraseToken
            tokens = graphResultsForPhraseToken.index.tolist()
            conceptNames = graphResultsForPhraseToken.columns.tolist()
            
            # Variables for concept - token
            x={}
    
            # Variables for relation - token, token
            y={}
                
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
                ilpOntSolver.__logger.info('Optimizing model with %i variables and %i constrains'%(m.NumVars, m. NumConstrs))
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
                    ilpOntSolver.__logger.info('Optimal solution was found - elapsed time: %ims'%(elapsedOptimize.microseconds/1000))
                elif m.status == GRB.Status.INFEASIBLE:
                     ilpOntSolver.__logger.warning('Model was proven to be infeasible.')
                elif m.status == GRB.Status.INF_OR_UNBD:
                     ilpOntSolver.__logger.warning('Model was proven to be infeasible or unbound.')
                elif m.status == GRB.Status.UNBOUNDED:
                     ilpOntSolver.__logger.warning('Model was proven to be unbound.')
                else:
                     ilpOntSolver.__logger.warning('Optimal solution not was found - error code %i'%(m.status))
            elif self.ilpSolver == "GEKKO":
                ilpOntSolver.__logger.info('Optimal solution was found - elapsed time: %ims'%(elapsedOptimize.microseconds/1000))

            # Collect results for tokens
            tokenResult = None
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
                                        
                                        ilpOntSolver.__logger.info('Solution \"%s\" is in relation \"%s\" with \"%s\"'%(token1,relationName,token))
    
                            relationsResult[relationName] = relationResult
            elif self.ilpSolver == "GEKKO":
                for relationName in relationNames:
                    relationResult = pd.DataFrame(0, index=tokens, columns=tokens)
                    
                    for token in tokens :
                        for token1 in tokens:
                            if token == token1:
                                continue
                            
                            if y[relationName, token, token1].value[0] > gekkoTresholdForTruth:
                                relationResult[token1][token] = 1
                                
                                ilpOntSolver.__logger.info('Solution \"%s\" is in relation \"%s\" with \"%s\"'%(token1,relationName,token))

                    relationsResult[relationName] = relationResult
                        
        except:
            ilpOntSolver.__logger.error('Error')
            raise
           
        end = datetime.datetime.now()
        elapsed = end - start
        ilpOntSolver.__logger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # return results of ILP optimization
        return tokenResult, relationsResult

    DataInstance = Dict[str, Tensor]
    
    def inferSelection(self, graph: Graph, data: DataInstance, vocab = None) -> DataInstance:
        
        concepts = []
        relations = []

        for prop in graph.poi:
            concept = prop.sup
            if concept.has_a():
                relations.append(concept.name)
            else:
                concepts.append(concept.name)

        groups = [
            concepts,
            relations
        ]
    
        tokens_sensors = graph.get_sensors(SentenceEmbedderSensor)
        name, tokens_sensor = tokens_sensors[0] # FIXME: considering problems with only one sentence
    
        mask = tokens_sensor.get_mask(data)
        mask_len = mask.sum(dim=1).clone().cpu().detach().numpy() # (b, )
    
        sentence = data[tokens_sensor.fullname + '_index'][tokens_sensor.key] # (b, l)
    
        # Table columns, as many table columns as groups
        tables = [[] for _ in groups]
        wild = []  # For those not in any group
        
        # For each subgraph.concept[prop] that has multiple assignment
        for prop in graph.get_multiassign(): # order? always check with table
            # find the group it goes to
            # TODO: update later with group discover?
            # For each assignment, [0] for label, [1] for pred consider only prediction here
            sensor = list(prop.values())[1]
            
            # How about conf?
            for group, table in zip(groups, tables):
                if prop.sup.name in group:
                    table.append(sensor)
                    break
            else: 
                # For group, table, no break belongs to no group still do something, differently
                wild.append(sensor)
    
        # Now we have (batch, ) in predictions, but inference may process one item at a time should organize in this way (batch, len, ..., t/f, column), then we can iter through batches
        # note that, t/f is a redundant dim, that gives 2 scalars: [1-p, p], maybe needed to squeeze using the pindex and index_select requires the above facts
        valuetables = []
        batch_size = None
        for table in tables:
            if len(table) == 0:
                continue
            
            # Assume all of them give same dims of output: (batch, len, ..., t/f)
            values = []
            for sensor in table:
                sensor(data)  # (batch, len, ..., t/f)
                value = data[sensor.fullname]
                # (b, l, c) - dim=2 / (b, l, l, c) - dim=3
                #value = torch.exp(value)
                value = torch.nn.functional.softmax(value, dim=-1)
                    
                # At t/f dim, 0 for 1-p, 1 for p
                pindex = torch.tensor(1, device=value.device).long()
                
                # (batch, len, ..., 1) # need tailing 1 for cat
                value = value.index_select(-1, pindex)
                
                # Get/check the batch_size
                if batch_size is None:
                    batch_size = value.size()[0]
                else:
                    assert batch_size == value.size()[0]
                values.append(value)
                
            values = torch.cat(values, dim=-1)  # (batch, len, ..., ncls)
            
            # Then it has the same order as tables, where we can refer to related concept
            valuetables.append(values)
            
        # We need all columns to be placed, for all tables, before inference now we have
    
        updated_valuetables_batch = [[] for _ in valuetables]
        
        # For each batch
        for batch_index in torch.arange(batch_size, device=values.device):
            inference_tables = []
            for values, table in zip(valuetables, tables):
                # Use only current batch 0 for batch, resulting (len, ..., ncls)
                values = values.index_select(0, batch_index)
                values = values.squeeze(dim=0)
                
                # Now values is the table we need and names is the list of grouped concepts (name of the columns)
                inference_tables.append((table, values))
    
            # Data structure convert
            phrase = None  # TODO: since it not using now. if it is needed later, will pass it somewhere else
    
            phrasetable = inference_tables[0][1].clone().cpu().detach().numpy()
            
            # Apply mask for phrase
            phrasetable = phrasetable[:mask_len[batch_index], :]
            if vocab:
                tokens = ['{}_{}'.format(i, vocab.get_token_from_index(int(sentence[batch_index,i]), namespace=tokens_sensor.key))
                          for i in torch.arange(phrasetable.shape[0], device=values.device)]
            else:
                tokens = [str(j) for j in range(phrasetable.shape[0])]
            concept_names = [sensor.sup.sup.name for sensor in tables[0]]
            
            graphResultsForPhraseToken = pd.DataFrame(phrasetable, index=tokens, columns=concept_names)
    
            graphResultsForPhraseRelation = dict()
            if len(tables[1]) > 0:
                graphtable = inference_tables[1][1].clone().cpu().detach().numpy()
                
                for i, sensor in enumerate(tables[1]):
                    # each relation - apply mask
                    graphResultsForPhraseRelation[sensor.sup.sup.name] = pd.DataFrame(graphtable[:mask_len[batch_index], :mask_len[batch_index], i], index=tokens, columns=tokens)
    
            # Do inference
            try:
                tokenResult, relationsResult = self.calculateILPSelection(phrase, graphResultsForPhraseToken, graphResultsForPhraseRelation)
                
                if tokenResult is None and relationsResult is None:
                    raise RuntimeError('No result from solver. Check any log from the solver.')
            except:
                print('-'*40)
                print(phrasetable)
                print(tokens)
                print(concept_names)
                print(graphResultsForPhraseToken)
                print(graphResultsForPhraseRelation)
                print(tokenResult, relationsResult)
                print('-'*40)
                raise
    
            # Convert back
            for i, (updated_batch, (table, values)) in enumerate(zip(updated_valuetables_batch, inference_tables)):
                # Values: tensor (len, ..., ncls) updated_batch: list of batches of result of tensor (len, ..., ncls)
                updated = torch.zeros(values.size(), device=values.device)
                
                # do something to query iplResults to fill updated - implement below
                if i == 0:
                    # tokenResult: [len, ncls], notice the order of ncls - updated: tensor(len, ncls)
                    result = tokenResult[[sensor.sup.sup.name for sensor in table]].to_numpy() # use the names to control the order
                    updated[:mask_len[batch_index],:] = torch.from_numpy(result)
                elif i == 1:
                    # relationsResult: dict(ncls)[len, len], order of len should not be changed - updated: tensor(len, len, ncls)
                    for j, sensor in zip(torch.arange(len(table)), table):
                        try:
                            result = relationsResult[sensor.sup.sup.name][tokens].to_numpy() # Use the tokens to enforce the order
                        except:
                            print('-'*40)
                            print(tokens)
                            print(graphResultsForPhraseToken)
                            print(graphResultsForPhraseRelation)
                            print(tokenResult)
                            print(relationsResult)
                            print(i)
                            print(name)
                            print('-'*40)
                            raise
    
                        updated[:mask_len[batch_index],:mask_len[batch_index],j] = torch.from_numpy(result)
                else:
                    # Should be nothing here
                    pass
    
                # Add to updated batch
                updated_batch.append(updated)
                
        # updated_valuetables_batch is List(tables)[List(batch_size)[Tensor(len, ..., ncls)]]
    
        # Put it back into one piece - we want List(tables)[List(ncls)[Tensor(batch, len, ..., 2)]] be careful of that the 2 need extra manipulation
        for updated_batch, table in zip(updated_valuetables_batch, tables):
            # No update then continue
            if len(table) == 0 or len(updated_batch) == 0:
                continue
    
            # Updated_batch: List(batch_size)[Tensor(len, ..., ncls)]
            updated_batch = [updated.unsqueeze(dim=0) for updated in updated_batch]
           
            # Tensor(batch, len, ..., ncls)
            updated_batch_tensor = torch.cat(updated_batch, dim=0)
    
            # For each class in ncls
            ncls = updated_batch_tensor.size()[-1]
            for icls, sensor in zip(torch.arange(ncls, device=updated_batch_tensor.device), table):
                # Tensor(batch, len, ..., 1)
                value = updated_batch_tensor.index_select(-1, icls)
               
                # Tensor(batch, len, ..., 2)
                value = torch.cat([1 - value, value], dim=-1)
    
                # Put it back finally
                logits_value = torch.log(value/(1-value)) # Go to +- inf
                
                data[sensor.sup.fullname] = logits_value
                data[sensor.fullname] = logits_value
    
        return data

def setup_solver_logger(log_filename='ilpOntSolver.log'):
    logger = logging.getLogger(__name__)

    # create file handler and set level to info
    ch = logging.FileHandler(log_filename)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(funcName)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

setup_solver_logger()

# --------- Testing

def main():
    setup_solver_logger()

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
    
    myData = inferSelection(allenEmrGraph, data, vocab=myVocab)

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
    tokenResult, relationsResult = myilpOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation)

    print("\nResults - ")
    print(tokenResult)
    
    if relationsResult != None :
        for name, result in relationsResult.items():
            print("\n")
            print(name)
            print(result)
    
if __name__ == '__main__' :
    main()
