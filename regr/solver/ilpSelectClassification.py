# numpy
import numpy as np

# pandas
import pandas as pd

# ontology
from owlready2 import *

import os
from pkg_resources import resource_filename

# path to Meta Graph ontology
graphMetaOntologyPathname = resource_filename('regr', 'ontology/ML')

# path
from pathlib import Path

if __package__ is None or __package__ == '':
    from regr.graph.concept import Concept, enum
    from regr.graph.graph import Graph
    #from regr.graph.relation import Relation, IsA, HasA
    from regr.solver.ilpBooleanMethods import *
else:
    from ..graph.concept import Concept, enum
    from ..graph.graph import Graph
    #from ..graph.relation import Relation, IsA, HasA
    from .ilpBooleanMethods import *

import logging
import datetime

class ilpOntSolver:
    __instances = {}
    __logger = logging.getLogger(__name__)
    
    __negVarTrashhold = 0.3
    
    @staticmethod
    def getInstance(graph):
        if (graph is not None) and (graph.ontology is not None):
            if graph.ontology.iri not in ilpOntSolver.__instances:
                ilpOntSolver(graph.ontology.iri, graph.ontology.local)
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

    def __init__(self, ontologyURL, ontologyPathname=None):
        if ontologyURL in ilpOntSolver.__instances:
             pass
        else:  
            ilpOntSolver.__logger.info("Creating new ilpOntSolver for ontology: %s"%(ontologyURL))

            self.loadOntology(ontologyURL, ontologyPathname)
            ilpOntSolver.__instances[ontologyURL] = self 
        
    def addTokenConstrains(self, m, tokens, conceptNames, x, graphResultsForPhraseToken):
        # Create variables for token - concept and negative variables
        for token in tokens:            
            for conceptName in conceptNames: 
                if ilpSolver=="Gurobi":
                    x[token, conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName))
                elif ilpSolver == "GEKKO":
                    x[token, conceptName]=m.Var(0, lb=0, ub=1, integer=True, name="x_%s_%s"%(token, conceptName))
                
                if graphResultsForPhraseToken[conceptName][token] < ilpOntSolver.__negVarTrashhold:
                    if ilpSolver=="Gurobi":
                        x[token, conceptName+'-neg']=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName+'-neg'))
                    elif ilpSolver == "GEKKO":
                        x[token, conceptName]=m.Var(0, lb=0, ub=1, integer=True, name="x_%s_%s"%(token, conceptName+'-neg'))
    
        # Add constraints forcing decision between variable and negative variables 
        for conceptName in conceptNames:
            for token in tokens:
                if (token, conceptName+'-neg') in x:
                    if ilpSolver=="Gurobi":
                        constrainName = 'c_%s_%sselfDisjoint'%(token, conceptName)
                        
                        m.addConstr(x[token, conceptName] + x[token, conceptName+'-neg'], GRB.LESS_EQUAL, 1, name=constrainName)
                    elif ilpSolver == "GEKKO":
                        m.Equation(x[token, conceptName] + x[token, conceptName+'-neg'] <= 1)
                    
        if ilpSolver=="Gurobi":       
            m.update()
        elif ilpSolver == "GEKKO":
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
                    if ilpSolver=="Gurobi":       
                        constrainName = 'c_%s_%s_Disjoint_%s'%(token, conceptName, disjointConcept)
                        
                        m.addConstr(nandVar(m, x[token, conceptName], x[token, disjointConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)
                    elif ilpSolver == "GEKKO":
                        m.Equation(nandVar(m, x[token, conceptName], x[token, disjointConcept]) >= 1)
                               
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
                    if ilpSolver=="Gurobi":       
                        constrainName = 'c_%s_%s_Equivalent_%s'%(token, conceptName, equivalentConcept.name)
                        
                        m.addConstr(andVar(m, x[token, conceptName], x[token, equivalentConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)
                    elif ilpSolver == "GEKKO":
                        m.Equation(andVar(m, x[token, conceptName], x[token, equivalentConcept]) >= 1)

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
                    if ilpSolver=="Gurobi":       
                        constrainName = 'c_%s_%s_Ancestor_%s'%(token, currentConcept, ancestorConcept.name)
                        
                        m.addConstr(ifVar(m, x[token, conceptName], x[token, ancestorConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)
                    elif ilpSolver == "GEKKO":
                        m.Equation(ifVar(m, x[token, conceptName], x[token, ancestorConcept]) >= 1)

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
                        
                        if ilpSolver=="Gurobi":
                            constrainName = 'c_%s_%s_Intersection'%(token, conceptName)
    
                            m.addConstr(andVar(m, andList), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif ilpSolver == "GEKKO":
                            m.Equation(andVar(m, andList) >= 1)
        
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
                        
                        if ilpSolver=="Gurobi": 
                            constrainName = 'c_%s_%s_Union'%(token, conceptName)
   
                            m.addConstr(orVar(m, orList), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif ilpSolver == "GEKKO":
                            m.Equation(orVar(m, orList) >= 1)
        
        # -- Add constraints based on concept objectComplementOf statements in ontology - xor(var1, var2)
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is Not :
                    
                    complementClass = conceptConstruct.Class

                    for token in tokens:       
                        if ilpSolver=="Gurobi":    
                            constrainName = 'c_%s_%s_ComplementOf_%s'%(token, conceptName, complementClass.name)
                            
                            m.addConstr(xorVar(m, x[token, conceptName], x[token, complementClass.name]), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif ilpSolver == "GEKKO":
                            m.Equation(xorVar(m, x[token, conceptName], x[token, complementClass.name]) >= 1)

                    ilpOntSolver.__logger.info("Created - objectComplementOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,complementClass.name))
                        
        # ---- No supported yet
    
            # -- Add constraints based on concept disjonitUnion statements in ontology -  Not supported by owlready2 yet
        
            # -- Add constraints based on concept oneOf statements in ontology - ?
    
        if ilpSolver=="Gurobi":  
            m.update()
        elif ilpSolver == "GEKKO":
            pass
        
        # Add objectives
        X_Q = None
        for token in tokens:
            for conceptName in conceptNames:
                if ilpSolver=="Gurobi":
                    X_Q += graphResultsForPhraseToken[conceptName][token]*x[token, conceptName]
                elif ilpSolver == "GEKKO":
                    if X_Q is None:
                        X_Q = graphResultsForPhraseToken[conceptName][token]*x[token, conceptName]
                    else:
                        X_Q += graphResultsForPhraseToken[conceptName][token]*x[token, conceptName]

                if (token, conceptName+'-neg') in x: 
                    if ilpSolver=="Gurobi":
                        X_Q += (1-graphResultsForPhraseToken[conceptName][token])*x[token, conceptName+'-neg']
                    elif ilpSolver == "GEKKO":
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
    
                    if ilpSolver=="Gurobi":
                        y[relationName, token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(relationName, token, token1))
                    elif ilpSolver == "GEKKO":
                        y[relationName, token, token1]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s_%s_%s"%(relationName, token, token1))

                    if graphResultsForPhraseRelation[relationName][token1][token] < ilpOntSolver.__negVarTrashhold:
                        if ilpSolver=="Gurobi":
                            y[relationName+'-neg', token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s-neg_%s_%s"%(relationName, token, token1))
                        elif ilpSolver == "GEKKO":
                            y[relationName+'-neg', token, token1]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s-neg_%s_%s"%(relationName, token, token1))
                        
        # Add constraints forcing decision between variable and negative variables 
        for relationName in relationNames:
            for token in tokens: 
                for token1 in tokens:
                    if token == token1:
                        continue
                    
                    if (relationName+'-neg', token, token1) in y: 
                        if ilpSolver=="Gurobi":
                            constrainName = 'c_%s_%s_%sselfDisjoint'%(token, token1, relationName)
                            m.addConstr(y[relationName, token, token1] + y[relationName+'-neg', token, token1], GRB.LESS_EQUAL, 1, name=constrainName)
                        elif ilpSolver == "GEKKO":
                            m.Equation(y[relationName, token, token1] + y[relationName+'-neg', token, token1] <= 1)
                                
        if ilpSolver=="Gurobi":
            m.update()
        elif ilpSolver == "GEKKO":
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
                             
                            if ilpSolver=="Gurobi":
                                constrainNameDomain = 'c_domain_%s_%s_%s'%(currentRelation, token, token1)
                                constrainNameRange = 'c_range_%s_%s_%s'%(currentRelation, token, token1)
                                
                                #m.addConstr(ifVar(m, y[currentRelation._name, token, token1], x[token, domain._name]), GRB.GREATER_EQUAL, 1, name=constrainNameDomain)
                                #m.addConstr(ifVar(m, y[currentRelation._name, token, token1], x[token1, range._name]), GRB.GREATER_EQUAL, 1, name=constrainNameRange)
                                
                                m.addConstr(x[token, domain._name] - y[currentRelation._name, token, token1], GRB.GREATER_EQUAL, 0, name=constrainNameDomain)
                                m.addConstr(x[token1, range._name] - y[currentRelation._name, token, token1], GRB.GREATER_EQUAL, 0, name=constrainNameRange)
                            elif ilpSolver == "GEKKO":
                                #m.Equation(ifVar(m, y[currentRelation._name, token, token1], x[token, domain._name]) >= 1)
                                #m.Equation(ifVar(m, y[currentRelation._name, token, token1], x[token, range._name]) >= 1)
                                
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
                        
                        if ilpSolver=="Gurobi":
                            constrainName = 'c_%s_%s_%s_SuperProperty_%s'%(token, token1, relationName, superProperty.name)
                            
                            #m.addConstr(ifVar(m, y[relationName, token, token1], y[superProperty.name, token, token1]), GRB.GREATER_EQUAL, 1, name=constrainName)
                            m.addConstr(y[superProperty.name, token, token1] - y[relationName, token, token1], GRB.GREATER_EQUAL, 0, name=constrainName)
                        elif ilpSolver == "GEKKO":
                            #m.Equation(ifVar(m, y[relationName, token, token1], y[superProperty.name, token, token1]) >= 1)
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
                        
                        if ilpSolver=="Gurobi":
                            constrainName = 'c_%s_%s_%s_EquivalentProperty_%s'%(token, token1, relationName, equivalentProperty.name)
                            
                            m.addConstr(andVar(m, y[relationName, token, token1], y[equivalentProperty.name, token, token1]), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif ilpSolver == "GEKKO":
                            m.Equation(andVar(m, y[relationName, token, token1], y[equivalentProperty.name, token, token1]) >= 1)
                            
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
                        
                        if ilpSolver=="Gurobi":
                            constrainName = 'c_%s_%s_%s_InverseProperty'%(token, token1, relationName)
                            
                            m.addGenConstrIndicator(y[relationName, token, token1], True, y[currentRelationInverse.name, token1, token], GRB.EQUAL, 1)
                            
                            m.addConstr(ifVar(m, y[equivalentProperty.name, token, token1], y[relationName, token, token1]), GRB.GREATER_EQUAL, 1, name=constrainName)
                        elif ilpSolver == "GEKKO":
                            m.Equation(ifVar(m, y[equivalentProperty.name, token, token1], y[relationName, token, token1]) >= 1)
                            
        # -- Add constraints based on property functionalProperty statements in ontology - at most one P(x,y) for x
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if ilpSolver=="Gurobi":
                functionalLinExpr =  LinExpr()
            elif ilpSolver == "GEKKO":
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
    
        # ---- Related to DataType properties - not sure yet if we needto support them
        
            # -- Add constraints based on property dataSomeValuesFrom statements in ontology
        
            # -- Add constraints based on property dataHasValue statements in ontology
    
            # -- Add constraints based on property dataAllValuesFrom statements in ontology
    
        if ilpSolver=="Gurobi":
            m.update()
        elif ilpSolver == "GEKKO":
            pass
    
        # Add objectives
        Y_Q  = None
        for relationName in relationNames:
            for token in tokens:
                for token1 in tokens:
                    if token == token1 :
                        continue
    
                    if ilpSolver=="Gurobi":
                        Y_Q += graphResultsForPhraseRelation[relationName][token1][token]*y[relationName, token, token1]
                    elif ilpSolver == "GEKKO":
                        if Y_Q is None:
                            Y_Q = graphResultsForPhraseRelation[relationName][token1][token]*y[relationName, token, token1]
                        else:
                            Y_Q += graphResultsForPhraseRelation[relationName][token1][token]*y[relationName, token, token1]

        
                    if (relationName+'-neg', token, token1) in y: 
                        if ilpSolver=="Gurobi":
                            Y_Q += (1-graphResultsForPhraseRelation[relationName][token1][token])*y[relationName+'-neg', token, token1]
                        elif ilpSolver == "GEKKO":
                            if Y_Q is None:
                                Y_Q = (1-graphResultsForPhraseRelation[relationName][token1][token])*y[relationName+'-neg', token, token1]
                            else:
                                Y_Q += (1-graphResultsForPhraseRelation[relationName][token1][token])*y[relationName+'-neg', token, token1]
        
        return Y_Q
        
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken, graphResultsForPhraseRelation):
    
        start = datetime.datetime.now()
        ilpOntSolver.__logger.info('Start for phrase %s'%(phrase))
        ilpOntSolver.__logger.info('graphResultsForPhraseToken \n%s'%(graphResultsForPhraseToken))
        for relation in graphResultsForPhraseRelation:
            ilpOntSolver.__logger.info('graphResultsForPhraseRelation for relation \"%s\" \n%s'%(relation, graphResultsForPhraseRelation[relation]))

        tokenResult = None
        relationsResult = None
        
        try:
            if ilpSolver=="Gurobi":
                # Create a new Gurobi model
                m = Model("decideOnClassificationResult")
                m.params.outputflag = 0
            elif ilpSolver == "GEKKO":
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
            
            if ilpSolver=="Gurobi":
                m.setObjective(Q, GRB.MAXIMIZE)
            elif ilpSolver == "GEKKO":
                m.Obj((-1) * Q)
    
            # Token is associated with a single concept
            #for token in tokens:
            #   constrainName = 'c_%s'%(token)
            #    m.addConstr(quicksum(x[token, conceptName] for conceptName in conceptNames), GRB.LESS_EQUAL, 1, name=constrainName)
            
            if ilpSolver=="Gurobi":
                m.update()
            elif ilpSolver == "GEKKO":
                pass
            
            startOptimize = datetime.datetime.now()
            if ilpSolver=="Gurobi":
                ilpOntSolver.__logger.info('Optimizing model with %i variables and %i constrains'%(m.NumVars, m. NumConstrs))
            elif ilpSolver == "GEKKO":
                pass

            if ilpSolver=="Gurobi":
                m.optimize()
            elif ilpSolver == "GEKKO":
                m.solve(disp=solverDisp)
            
            endOptimize = datetime.datetime.now()
            elapsedOptimize = endOptimize - startOptimize

            if ilpSolver=="Gurobi":
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
            elif ilpSolver == "GEKKO":
                ilpOntSolver.__logger.info('Optimal solution was found - elapsed time: %ims'%(elapsedOptimize.microseconds/1000))

            # Collect results for tokens
            tokenResult = None
            tokenResult = pd.DataFrame(0, index=tokens, columns=conceptNames)
            if ilpSolver=="Gurobi":
                if x or True:
                    if m.status == GRB.Status.OPTIMAL:
        
                        solution = m.getAttr('x', x)
                        
                        for token in tokens :
                            for conceptName in conceptNames:
                                if solution[token, conceptName] == 1:                                    
                                    tokenResult[conceptName][token] = 1
                                    ilpOntSolver.__logger.info('Solution \"%s\" is \"%s\"'%(token,conceptName))
                                    
            elif ilpSolver == "GEKKO":
                for token in tokens :
                    for conceptName in conceptNames:
                        if x[token, conceptName].value[0] > gekkoTresholdForTruth:
                            tokenResult[conceptName][token] = 1
                            ilpOntSolver.__logger.info('Solution \"%s\" is \"%s\"'%(token,conceptName))

            # Collect results for relations
            relationsResult = {}
            relationNames = graphResultsForPhraseRelation.keys()
            if ilpSolver=="Gurobi":
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
            elif ilpSolver == "GEKKO":
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
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

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