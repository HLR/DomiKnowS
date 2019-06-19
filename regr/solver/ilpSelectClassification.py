# numpy
import numpy as np

# pandas
import pandas as pd

# Gurobi
from gurobipy import *

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
else:
    from ..graph.concept import Concept, enum
    from ..graph.graph import Graph
    #from ..graph.relation import Relation, IsA, HasA

import logging
import datetime

class iplOntSolver:
    __instances = {}
    __logger = None
    
    __negVarTrashhold = 0.3
    
    @staticmethod
    def getInstance(graph, ontologyPathname):    
        if (graph is not None) and (graph.ontology is not None):
            if graph.ontology not in iplOntSolver.__instances:
                iplOntSolver(graph.ontology, ontologyPathname)
                iplOntSolver.__instances[graph.ontology].myGraph = graph
            else:
                iplOntSolver.__logger.info("Returning existing iplOntSolver for %s"%(graph.ontology))
            
        return iplOntSolver.__instances[graph.ontology]
       
    def loadOntology(self, ontologyURL, ontologyPathname = "./"):
        start = datetime.datetime.now()
        iplOntSolver.__logger.info('Start')
        
        currentPath = Path(os.path.normpath("./")).resolve()
        
        # Check if Graph Meta ontology path is correct
        graphMetaOntologyPath = Path(os.path.normpath(graphMetaOntologyPathname))
        graphMetaOntologyPath = graphMetaOntologyPath.resolve()
        if not os.path.isdir(graphMetaOntologyPath):
            iplOntSolver.__logger.error("Path to load Graph ontology: %s does not exists in current directory %s"%(graphMetaOntologyPath,currentPath))
            exit()
            
        # Check if specific ontology path is correct
        ontologyPath = Path(os.path.normpath(ontologyPathname))
        ontologyPath = ontologyPath.resolve()
        if not os.path.isdir(ontologyPath):
            iplOntSolver.__logger.error("Path to load ontology: %s does not exists in current directory %s"%(ontologyURL,currentPath))
            exit()
    
        onto_path.append(graphMetaOntologyPath)  # the folder with the Graph Meta ontology
        onto_path.append(ontologyPath) # the folder with the ontology for the specific  graph
    
        # Load specific ontology
        try :
            self.myOnto = get_ontology(ontologyURL)
            self.myOnto.load(only_local = True, fileobj = None, reload = False, reload_if_newer = False)
        except FileNotFoundError as e:
            iplOntSolver.__logger.warning("Error when loading - %s from: %s"%(ontologyURL,ontologyPath))
    
        end = datetime.datetime.now()
        elapsed = end - start
        iplOntSolver.__logger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        return self.myOnto

    def __init__(self, ontologyURL, ontologyPathname):
        if iplOntSolver.__logger == None:
            # create logger
            iplOntSolver.__logger = logging.getLogger('iplOntSolver')
            iplOntSolver.__logger.setLevel(logging.DEBUG)

            # create console handler and set level to debug
            ch = logging.FileHandler('iplOntSolver.log')
            ch.setLevel(logging.INFO)

            # create formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(funcName)s - %(message)s')

            # add formatter to ch
            ch.setFormatter(formatter)

            # add ch to logger
            iplOntSolver.__logger.addHandler(ch)
            
        if ontologyURL in iplOntSolver.__instances:
             pass
        else:  
            iplOntSolver.__logger.info("Creating new iplOntSolver for ontology: %s"%(ontologyURL))

            self.loadOntology(ontologyURL, ontologyPathname)
            iplOntSolver.__instances[ontologyURL] = self 
        
    def addTokenConstrains(self, m, tokens, conceptNames, x, graphResultsForPhraseToken):
            
        # Create variables for token - concept and negative variables
        for token in tokens:            
            for conceptName in conceptNames: 
                x[token, conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName))
                
                if graphResultsForPhraseToken[conceptName][token] < iplOntSolver.__negVarTrashhold:
                    x[token, conceptName+'-neg']=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName+'-neg'))
    
        # Add constraints forcing decision between variable and negative variables 
        for conceptName in conceptNames:
            for token in tokens:
                if (token, conceptName+'-neg') in x: 
                    constrainName = 'c_%s_%sselfDisjoint'%(token, conceptName)
                    m.addConstr(x[token, conceptName] + x[token, conceptName+'-neg'], GRB.LESS_EQUAL, 1, name=constrainName)
                
        m.update()
         
        iplOntSolver.__logger.info("Created %i ipl variables for tokens"%(len(x)))
        
        # -- Add constraints based on concept disjoint statements in ontology - not(and(var1, var2)) = nand(var1, var2)
        foundDisjoint = dict() # too eliminate duplicates
        for conceptName in conceptNames:
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
            
            if currentConcept is None :
                continue
            
            iplOntSolver.__logger.debug("Concept \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentConcept.name, conceptName))
                
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
                    constrainName = 'c_%s_%s_Disjoint_%s'%(token, conceptName, disjointConcept)
                    m.addConstr(x[token, conceptName] + x[token, disjointConcept], GRB.LESS_EQUAL, 1, name=constrainName)
                                                
                if not (conceptName in foundDisjoint):
                    foundDisjoint[conceptName] = {disjointConcept}
                else:
                    foundDisjoint[conceptName].add(disjointConcept)
                           
            if conceptName in foundDisjoint:
                iplOntSolver.__logger.info("Created - disjoint - constrains between concept \"%s\" and concepts %s"%(conceptName,foundDisjoint[conceptName]))

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
                    constrainName = 'c_%s_%s_Equivalent_%s'%(token, conceptName, equivalentConcept.name)
                    _varAnd = m.addVar(name="andVar_%s"%(constrainName))
        
                    m.addGenConstrAnd(_varAnd, [x[token, conceptName], x[token, equivalentConcept.name]], constrainName)
                    #m.addConstr(x[token, conceptName] + x[token, equivalentConcept], GRB.LESS_EQUAL, 1, name=constrainName) #?
                                             
                if not (conceptName in foundEquivalent):
                    foundEquivalent[conceptName] = {equivalentConcept.name}
                else:
                    foundEquivalent[conceptName].add(equivalentConcept.name)
           
            if conceptName in foundEquivalent:
                iplOntSolver.__logger.info("Created - equivalent - constrains between concept \"%s\" and concepts %s"%(conceptName,foundEquivalent[conceptName]))
    
        # -- Add constraints based on concept subClassOf statements in ontology - var1 -> var2
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for ancestorConcept in currentConcept.ancestors(include_self = False) :
                if ancestorConcept.name not in graphResultsForPhraseToken.columns :
                     continue
                            
                for token in tokens:
                    constrainName = 'c_%s_%s_Ancestor_%s'%(token, currentConcept, ancestorConcept.name)
                    m.addGenConstrIndicator(x[token, conceptName], True, x[token, ancestorConcept.name], GRB.EQUAL, 1)
                
                iplOntSolver.__logger.info("Created - subClassOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,ancestorConcept.name))

        # -- Add constraints based on concept intersection statements in ontology - and(var1, var2, var3, ..)
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is And :
                    
                    for token in tokens:
                        constrainName = 'c_%s_%s_Intersection'%(token, conceptName)
                        _varAnd = m.addVar(name="andVar_%s"%(constrainName))
        
                        andList = []
                    
                        for currentClass in conceptConstruct.Classes :
                            andList.append(x[token, currentClass.name])
    
                        andList.append(x[token, conceptName])
                        
                        m.addGenConstrAnd(_varAnd, andList, constrainName)
        
        # -- Add constraints based on concept union statements in ontology -  or(var1, var2, var3, ..)
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is Or :
                    
                    for token in tokens:
                        constrainName = 'c_%s_%s_Union'%(token, conceptName)
                        _varOr = m.addVar(name="orVar_%s"%(constrainName))
        
                        orList = []
                    
                        for currentClass in conceptConstruct.Classes :
                            orList.append(x[token, currentClass.name])
    
                        orList.append(x[token, conceptName])
                        
                        m.addGenConstrOr(_varOr, orList, constrainName)
        
        # -- Add constraints based on concept objectComplementOf statements in ontology - var1 + var 2 = 1
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is Not :
                    
                    complementClass = conceptConstruct.Class

                    for token in tokens:                    
                        constrainName = 'c_%s_%s_ComplementOf_%s'%(token, conceptName, complementClass.name)
                        m.addConstr(x[token, conceptName] + x[token, complementClass.name], GRB.EQUAL, 1, name=constrainName)
                        
                    iplOntSolver.__logger.info("Created - objectComplementOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,complementClass.name))
                        
        # ---- No supported yet
    
            # -- Add constraints based on concept disjonitUnion statements in ontology -  Not supported by owlready2 yet
        
            # -- Add constraints based on concept oneOf statements in ontology - ?
    
        m.update()
                
        # Add objectives
        X_Q = None
        for token in tokens :
            for conceptName in conceptNames :
                X_Q += graphResultsForPhraseToken[conceptName][token]*x[token, conceptName]
                
                if (token, conceptName+'-neg') in x: 
                    X_Q += (1-graphResultsForPhraseToken[conceptName][token])*x[token, conceptName+'-neg']
    
        return X_Q
        
    def addRelationsConstrains(self, m, tokens, conceptNames, x, y, graphResultsForPhraseRelation):
        
        relationNames = graphResultsForPhraseRelation.keys()
            
        for relationName in relationNames:            
            for token in tokens: 
                for token1 in tokens:
                    if token == token1:
                        continue
    
                    y[relationName, token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(relationName, token, token1))
                    
                    if graphResultsForPhraseRelation[relationName][token1][token] < iplOntSolver.__negVarTrashhold:
                        y[relationName+'-neg', token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s-neg_%s_%s"%(relationName, token, token1))
              
        # Add constraints forcing decision between variable and negative variables 
        for relationName in relationNames:
            for token in tokens: 
                for token1 in tokens:
                    if token == token1:
                        continue
                    
                    if (relationName+'-neg', token, token1) in y: 
                        constrainName = 'c_%s_%s_%sselfDisjoint'%(token, token1, relationName)
                        m.addConstr(y[relationName, token, token1] + y[relationName+'-neg', token, token1], GRB.LESS_EQUAL, 1, name=constrainName)
    
        m.update()
    
        iplOntSolver.__logger.info("Created %i ipl variables for relations"%(len(y)))

        # -- Add constraints based on property domain and range statements in ontology - P(x,y) -> D(x), R(y)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue
    
            iplOntSolver.__logger.debug("Relation \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentRelation.name, relationName))
    
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
                                    
                            constrainName = 'c_%s_%s_%s'%(currentRelation, token, token1)
                            currentConstrain = y[currentRelation._name, token, token1] + x[token, domain._name] + x[token1, range._name]
                            m.addConstr(currentConstrain, GRB.GREATER_EQUAL, 3 * y[currentRelation._name, token, token1], name=constrainName)
                    
                    iplOntSolver.__logger.info("Created - domain-range - constrains for relation \"%s\" for domain \"%s\" and range \"%s\""%(relationName,domain._name,range._name))

          # -- Add constraints based on property subProperty statements in ontology P subproperty of S - P(x, y) -> S(x, y)
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
                        
                        constrainName = 'c_%s_%s_%s_SuperProperty_%s'%(token, token1, relationName, superProperty.name)
                        m.addGenConstrIndicator(y[relationName, token, token1], True, y[superProperty.name, token, token1], GRB.EQUAL, 1)
            
        # -- Add constraints based on property equivalentProperty statements in ontology -  and(var1, av2)
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
                    
                        constrainName = 'c_%s_%s_%s_EquivalentProperty_%s'%(token, token1, relationName, equivalentProperty.name)
                        _varAnd = m.addVar(name="andVar_%s"%(constrainName))
        
                        m.addGenConstrAnd(_varAnd, [y[relationName, token, token1], y[equivalentProperty.name, token, token1]], constrainName)
                                             
                    if not (relationName in foundEquivalent):
                        foundEquivalent[relationName] = {equivalentProperty.name}
                    else:
                        foundEquivalent[relationName].add(equivalentProperty.name)
        
        # -- Add constraints based on property inverseProperty statements in ontology - S(x,y) -> P(y,x)
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
                        
                        constrainName = 'c_%s_%s_%s_InverseProperty'%(token, token1, relationName)
                        m.addGenConstrIndicator(y[relationName, token, token1], True, y[currentRelationInverse.name, token1, token], GRB.EQUAL, 1)
                            
        # -- Add constraints based on property functionalProperty statements in ontology - at most one P(x,y) for x
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            functionalLinExpr =  LinExpr()
    
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
    
        m.update()
    
        # Add objectives
        Y_Q  = None
        for relationName in relationNames:
            for token in tokens:
                for token1 in tokens:
                    if token == token1 :
                        continue
    
                    Y_Q += graphResultsForPhraseRelation[relationName][token1][token]*y[relationName, token, token1]
                    
                    if (relationName+'-neg', token, token1) in y: 
                        Y_Q += (1-graphResultsForPhraseRelation[relationName][token1][token])*y[relationName+'-neg', token, token1]
        
        return Y_Q
        
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken, graphResultsForPhraseRelation):
    
        start = datetime.datetime.now()
        iplOntSolver.__logger.info('Start for phrase %s'%(phrase))
        iplOntSolver.__logger.info('graphResultsForPhraseToken \n%s'%(graphResultsForPhraseToken))
        for relation in graphResultsForPhraseRelation:
            iplOntSolver.__logger.info('graphResultsForPhraseRelation for relation \"%s\" \n%s'%(relation, graphResultsForPhraseRelation[relation]))

        tokenResult = None
        relationsResult = None
        
        try:
            # Create a new Gurobi model
            m = Model("decideOnClassificationResult")
            m.params.outputflag = 0
                
            # Get list of tokens and concepts from panda dataframe graphResultsForPhraseToken
            tokens = graphResultsForPhraseToken.index.tolist()
            conceptNames = graphResultsForPhraseToken.columns.tolist()
            
            # Gurobi variables for concept - token
            x={}
    
            # Gurobi variables for relation - token, token
            y={}
                
            # -- Set objective - maximize 
            Q = None
            
            X_Q = self.addTokenConstrains(m, tokens, conceptNames, x, graphResultsForPhraseToken)
            if X_Q is not None:
                Q += X_Q
            
            Y_Q = self.addRelationsConstrains(m, tokens, conceptNames, x, y, graphResultsForPhraseRelation)
            if Y_Q is not None:
                Q += Y_Q
            
            m.setObjective(Q, GRB.MAXIMIZE)
    
            # Token is associated with a single concept
            #for token in tokens:
            #   constrainName = 'c_%s'%(token)
            #    m.addConstr(quicksum(x[token, conceptName] for conceptName in conceptNames), GRB.LESS_EQUAL, 1, name=constrainName)
            
            m.update()
            
            startOptimize = datetime.datetime.now()
            iplOntSolver.__logger.info('Optimizing model with %i variables and %i constrains'%(m.NumVars, m. NumConstrs))

            m.optimize()
            endOptimize = datetime.datetime.now()
            elapsedOptimize = endOptimize - startOptimize

            if m.status == GRB.Status.OPTIMAL:
                iplOntSolver.__logger.info('Optimal solution was found - elapsed time: %ims'%(elapsedOptimize.microseconds/1000))
            elif m.status == GRB.Status.INFEASIBLE:
                 iplOntSolver.__logger.warning('Model was proven to be infeasible.')
            elif m.status == GRB.Status.INF_OR_UNBD:
                 iplOntSolver.__logger.warning('Model was proven to be infeasible or unbound.')
            elif m.status == GRB.Status.UNBOUNDED:
                 iplOntSolver.__logger.warning('Model was proven to be unbound.')
            else:
                 iplOntSolver.__logger.warning('Optimal solution not was found - error code %i'%(m.status))
                 
            # Collect results for tokens
            tokenResult = None
            if x or True:
                if m.status == GRB.Status.OPTIMAL:
                    tokenResult = pd.DataFrame(0, index=tokens, columns=conceptNames)
    
                    solution = m.getAttr('x', x)
                    
                    for token in tokens :
                        for conceptName in conceptNames:
                            if solution[token, conceptName] == 1:
                                #print("The  %s is classified as %s" % (token, conceptName))
                                
                                tokenResult[conceptName][token] = 1
                                
                                iplOntSolver.__logger.info('Solution \"%s\" is \"%s\"'%(token,conceptName))

    
            # Collect results for relations
            relationsResult = {}
            if y or True:
                if m.status == GRB.Status.OPTIMAL:
                    solution = m.getAttr('x', y)
                    relationNames = graphResultsForPhraseRelation.keys()
                    
                    for relationName in relationNames:
                        relationResult = pd.DataFrame(0, index=tokens, columns=tokens)
                        
                        for token in tokens :
                            for token1 in tokens:
                                if token == token1:
                                    continue
                                
                                if solution[relationName, token, token1] == 1:
                                    relationResult[token1][token] = 1
                                    
                                    iplOntSolver.__logger.info('Solution \"%s\" is in relation \"%s\" with \"%s\"'%(token1,relationName,token))

                        relationsResult[relationName] = relationResult
                        
        except GurobiError as e:
            iplOntSolver.__logger.error('GurobiError')
            raise
        
        except AttributeError:
            iplOntSolver.__logger.error('Gurobi AttributeError')
            raise
           
        end = datetime.datetime.now()
        elapsed = end - start
        iplOntSolver.__logger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # return results of ILP optimization
        return tokenResult, relationsResult

# --------- Testing

def main() :
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
    
    myiplOntSolver = iplOntSolver.getInstance(test_graph, ontologyPathname="./examples/emr/")
    tokenResult, relationsResult = myiplOntSolver.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation)

    myiplOntSolver2 = iplOntSolver.getInstance(ling_graph, ontologyPathname="./examples/emr/")
    tokenResult, relationsResult = myiplOntSolver2.calculateILPSelection(test_phrase, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation)
    
    print("\nResults - ")
    print(tokenResult)
    
    if relationsResult != None :
        for name, result in relationsResult.items():
            print("\n")
            print(name)
            print(result)
    
if __name__ == '__main__' :
    main()