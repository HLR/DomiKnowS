from six import string_types
from itertools import product, permutations, groupby
import logging
import datetime

# numpy
import numpy as np

# ontology
from owlready2 import *

# Gurobi
from gurobipy import *

from regr.graph.concept import Concept
from regr.solver.ilpOntSolver import ilpOntSolver
from regr.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from regr.graph import LogicalConstrain, andL, orL, ifL, existsL, notL

class gurobiILPOntSolver(ilpOntSolver):
    ilpSolver = 'Gurobi'

    def __init__(self, graph, ontologiesTuple, _ilpConfig) -> None:
        super().__init__(graph, ontologiesTuple, _ilpConfig)
        self.myIlpBooleanProcessor = gurobiILPBooleanProcessor()
        
    def valueToBeSkipped(self, x):
        return ( 
                x != x or  # nan 
                abs(x) == float('inf')  # inf 
                ) 
               
    def addTokenConstrains(self, m, conceptNames, tokens, x, graphResultsForPhraseToken):
        if graphResultsForPhraseToken is None:
            return None
        
        self.myLogger.info('Starting method addTokenConstrains')
        self.myLogger.debug('graphResultsForPhraseToken')
        padding = max([len(str(t)) for t in tokens])
        spacing = max([len(str(c)) for c in conceptNames]) + 1
        self.myLogger.debug("{:^{}}".format("", spacing) + ' '.join(map('{:^10}'.format, ['\''+ str(t) + '\'' for t in tokens])))
        for concept, tokenTable in graphResultsForPhraseToken.items():
            self.myLogger.debug("{:<{}}".format(concept, spacing) + ' '.join(map('{:^10f}'.format, [t[1] for t in tokenTable])))

        # Create variables for token - concept and negative variables
        for conceptName in conceptNames: 
            for tokenIndex, token in enumerate(tokens):            
                currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                # Check if probability is NaN or if and has to be skipped
                if self.valueToBeSkipped(currentProbability[1]):
                    self.myLogger.info("Probability is %f for concept %s and token %s - skipping it"%(currentProbability[1],token,conceptName))
                    continue

                # Create variable
                x[conceptName, token]=m.addVar(vtype=GRB.BINARY,name="x_%s_is_%s"%(token, conceptName))             
                #self.myLogger.debug("Created ILP variable for concept %s and token %s it's probability is %f"%(conceptName,token,currentProbability[1]))

                # Check if probability is NaN or if and has to be created based on positive value
                if self.valueToBeSkipped(currentProbability[0]):
                    currentProbability[0] = 1 - currentProbability[1]
                    self.myLogger.info("No ILP negative variable for concept %s and token %s - created based on positive value %f"%(token, conceptName, currentProbability[0]))

                # Create negative variable
                if True: # ilpOntSolver.__negVarTrashhold:
                    x['Not_'+conceptName, token]=m.addVar(vtype=GRB.BINARY,name="x_%s_is_not_%s"%(token, conceptName))
                else:
                    self.myLogger.info("No ILP negative variable for concept %s and token %s created"%(token, conceptName))

        # Add constraints forcing decision between variable and negative variables 
        for conceptName in conceptNames:
            for token in tokens:
                if ('Not_'+conceptName, token) in x:
                    currentConstrLinExpr = x[conceptName, token] + x['Not_'+conceptName, token]
                    m.addConstr(currentConstrLinExpr == 1, name='c_%s_%sselfDisjoint'%(conceptName, token))
                    #self.myLogger.debug("Disjoint constrain between token %s is concept %s and token %s is concept - %s == %i"%(token,conceptName,token,'Not_'+conceptName,1))
                    
        m.update()

        if len(x):
            self.myLogger.info("Created %i ILP variables for tokens"%(len(x)))
        else:
            self.myLogger.warning("No ILP variables created for tokens")
            return

        if hasattr(self, 'myOnto'): # --- Use Ontology as a source of constrains 
            # -- Add constraints based on concept disjoint statements in ontology - not(and(var1, var2)) = nand(var1, var2)
            foundDisjoint = dict() # too eliminate duplicates
            for conceptName in conceptNames:
                
                currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
                if currentConcept is None :
                    continue
                
                #self.myLogger.debug("Concept \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentConcept.name, conceptName))
                    
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
                                
                    for tokenIndex, token in enumerate(tokens):
                        if (conceptName, token) not in x:
                            continue
                        
                        self.myIlpBooleanProcessor.nandVar(m, x[conceptName, token], x[disjointConcept, token], onlyConstrains = True)
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
                                
                    for tokenIndex, token in enumerate(tokens):
                        if (conceptName, token) not in x:
                            continue
                        
                        self.myIlpBooleanProcessor.andVar(m, x[conceptName, token], x[equivalentConcept, token], onlyConstrains = True)
                    if not (conceptName in foundEquivalent):
                        foundEquivalent[conceptName] = {equivalentConcept.name}
                    else:
                        foundEquivalent[conceptName].add(equivalentConcept.name)
               
                if conceptName in foundEquivalent:
                    self.myLogger.info("Created - equivalent - constrains between concept \"%s\" and concepts %s"%(conceptName,foundEquivalent[conceptName]))
        
            # -- Add constraints based on concept subClassOf statements in ontology - var1 -> var2
            for conceptName in conceptNames:
                
                currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                    
                if currentConcept is None :
                    continue
                    
                for ancestorConcept in currentConcept.ancestors(include_self = False):
                    if ancestorConcept.name not in conceptNames:
                         continue
                                
                    for tokenIndex, token in enumerate(tokens):
                        if (conceptName, token) not in x:
                            continue
                        
                        if (ancestorConcept, token) not in x:
                            continue
                        
                        self.myIlpBooleanProcessor.ifVar(m, x[conceptName, token], x[ancestorConcept.name, token], onlyConstrains = True)
                            
                    self.myLogger.info("Created - subClassOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,ancestorConcept.name))
    
            # -- Add constraints based on concept intersection statements in ontology - and(var1, var2, var3, ..)
            for conceptName in conceptNames:
                
                currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                    
                if currentConcept is None:
                    continue
                    
                for conceptConstruct in currentConcept.constructs(Prop = None):
                    if type(conceptConstruct) is And:
                        
                        for tokenIndex, token in enumerate(tokens):
                            if (conceptName, token) not in x:
                                continue
                        
                            _varAnd = m.addVar(name="andVar_%s_%s"%(conceptName, token))
            
                            andList = []
                        
                            for currentClass in conceptConstruct.Classes :
                                if (currentClass.name, token) not in x:
                                    continue
                            
                                andList.append(x[currentClass.name, token])
        
                            andList.append(x[conceptName, token])
                            
                            self.myIlpBooleanProcessor.andVar(m, andList, onlyConstrains = True)
                            
            # -- Add constraints based on concept union statements in ontology -  or(var1, var2, var3, ..)
            for conceptName in conceptNames:
                
                currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                    
                if currentConcept is None:
                    continue
                    
                for conceptConstruct in currentConcept.constructs(Prop = None):
                    if type(conceptConstruct) is Or:
                        
                        for tokenIndex, token in enumerate(tokens):    
                            if (conceptName, token) not in x:
                                continue
                            
                            _varOr = m.addVar(name="orVar_%s_%s"%(conceptName, token))
            
                            orList = []
                        
                            for currentClass in conceptConstruct.Classes :
                                if (currentClass.name, token) not in x:
                                    continue
                                
                                orList.append(x[currentClass.name, token])
        
                            orList.append(x[conceptName, token])
                            
                            self.myIlpBooleanProcessor.orVar(m, orList, onlyConstrains = True)
            
            # -- Add constraints based on concept objectComplementOf statements in ontology - xor(var1, var2)
            for conceptName in conceptNames:
                
                currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                    
                if currentConcept is None:
                    continue
                    
                for conceptConstruct in currentConcept.constructs(Prop = None):
                    if type(conceptConstruct) is Not:
                        
                        complementClass = conceptConstruct.Class
    
                        for tokenIndex, token in enumerate(tokens):          
                            if (conceptName, token) not in x:
                                continue
                            
                            if (complementClass.name, token) not in x:
                                continue
                              
                            self.myIlpBooleanProcessor.xorVar(m, x[conceptName, token], x[complementClass.name, token], onlyConstrains = True)
    
                        self.myLogger.info("Created - objectComplementOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,complementClass.name))
                            
            # ---- No supported yet
        
            # -- Add constraints based on concept disjonitUnion statements in ontology -  Not supported by owlready2 yet
            
            # -- Add constraints based on concept oneOf statements in ontology - ?
        else: # ---------- no Ontology
            concepts = set()
            
            # Get concept based on concept names
            for graph in self.myGraph:
                for conceptName in conceptNames:
                    if conceptName in graph.concepts:
                        concepts.add(graph.concepts[conceptName])
                        
                for subGraphKey in graph._objs:
                    subGraph = graph._objs[subGraphKey]
                    for conceptName in conceptNames:
                            if conceptName in subGraph.concepts:
                                concepts.add(subGraph.concepts[conceptName])
            
            # Create subclass constrains
            for concept in concepts:
                for rel in concept.is_a():
                    # A is_a B : if(A, B) : A(x) <= B(x)
                    for token in tokens:
                        if (rel.src.name, token) not in x: # subclass (A)
                            continue
                        
                        if (rel.dst.name, token) not in x: # superclass (B)
                            continue
                                                                        
                        self.myIlpBooleanProcessor.ifVar(m, x[rel.src.name, token], x[rel.dst.name, token], onlyConstrains = True)
                        self.myLogger.info("Created - subclass - constrains between concept \"%s\" and concepts %s"%(rel.src.name,rel.dst.name))

            # Create disjoint constraints
            foundDisjoint = dict() # To eliminate duplicates
            for concept in concepts:     
                for rel in concept.not_a():
                    conceptName = concept.name
                    disjointConcept = rel.dst.name
                        
                    if disjointConcept not in conceptNames:
                         continue
                            
                    if conceptName in foundDisjoint:
                        if disjointConcept in foundDisjoint[conceptName]:
                            continue
                    
                    if disjointConcept in foundDisjoint:
                        if conceptName in foundDisjoint[disjointConcept]:
                            continue
                                
                    for tokenIndex, token in enumerate(tokens):
                        if (conceptName, token) not in x:
                            continue
                        
                        if (disjointConcept, token) not in x:
                            continue
                            
                        self.myIlpBooleanProcessor.nandVar(m, x[conceptName, token], x[disjointConcept, token], onlyConstrains = True)
                            
                    if not (conceptName in foundDisjoint):
                        foundDisjoint[conceptName] = {disjointConcept}
                    else:
                        foundDisjoint[conceptName].add(disjointConcept)
                               
                if conceptName in foundDisjoint:
                    self.myLogger.info("Created - disjoint - constrains between concept \"%s\" and concepts %s"%(conceptName,foundDisjoint[conceptName]))
            
        m.update()

        # Add objectives
        X_Q = None
        for tokenIndex, token in enumerate(tokens):
            for conceptName in conceptNames:      
                if (conceptName, token) not in x:
                    continue
                           
                currentQElement = graphResultsForPhraseToken[conceptName][tokenIndex][1] * x[conceptName, token]

                X_Q += currentQElement
                #self.myLogger.debug("Created objective element %s"%(currentQElement))

                if ('Not_'+conceptName, token) in x: 
                    currentQElement = graphResultsForPhraseToken[conceptName][tokenIndex][0]*x['Not_'+conceptName, token]

                    X_Q += currentQElement
                    #self.myLogger.debug("Created objective element %s"%(currentQElement))

        return X_Q
     
    def addRelationsConstrains(self, m, conceptNames, tokens, x, y, graphResultsForPhraseRelation):
        if graphResultsForPhraseRelation is None:
            return None

        self.myLogger.info('Starting method addRelationsConstrains with graphResultsForPhraseToken')

        if graphResultsForPhraseRelation is not None:
            for relation in graphResultsForPhraseRelation:
                #self.myLogger.debug('graphResultsForPhraseRelation for relation \"%s\" \n%s'%(relation, np.column_stack( (["   "] + tokens, np.vstack((tokens, graphResultsForPhraseRelation[relation])))) ))
                pass
                
        relationNames = list(graphResultsForPhraseRelation)
            
        # Create variables for relation - token - token and negative variables
        for relationName in relationNames:            
            for token1Index, token1 in enumerate(tokens): 
                for token2Index, token2 in enumerate(tokens):
                    if token1 == token2:
                        continue
    
                    # Check if probability is NaN or if and has to be skipped
                    currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                    if self.valueToBeSkipped(currentProbability[1]):
                        self.myLogger.info("Probability is %f for relation %s and tokens %s %s - skipping it"%(currentProbability[1],relationName,token1,token2))
                        continue

                    # Create variable
                    y[relationName, token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(token1, relationName, token2))
                    #self.myLogger.debug("Probability for token %s in relation %s to token %s is %f"%(token1,relationName,token2, currentProbability[1]))
                    
                    # Check if probability is NaN or if and has to be created based on positive value
                    if self.valueToBeSkipped(currentProbability[0]):
                        currentProbability[0] = 1 - currentProbability[1]
                        self.myLogger.info("No ILP negative variable for relation %s and tokens %s %s - created based on positive value %f"%(relationName,token1,token2, currentProbability[0]))
                
                    # Create negative variable
                    if True: # ilpOntSolver.__negVarTrashhold:
                        y[relationName+'-neg', token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s_not_%s_%s"%(token1, relationName, token2))
                    else:
                        self.myLogger.info("No ILP negative variable for relation %s and tokens %s %s created"%(relationName,token1,token2))
                        
        # Add constraints forcing decision between variable and negative variables 
        for relationName in relationNames:
            for token1 in tokens: 
                for token2 in tokens:
                    if token2 == token1:
                        continue
                    
                    if (relationName+'-neg', token1, token2) in y: 
                        m.addConstr(y[relationName, token1, token2] + y[relationName+'-neg', token1, token2] == 1, name='c_%s_%s_%sselfDisjoint'%(token1, token2, relationName))
                        #self.myLogger.debug("Disjoint constrain between relation %s and not relation %s between tokens - %s %s == %i"%(relationName,relationName,token1,token2,1))

        m.update()
   
        self.myLogger.info("Created %i ilp variables for relations"%(len(y)))

        if hasattr(self, 'myOnto'): # --- Use Ontology as a source of constrains 
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
                                
                        for token1Index, token1 in enumerate(tokens): 
                            for token2Index, token2 in enumerate(tokens):
                                if token1 == token2:
                                    continue
    
                                if (domain.name, token1) not in x:
                                     continue
                                 
                                if (range.name, token2) not in x:
                                     continue
                                 
                                if (currentRelation.name, token1, token2) not in y:
                                     continue
                                    
                                self.myIlpBooleanProcessor.ifVar(m, y[currentRelation._name, token1, token2], x[domain._name, token1], onlyConstrains = True)
                                self.myIlpBooleanProcessor.ifVar(m, y[currentRelation._name, token1, token2], x[range._name, token2],  onlyConstrains = True)
                                
                        self.myLogger.info("Created - domain-range - constrains for relation \"%s\" for domain \"%s\" and range \"%s\""%(relationName,domain._name,range._name))
    
            # -- Add constraints based on property subProperty statements in ontology R subproperty of S - R(x, y) -> S(x, y)
            for relationName in graphResultsForPhraseRelation:
                currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                    
                if currentRelation is None:
                    continue   
                
                for superProperty in currentRelation.is_a:
                    if superProperty.name not in graphResultsForPhraseRelation:
                        continue
                    
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token1 == token2:
                                continue
        
                            if (superProperty.name, token1, token2) not in y:
                                continue
                            
                            self.myIlpBooleanProcessor.ifVar(m, y[relationName, token1, token2], y[superProperty.name, token1, token2], onlyConstrains = True)
                
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
                                
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token1 == token2:
                                continue
        
                            if (relationName, token1, token2) not in y:
                                continue
                            
                            if (equivalentProperty.name, token1, token2) not in y:
                                continue
                            
                            self.myIlpBooleanProcessor.andVar(m, y[relationName, token1, token2], y[equivalentProperty.name, token1, token2], onlyConstrains = True)
                                
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
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token1 == token2:
                                continue
     
                            if (relationName, token1, token2) not in y:
                                continue
                            
                            if (currentRelationInverse.name, token1, token2) not in y:
                                continue
                            
                            self.myIlpBooleanProcessor.ifVar(m, y[currentRelationInverse.name, token1, token2], y[relationName, token1, token2], onlyConstrains = True)
                                                            
            # -- Add constraints based on property functionalProperty statements in ontology - at most one P(x,y) for x
            for relationName in graphResultsForPhraseRelation:
                currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                    
                if currentRelation is None:
                    continue   
                
                functionalLinExpr =  LinExpr()
    
                if FunctionalProperty in currentRelation.is_a:
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token1 == token2:
                                continue
        
                            if (relationName, token1, token2) not in y:
                                continue
                            
                            functionalLinExpr += y[relationName, token1, token2]
                                    
                    if functionalLinExpr:
                        constrainName = 'c_%s_FunctionalProperty'%(relationName)
                        m.addConstr(functionalLinExpr <= 1, name=constrainName)
            
            # -- Add constraints based on property inverseFunctionaProperty statements in ontology - at most one P(x,y) for y
            for relationName in graphResultsForPhraseRelation:
                currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                    
                if currentRelation is None:
                    continue   
                
                if InverseFunctionalProperty in currentRelation.is_a:
                    for token1 in tokens: 
                        for token2 in tokens:
                            if token1 == token2:
                                continue
        
                            if (relationName, token1, token2) not in y:
                                continue
                            
                            functionalLinExpr += y[relationName, token2, token1]
        
                    if functionalLinExpr:
                        constrainName = 'c_%s_InverseFunctionalProperty'%(relationName)
                        m.addConstr(functionalLinExpr <= 1, name=constrainName)
            
            # -- Add constraints based on property reflexiveProperty statements in ontology - P(x,x)
            for relationName in graphResultsForPhraseRelation:
                currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                    
                if currentRelation is None:
                    continue   
                
                if ReflexiveProperty in currentRelation.is_a:
                    for tokenIndex, token in enumerate(tokens):
                        
                        if (relationName, token1, token2) not in y:
                            continue
                            
                        constrainName = 'c_%s_%s_ReflexiveProperty'%(token, relationName)
                        m.addConstr(y[relationName, token, token] == 1, name=constrainName)  
                            
            # -- Add constraints based on property irreflexiveProperty statements in ontology - not P(x,x)
            for relationName in graphResultsForPhraseRelation:
                currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                    
                if currentRelation is None:
                    continue   
                
                if IrreflexiveProperty in currentRelation.is_a:
                    for tokenIndex, token in enumerate(tokens):
                        
                        if (relationName, token1, token2) not in y:
                            continue
                        
                        constrainName = 'c_%s_%s_ReflexiveProperty'%(token, relationName)
                        m.addConstr(y[relationName, token, token] == 0, name=constrainName)  
                        
            # -- Add constraints based on property symetricProperty statements in ontology - R(x, y) -> R(y,x)
            for relationName in graphResultsForPhraseRelation:
                currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                    
                if currentRelation is None:
                    continue   
                
                if SymmetricProperty in currentRelation.is_a:
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token1 == token2:
                                continue
        
                            if (relationName, token1, token2) not in y:
                                continue
                            
                            constrainName = 'c_%s_%s_%s_SymmetricProperty'%(token1, token2, relationName)
                            m.addGenConstrIndicator(y[relationName, token1, token2], True, y[relationName, token2, token1] == 1)
            
            # -- Add constraints based on property asymetricProperty statements in ontology - not R(x, y) -> R(y,x)
            for relationName in graphResultsForPhraseRelation:
                currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                    
                if currentRelation is None:
                    continue   
                
                if AsymmetricProperty in currentRelation.is_a:
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token1 == token2:
                                continue
        
                            if (relationName, token1, token2) not in y:
                                continue
                            
                            constrainName = 'c_%s_%s_%s_AsymmetricProperty'%(token1, token2, relationName)
                            m.addGenConstrIndicator(y[relationName, token1, token2], True, y[relationName, token2, token1] == 0)  
                            
            # -- Add constraints based on property transitiveProperty statements in ontology - P(x,y) and P(y,z) - > P(x,z)
            for relationName in graphResultsForPhraseRelation:
                currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                    
                if currentRelation is None:
                    continue   
                
                if TransitiveProperty in currentRelation.is_a:
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token1 == token2:
                                continue
        
                            if (relationName, token1, token2) not in y:
                                continue
                            
                            constrainName = 'c_%s_%s_%s_TransitiveProperty'%(token1, token2, relationName)
                            #m.addGenConstrIndicator(y[relationName, token, token1], True, y[relationName, token1, token] == 1)  
                                   
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
        else: # ------ No Ontology
            relations = set()
            
            for graph in self.myGraph:
                for currentGraphConceptName in graph.concepts:
                    for relationName in graphResultsForPhraseRelation:
                        if relationName in graph.concepts:
                            relations.add(graph.concepts[relationName])
                            
                for subGraphKey in graph._objs:
                    subGraph = graph._objs[subGraphKey]
                    for relationName in graphResultsForPhraseRelation:
                        if relationName in subGraph.concepts:
                            relations.add(subGraph.concepts[relationName])
                            
            for relation in relations:
                for arg_id, rel in enumerate(relation.has_a()): 
                    # TODO: need to include indirect ones like sp_tr is a tr while tr has a lm
                    # A has_a B : A(x,y,...) <= B(x)
                    #for xy in candidates[rel.src]:
                    #x = xy[arg_id]
                    relationName = rel.src.name
                    conceptName = rel.dst.name
                    
                    for token1 in tokens: 
                        for token2 in tokens:
                            if token1 == token2:
                                continue
                            else:
                                if (relationName, token1, token2) not in y: 
                                    continue
                                
                                if arg_id is 0: # Domain
                                    if (conceptName, token1) not in x: 
                                        continue
                                    
                                    self.myIlpBooleanProcessor.ifVar(m, y[relationName, token1, token2], x[conceptName, token1], onlyConstrains = True)
                                    #self.myLogger.info("Created - domain - constrains for relation \"%s\" and \"%s\""%(y[relationName, token1, token2].VarName,x[conceptName, token1].VarName))
                                    
                                elif arg_id is 1: # Range
                                    if (conceptName, token2) not in x: 
                                        continue
                                
                                    self.myIlpBooleanProcessor.ifVar(m, y[relationName, token1, token2], x[conceptName, token2], onlyConstrains = True)
                                    #self.myLogger.info("Created - range - constrains for relation \"%s\" and \"%s\""%(y[relationName, token1, token2].VarName,x[conceptName, token2].VarName))
                                    
                                else:
                                    self.myLogger.warn("When creating domain-range constrains for relation \"%s\" for concept \"%s\" received more then two concepts"%(relationName,conceptName))

        m.update()
    
        # Add objectives
        Y_Q  = None
        for relationName in relationNames:
            for token1Index, token1 in enumerate(tokens):
                for token2Index, token2 in enumerate(tokens):
                    if token1 == token2 :
                        continue
    
                    if (relationName, token1, token2) not in y:
                        continue
                        
                    currentQElement = graphResultsForPhraseRelation[relationName][token1Index][token2Index][1]*y[relationName, token1, token2]

                    Y_Q += currentQElement
                    #self.myLogger.debug("Created objective element %s"%(currentQElement))

                    if (relationName+'-neg', token1, token2) in y: 
                        currentQElement = graphResultsForPhraseRelation[relationName][token1Index][token2Index][0]*y[relationName+'-neg', token1, token2]

                        Y_Q += currentQElement
                        #self.myLogger.debug("Created objective element %s"%(currentQElement))

        return Y_Q
    
    def addTripleRelationsConstrains(self, m, conceptNames, tokens, x, y, z, graphResultsForPhraseTripleRelation):
        if graphResultsForPhraseTripleRelation is None:
            return None

        self.myLogger.info('Starting method addTripleRelationsConstrains with graphResultsForPhraseTripleRelation')
        if graphResultsForPhraseTripleRelation is not None:
            for tripleRelation in graphResultsForPhraseTripleRelation:
                self.myLogger.debug('graphResultsForPhraseTripleRelation for relation \"%s"'%(tripleRelation))

                for token1Index, token1 in enumerate(tokens):
                    #self.myLogger.debug('for token \"%s \n%s"'%(token1, np.column_stack( (["   "] + tokens, np.vstack((tokens, graphResultsForPhraseTripleRelation[tripleRelation][token1Index]))))))
                    pass
                
        tripleRelationNames = list(graphResultsForPhraseTripleRelation)
            
        # Create variables for relation - token - token -token and negative variables
        for tripleRelationName in tripleRelationNames:            
            for token1Index, token1 in enumerate(tokens): 
                for token2Index, token2 in enumerate(tokens):
                    if token2 == token1:
                        continue
                        
                    for token3Index, token3 in enumerate(tokens):
                        if token3 == token2:
                            continue
                        
                        if token3 == token1:
                            continue
                        
                        # Check if probability is NaN or if and has to be skipped
                        currentProbability = graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index]
                        #self.myLogger.info("Probability is %f for relation %s and tokens %s %s %s"%(currentProbability[1],tripleRelationName,token1,token2,token3))

                        if self.valueToBeSkipped(currentProbability[1]):
                            self.myLogger.info("Probability is %f for relation %s and tokens %s %s %s - skipping it"%(currentProbability[1],tripleRelationName,token1,token2, token3))
                            continue

                        # Create variable
                        z[tripleRelationName, token1, token2, token3]=m.addVar(vtype=GRB.BINARY,name="z_%s_%s_%s_%s"%(tripleRelationName, token1, token2, token3))
                        #self.myLogger.debug("Created variable for relation %s between tokens %s %s %s, probability is %f"%(tripleRelationName,token1, token2, token3, currentProbability[1]))

                         # Check if probability is NaN or if and has to be created based on positive value
                        if self.valueToBeSkipped(currentProbability[0]):
                            currentProbability[0] = 1 - currentProbability[1]
                            self.myLogger.info("No ILP negative variable for relation %s and tokens %s %s %s - created based on positive value %f"%(tripleRelationName,token1,token2,token3, currentProbability[0]))
                        
                        # Create negative variable
                        if True: #ilpOntSolver.__negVarTrashhold:
                            z[tripleRelationName+'-neg', token1, token2, token3]=m.addVar(vtype=GRB.BINARY,name="y_%s_not_%s_%s_%s"%(tripleRelationName, token1, token2, token3))
                        else:
                            self.myLogger.info("No ILP negative variable for relation %s and tokens %s %s %s created"%(tripleRelationName,token1,token2,token3))
                            
        # Add constraints forcing decision between variable and negative variables 
        for tripleRelationName in tripleRelationNames:            
            for token1Index, token1 in enumerate(tokens): 
                for token2Index, token2 in enumerate(tokens):
                    if token2 == token1:
                        continue
                        
                    for token3Index, token3 in enumerate(tokens):
                        if token3 == token2:
                            continue
                        
                        if token3 == token1:
                            continue
                        
                        currentProbability = graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index]
                        if currentProbability[1] == 0:
                            continue
                        
                        if (tripleRelationName+'-neg', token1, token2, token3) in z: 
                            m.addConstr(z[tripleRelationName, token1, token2, token3] + z[tripleRelationName+'-neg', token1, token2, token3] == 1, name='c_%s_%s_%s_%sselfDisjoint'%(token1, token2, token3, tripleRelationName))
                            #self.myLogger.debug("Disjoint constrain between relation %s and not relation %s between tokens - %s %s %s == %i"%(tripleRelationName,tripleRelationName,token1,token2,token3,1))

        m.update()
    
        self.myLogger.info("Created %i ilp variables for triple relations"%(len(z)))

        if hasattr(self, 'myOnto'): # --- Use Ontology as a source of constrains 
            
            # -- Add constraints based on concept subClassOf statements in ontology - var1 -> var2
            for tripleRelationName in tripleRelationNames:
                
                tripleRelation = self.myOnto.search_one(iri = "*%s"%(tripleRelationName))
                    
                if tripleRelation is None :
                    continue
                    
                for ancestorTripleRelation in tripleRelation.ancestors(include_self = False) :
                    if ancestorTripleRelation.name not in tripleRelationNames:
                         continue
                                
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token2 == token1:
                                continue
                        
                            for token3Index, token3 in enumerate(tokens):
                                if token3 == token2:
                                    continue
                        
                                if token3 == token1:
                                    continue
                                
                                if (tripleRelationName, token1, token2, token3) not in z:
                                    continue
                                
                                if (ancestorTripleRelation.name, token1, token2, token3) not in z:
                                    continue
                        
                                self.myIlpBooleanProcessor.ifVar(m, z[tripleRelationName, token1, token2, token3], z[ancestorTripleRelation.name, token1, token2, token3], onlyConstrains = True)
                            
                    self.myLogger.info("Created - subClassOf - constrains between triple relations \"%s\" -> \"%s\""%(tripleRelationName,ancestorTripleRelation.name))
                    
            # -- Add constraints based on concept disjoint statements in ontology - not(and(var1, var2)) = nand(var1, var2)
            foundDisjoint = dict() # too eliminate duplicates
            for tripleRelationName in tripleRelationNames:
                
                tripleRelation = self.myOnto.search_one(iri = "*%s"%(tripleRelationName))
                    
                if tripleRelation is None :
                    continue
                                    
                for d in tripleRelation.disjoints():
                    disjointTriple = d.entities[1]._name
                        
                    if tripleRelationName == disjointTriple:
                        disjointTriple = d.entities[0]._name
                            
                        if tripleRelationName == disjointTriple:
                            continue
                            
                    if disjointTriple not in tripleRelationNames:
                         continue
                            
                    if tripleRelationName in foundDisjoint:
                        if disjointTriple in foundDisjoint[tripleRelationName]:
                            continue
                    
                    if disjointTriple in foundDisjoint:
                        if tripleRelationName in foundDisjoint[disjointTriple]:
                            continue
                                
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token2 == token1:
                                continue
                        
                            for token3Index, token3 in enumerate(tokens):
                                if token3 == token2:
                                    continue
                        
                                if token3 == token1:
                                    continue
                                
                                if (tripleRelationName, token1, token2, token3) not in z:
                                    continue
                                
                                if (disjointTriple, token1, token2, token3) not in z:
                                    continue
                        
                                self.myIlpBooleanProcessor.nandVar(m, z[tripleRelationName, token1, token2, token3], z[disjointTriple, token1, token2, token3], onlyConstrains = True)
                    if not (tripleRelationName in foundDisjoint):
                        foundDisjoint[tripleRelationName] = {disjointTriple}
                    else:
                        foundDisjoint[tripleRelationName].add(disjointTriple)
                               
                if tripleRelationName in foundDisjoint:
                    self.myLogger.info("Created - disjoint - constrains between triples \"%s\" and  %s"%(tripleRelationName,foundDisjoint[tripleRelationName]))
    
            # -- Add constraints based on triple relation ranges
            for tripleRelationName in graphResultsForPhraseTripleRelation:
                currentTripleRelation = self.myOnto.search_one(iri = "*%s"%(tripleRelationName))
                    
                if currentTripleRelation is None:
                    continue
        
                #self.myLogger.debug("Triple Relation \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentTripleRelation.name, currentTripleRelation))
                
                ancestorConcept = None
                for _ancestorConcept in currentTripleRelation.ancestors(include_self = True):
                    if _ancestorConcept.name == "Thing":
                         continue
                     
                    ancestorConcept = _ancestorConcept
                    break
                
                if ancestorConcept is None:
                    break
                
                tripleProperties = {}
                triplePropertiesRanges = {}    
                noTriplePropertiesRanges = 0 
                             
                for property in self.myOnto.object_properties():
                    _domain = property.domain
                    
                    if _domain is None:
                        break
                    
                    domain = _domain[0]._name
                    if domain is ancestorConcept.name:                    
                        for superProperty in property.is_a:
                            if superProperty is None:
                                continue
                            
                            if superProperty.name == "ObjectProperty":
                                continue
                             
                            if superProperty.name == 'first':
                                tripleProperties['1'] = property
                                _range = property.range
                    
                                if _range is None:
                                    break
                    
                                range = _range[0]._name
                                triplePropertiesRanges['1'] = range
                                noTriplePropertiesRanges = noTriplePropertiesRanges + 1
                            elif superProperty.name == 'second':
                                tripleProperties['2'] = property
                                _range = property.range
                    
                                if _range is None:
                                    break
                    
                                range = _range[0]._name
                                triplePropertiesRanges['2'] = range
                                noTriplePropertiesRanges = noTriplePropertiesRanges + 1
                            elif superProperty.name == 'third':
                                tripleProperties['3'] = property
                                _range = property.range
                    
                                if _range is None:
                                    break
                    
                                range = _range[0]._name
                                triplePropertiesRanges['3'] = range
                                noTriplePropertiesRanges = noTriplePropertiesRanges + 1
                                  
                if noTriplePropertiesRanges < 3:
                    self.myLogger.warning("Problem with creation of constrains for relation \"%s\" - not found its full definition %s"%(tripleRelationName,triplePropertiesRanges))
                    self.myLogger.warning("Abandon it - going to the next relation")
    
                    continue
                else:
                    self.myLogger.info("Found definition for relation \"%s\" - %s"%(tripleRelationName,triplePropertiesRanges))
    
                tripleConstrainsNo = 0
                for token1Index, token1 in enumerate(tokens): 
                    for token2Index, token2 in enumerate(tokens):
                        if token2 == token1:
                            continue
                            
                        for token3Index, token3 in enumerate(tokens):
                            if token3 == token2:
                                continue
                            
                            if token3 == token1:
                                continue
                            
                            if (tripleRelationName, token1, token2, token3) not in z: 
                                continue
                            
                            if ((triplePropertiesRanges['1'], token1) not in x) or ((triplePropertiesRanges['2'], token2) not in x)  or ((triplePropertiesRanges['3'], token3) not in x):
                                continue
                            
                            r1 = x[triplePropertiesRanges['1'], token1]
                            r2 = x[triplePropertiesRanges['2'], token2] 
                            r3 = x[triplePropertiesRanges['3'], token3]
                            rel = z[tripleRelationName, token1, token2, token3]
                            
                            currentConstrLinExprRange = r1 + r2 + r3 - 3 * rel
                            m.addConstr(currentConstrLinExprRange >= 0, name='c_triple_%s_%s_%s_%s'%(tripleRelationName, token1, token2, token3))
                                        
                            self.myLogger.debug("Created constrains for relation \"%s\" for tokens \"%s\", \"%s\", \"%s\""%(tripleRelationName,token1,token2,token3))
                            tripleConstrainsNo = tripleConstrainsNo+1
                
                self.myLogger.info("Created %i constrains for relation \"%s\""%(tripleConstrainsNo,tripleRelationName))
        else: # ------- No Ontology
            tripleRelations = set()
            
            for graph in self.myGraph:
                for currentGraphConceptName in graph.concepts:
                    for tripleRelationName in graphResultsForPhraseTripleRelation:
                        if tripleRelationName in graph.concepts:
                            tripleRelations.add(graph.concepts[tripleRelationName])
                            
                for subGraphKey in graph._objs:
                    subGraph = graph._objs[subGraphKey]
                    for tripleRelationName in graphResultsForPhraseTripleRelation:
                        if tripleRelationName in subGraph.concepts:
                            tripleRelations.add(subGraph.concepts[tripleRelationName])
                            
            for relation in tripleRelations:
                for arg_id, rel in enumerate(relation.has_a()): 
                    # TODO: need to include indirect ones like sp_tr is a tr while tr has a lm
                    # A has_a B : A(x,y,...) <= B(x)
                    #for xy in candidates[rel.src]:
                    #x = xy[arg_id]
                    tripleRelationName = rel.src.name
                    conceptName = rel.dst.name
                    
                    if arg_id > 2:
                        self.myLogger.warn("When creating triple relation constrains for relation \"%s\" for concept \"%s\" received more then three concepts"%(tripleRelationName,conceptName))
                        continue
                    
                    for triple in permutations(tokens, r=3):
                        if  (tripleRelationName, triple[0], triple[1], triple[2]) not in z:
                            continue
                        
                        if (conceptName, triple[arg_id]) not in x: 
                            continue
                            
                        self.myIlpBooleanProcessor.ifVar(m, z[tripleRelationName, triple[0], triple[1], triple[2]], x[conceptName, triple[arg_id]], onlyConstrains = True)
                        #self.myLogger.info("Created - domain - constrains for relation \"%s\" and \"%s\""%(y[tripleRelationName, token1, token2].VarName,x[conceptName, token1].VarName))
            
        m.update()

        # Add objectives
        Z_Q  = None
        for tripleRelationName in tripleRelationNames:
            for token1Index, token1 in enumerate(tokens): 
                for token2Index, token2 in enumerate(tokens):
                    if token2 == token1:
                        continue
                        
                    for token3Index, token3 in enumerate(tokens):
                        if token3 == token1:
                            continue
                        
                        if token3 == token2:
                            continue

                        if (tripleRelationName, token1, token2, token3) not in z: 
                            continue
                        
                        Z_Q += graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index][1]*z[tripleRelationName, token1, token2, token3]
    
                        if (tripleRelationName+'-neg', token1, token2, token3) in z: 
                            Z_Q += graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index][0]*z[tripleRelationName+'-neg', token1, token2, token3]

        return Z_Q
        
    def addLogicalConstrains(self, m, concepts, tokens, x, y, z, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation):
        self.myLogger.info('Starting method addLogicalConstrains')
        
        for graph in self.myGraph:
            for lcKey, lc in graph.logicalConstrains.items():
                if not lc.active:
                    continue
                    
                self.myLogger.info('Processing Logical Constrain %s - %s - %s'%(lc.lcName, lc, [str(e) for e in lc.e]))
                self._constructLogicalConstrains(lc, m, concepts, tokens, x, y, z, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation, headLC = True)
                
    def _constructLogicalConstrains(self, lc, m, concepts, tokens, x, y, z, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation, resultVariableName='Final', headLC = False):
        lcVariables = []
        
        for eIndex, e in enumerate(lc.e): 
            if isinstance(e, Concept) or isinstance(e, LogicalConstrain): 
                
                # Look on step ahead in the parsed logical constrain and get variables names (if present) after the current concept
                variablesNames = None
                if eIndex + 1 < len(lc.e):
                    if isinstance(lc.e[eIndex+1], tuple):
                        variablesNames = lc.e[eIndex+1]
                        
                if isinstance(e, Concept): # -- Concept or Relation
                    typeOfConcept, conceptTypes = self._typeOfConcept(e)
                    conceptName = e.name
                    
                    if not variablesNames:
                        #self.myLogger.info('Logical Constrain %s has no variables set for %s'%(lc.lcName,conceptName))
                        variablesNames = ()
                    
                    if typeOfConcept == 'concept':
                        if len(variablesNames) == 0:
                            variablesNames = ('x', )
                            
                        if len(variablesNames) > 1:
                            self.myLogger.warning('Logical Constrain %s has incorrect variables set %s for %s'%(lc.lcName,variablesNames,conceptName))
                            
                        conceptVariables = {}
                                    
                        for token in tokens:
                            if (conceptName, token) in x:
                                conceptVariables[(token, )] = x[(conceptName, token)]
                            else:
                                conceptVariables[(token, )] = None
   
                        _lcVariables = {}
                        _lcVariables[variablesNames] = conceptVariables
                        
                        lcVariables.append(_lcVariables)
                        
                    elif typeOfConcept == 'pair':
                        if len(variablesNames) == 0:
                            variablesNames = ('x', 'y')
                            
                        if len(variablesNames) > 2:
                            self.myLogger.warn('Logical Constrain %s has incorrect variables set %s for %s'%(lc.lcName,variablesNames,conceptName))
                            
                        conceptVariables = {}

                        for tokensPair in permutations(tokens, r=2):
                            if (conceptName, *tokensPair) in y:
                                conceptVariables[tokensPair] = y[(conceptName, *tokensPair)]
                            else:
                                conceptVariables[tokensPair] = None
                       
                        _lcVariables = {}
                        _lcVariables[variablesNames] = conceptVariables
                        
                        lcVariables.append(_lcVariables)
    
                    elif typeOfConcept == 'triplet':                        
                        _lcVariables = {}
                        
                        if len(variablesNames) == 0:
                            variablesNames = ('x', 'y', 'z')
                            
                        if len(variablesNames) > 3:
                            self.myLogger.warn('Logical Constrain %s has incorrect variables set %s for %s'%(lc.lcName,variablesNames,conceptName))

                        conceptVariables = {}

                        for tokensPermutation in permutations(tokens, r=3):                            
                            if (conceptName, *tokensPermutation) in z:
                                conceptVariables[tokensPermutation] = z[(conceptName, *tokensPermutation)]
                            else:
                                conceptVariables[tokensPermutation] = None
                                
                        _lcVariables = {}
                        _lcVariables[variablesNames] = conceptVariables
                        
                        lcVariables.append(_lcVariables)
                        
                elif isinstance(e, LogicalConstrain): # LogicalConstrain - process recursively
                    if not variablesNames:
                        #self.myLogger.info('Logical Constrain %s has no variables set for %s'%(lc.lcName,e.lcName))
                        variablesNames = ()
                        
                    if len(variablesNames) == 0:
                        variablesNames = ('variableName1', )
                            
                    if len(variablesNames) > 1:
                        self.myLogger.warning('Logical Constrain %s has incorrect variables set %s for %s'%(lc.lcName,variablesNames,e))
                    
                    lcVariables.append(self._constructLogicalConstrains(e, m, concepts, tokens, x, y, z, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation, resultVariableName = variablesNames[0]))
            elif isinstance(e, tuple): # tuple with named variable - skip for now
                pass # Already processed in the previous iteration
            else:
                self.myLogger.error('Logical Constrain %s has incorrect element %s'%(lcKey,e))

        return lc(m, self.myIlpBooleanProcessor, lcVariables, resultVariableName=resultVariableName, headConstrain = headLC)
                
    def _typeOfConcept (self, e):
        for is_a in e.is_a():
            is_a_dst = is_a.dst.name
            
            if is_a_dst is 'pair':
                pairConcepts = []
                for has_a in e.has_a():
                    pairConcepts.append(has_a.dst.name)
                return 'pair', pairConcepts
            elif is_a_dst is 'triplet':
                tripletConcepts = []
                for has_a in e.has_a():
                    tripletConcepts.append(has_a.dst.name)
                return 'triplet', tripletConcepts
        
        return 'concept', []
                
    def checkIContainNegativeProbability(self, concepts, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None):
        
        if graphResultsForPhraseToken:
            correctN = True
            for c in concepts:
                if len(graphResultsForPhraseToken[c].shape) < 2:
                    correctN = False
                break
            
            if not correctN:
                for c in concepts:
                    graphResultsForPhraseToken[c] = np.expand_dims(graphResultsForPhraseToken[c], axis=0)
                    graphResultsForPhraseToken[c] = np.concatenate((graphResultsForPhraseToken[c], graphResultsForPhraseToken[c]))
                    graphResultsForPhraseToken[c] = np.swapaxes(graphResultsForPhraseToken[c], 1, 0)
                    
                    for i in range(len(graphResultsForPhraseToken[c])):
                        graphResultsForPhraseToken[c][i][0] = 1 - graphResultsForPhraseToken[c][i][0]
                        
        if graphResultsForPhraseRelation:
            correctN = True
            relationNames = list(graphResultsForPhraseRelation)
            for c in relationNames:
                temp = np.asarray(graphResultsForPhraseRelation[c][0])
                if len(temp.shape) < 2:
                    correctN = False
                break
            
            if not correctN:
                for c in relationNames:
                    for i in range(len(graphResultsForPhraseRelation[c])):
                        ter1 = np.expand_dims(graphResultsForPhraseRelation[c][i], axis=0) # can also use [np.newaxis]
                        ter2 = np.concatenate((ter1, ter1)) 
                        ter3 = np.swapaxes(ter2, 1, 0)
                        
                        for j in range(len(graphResultsForPhraseRelation[c][i])):
                            ter3[j][0] = 1 - ter3[j][0]
                            
                        ter3 = np.expand_dims(ter3, axis=0)
                        if i == 0:
                            temp = ter3
                        else:
                            temp = np.append(temp, ter3, axis=0)    
                                
                    graphResultsForPhraseRelation[c] = temp
                    
        if graphResultsForPhraseTripleRelation:
            correctN = True
            tripleRelationNames = list(graphResultsForPhraseTripleRelation)

            for c in tripleRelationNames:
                temp = np.asarray(graphResultsForPhraseTripleRelation[c][0][0])
                if len(temp.shape) < 2:
                    correctN = False
                break
            
            if not correctN:
                for c in tripleRelationNames:
                    temp = None
                    for i in range(len(graphResultsForPhraseTripleRelation[c])):
                        temp1 = None
                        for j in range(len(graphResultsForPhraseTripleRelation[c][i])):
                            ter1 = np.expand_dims(graphResultsForPhraseTripleRelation[c][i][j], axis=0) # can also use [np.newaxis]
                            ter2 = np.concatenate((ter1, ter1)) 
                            ter3 = np.swapaxes(ter2, 1, 0)
                            
                            for k in range(len(graphResultsForPhraseTripleRelation[c][i][j])):
                                ter3[k][0] = 1 - ter3[k][0]
                                
                            ter3 = np.expand_dims(ter3, axis=0)
                            if j == 0:
                                temp1 = ter3
                            else:
                                temp1 = np.append(temp1, ter3, axis=0)    
                        
                        temp1 = np.expand_dims(temp1, axis=0)
                        if i == 0:
                            temp = temp1
                        else:
                            temp = np.append(temp, temp1, axis=0)
                                    
                    graphResultsForPhraseTripleRelation[c] = temp        
                    
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None):
    
        if self.ilpSolver == None:
            self.myLogger.warning('ILP solver not provided - returning unchanged results')
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation
        
        start = datetime.datetime.now()
        self.myLogger.info('Start for phrase %s'%(phrase))

        if graphResultsForPhraseToken is None:
            self.myLogger.warning('graphResultsForPhraseToken is None - returning unchanged results')
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation
              
        concepts = [k for k in graphResultsForPhraseToken.keys()]

        self.checkIContainNegativeProbability(concepts, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation)
        
        tokens = None
        if all(isinstance(item, tuple) for item in phrase):
            tokens = [x for x, _ in phrase]
        elif all((isinstance(item, string_types) or isinstance(item, int)) for item in phrase):
            tokens = phrase
        else:
            self.myLogger.warning('Phrase type is not supported %s - returning unchanged results'%(phrase))
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation

        try:
            # Create a new Gurobi model
            self.myIlpBooleanProcessor.resetCaches()
            m = Model("decideOnClassificationResult" + str(start))
            m.params.outputflag = 0
            
            # Variables for concept - token
            x={}
    
            # Variables for relation - token, token
            y={}
            
            # Variables for relation - token, token, token
            z={}
                
            # -- Set objective
            Q = None
            
            X_Q = self.addTokenConstrains(m, concepts, tokens, x, graphResultsForPhraseToken)
            if X_Q is not None:
                if Q is None:
                    Q = X_Q
                else:
                    Q += X_Q
            
            Y_Q = self.addRelationsConstrains(m, concepts, tokens, x, y, graphResultsForPhraseRelation)
            if Y_Q is not None:
                Q += Y_Q
                
            Z_Q = self.addTripleRelationsConstrains(m, concepts, tokens, x, y, z, graphResultsForPhraseTripleRelation)
            if Z_Q is not None:
                Q += Z_Q
            
            if not hasattr(self, 'myOnto'): # --- Not Using Ontology as a source of constrains 
                self.addLogicalConstrains(m, concepts, tokens, x, y, z, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation)

            m.setObjective(Q, GRB.MAXIMIZE)

            # Token is associated with a single concept
            #for token in tokens:
            #   constrainName = 'c_%s'%(token)
            #    m.addConstr(quicksum(x[conceptName, token] for conceptName in conceptNames) <= 1, name=constrainName)
            
            m.update()

            startOptimize = datetime.datetime.now()
            self.myLogger.info('Optimizing model with %i variables and %i constrains'%(m.NumVars, m. NumConstrs))

            m.optimize()
            
            endOptimize = datetime.datetime.now()
            elapsedOptimize = endOptimize - startOptimize

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

            # Collect results for tokens
            tokenResult = None
            if graphResultsForPhraseToken is not None:
                tokenResult = dict()
                conceptNames = [k for k in graphResultsForPhraseToken.keys()]

                if x or True:
                    if m.status == GRB.Status.OPTIMAL:
                        solution = m.getAttr('x', x)

                        self.myLogger.info('')
                        self.myLogger.info('---- Token Solutions ----')

                        for conceptName in conceptNames:
                            tokenResult[conceptName] = np.zeros(len(tokens))
                            
                            for tokenIndex, token in enumerate(tokens):
                                if ((conceptName, token) in solution) and (solution[conceptName, token] == 1):                                    
                                    tokenResult[conceptName][tokenIndex] = 1
                                    self.myLogger.info('\"%s\" is \"%s\"'%(token,conceptName))

            # Collect results for relations
            relationResult = None
            if graphResultsForPhraseRelation is not None: 
                relationResult = dict()
                relationNames = [k for k in graphResultsForPhraseRelation.keys()]
                
                if y or True:
                    if m.status == GRB.Status.OPTIMAL:
                        solution = m.getAttr('x', y)
                                                
                        self.myLogger.info('')
                        self.myLogger.info('---- Relation Solutions ----')

                        for relationName in relationNames:
                            relationResult[relationName] = np.zeros((len(tokens), len(tokens)))
                            
                            for token1Index, token1 in enumerate(tokens):
                                for token2Index, token2 in enumerate(tokens):
                                    if token2 == token1:
                                        continue
                                    
                                    if ((relationName, token1, token2) in solution) and (solution[relationName, token1, token2] == 1):
                                        relationResult[relationName][token1Index][token2Index] = 1
                                        
                                        self.myLogger.info('\"%s\" \"%s\" \"%s\"'%(token1,relationName,token2))
        
            # Collect results for triple relations
            tripleRelationResult = None
            if graphResultsForPhraseTripleRelation is not None:
                tripleRelationResult = {}
                tripleRelationNames = [k for k in graphResultsForPhraseTripleRelation.keys()]
                
                for tripleRelationName in tripleRelationNames:
                    tripleRelationResult[tripleRelationName] = np.zeros((len(tokens), len(tokens), len(tokens)))
                
                if z or True:
                    if m.status == GRB.Status.OPTIMAL:
                        solution = m.getAttr('x', z)
                        
                        self.myLogger.info('')
                        self.myLogger.info('---- Triple Relation Solutions ----')

                        for tripleRelationName in tripleRelationNames:
                            self.myLogger.info('Solutions for relation %s\n'%(tripleRelationName))

                            for token1Index, token1 in enumerate(tokens):
                                for token2Index, token2 in enumerate(tokens):
                                    if token1 == token2:
                                        continue
                                    
                                    for token3Index, token3 in enumerate(tokens):
                                        if token3 == token2:
                                            continue
                                        
                                        if token3 == token1:
                                            continue
                                    
                                        if ((tripleRelationName, token1, token2, token3) in solution) and (solution[tripleRelationName, token1, token2, token3] == 1):
                                            tripleRelationResult[tripleRelationName][token1Index, token2Index, token3Index] = 1
                                        
                                            self.myLogger.info('\"%s\" and \"%s\" and \"%s\" is in triple relation %s'%(token1,token2,token3,tripleRelationName))
        
        except:
            self.myLogger.error('Error returning solutions')
            raise
           
        end = datetime.datetime.now()
        elapsed = end - start
        self.myLogger.info('')
        self.myLogger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # Return results of ILP optimization
        return tokenResult, relationResult, tripleRelationResult