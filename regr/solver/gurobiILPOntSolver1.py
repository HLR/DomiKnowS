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
from regr.solver.ilpConfig import ilpConfig
from regr.solver.ilpOntSolver import ilpOntSolver
from regr.solver.gurobiILPBooleanMethods1 import gurobiILPBooleanProcessor
from regr.graph import LogicalConstrain, andL, orL, ifL, existsL, notL
from click.decorators import group

class gurobiILPOntSolver(ilpOntSolver):
    ilpSolver = 'Gurobi1'

    def __init__(self) -> None:
        super().__init__()
        self.myIlpBooleanProcessor = gurobiILPBooleanProcessor()
               
    def addTokenConstrains(self, m, conceptNames, tokens, x, graphResultsForPhraseToken):
        if graphResultsForPhraseToken is None:
            return None

        padding = max([len(str(t)) for t in tokens])
        spacing = max([len(str(c)) for c in conceptNames]) + 1
        # for concept, tokenTable in graphResultsForPhraseToken.items():
            # self.myLogger.debug("{:<{}}".format(concept, spacing) + ' '.join(map('{:^10f}'.format, [t for t in tokenTable])))

        # Create variables for token - concept and negative variables
        for conceptName in conceptNames: 
            for tokenIndex, token in enumerate(tokens):            
                currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                # Check if probability not zero
                if currentProbability[1] == 0:
                    continue
                # elif currentProbability[1] > 1:
                #     self.myLogger.error("Probability %f is above 1 for concept %s and token %s created"%(currentProbability,token, conceptName))
                #     continue

                # Create variable
                x[conceptName, token]=m.addVar(vtype=GRB.BINARY,name="x_%s_is_%s"%(token, conceptName))             
                #self.myLogger.debug("Created ILP variable for concept %s and token %s it's probability is %f"%(conceptName,token,currentProbability))

                # Create negative variable
                if True: # ilpOntSolver.__negVarTrashhold:
                    x['Not_'+conceptName, token]=m.addVar(vtype=GRB.BINARY,name="x_%s_is_not_%s"%(token, conceptName))

        # Add constraints forcing decision between variable and negative variables 
        for conceptName in conceptNames:
            for token in tokens:
                if ('Not_'+conceptName, token) in x:
                    currentConstrLinExpr = x[conceptName, token] + x['Not_'+conceptName, token]
                    m.addConstr(currentConstrLinExpr == 1, name='c_%s_%sselfDisjoint'%(conceptName, token))
                    #self.myLogger.debug("Disjoint constrain between token %s is concept %s and token %s is concept - %s == %i"%(token,conceptName,token,'Not_'+conceptName,1))
                    
        m.update()

        if len(x):
            pass
        else:
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
                    pass

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
                    pass
        
            # -- Add constraints based on concept subClassOf statements in ontology - var1 -> var2
            for conceptName in conceptNames :
                
                currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                    
                if currentConcept is None :
                    continue
                    
                for ancestorConcept in currentConcept.ancestors(include_self = False) :
                    if ancestorConcept.name not in conceptNames :
                         continue
                                
                    for tokenIndex, token in enumerate(tokens):
                        if (conceptName, token) not in x:
                            continue
                        
                        self.myIlpBooleanProcessor.ifVar(m, x[conceptName, token], x[ancestorConcept, token], onlyConstrains = True)
                            

    
            # -- Add constraints based on concept intersection statements in ontology - and(var1, var2, var3, ..)
            for conceptName in conceptNames :
                
                currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                    
                if currentConcept is None :
                    continue
                    
                for conceptConstruct in currentConcept.constructs(Prop = None) :
                    if type(conceptConstruct) is And :
                        
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
            for conceptName in conceptNames :
                
                currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                    
                if currentConcept is None :
                    continue
                    
                for conceptConstruct in currentConcept.constructs(Prop = None) :
                    if type(conceptConstruct) is Or :
                        
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
                    
                if currentConcept is None :
                    continue
                    
                for conceptConstruct in currentConcept.constructs(Prop = None) :
                    if type(conceptConstruct) is Not :
                        
                        complementClass = conceptConstruct.Class
    
                        for tokenIndex, token in enumerate(tokens):          
                            if (conceptName, token) not in x:
                                continue
                            
                            if (complementClass.name, token) not in x:
                                continue
                              
                            self.myIlpBooleanProcessor.xorVar(m, x[conceptName, token], x[complementClass.name, token], onlyConstrains = True)
    

                            
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
                    pass
            
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


        # if graphResultsForPhraseRelation is not None:
        #     for relation in graphResultsForPhraseRelation:
        #         self.myLogger.debug('graphResultsForPhraseRelation for relation \"%s\" \n%s'%(relation, np.column_stack( (["   "] + tokens, np.vstack((tokens, graphResultsForPhraseRelation[relation])))) ))
        #
        relationNames = list(graphResultsForPhraseRelation)
            
        # Create variables for relation - token - token and negative variables
        for relationName in relationNames:            
            for token1Index, token1 in enumerate(tokens): 
                for token2Index, token2 in enumerate(tokens):
                    if token1 == token2:
                        continue
    
                    # Check if probability not zero
                    currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                    # if currentProbability == 0:
                    #     continue
                    # elif currentProbability > 1:
                    #     self.myLogger.error("Probability %f is above 1 for relation %s and tokens %s %s created"%(currentProbability,relationName,token1,token2))
                    #     continue

                    # Create variable
                    y[relationName, token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(token1, relationName, token2))
                    #self.myLogger.debug("Probability for token %s in relation %s to token %s is %f"%(token1,relationName,token2, currentProbability))
                    
                    # Create negative variable
                    if True: # ilpOntSolver.__negVarTrashhold:
                        y[relationName+'-neg', token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s_not_%s_%s"%(token1, relationName, token2))
                    else:
                        pass
                        
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
                                   pass

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

        if graphResultsForPhraseTripleRelation is not None:
            for tripleRelation in graphResultsForPhraseTripleRelation:

                for token1Index, token1 in enumerate(tokens):
                    self.myLogger.debug('for token \"%s \n%s"'%(token1, np.column_stack( (["   "] + tokens, np.vstack((tokens, graphResultsForPhraseTripleRelation[tripleRelation][token1Index]))))))

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
                        
                        # Check if probability not zero
                        currentProbability = graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index]
                        #self.myLogger.info("Probability is %f for relation %s and tokens %s %s %s"%(currentProbability,tripleRelationName,token1,token2,token3))

                        if currentProbability == 0:
                            self.myLogger.debug("Probability is %f for relation %s and tokens %s %s %s - no variable created"%(currentProbability,tripleRelationName,token1,token2,token3))
                            continue
                        elif currentProbability > 1:
                            self.myLogger.error("Probability %f is above 1 for relation %s and tokens %s %s %s - no variable created"%(currentProbability,tripleRelationName,token1,token2,token3))
                            continue

                        # Create variable
                        z[tripleRelationName, token1, token2, token3]=m.addVar(vtype=GRB.BINARY,name="z_%s_%s_%s_%s"%(tripleRelationName, token1, token2, token3))
                        self.myLogger.debug("Created variable for relation %s between tokens %s %s %s, probability is %f"%(tripleRelationName,token1, token2, token3, currentProbability))

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
                        if currentProbability == 0:
                            continue
                        
                        if (tripleRelationName+'-neg', token1, token2, token3) in z: 
                            m.addConstr(z[tripleRelationName, token1, token2, token3] + z[tripleRelationName+'-neg', token1, token2, token3] == 1, name='c_%s_%s_%s_%sselfDisjoint'%(token1, token2, token3, tripleRelationName))
                            #self.myLogger.debug("Disjoint constrain between relation %s and not relation %s between tokens - %s %s %s == %i"%(tripleRelationName,tripleRelationName,token1,token2,token3,1))

        m.update()
    
        self.myLogger.info("Created %i ilp variables for triple relations"%(len(z)))

        if hasattr(self, 'myOnto'): # --- Use Ontology as a source of constrains 
            # -- Add constraints 
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
                        
                        Z_Q += graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index]*z[tripleRelationName, token1, token2, token3]
    
                        if (tripleRelationName+'-neg', token1, token2, token3) in z: 
                            Z_Q += (1-graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index])*z[tripleRelationName+'-neg', token1, token2, token3]

        return Z_Q
        
    def addLogicalConstrains(self, m, concepts, tokens, x, y, z, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation):

        for graph in self.myGraph:
            for lcKey, lc in graph.logicalConstrains.items():
                if not lc.active:
                    continue
                    
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
                            pass
                            
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
                            pass
                            
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
                            pass

                        conceptVariables = {}

                        for tokensPermutation in permutations(tokens, r=3):                            
                            if (conceptName, *tokensPair) in z:
                                conceptVariables[tokensPair] = z[(conceptName, *tokensPair)]
                            else:
                                conceptVariables[tokensPair] = None
                                
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
                        pass
                    
                    lcVariables.append(self._constructLogicalConstrains(e, m, concepts, tokens, x, y, z, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation, resultVariableName = variablesNames[0]))
            elif isinstance(e, tuple): # tuple with named variable - skip for now
                pass # Already processed in the previous iteration
            else:
                pass

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
                
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None):
    
        if self.ilpSolver == None:
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation
        
        start = datetime.datetime.now()

        if graphResultsForPhraseToken is None:
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation
              
        concepts = [k for k in graphResultsForPhraseToken.keys()]

        tokens = None
        if all(isinstance(item, tuple) for item in phrase):
            tokens = [x for x, _ in phrase]
        elif all((isinstance(item, string_types) or isinstance(item, int)) for item in phrase):
            tokens = phrase
        else:
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

            m.optimize()
            
            endOptimize = datetime.datetime.now()
            elapsedOptimize = endOptimize - startOptimize

            if m.status == GRB.Status.OPTIMAL:
                pass
            elif m.status == GRB.Status.INFEASIBLE:
                 pass
            elif m.status == GRB.Status.INF_OR_UNBD:
                 pass
            elif m.status == GRB.Status.UNBOUNDED:
                 pass
            else:
                 pass

            # Collect results for tokens
            tokenResult = None
            if graphResultsForPhraseToken is not None:
                tokenResult = dict()
                conceptNames = [k for k in graphResultsForPhraseToken.keys()]

                if x or True:
                    if m.status == GRB.Status.OPTIMAL:
                        solution = m.getAttr('x', x)


                        for conceptName in conceptNames:
                            tokenResult[conceptName] = np.zeros(len(tokens))
                            
                            for tokenIndex, token in enumerate(tokens):
                                if ((conceptName, token) in solution) and (solution[conceptName, token] == 1):                                    
                                    tokenResult[conceptName][tokenIndex] = 1

            # Collect results for relations
            relationResult = None
            if graphResultsForPhraseRelation is not None: 
                relationResult = dict()
                relationNames = [k for k in graphResultsForPhraseRelation.keys()]
                
                if y or True:
                    if m.status == GRB.Status.OPTIMAL:
                        solution = m.getAttr('x', y)
                                                

                        for relationName in relationNames:
                            relationResult[relationName] = np.zeros((len(tokens), len(tokens)))
                            
                            for token1Index, token1 in enumerate(tokens):
                                for token2Index, token2 in enumerate(tokens):
                                    if token2 == token1:
                                        continue
                                    
                                    if ((relationName, token1, token2) in solution) and (solution[relationName, token1, token2] == 1):
                                        relationResult[relationName][token1Index][token2Index] = 1
                                        

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
                        

                        for tripleRelationName in tripleRelationNames:

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
                                        

        except:
            self.myLogger.error('Error returning solutions')
            raise
           
        end = datetime.datetime.now()
        elapsed = end - start
        self.myLogger.info('')
        self.myLogger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # Return results of ILP optimization
        return tokenResult, relationResult, tripleRelationResult