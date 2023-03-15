# numpy
import numpy as np

# ontology
from owlready2 import *

from gekko import GEKKO

if __package__ is None or __package__ == '': 
    from domiknows.solver.ilpConfig import ilpConfig
    from domiknows.solver.ilpOntSolver import ilpOntSolver
 
    from domiknows.solver.gekkoILPBooleanMethods import gekkoILPBooleanProcessor
else:
    from .ilpConfig import ilpConfig 
    from .ilpOntSolver import ilpOntSolver

    from .gekkoILPBooleanMethods import gekkoILPBooleanProcessor

import logging
import datetime

class gekkoILPOntSolver(ilpOntSolver):
    ilpSolver = 'GEKKO'

    def __init__(self, graph, ontologiesTuple, _ilpConfig=ilpConfig) -> None:
        super().__init__(graph, ontologiesTuple, _ilpConfig=ilpConfig)
        self.myIlpBooleanProcessor = gekkoILPBooleanProcessor()
    
    def addTokenConstrains(self, m, tokens, conceptNames, x, graphResultsForPhraseToken):
        if graphResultsForPhraseToken is None:
            return None
        
        # Create variables for token - concept and negative variables
        for tokenIndex, token in enumerate(tokens):            
            for conceptName in conceptNames: 
                x[token, conceptName]=m.Var(0, lb=0, ub=1, integer=True, name="x_%s_%s"%(token, conceptName))
                #it = np.nditer(graphResultsForPhraseToken[conceptName], flags=['c_index', 'multi_index'])
                
                currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                self.myLogger.info("Probability for concept %s and token %s is %f"%(conceptName,token,currentProbability))

                if currentProbability < 1.0: #ilpOntSolver.__negVarTrashhold:
                    x[token, conceptName]=m.Var(0, lb=0, ub=1, integer=True, name="x_%s_%s"%(token, conceptName+'-neg'))

        # Add constraints forcing decision between variable and negative variables 
        for conceptName in conceptNames:
            for token in tokens:
                if (token, conceptName+'-neg') in x:
                    m.Equation(x[token, conceptName] + x[token, conceptName+'-neg'] <= 1)
         
        self.myLogger.info("Created %i ILP variables for tokens"%(len(x)))
        
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
                        m.Equation(self.myIlpBooleanProcessor.xorVar(m, x[token, conceptName], x[token, complementClass.name]) >= 1)

                    self.myLogger.info("Created - objectComplementOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,complementClass.name))
                        
        # ---- No supported yet
    
        # -- Add constraints based on concept disjonitUnion statements in ontology -  Not supported by owlready2 yet
        
        # -- Add constraints based on concept oneOf statements in ontology - ?
        
        # Add objectives
        X_Q = None
        for tokenIndex, token in enumerate(tokens):
            for conceptName in conceptNames:
                if X_Q is None:
                    X_Q = graphResultsForPhraseToken[conceptName][tokenIndex]*x[token, conceptName]
                else:
                    X_Q += graphResultsForPhraseToken[conceptName][tokenIndex]*x[token, conceptName]

                if (token, conceptName+'-neg') in x: 
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
    
                    y[relationName, token1, token2]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s_%s_%s"%(relationName, token1, token2))

                    if graphResultsForPhraseRelation[relationName][token2Index][token1Index] < 1.0: #ilpOntSolver.__negVarTrashhold:
                        y[relationName+'-neg', token1, token2]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s-neg_%s_%s"%(relationName, token1, token2))

        # Add constraints forcing decision between variable and negative variables 
        for relationName in relationNames:
            for token in tokens: 
                for token1 in tokens:
                    if token == token1:
                        continue
                    
                    if (relationName+'-neg', token, token1) in y: 
                        m.Equation(y[relationName, token, token1] + y[relationName+'-neg', token, token1] <= 1)
                                
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
                        
                        m.Equation(self.myIlpBooleanProcessor.ifVar(m, y[equivalentProperty.name, token, token1], y[relationName, token, token1]) >= 1)
  
        # -- Add constraints based on property functionalProperty statements in ontology - at most one P(x,y) for x
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
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
    
        # Add objectives
        Y_Q  = None
        for relationName in relationNames:
            for token1Index, token1 in enumerate(tokens):
                for token2Index, token2 in enumerate(tokens):
                    if token1 == token2 :
                        continue
    
                    if Y_Q is None:
                        Y_Q = graphResultsForPhraseRelation[relationName][token2Index][token1Index]*y[relationName, token1, token2]
                    else:
                        Y_Q += graphResultsForPhraseRelation[relationName][token2Index][token1Index]*y[relationName, token1, token2]

                    if (relationName+'-neg', token, token1) in y: 
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
                    
                        z[tripleRelationName, token, token1, token2]=m.Var(0, lb=0, ub=1, integer=True, name="y_%s_%s_%s_%s"%(tripleRelationName, token, token1, token2))

    
                        if graphResultsForPhraseTripleRelation[tripleRelationName][token2Index][token1Index][tokenIndex] < ilpOntSolver.__negVarTrashhold:
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
                            m.Equation(z[tripleRelationName, token, token1, token2] + z[tripleRelationName+'-neg', token, token1, token2] <= 1)
    
        self.myLogger.info("Created %i ilp variables for triple relations"%(len(z)))

        # -- Add constraints 
        for tripleRelationName in graphResultsForPhraseTripleRelation:
            currentTripleRelation = self.myOnto.search_one(iri = "*%s"%(tripleRelationName))
                
            if currentTripleRelation is None:
                continue
    
            self.myLogger.debug("Triple Relation \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentTripleRelation.name, currentTripleRelation))
            
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
                     
                        # TODO
                                    
                        self.myLogger.info("Created - triple - constrains for relation \"%s\" for tokens \"%s\", \"%s\", \"%s\""%(tripleRelationName,token1,token2,token3))
    
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

                        if Z_Q is None:
                            Z_Q = graphResultsForPhraseTripleRelation[tripleRelationName][token3Index][token2Index][token1Index]*z[tripleRelationName, token1, token2, token3]
                        else:
                            Z_Q += graphResultsForPhraseTripleRelation[tripleRelationName][token3Index][token2Index][token1Index]*z[tripleRelationName, token1, token2, token3]
    
                        if (tripleRelationName+'-neg', token1, token2, token3) in z: 
                            if Z_Q is None:
                                Z_Q = (1-graphResultsForPhraseTripleRelation[tripleRelationName][token3Index][token2Index][token1Index])*z[tripleRelationName+'-neg', token1, token2, token3]
                            else:
                                Z_Q += (1-graphResultsForPhraseTripleRelation[tripleRelationName][token3Index][token2Index][token1Index])*z[tripleRelationName+'-neg', token1, token2, token3]
            
        return Z_Q
        
    def calculateILPSelection(self, phrase, fun=None, epsilon = 0.00001, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None):
        start = datetime.datetime.now()
        self.myLogger.info('Start for phrase %s'%(phrase))

        if graphResultsForPhraseRelation is not None:
            for relation in graphResultsForPhraseRelation:
                self.myLogger.info('graphResultsForPhraseRelation for relation \"%s\" \n%s'%(relation, graphResultsForPhraseRelation[relation]))
        
        if graphResultsForPhraseTripleRelation is not None:
            for tripleRelation in graphResultsForPhraseTripleRelation:
                self.myLogger.info('graphResultsForPhraseTripleRelation for relation \"%s\" \n%s'%(tripleRelation, graphResultsForPhraseTripleRelation[tripleRelation]))

        conceptNames = graphResultsForPhraseToken.keys()
        tokens = [x for x, _ in phrase]
                    
        try:
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
            
            m.Obj((-1) * Q)

            # Token is associated with a single concept
            #for token in tokens:
            #   constrainName = 'c_%s'%(token)
            #    m.addConstr(quicksum(x[token, conceptName] for conceptName in conceptNames), GRB.LESS_EQUAL, 1, name=constrainName)
            
            startOptimize = datetime.datetime.now()

            m.solve(disp=solverDisp)

            endOptimize = datetime.datetime.now()
            elapsedOptimize = endOptimize - startOptimize

            self.myLogger.info('Optimal solution was found - elapsed time: %ims'%(elapsedOptimize.microseconds/1000))

            # Collect results for tokens
            tokenResult = None
            if graphResultsForPhraseToken is not None:
                tokenResult = dict()
                
                for conceptName in conceptNames:
                    tokenResult[conceptName] = np.zeros(len(tokens))
                            
                    for tokenIndex, token in enumerate(tokens):             
                        if x[token, conceptName].value[0] > gekkoTresholdForTruth:
                            tokenResult[conceptName][tokenIndex] = 1
                            self.myLogger.info('Solution \"%s\" is \"%s\"'%(token,conceptName))

            # Collect results for relations
            relationResult = None
            if graphResultsForPhraseRelation is not None: 
                relationResult = dict()
                relationNames = graphResultsForPhraseRelation.keys()
                
                for relationName in relationNames:                        
                    for token1Index, token1 in enumerate(tokens):
                        for token2Index, token2 in enumerate(tokens):
                            if token2 == token1:
                                continue
                            
                            if y[relationName, token1, token2].value[0] > gekkoTresholdForTruth:
                                relationResult[relationName][token2Index][token1Index] = 1
                                
                                self.myLogger.info('Solution \"%s\" is in relation \"%s\" with \"%s\"'%(token2,relationName,token1))
                        
            # Collect results for triple relations
            tripleRelationResult = None
            if graphResultsForPhraseTripleRelation is not None:
                tripleRelationResult = {}
                tripleRelationNames = graphResultsForPhraseTripleRelation.keys()
                
                for tripleRelationName in tripleRelationNames:
                    tripleRelationResult[tripleRelationName] = np.zeros((len(tokens), len(tokens), len(tokens)))
                
                pass # TODO
        
        except:
            self.myLogger.error('Error')
            raise
           
        end = datetime.datetime.now()
        elapsed = end - start
        self.myLogger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # return results of ILP optimization
        return tokenResult, relationResult, tripleRelationResult