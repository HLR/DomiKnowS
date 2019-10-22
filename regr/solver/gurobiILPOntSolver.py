from six import string_types

# numpy
import numpy as np

# ontology
from owlready2 import *

# Gurobi
from gurobipy import *
from examples.emr.emr.graph import phrase

if __package__ is None or __package__ == '': 
    from regr.solver.ilpConfig import ilpConfig
    from regr.solver.ilpOntSolver import ilpOntSolver
 
    from regr.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
else:
    from .ilpConfig import ilpConfig 
    from .ilpOntSolver import ilpOntSolver

    from .gurobiILPBooleanMethods import gurobiILPBooleanProcessor
    
import logging
import datetime

class gurobiILPOntSolver(ilpOntSolver):
    def __init__(self) -> None:
        super().__init__()
        self.myIlpBooleanProcessor = gurobiILPBooleanProcessor()
               
    def addTokenConstrains(self, m, tokens, conceptNames, x, graphResultsForPhraseToken):
        if graphResultsForPhraseToken is None:
            return None
        
        self.myLogger.info('Starting method addTokenConstrains')
        self.myLogger.info('graphResultsForPhraseToken')
        padding = max([len(str(x)) for x in tokens])
        spacing = max([len(str(x)) for x in conceptNames]) + 1
        self.myLogger.info("{:^{}}".format("", spacing) + ' '.join(map('{:^10}'.format, ['\''+ str(t) + '\'' for t in tokens])))
        for concept, tokenTable in graphResultsForPhraseToken.items():
            self.myLogger.info("{:<{}}".format(concept, spacing) + ' '.join(map('{:^10f}'.format, [t for t in tokenTable])))

        # Create variables for token - concept and negative variables
        for tokenIndex, token in enumerate(tokens):            
            for conceptName in conceptNames: 
                currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                if currentProbability == 0:
                    continue

                x[token, conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_is_%s"%(token, conceptName))
                                
                self.myLogger.info("Probability for concept %s and token %s is %f"%(conceptName,token,currentProbability))

                if currentProbability < 1.0: # ilpOntSolver.__negVarTrashhold:
                    x[token, 'Not_'+conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_is_not_%s"%(token, conceptName))

        # Add constraints forcing decision between variable and negative variables 
        for conceptName in conceptNames:
            for token in tokens:
                if (token, 'Not_'+conceptName) in x:
                    constrainName = 'c_%s_%sselfDisjoint'%(token, conceptName)  
                    currentConstrLinExpr = x[token, conceptName] + x[token, 'Not_'+conceptName]
                    m.addConstr(currentConstrLinExpr, GRB.LESS_EQUAL, 1, name=constrainName)
                    self.myLogger.info("Disjoint constrain between token %s is concept %s and token %s is concept %s - %s %i"%(token,conceptName,token,'Not_'+conceptName,GRB.LESS_EQUAL,1))
                    
        m.update()

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
                            
                for tokenIndex, token in enumerate(tokens):
                    currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                    if currentProbability == 0:
                        continue
                
                    currentConstrName = 'c_%s_%s_Disjoint_%s'%(token, conceptName, disjointConcept)
                        
                    # Version of the disjoint constrain using logical function library
                    #m.addConstr(self.myIlpBooleanProcessor.nandVar(m, x[token, conceptName], x[token, disjointConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)

                    # Short version ensuring that logical expression is SATISFY - no generating variable holding the result of evaluating the expression
                    currentConstrLinExpr = x[token, conceptName] + x[token, disjointConcept]
                    m.addConstr(currentConstrLinExpr, GRB.LESS_EQUAL, 1, name=currentConstrName)
                    self.myLogger.info("Disjoint constrain between concept \"%s\" and concept %s - %s %s %i"%(conceptName,disjointConcept,currentConstrLinExpr,GRB.LESS_EQUAL,1))
                               
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
                    currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                    if currentProbability == 0:
                        continue
                    
                    constrainName = 'c_%s_%s_Equivalent_%s'%(token, conceptName, equivalentConcept.name)
                    m.addConstr(self.myIlpBooleanProcessor.andVar(m, x[token, conceptName], x[token, equivalentConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)

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
                            
                for tokenIndex, token in enumerate(tokens):
                    currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                    if currentProbability == 0:
                        continue
                    
                    constrainName = 'c_%s_%s_Ancestor_%s'%(token, currentConcept, ancestorConcept.name)
                    m.addConstr(self.myIlpBooleanProcessor.ifVar(m, x[token, conceptName], x[token, ancestorConcept]), GRB.GREATER_EQUAL, 1, name=constrainName)

                self.myLogger.info("Created - subClassOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,ancestorConcept.name))

        # -- Add constraints based on concept intersection statements in ontology - and(var1, var2, var3, ..)
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is And :
                    
                    for tokenIndex, token in enumerate(tokens):
                        currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                        if currentProbability == 0:
                            continue
                        
                        _varAnd = m.addVar(name="andVar_%s"%(constrainName))
        
                        andList = []
                    
                        for currentClass in conceptConstruct.Classes :
                            andList.append(x[token, currentClass.name])
    
                        andList.append(x[token, conceptName])
                        
                        constrainName = 'c_%s_%s_Intersection'%(token, conceptName)
                        m.addConstr(self.myIlpBooleanProcessor.andVar(m, andList), GRB.GREATER_EQUAL, 1, name=constrainName)
                        
        # -- Add constraints based on concept union statements in ontology -  or(var1, var2, var3, ..)
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is Or :
                    
                    for tokenIndex, token in enumerate(tokens):
                        currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                        if currentProbability == 0:
                            continue
                    
                        _varOr = m.addVar(name="orVar_%s"%(constrainName))
        
                        orList = []
                    
                        for currentClass in conceptConstruct.Classes :
                            orList.append(x[token, currentClass.name])
    
                        orList.append(x[token, conceptName])
                        
                        constrainName = 'c_%s_%s_Union'%(token, conceptName)
                        m.addConstr(self.myIlpBooleanProcessor.orVar(m, orList), GRB.GREATER_EQUAL, 1, name=constrainName)
        
        # -- Add constraints based on concept objectComplementOf statements in ontology - xor(var1, var2)
        for conceptName in conceptNames :
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for conceptConstruct in currentConcept.constructs(Prop = None) :
                if type(conceptConstruct) is Not :
                    
                    complementClass = conceptConstruct.Class

                    for tokenIndex, token in enumerate(tokens):
                        currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                        if currentProbability == 0:
                            continue   
                            
                        constrainName = 'c_%s_%s_ComplementOf_%s'%(token, conceptName, complementClass.name) 
                        m.addConstr(self.myIlpBooleanProcessor.xorVar(m, x[token, conceptName], x[token, complementClass.name]), GRB.GREATER_EQUAL, 1, name=constrainName)

                    self.myLogger.info("Created - objectComplementOf - constrains between concept \"%s\" and concept \"%s\""%(conceptName,complementClass.name))
                        
        # ---- No supported yet
    
        # -- Add constraints based on concept disjonitUnion statements in ontology -  Not supported by owlready2 yet
        
        # -- Add constraints based on concept oneOf statements in ontology - ?
    
        m.update()

        # Add objectives
        X_Q = None
        for tokenIndex, token in enumerate(tokens):
            for conceptName in conceptNames:
                
                currentProbability = graphResultsForPhraseToken[conceptName][tokenIndex]
                
                if currentProbability == 0:
                     continue   
                        
                currentQElement =  graphResultsForPhraseToken[conceptName][tokenIndex]*x[token, conceptName]
                self.myLogger.info("Created objective element %s"%(currentQElement))
                X_Q += currentQElement

                if (token, 'Not_'+conceptName) in x: 
                    currentQElement = (1-graphResultsForPhraseToken[conceptName][tokenIndex])*x[token, 'Not_'+conceptName]
                    self.myLogger.info("Created objective element %s"%(currentQElement))
                    X_Q += currentQElement

        return X_Q
     
    def addRelationsConstrains(self, m, tokens, conceptNames, x, y, graphResultsForPhraseRelation):
        if graphResultsForPhraseRelation is None:
            return None
        
        self.myLogger.info('Starting method addRelationsConstrains with graphResultsForPhraseToken')

        if graphResultsForPhraseRelation is not None:
            for relation in graphResultsForPhraseRelation:
                self.myLogger.info('graphResultsForPhraseRelation for relation \"%s\" \n%s'%(relation, np.column_stack( (["   "] + tokens, np.vstack((tokens, graphResultsForPhraseRelation[relation])))) ))
                
        relationNames = list(graphResultsForPhraseRelation)
            
        # Create variables for relation - token - token and negative variables
        for relationName in relationNames:            
            for token1Index, token1 in enumerate(tokens): 
                for token2Index, token2 in enumerate(tokens):
                    if token1 == token2:
                        continue
    
                    currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                    if currentProbability == 0:
                        continue   

                    y[relationName, token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(token1, relationName, token2))

                    self.myLogger.info("Probability for token %s in relation %s to token %s is %f"%(token1,relationName,token2, currentProbability))
                    
                    if currentProbability < 1.0: # ilpOntSolver.__negVarTrashhold:
                        y[relationName+'-neg', token1, token2]=m.addVar(vtype=GRB.BINARY,name="y_%s_not_%s_%s"%(token1, relationName, token2))
                        self.myLogger.info("Disjoint constrain between relation %s and not relation %s between tokens %s %s  - %s %i"%(relationName,relationName,token1,token2,GRB.LESS_EQUAL,1))

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
                            
                    for token1Index, token1 in enumerate(tokens): 
                        for token2Index, token2 in enumerate(tokens):
                            if token1 == token2:
                                continue
    
                            currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                            if currentProbability == 0:
                                continue   
                             
                            currentConstrNameDomain = 'c_domain_%s_%s_%s'%(currentRelation, token1, token2)
                            currentConstrNameRange = 'c_range_%s_%s_%s'%(currentRelation, token1, token2)
                                
                            # Version of the domain and range constrains using logical function library
                            #m.addConstr(self.myIlpBooleanProcessor.ifVar(m, y[currentRelation._name, token1, token2], x[token1, domain._name]), GRB.GREATER_EQUAL, 1, name=constrainNameDomain)
                            #m.addConstr(self.myIlpBooleanProcessor.ifVar(m, y[currentRelation._name, token1, token2], x[token2, range._name]), GRB.GREATER_EQUAL, 1, name=constrainNameRange)
                                
                            # Short version ensuring that logical expression is SATISFY - no generating variable holding the result of evaluating the expression
                            currentConstrLinExprDomain = x[token1, domain._name] - y[currentRelation._name, token1, token2]
                            m.addConstr(currentConstrLinExprDomain, GRB.GREATER_EQUAL, 0, name=currentConstrNameDomain)
                            self.myLogger.info("Domain constrain between relation \"%s\" and domain %s - %s %s %i"%(relationName,domain._name,currentConstrLinExprDomain,GRB.GREATER_EQUAL,0))

                            currentConstrLinExprRange = x[token2, range._name] - y[currentRelation._name, token1, token2]
                            m.addConstr(currentConstrLinExprRange, GRB.GREATER_EQUAL, 0, name=currentConstrNameRange)
                            self.myLogger.info("Range constrain between relation \"%s\" and range %s - %s %s %i"%(relationName,range._name,currentConstrLinExprRange,GRB.GREATER_EQUAL,0))
                                
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
    
                        currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                        if currentProbability == 0:
                            continue
                        
                        constrainName = 'c_%s_%s_%s_SuperProperty_%s'%(token1, token2, relationName, superProperty.name)
                            
                        #m.addConstr(self.myIlpBooleanProcessor.ifVar(m, y[relationName, token1, token2], y[superProperty.name, token1, token2]), GRB.GREATER_EQUAL, 1, name=constrainName)
                        m.addConstr(y[superProperty.name, token1, token2] - y[relationName, token1, token2], GRB.GREATER_EQUAL, 0, name=constrainName)
            
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
    
                        currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                        if currentProbability == 0:
                            continue
                        
                        constrainName = 'c_%s_%s_%s_EquivalentProperty_%s'%(token1, token2, relationName, equivalentProperty.name)
                        m.addConstr(self.myIlpBooleanProcessor.andVar(m, y[relationName, token1, token2], y[equivalentProperty.name, token1, token2]), GRB.GREATER_EQUAL, 1, name=constrainName)
                            
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
    
                        currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                        if currentProbability == 0:
                            continue
                        
                        constrainName = 'c_%s_%s_%s_InverseProperty'%(token1, token2, relationName)
                        m.addGenConstrIndicator(y[relationName, token1, token2], True, y[currentRelationInverse.name, token2, token1], GRB.EQUAL, 1)
                        m.addConstr(self.myIlpBooleanProcessor.ifVar(m, y[equivalentProperty.name, token1, token2], y[relationName, token1, token2]), GRB.GREATER_EQUAL, 1, name=constrainName)
                            
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
    
                        currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                        if currentProbability == 0:
                            continue
                        
                        functionalLinExpr += y[relationName, token1, token2]
                
                constrainName = 'c_%s_%s_%s_FunctionalProperty'%(token1, token2, relationName)
                m.addConstr(functionalLinExpr, GRB.LESS_EQUAL, 1, name=constrainName)
        
        # -- Add constraints based on property inverseFunctionaProperty statements in ontology - at most one P(x,y) for y
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if InverseFunctionalProperty in currentRelation.is_a:
                for token1Index, token1 in enumerate(tokens): 
                    for token2Index, token2 in enumerate(tokens):
                        if token1 == token2:
                            continue
    
                        currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                        if currentProbability == 0:
                            continue
                        
                        functionalLinExpr += y[relationName, token2, token1]
    
                constrainName = 'c_%s_%s_%s_InverseFunctionalProperty'%(token1, token2, relationName)
                m.addConstr(functionalLinExpr, GRB.LESS_EQUAL, 1, name=constrainName)
        
        # -- Add constraints based on property reflexiveProperty statements in ontology - P(x,x)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if ReflexiveProperty in currentRelation.is_a:
                for tokenIndex, token in enumerate(tokens):
                    
                    currentProbability = graphResultsForPhraseRelation[relationName][tokenIndex][tokenIndex]
                    if currentProbability == 0:
                        continue
                        
                    constrainName = 'c_%s_%s_ReflexiveProperty'%(token, relationName)
                    m.addConstr(y[relationName, token, token], GRB.EQUAL, 1, name=constrainName)  
                        
        # -- Add constraints based on property irreflexiveProperty statements in ontology - not P(x,x)
        for relationName in graphResultsForPhraseRelation:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue   
            
            if IrreflexiveProperty in currentRelation.is_a:
                for tokenIndex, token in enumerate(tokens):
                    
                    currentProbability = graphResultsForPhraseRelation[relationName][tokenIndex][tokenIndex]
                    if currentProbability == 0:
                        continue
                    
                    constrainName = 'c_%s_%s_ReflexiveProperty'%(token, relationName)
                    m.addConstr(y[relationName, token, token], GRB.EQUAL, 0, name=constrainName)  
                    
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
    
                        currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                        if currentProbability == 0:
                            continue
                        
                        constrainName = 'c_%s_%s_%s_SymmetricProperty'%(token1, token2, relationName)
                        m.addGenConstrIndicator(y[relationName, token1, token2], True, y[relationName, token2, token1], GRB.EQUAL, 1)
        
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
    
                        currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                        if currentProbability == 0:
                            continue
                        
                        constrainName = 'c_%s_%s_%s_AsymmetricProperty'%(token1, token2, relationName)
                        m.addGenConstrIndicator(y[relationName, token1, token2], True, y[relationName, token2, token1], GRB.EQUAL, 0)  
                        
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
    
                        currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                        if currentProbability == 0:
                            continue
                        
                        constrainName = 'c_%s_%s_%s_TransitiveProperty'%(token1, token2, relationName)
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
    
        m.update()
    
        # Add objectives
        Y_Q  = None
        for relationName in relationNames:
            for token1Index, token1 in enumerate(tokens):
                for token2Index, token2 in enumerate(tokens):
                    if token1 == token2 :
                        continue
    
                    currentProbability = graphResultsForPhraseRelation[relationName][token1Index][token2Index]
                    if currentProbability == 0:
                        continue
                        
                    currentQElement = graphResultsForPhraseRelation[relationName][token1Index][token2Index]*y[relationName, token1, token2]
                    self.myLogger.info("Created objective element %s"%(currentQElement))
                    Y_Q += currentQElement
                    
                    if (relationName+'-neg', token, token1) in y: 
                        currentQElement = (1-graphResultsForPhraseRelation[relationName][token1Index][token2Index])*y[relationName+'-neg', token1, token2]
                        self.myLogger.info("Created objective element %s"%(currentQElement))
                        Y_Q += currentQElement
        
        return Y_Q
    
    def addTripleRelationsConstrains(self, m, tokens, conceptNames, x, y, z, graphResultsForPhraseTripleRelation):
        if graphResultsForPhraseTripleRelation is None:
            return None
        
        self.myLogger.info('Starting method addTripleRelationsConstrains with graphResultsForPhraseTripleRelation')
        if graphResultsForPhraseTripleRelation is not None:
            for tripleRelation in graphResultsForPhraseTripleRelation:
                self.myLogger.info('graphResultsForPhraseTripleRelation for relation \"%s"'%(tripleRelation))

                for token1Index, token1 in enumerate(tokens):
                    self.myLogger.info('for token \"%s \n%s"'%(token1, np.column_stack( (["   "] + tokens, np.vstack((tokens, graphResultsForPhraseTripleRelation[tripleRelation][token1Index]))))))

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
                        
                        currentProbability = graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index]
                        if currentProbability == 0:
                            continue
                        
                        z[tripleRelationName, token1, token2, token3]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s_%s"%(tripleRelationName, token1, token2, token3))
    
                        currentProbability = graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index]
                        self.myLogger.info("Probability for relation %s between tokens %s %s %s is %f"%(tripleRelationName,token1, token2, token3, currentProbability))

                        if currentProbability < 1.0: #ilpOntSolver.__negVarTrashhold:
                            z[tripleRelationName+'-neg', token1, token2, token3]=m.addVar(vtype=GRB.BINARY,name="y_%s_not_%s_%s_%s"%(tripleRelationName, token1, token2, token3))

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
                            constrainName = 'c_%s_%s_%s_%sselfDisjoint'%(token1, token2, token3, tripleRelationName)
                            m.addConstr(z[tripleRelationName, token1, token2, token3] + z[tripleRelationName+'-neg', token1, token2, token3], GRB.LESS_EQUAL, 1, name=constrainName)
                            
        m.update()
    
        self.myLogger.info("Created %i ilp variables for triple relations"%(len(z)))

        # -- Add constraints 
        for tripleRelationName in graphResultsForPhraseTripleRelation:
            currentTripleRelation = self.myOnto.search_one(iri = "*%s"%(tripleRelationName))
                
            if currentTripleRelation is None:
                continue
    
            self.myLogger.debug("Triple Relation \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentTripleRelation.name, currentTripleRelation))
            
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
                self.myLogger.warn("Problem with creation of - triple - constrains for relation \"%s\" - not found its full definition %s"%(tripleRelationName,triplePropertiesRanges))
                self.myLogger.warn("Abandon it - going to the next relation")

                break
            else:
                self.myLogger.info("Found definition for relation %s - %s"%(tripleRelationName,triplePropertiesRanges))

            for token1 in tokens:
                for token2 in tokens:
                    if token2 == token1:
                        continue
                        
                    for token3 in tokens:
                        if token3 == token2:
                            continue
                        
                        if token3 == token1:
                            continue
                     
                        currentProbability = graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index]
                        if currentProbability == 0:
                            continue
                        
                        constrainNameTriple = 'c_triple_%s_%s_%s_%s'%(tripleRelationName, token1, token2, token3)
                        r1 = x[token1, triplePropertiesRanges['1']]
                        r2 = x[token2, triplePropertiesRanges['2']] 
                        r3 = x[token3, triplePropertiesRanges['3']]
                        rel = z[tripleRelationName, token1, token2, token3]
                        
                        currentConstrLinExprRange = r1 + r2 + r3 - 3 * rel
                        m.addConstr(currentConstrLinExprRange, GRB.GREATER_EQUAL, 0, name=constrainNameTriple)
                                    
                        self.myLogger.info("Created - triple - constrains for relation \"%s\" for tokens \"%s\", \"%s\", \"%s\""%(tripleRelationName,token1,token2,token3))
    
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

                        currentProbability = graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index]
                        if currentProbability == 0:
                            continue
                        
                        Z_Q += graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index]*z[tripleRelationName, token1, token2, token3]
    
                        if (tripleRelationName+'-neg', token1, token2, token3) in z: 
                            Z_Q += (1-graphResultsForPhraseTripleRelation[tripleRelationName][token1Index][token2Index][token3Index])*z[tripleRelationName+'-neg', token1, token2, token3]

        return Z_Q
        
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None):
    
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
        elif all((isinstance(item, string_types) or isinstance(item, int)) for item in phrase):
            tokens = phrase
        else:
            self.myLogger.info('Phrase type is not supported %s - returning unchanged results'%(phrase))
            return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation

        try:
            # Create a new Gurobi model
            m = Model("decideOnClassificationResult")
            m.params.outputflag = 0
            
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
            
            m.setObjective(Q, GRB.MAXIMIZE)

            # Token is associated with a single concept
            #for token in tokens:
            #   constrainName = 'c_%s'%(token)
            #    m.addConstr(quicksum(x[token, conceptName] for conceptName in conceptNames), GRB.LESS_EQUAL, 1, name=constrainName)
            
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
                if x or True:
                    if m.status == GRB.Status.OPTIMAL:
                        solution = m.getAttr('x', x)

                        self.myLogger.info('Token Solutions\n')

                        for conceptName in conceptNames:
                            tokenResult[conceptName] = np.zeros(len(tokens))
                            
                            for tokenIndex, token in enumerate(tokens):
                                if ((token, conceptName) in solution) and (solution[token, conceptName] == 1):                                    
                                    tokenResult[conceptName][tokenIndex] = 1
                                    self.myLogger.info('Solution \"%s\" is \"%s\"'%(token,conceptName))

            # Collect results for relations
            relationResult = None
            if graphResultsForPhraseRelation is not None: 
                relationResult = dict()
                relationNames = graphResultsForPhraseRelation.keys()
                
                if y or True:
                    if m.status == GRB.Status.OPTIMAL:
                        solution = m.getAttr('x', y)
                        self.myLogger.info('Relation Solutions\n')

                        for relationName in relationNames:
                            relationResult[relationName] = np.zeros((len(tokens), len(tokens)))
                            
                            for token1Index, token1 in enumerate(tokens):
                                for token2Index, token2 in enumerate(tokens):
                                    if token2 == token1:
                                        continue
                                    
                                    if ((relationName, token1, token2) in solution) and (solution[relationName, token1, token2] == 1):
                                        relationResult[relationName][token1Index][token2Index] = 1
                                        
                                        self.myLogger.info('Solution \"%s\" \"%s\" \"%s\"'%(token1,relationName,token2))
        
            # Collect results for triple relations
            tripleRelationResult = None
            if graphResultsForPhraseTripleRelation is not None:
                tripleRelationResult = {}
                tripleRelationNames = graphResultsForPhraseTripleRelation.keys()
                
                for tripleRelationName in tripleRelationNames:
                    tripleRelationResult[tripleRelationName] = np.zeros((len(tokens), len(tokens), len(tokens)))
                
                if z or True:
                    if m.status == GRB.Status.OPTIMAL:
                        solution = m.getAttr('x', z)
                        self.myLogger.info('Triple Relation Solutions\n')

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
                                        
                                            self.myLogger.info('Solution \"%s\" and \"%s\" and \"%s\" is in triple relation %s'%(token1,token2,token3,tripleRelationName))
        
        except:
            self.myLogger.error('Error')
            raise
           
        end = datetime.datetime.now()
        elapsed = end - start
        self.myLogger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # return results of ILP optimization
        return tokenResult, relationResult, tripleRelationResult
    
        def inferILPConstrains(self, model_trail, *conceptsRelations): 
            if len(conceptsRelations) == 0:
                return model_trail
            
            return conceptsRelations[0].inferILPConstrains(self, model_trail, conceptsRelations[1:])