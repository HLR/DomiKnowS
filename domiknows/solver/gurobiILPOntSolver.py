import math
from time import perf_counter, perf_counter_ns
from collections import OrderedDict
import logging

# pytorch
import torch

# Gurobi
from gurobipy import GRB, Model, Var, Env
import gurobipy
import collections

from domiknows.graph.concept import Concept, EnumConcept
from domiknows.solver.ilpOntSolver import ilpOntSolver
from domiknows.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods
from domiknows.solver.lcLossSampleBooleanMethods import lcLossSampleBooleanMethods
from domiknows.solver.ilpBooleanMethodsCalculator import booleanMethodsCalculator

from domiknows.graph import LcElement, LogicalConstrain, V, fixedL, ifL, forAllL
from domiknows.graph import CandidateSelection
from domiknows.utils import getReuseModel
from domiknows.utils import getDnSkeletonMode

from domiknows.graph.candidates import getCandidates


class gurobiILPOntSolver(ilpOntSolver):
    ilpSolver = 'Gurobi'

    def __init__(self, graph, ontologiesTuple, _ilpConfig, reuse_model=False) -> None:
        super().__init__(graph, ontologiesTuple, _ilpConfig)
        self.myIlpBooleanProcessor = gurobiILPBooleanProcessor()
        self.myLcLossBooleanMethods = lcLossBooleanMethods()
        self.myLcLossSampleBooleanMethods = lcLossSampleBooleanMethods()
        self.booleanMethodsCalculator = booleanMethodsCalculator()
        self.logical_constraints = {}
        for g in graph:
            self.logical_constraints = {**self.logical_constraints, **g.logicalConstrains}

        self.reuse_model = reuse_model
        if getReuseModel():
            self.reuse_model = True
            
        self.model = collections.deque([], 20)
        
    def set_logical_constraints(self, new_logical_constraints):
        self.logical_constraints = new_logical_constraints
        
    def get_logical_constraints(self, ):
        return self.logical_constraints
    
    def reset_logical_constraints(self, ):
        self.logical_constraints = self.myGraph[0].logicalConstrains ### Can myGraph really be multiple graphs?
        
    def getConcept(self, concept):
        return concept[0]
    
    def getConceptName(self, concept):
        return concept[0].name
    
    def conceptIsBinary(self, concept):
        return concept[2] is None
    
    def conceptIsMultiClass(self, concept):
        return concept[2] is not None
    
    # Check if value is NaN or if and has to be skipped
    def valueToBeSkipped(self, x):
        return math.isnan(x) or math.isinf(x)
        
    # Get Ground Truth for provided concept
    def __getLabel(self, dn, conceptRelation, fun=None, epsilon = None):
        value = dn.getAttribute(conceptRelation, 'label')
        return value
        
    # Get and tailor probability for provided concept and instance (represented as dn)
    def getProbability(self, dn, conceptRelation, key = ("local" , "softmax"), fun=None, epsilon = 0.00001):
        valueI = dn.getAttribute(conceptRelation, *key) if dn else None
        labelIndex = conceptRelation[2]
        
        if valueI is None:
            value = None
        else:
            labelIndex = conceptRelation[2]
            # If value is for multi-class concept, then we need to convert it to binary
            value = torch.zeros(2, dtype=torch.float) if self.conceptIsMultiClass(conceptRelation) else valueI
            value[0] = 1 - valueI[labelIndex] if self.conceptIsMultiClass(conceptRelation) else value[0]
            value[1] = valueI[labelIndex] if self.conceptIsMultiClass(conceptRelation) else value[1]
            
        # If softmax process probability through function and apply epsilon
        if "softmax" in key and value is not None and not math.isnan(value[0]) and epsilon is not None:
            value[0] = max(epsilon, min(1-epsilon, value[0]))
            value[1] = max(epsilon, min(1-epsilon, value[1]))
                    
            # Apply fun on probabilities if defined
            if fun is not None:
                value = fun(value)
        
        return value # Return probability
    
    def createILPVariable(self, m, dn, currentConceptRelation, notV = False):
        if notV:
            xkey = '<' + self.getConceptName(currentConceptRelation) + '>/ILP/notx'
        else:
            xkey = '<' + self.getConceptName(currentConceptRelation) + '>/ILP/x'  

        # Create a placeholder in Datanode for variable if it does not exist
        currentConceptLabelLength =  currentConceptRelation[3]
        dn.attributes.setdefault(xkey, [None] * currentConceptLabelLength)
        
        # Check Datanode if ILP variable is already created
        if self.conceptIsMultiClass(currentConceptRelation):
            xNew = dn.attributes[xkey][currentConceptRelation[2]]
        else:
            xNew =  dn.attributes[xkey][0]
            
        # Return ILP variable if it already exists
        if xNew is not None:
            return xNew
            
        # --- Else create ILP variable
        currentLabel = currentConceptRelation[1]

        if notV:
            xVarName = "%s_%s_is_not_%s"%(dn.getOntologyNode(), dn.getInstanceID(), currentLabel)
        else:
            xVarName = "%s_%s_is_%s"%(dn.getOntologyNode(), dn.getInstanceID(), currentLabel)
            
        xNew = m.addVar(vtype=GRB.BINARY,name=xVarName) 

        # Add ILP Variable to Datanode
        if self.conceptIsMultiClass(currentConceptRelation):
            dn.attributes[xkey][currentConceptRelation[2]] = xNew
        else:
            dn.attributes[xkey][0] = xNew
        
        return xNew
        
    def getDatanodesForConcept(self, rootDn, currentName, conceptToDNSCash=None):
        if conceptToDNSCash is None or currentName is None:
            conceptToDNSCash = {}
           
            if currentName is None:
                return  # Just reset cash
            
        if currentName in conceptToDNSCash:
            dns = conceptToDNSCash[currentName]
        else:
            rootConcept = rootDn.findRootConceptOrRelation(currentName)
            dns = rootDn.findDatanodes(select=rootConcept)
            conceptToDNSCash[currentName] = dns
            
        return dns
    
    # Count number of variables for logical constraints
    def countLCVariables(self, rootDn, *conceptsRelations):
        ilpVarCount = {}
        
        for currentConceptRelation in conceptsRelations: 
            currentName = self.getConceptName(currentConceptRelation)
            currentLabel = currentConceptRelation[1]

            dns = self.getDatanodesForConcept(rootDn, currentName)
                
            currentConceptName = currentName + "_" + currentLabel
            ilpVarCount[currentConceptName] = len(dns)
            
            if self.conceptIsMultiClass(currentConceptName):
                ilpVarCount[currentConceptName] += len(dns)
       
        return ilpVarCount
  
    # Create ILP variables for logical constraints and objective
    def createILPVariables(self, m, x, rootDn, *conceptsRelations, key = ("local" , "softmax"), fun=None, epsilon = 0.00001):
        Q = None
        # Reset the cash for concept to datanodes
        self.getDatanodesForConcept(rootDn, None)
        
        # Create ILP variables 
        for currentConceptRelation in conceptsRelations: 
            currentLabel = currentConceptRelation[1]
            if self.conceptIsMultiClass(currentConceptRelation):
                currentLabelIndex = currentConceptRelation[2]
            else:
                currentLabelIndex = 0
            
            if currentLabel is None:
                self.myLogger.debug("Concept %s does not have label"%currentConceptRelation)
       
            # Get datanodes for concept
            dns = self.getDatanodesForConcept(rootDn, self.getConceptName(currentConceptRelation))
            for dn in dns:
                xNew = x.get((self.getConcept(currentConceptRelation), currentLabel, dn.getInstanceID(), currentLabelIndex), None)
                if self.conceptIsBinary(currentConceptRelation):
                    xNotNew = x.get((self.getConcept(currentConceptRelation), 'Not_' + currentLabel, dn.getInstanceID(), currentLabelIndex), None)

                if xNew is None:
                    # Create ILP variable
                    xNew = self.createILPVariable(m, dn, currentConceptRelation)
                    if self.conceptIsBinary(currentConceptRelation):
                        xNotNew  = self.createILPVariable(m, dn, currentConceptRelation, notV=True)

                # ---- Prepare ILP variables for constraints and objective for the current run
                
                # Create placeholder for variable in the current x set for run
                x[self.getConcept(currentConceptRelation), currentLabel, dn.getInstanceID(), currentLabelIndex] = None
                if self.conceptIsBinary(currentConceptRelation):
                    x[self.getConcept(currentConceptRelation), 'Not_'+  currentLabel, dn.getInstanceID(), currentLabelIndex]  = None
                    
                # Get probability for variable in the current run
                currentProbability = self.getProbability(dn, currentConceptRelation, key=key, fun=fun, epsilon=epsilon)
                
                # Skip variable for the current run if probability is None
                if currentProbability == None or (torch.is_tensor(currentProbability) and currentProbability.dim() == 0) or len(currentProbability) < 2:
                    self.myLogger.warning("Probability not provided for variable concept %s in dataNode %s - skipping it"%(self.getConceptName(currentConceptRelation),dn.getInstanceID()))
                # Skip variable for the current run if probability is NaN or Inf
                elif self.valueToBeSkipped(currentProbability[1]):
                    self.myLogger.info("Probability is %f for concept %s and dataNode %s - skipping it"%(currentProbability[1],  currentLabel, dn.getInstanceID()))
                else:    
                    # Add variable to the x set for current run
                    x[self.getConcept(currentConceptRelation),  currentLabel, dn.getInstanceID(), currentLabelIndex] = xNew
                    if self.conceptIsBinary(currentConceptRelation):
                        x[self.getConcept(currentConceptRelation), 'Not_' + currentLabel, dn.getInstanceID(), currentLabelIndex] = xNotNew

                    # Add variable to objective
                    if Q is None:
                        Q = currentProbability[1] * xNew
                    else:
                        Q += currentProbability[1] * xNew       
                            
                    if self.conceptIsBinary(currentConceptRelation): 
                        Q += currentProbability[0] * xNotNew    
                
            if self.conceptIsMultiClass(currentConceptRelation):
                self.myLogger.debug("No creating ILP negative variables for multiclass concept %s"%(currentLabel))
                
            if getDnSkeletonMode() and "variableSet" in rootDn.attributes:
                rootConcept = rootDn.findRootConceptOrRelation(currentConceptRelation)
                rootConceptName = rootConcept.name

                xkey = '<' + self.getConceptName(currentConceptRelation) + '>/ILP/x'
                xkeyInVariableSet = rootConceptName + "/" + xkey
                if not rootDn.hasAttribute(xkey):
                    rootDn.attributes["variableSet"][xkeyInVariableSet] = []
                xForConcept = [value for key, value in x.items() if key[0] == self.getConcept(currentConceptRelation) and key[1] == currentLabel]
                rootDn.attributes["variableSet"][xkeyInVariableSet].append(xForConcept)
                    
                if self.conceptIsBinary(currentConceptRelation):
                    notxkey = '<' + self.getConceptName(currentConceptRelation) + '>/ILP/notx'
                    notxkeyInVariableSet = rootConceptName + "/" + notxkey
                    if not rootDn.hasAttribute(notxkey):
                        rootDn.attributes["variableSet"][notxkeyInVariableSet] = []
                    
                    xnotForConcept = [value for key, value in x.items() if key[0] == self.getConcept(currentConceptRelation) and key[1] == 'Not_' + currentLabel]
                    rootDn.attributes["variableSet"][notxkeyInVariableSet].append(xnotForConcept)


        m.update()

        if len(x):
            self.myLogger.info("Created %i ILP variables"%(len(x)))
        else:
            self.myLogger.warning("No ILP variables created")
            
        return Q
    
    def addMulticlassExclusivity(self, conceptsRelations, rootDn, m):
        m.update()

        multiclassDict = {}
        
        for c in conceptsRelations:
            if c[2] is None:
                continue
            
            if c[0] not in multiclassDict:
                multiclassDict[c[0]] = []
            
            multiclassDict[c[0]].append(c)
            
        for mc in multiclassDict:
            # Add constraint which restrict number of muliclass labels assigned to one for the given instance
            rootConcept = rootDn.findRootConceptOrRelation(mc)
            dns = rootDn.findDatanodes(select = rootConcept)
            xkey = '<' + mc.name + '>/ILP/x'  
            
            for dn in dns:
                var = []
                for mlabel in multiclassDict[mc]: 
                    
                    if xkey not in dn.attributes:
                        continue
                    
                    cILPs =  dn.attributes[xkey]
                    cIndex = mlabel[2]
                    
                    currentV = cILPs[cIndex]
                    
                    if currentV:
                        var.append(currentV)
                     
                if var: 
                    self.myIlpBooleanProcessor.countVar(m, *var, onlyConstrains = True, limitOp = '==', limit = 1, logicMethodName = "MultiClass")

    def addGraphConstrains(self, m, rootDn, *conceptsRelations):
        # Add constraint based on probability 
        for _conceptRelation in conceptsRelations: 
            rootConcept = rootDn.findRootConceptOrRelation(_conceptRelation[0])
            dns = rootDn.findDatanodes(select = rootConcept)
            
            conceptRelation = _conceptRelation[0].name

            xkey = '<' + conceptRelation + '>/ILP/x'
            notxkey = '<' + conceptRelation + '>/ILP/notx'
            
            for dn in dns:
                if notxkey not in dn.attributes:
                    continue
                
                if _conceptRelation[2] is None:
                    x = dn.getAttribute(xkey)[0]
                    notx = dn.getAttribute(notxkey)[0]
                else:
                    x = dn.getAttribute(xkey)[_conceptRelation[2]]
                    notx = dn.getAttribute(notxkey)[_conceptRelation[2]]
                   
                currentConstrLinExpr = x + notx 
                
                m.addConstr(currentConstrLinExpr == 1, name='Disjoint: %s and %s'%(_conceptRelation[1], 'Not_'+_conceptRelation[1]))
                self.myLogger.debug("Disjoint constraint between variable %s is  %s and variable %s is not - %s == %i"
                                    %(dn.getInstanceID(),_conceptRelation[1],dn.getInstanceID(),'Not_'+_conceptRelation[1],1))

        m.update()
        
        # Create subclass constraints
        for concept in conceptsRelations:
            
            # Skip if multiclass
            if concept[2] is not None:
                continue
            
            rootConcept = rootDn.findRootConceptOrRelation(concept[0])
            dns = rootDn.findDatanodes(select = rootConcept)
            
            for rel in concept[0].is_a():
                if  not rel.auto_constraint:
                    continue
                
                # A is_a B : if(A, B) : A(x) <= B(x)
                
                sxkey = '<' + rel.src.name + '>/ILP/x'
                dxkey = '<' + rel.dst.name + '>/ILP/x'

                for dn in dns:
                     
                    if sxkey not in dn.attributes: # subclass (A)
                        continue
                    
                    if dxkey not in dn.attributes: # superclass (B)
                        continue
                                                                    
                    self.myIlpBooleanProcessor.ifVar(m, dn.getAttribute(sxkey)[0], dn.getAttribute(dxkey)[0], onlyConstrains = True)
                    self.myLogger.info("Created - subclass - constraints between concept \"%s\" and concepts %s"%(rel.src.name,rel.dst.name))

        # Create disjoint constraints
        foundDisjoint = dict() # To eliminate duplicates
        for concept in conceptsRelations:
            
            # Skip if multiclass
            if concept[2] is not None:
                continue
            
            rootConcept = rootDn.findRootConceptOrRelation(concept[0])
            dns = rootDn.findDatanodes(select = rootConcept)
               
            for rel in concept[0].not_a():
                if  not rel.auto_constraint:
                    continue
                
                conceptName = concept[1]
                
                relDestFound = False
                for c in conceptsRelations:
                    if rel.dst == c[0]:
                        relDestFound = True
                        break
                if not relDestFound:
                    continue
                
                disjointConcept = rel.dst.name
                    
                if conceptName in foundDisjoint:
                    if disjointConcept in foundDisjoint[conceptName]:
                        continue
                
                if disjointConcept in foundDisjoint:
                    if conceptName in foundDisjoint[disjointConcept]:
                        continue
                            
                cxkey = '<' + conceptName + '>/ILP/x'
                dxkey = '<' + disjointConcept + '>/ILP/x'
                
                for dn in dns:
                    if cxkey not in dn.attributes:
                        continue
                    
                    if dxkey not in dn.attributes:
                        continue
                        
                    self.myIlpBooleanProcessor.countVar(m, dn.getAttribute(cxkey)[0], dn.getAttribute(dxkey)[0], onlyConstrains = True, limitOp = '<=', limit = 1, logicMethodName = "atMostL")
                        
                if not (conceptName in foundDisjoint):
                    foundDisjoint[conceptName] = {disjointConcept}
                else:
                    foundDisjoint[conceptName].add(disjointConcept)
                           
            if concept[1] in foundDisjoint:
                self.myLogger.info("Created - disjoint - constraints between concept \"%s\" and concepts %s"%(conceptName,foundDisjoint[conceptName]))
                
        # Create relation links constraints
        for concept in conceptsRelations:
            
            # Skip if multiclass
            if concept[2] is not None:
                continue
            
            isConstrainCreated = False
            
            for arg_id, rel in enumerate(concept[0].has_a()): 
                
                if  not rel.auto_constraint:
                    continue
                
                isConstrainCreated = True
                
                # TODO: need to include indirect ones like sp_tr is a tr while tr has a lm
                # A has_a B : A(x,y,...) <= B(x)
                #for xy in candidates[rel.src]:
                #x = xy[arg_id]
                
                relationName = rel.src.name
                rootRelation = rootDn.findRootConceptOrRelation(relationName)
                dnsR = rootDn.findDatanodes(select = rootRelation)
                _relAttr = rootDn.getRelationAttrNames(rootRelation)
                relAttr = list(_relAttr.keys())
                
                conceptName = rel.dst.name
                rootConcept = rootDn.findRootConceptOrRelation(conceptName)
                
                rxkey = '<' + relationName + '>/ILP/x'
                cxkey = '<' + conceptName + '>/ILP/x'
                
                for dn in dnsR:
                    if rxkey not in dn.attributes:
                        continue
                    
                    cdn = dn.relationLinks[relAttr[arg_id]]
                    
                    if not cdn or cxkey not in cdn[0].attributes:
                        continue
                    
                    self.myIlpBooleanProcessor.ifVar(m,dn.getAttribute(rxkey)[0], cdn[0].getAttribute(cxkey)[0], onlyConstrains = True)
               
            if isConstrainCreated:             
                self.myLogger.info("Created - doman/range constraints for concepts %s"%(concept[1]))
                    
        m.update()
        
    def addOntologyConstrains(self, m, rootDn, *_conceptsRelations):
        
        if not hasattr(self, 'myOnto'): 
            return
        
        conceptsRelations = [cr[0].name for cr in _conceptsRelations]
        
        # -- Add constraints based on concept disjoint statements in ontology - not(and(var1, var2)) = nand(var1, var2)
        foundDisjoint = dict() # too eliminate duplicates
        for conceptName in conceptsRelations:
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
            
            if currentConcept is None :
                continue
            
            rootConcept = rootDn.findRootConceptOrRelation(conceptName)
            dns = rootDn.findDatanodes(select = rootConcept)
            
            #self.myLogger.debug("Concept \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentConcept.name, conceptName))
                
            for currentD in currentConcept.disjoints():
                currentDisjoints = []
                
                for e in currentD.entities:
                    if e != currentConcept:
                        currentDisjoints.append(e)
                                    
                for currentDisjoint in currentDisjoints:
                    extendedCurrentDisjoints = currentDisjoint.descendants()
            
                    for d in extendedCurrentDisjoints:
                        disjointConcept = d._name
                                
                        if disjointConcept not in conceptsRelations:
                            continue
                                
                        if conceptName in foundDisjoint:
                            if disjointConcept in foundDisjoint[conceptName]:
                                continue
                        
                        if disjointConcept in foundDisjoint:
                            if conceptName in foundDisjoint[disjointConcept]:
                                continue
                         
                        rootDisjointConcept = rootDn.findRootConceptOrRelation(disjointConcept)
                        dnsDC = rootDn.findDatanodes(select = rootDisjointConcept)           
                                    
                        cxkey = '<' + conceptName + '>/ILP/x'
                        dxkey = '<' + disjointConcept + '>/ILP/x'
                        
                        for dn in zip(dns,dnsDC):
                            if cxkey not in dn[0].attributes:
                                continue
                            
                            if dxkey not in dn[1].attributes:
                                continue
                                
                            self.myIlpBooleanProcessor.nandVar(m, dn[0].getAttribute(cxkey)[0], dn[1].getAttribute(dxkey)[0], onlyConstrains = True)
                            
                        if not (conceptName in foundDisjoint):
                            foundDisjoint[conceptName] = {disjointConcept}
                        else:
                            foundDisjoint[conceptName].add(disjointConcept)
                           
            if conceptName in foundDisjoint:
                self.myLogger.info("Created - disjoint - constraints between concept \"%s\" and concepts %s"%(conceptName,foundDisjoint[conceptName]))

        # -- Add constraints based on concept equivalent statements in ontology - and(var1, av2)
        foundEquivalent = dict() # too eliminate duplicates
        for conceptName in conceptsRelations:
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for equivalentConcept in currentConcept.equivalent_to:
                equivalentConceptName = equivalentConcept.name
                if equivalentConceptName not in conceptsRelations:
                    continue
                        
                if conceptName in foundEquivalent:
                    if equivalentConceptName in foundEquivalent[conceptName]:
                        continue
                
                if equivalentConceptName in foundEquivalent:
                    if conceptName in foundEquivalent[equivalentConceptName]:
                        continue
                            
                rootEquivalentConcept = rootDn.findRootConceptOrRelation(equivalentConceptName)
                dnsEC = rootDn.findDatanodes(select = rootEquivalentConcept)           
                        
                cxkey = '<' + conceptName + '>/ILP/x'
                exkey = '<' + equivalentConceptName + '>/ILP/x'
                
                for dn in zip(dns,dnsEC):
                    if cxkey not in dn[0].attributes:
                        continue
                    
                    if exkey not in dn[1].attributes:
                        continue
                    
                    self.myIlpBooleanProcessor.andVar(m, dn[0].getAttribute(cxkey)[0], dn[1].getAttribute(exkey)[0], onlyConstrains = True)
                    
                if not (conceptName in foundEquivalent):
                    foundEquivalent[conceptName] = {equivalentConceptName}
                else:
                    foundEquivalent[conceptName].add(equivalentConceptName)
           
            if conceptName in foundEquivalent:
                self.myLogger.info("Created - equivalent - constraints between concept \"%s\" and concepts %s"%(conceptName,foundEquivalent[conceptName]))
    
        # -- Add constraints based on concept subClassOf statements in ontology - var1 -> var2
        for conceptName in conceptsRelations:
            
            currentConcept = self.myOnto.search_one(iri = "*%s"%(conceptName))
                
            if currentConcept is None :
                continue
                
            for ancestorConcept in currentConcept.ancestors(include_self = False):
                ancestorConceptName = ancestorConcept.name
                if ancestorConceptName not in conceptsRelations:
                    continue
                
                rootAncestorConcept = rootDn.findRootConceptOrRelation(ancestorConceptName)
                dnsAC = rootDn.findDatanodes(select = rootAncestorConcept)           
                        
                cxkey = '<' + conceptName + '>/ILP/x'
                axkey = '<' + ancestorConceptName + '>/ILP/x'
                
                for dn in zip(dns,dnsAC):
                    if cxkey not in dn[0].attributes:
                        continue
                    
                    if axkey not in dn[1].attributes:
                        continue
                        
                    self.myIlpBooleanProcessor.ifVar(m,  dn[0].getAttribute(cxkey)[0], dn[1].getAttribute(axkey)[0], onlyConstrains = True)
                        
                self.myLogger.info("Created - subClassOf - constraints between concept \"%s\" and concept \"%s\""%(conceptName,ancestorConceptName))
    
        # ---- ------No supported yet ontology concept constraints --------
        
        # -- Add constraints based on concept intersection statements in ontology - and(var1, var2, var3, ..)
        # -- Add constraints based on concept union statements in ontology -  or(var1, var2, var3, ..)
        # -- Add constraints based on concept objectComplementOf statements in ontology - xor(var1, var2)
        # -- Add constraints based on concept disjonitUnion statements in ontology -  Not supported by owlready2 yet
        # -- Add constraints based on property domain and range statements in ontology - P(x,y) -> D(x), R(y)
        # -- Add constraints based on concept oneOf statements in ontology - ?

        # -- Add constraints based on property domain and range statements in ontology - P(x,y) -> D(x), R(y)
        for relationName in conceptsRelations:
            currentRelation = self.myOnto.search_one(iri = "*%s"%(relationName))
                
            if currentRelation is None:
                continue
    
            try:
                currentRelationDomain = currentRelation.get_domain() # domains_indirect()
                currentRelationRange = currentRelation.get_range()
            except AttributeError:
                continue
                    
            self.myLogger.debug("Relation \"%s\" from data set mapped to \"%s\" concept in ontology"%(currentRelation.name, relationName))

            rootRelation = rootDn.findRootConceptOrRelation(relationName)
            dnsR = rootDn.findDatanodes(select = rootRelation)
            _relAttr = rootDn.getRelationAttrNames(rootRelation)
            relAttr = list(_relAttr.keys())
            
            cxkey = '<' + currentRelation.name + '>/ILP/x'

            for rel_domain in currentRelationDomain:
                if rel_domain._name not in conceptsRelations:
                    continue
                        
                dxkey = '<' + rel_domain._name + '>/ILP/x'

                for rel_range in currentRelationRange:
                    if rel_range.name not in conceptsRelations:
                        continue
                         
                    rxkey = '<' + rel_range._name + '>/ILP/x'
                    
                    for dn in dnsR:
                        if cxkey not in dn.attributes:
                            continue
                        
                        ddn = dn.relationLinks[relAttr[0]]
                        
                        if not ddn or dxkey not in ddn[0].attributes:
                            continue
                        
                        self.myIlpBooleanProcessor.ifVar(m, dn.getAttribute(cxkey)[0], ddn[0].getAttribute(dxkey)[0], onlyConstrains = True)
                        
                        rdn = dn.relationLinks[relAttr[1]]
                        
                        if not rdn or rxkey not in rdn[0].attributes:
                            continue
                        
                        self.myIlpBooleanProcessor.ifVar(m, dn.getAttribute(cxkey)[0], rdn[0].getAttribute(rxkey)[0], onlyConstrains = True)
                        
                self.myLogger.info("Created - domain-range - constraints for relation \"%s\" for domain \"%s\" and range \"%s\""%(relationName,rel_domain._name,rel_range._name))
        
        m.update()
        
    def addLogicalConstrains(self, m, dn, lcs, p, key = None):        
        if key == None:
            key = "/ILP/xP" # to get ILP variable from datanodes
        
        for lc in lcs:   
            startLC = perf_counter_ns()
            m.update()
            startNumConstrs = m.NumConstrs
            
            if lc.active:
                self.myLogger.info('Processing %s(%s) - %s'%(lc.lcName, lc, lc.strEs()))
            else:
                self.myLogger.debug('Skipping not active Logical Constraint %s(%s) - %s'%(lc.lcName, lc, [str(e) for e in lc.e]))
                continue

            result = self.__constructLogicalConstrains(lc, self.myIlpBooleanProcessor, m, dn, p, key = key, headLC = True)
            
            m.update()
            endNumConstrs = m.NumConstrs
            newNumConstrs = endNumConstrs - startNumConstrs
            endLC = perf_counter_ns()
            elapsedInNsLC = endLC - startLC
            elapsedInMsLC = elapsedInNsLC/1000000
            
            if result != None and isinstance(result, list):
                if newNumConstrs:
                    self.myLogger.info('Successfully added Logical Constraint %s - created %i new ILP constraint\n'%(lc.lcName,newNumConstrs))
                    self.myLoggerTime.info('Processing time for %s(%s) is: %ims - created %i new ILP constraint'%(lc.lcName, lc, elapsedInMsLC,newNumConstrs))
                else:
                    self.myLogger.info('Finished processing Logical Constraint %s - no new ILP constraint was created for it\n'%(lc.lcName))
                    self.myLoggerTime.info('Processing time for %s(%s) is: %ims - no new ILP constraint was created'%(lc.lcName, lc, elapsedInMsLC))
            else:
                self.myLogger.error('Failed to add Logical Constraint %s\n'%(lc.lcName))
                self.myLoggerTime.error('Failed to add Logical Constraint %s'%(lc.lcName))

    def isConceptFixed(self, conceptName):
        for graph in self.myGraph: # Loop through graphs
            for _, lc in graph.logicalConstrains.items(): # loop trough lcs in the graph
                if not lc.headLC or not lc.active: # Process only active and head lcs
                    continue
                    
                if type(lc) is not fixedL: # Skip not fixedL lc
                    continue
                
                if not lc.e:
                    continue
                
                if lc.e[0][1] == conceptName:
                    return True
            
        return False
                
    def isVariableFixed(self, dn, conceptName, e):
        fixedAttribute= None
        fixedValue = None
        
        for graph in self.myGraph: # Loop through graphs
            for _, lc in graph.logicalConstrains.items(): # loop trough lcs in the graph
                if not lc.headLC or not lc.active: # Process only active and head lcs
                    continue
                    
                if type(lc) is not fixedL: # Skip not fixedL lc
                    continue
                
                if not lc.e:
                    continue
                
                if lc.e[0][1] != conceptName:
                    continue
                
                fixedAttribute = lc.e[1].v[1].e[1]
                fixedValue = lc.e[1].v[1].e[2]
                break
                
        if fixedAttribute == None or fixedValue == None:
            return None
                      
        # For spec
        if fixedAttribute not in dn.getAttributes():
            return None
        
        attributeValue = dn.getAttribute(fixedAttribute).item()
        
        if attributeValue in fixedValue:
            pass
        elif (True in  fixedValue ) and attributeValue == 1:
            pass
        elif (False in  fixedValue ) and attributeValue == 0:
            pass
        else:
            return None
       
        vDnLabel = self.__getLabel(dn, conceptName).item()

        if vDnLabel == e[2]:
            return 1
        else:
            return 0
                    
    def getMLResult(self, dn, xPkey, e, p, loss = False, sample = False):
        if dn == None:
            raise Exception("No datanode provided")
            
        conceptName = e[0]
        
        sampleKey = '<' + conceptName + ">/sample" 
        if sample and sampleKey not in dn.getAttributes():
            dn.getAttributes()[sampleKey] = {}
        
        if dn.ontologyNode.name == conceptName:
            if not sample:
                if "xP" in xPkey:
                    return 1
                elif loss:
                    tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=torch.float64)
                    
                else:
                    tOne = torch.ones(1, device=self.current_device, requires_grad=False)
                    
                tOneSqueezed = torch.squeeze(tOne)
                return tOneSqueezed
            else:
                sampleSize = p
                
                if sampleSize not in dn.getAttributes()[sampleKey]: 
                    dn.getAttributes()[sampleKey][sampleSize] = {}
                    
                xVarName = "%s_%s_is_%s"%(e[0], dn.getInstanceID(), e[1])

                dn.getAttributes()[sampleKey][sampleSize][e[1]] = torch.ones(sampleSize, dtype=torch.bool, device = self.current_device)
                xP = torch.ones(sampleSize, device = self.current_device)
                
                return (dn.getAttributes()[sampleKey][sampleSize][e[1]], (xP, dn.getAttributes()[sampleKey][sampleSize][e[1]], xVarName))
        
        if dn.getAttribute(xPkey) == None:
            if not sample:
                return None
            else:   
                return ([None], (None, [None]))
        
        if not loss: # ------- If ILP inference
            if "xP" in xPkey:
                vDn = dn.getAttribute(xPkey)[p][e[2]] # Get ILP variable for the concept
            elif "local/argmax" in xPkey:
                vDn = dn.getAttribute(xPkey)[e[1]] # Get ILP variable for the concep
            else:
                vDn = dn.getAttribute(xPkey)[e[2]] # Get ILP variable for the concept
                
            return vDn # ILP variable for ILP inference
        
        # ----- Loss calculation
        
        isFiexd = self.isVariableFixed(dn, conceptName, e)

        if isFiexd != None:
            if isFiexd == 1:
                vDn = torch.tensor(1.0, device=self.current_device, requires_grad=True)
            else:
                vDn = torch.tensor(0.0, device=self.current_device, requires_grad=True)
        else:
            try:
                vDn = dn.getAttribute(xPkey)[e[1]] # Get value for the concept 
            except IndexError: 
                vDn = None
    
        if not sample:
            return vDn # Return here if not sample
        
        # --- Generate sample 
        if torch.is_tensor(vDn) and (len(vDn.shape) == 0 or len(vDn.shape) == 1 and vDn.shape[0] == 1):
            vDn = vDn.item()  
             
        sampleSize = p

        xVarName = "%s_%s_is_%s"%(e[0], dn.getInstanceID(), e[1])
                
        usedSampleSize = sampleSize
        if sampleSize == -1:
            usedSampleSize = dn.getAttributes()[sampleKey][-1][e[1]].shape[0]
        if isFiexd != None:  
            if isFiexd == 1:
                xP = torch.ones(usedSampleSize, device = self.current_device, requires_grad=True)
            else:
                xP = torch.zeros(usedSampleSize, device = self.current_device, requires_grad=True)
        else:
            xV = dn.getAttribute(xPkey)
            xEp = dn.getAttribute(xPkey).expand(usedSampleSize, len(xV))
            xP = xEp[:,e[1]]
          
        if sampleSize > -1: 
            if sampleSize not in dn.getAttributes()[sampleKey]: 
                dn.getAttributes()[sampleKey][sampleSize] = {}
                
            if e[1] not in dn.getAttributes()[sampleKey][sampleSize]:
                # check if not already generated
                if vDn == None or vDn != vDn:
                    dn.getAttributes()[sampleKey][sampleSize][e[1]] = [None]
                else:
                    # Create sample for this concept and sample size
                    dn.getAttributes()[sampleKey][sampleSize][e[1]] = torch.bernoulli(xP)
            
        return (dn.getAttributes()[sampleKey][sampleSize][e[1]], (xP, dn.getAttributes()[sampleKey][sampleSize][e[1]], xVarName)) # Return sample data and probability info
                      
    def fixedLSupport(self, _dn, conceptName, vDn, i, m):
        vDnLabel = self.__getLabel(_dn, conceptName).item()

        if isinstance(vDn, Var):                                 
            if vDnLabel == -100:
                vDn.VTag = "None" + vDn.VarName
            elif vDnLabel == i:
                vDn.VTag = "True" + vDn.VarName
            else:
                vDn.VTag = "False" + vDn.VarName
                
            m.update()
            return vDn
        elif torch.is_tensor(vDn):
            if vDnLabel == -100:
                return None
            elif vDnLabel == i:
                ones = torch.ones(vDn.shape[0])
                return ones
            else:
                zeros = torch.zeros(vDn.shape[0])
                return zeros
        else:
            if vDnLabel == -100:
                return None
            elif vDnLabel == i:
                return 1
            else:
                return 0
        
        
    def __addLossTovDns(self, loss, vDns):
        if loss and vDns:
            vDnsOriginal = vDns
            vDnsList = [v[0] for v in  vDns]
            
            updatedVDns = []
            try:
                if len(vDnsList) > 1:
                    tStack = torch.stack(vDnsList, dim=1)
                else:
                    tStack = vDnsList[0]
                tsqueezed = torch.squeeze(tStack, dim=0)

            except IndexError as ie:
                tsqueezed = torch.stack(vDnsList, dim=0)
                pass
        
            if not len(tsqueezed.shape):
                tsqueezed = torch.unsqueeze(tsqueezed, 0)
                
            tList = [tsqueezed]
            updatedVDns.append(tList)
            
            return updatedVDns
        else:
            return vDns
    
    def __constructLogicalConstrains(self, lc, booleanProcessor, m, dn, p, key = None, lcVariablesDns = None, headLC = False, loss = False, sample = False, vNo = None, verify=False):
        if key == None:
            key = ""
            
        if lcVariablesDns == None:
            lcVariablesDns = OrderedDict()
            
        lcVariables = OrderedDict()
        
        if sample:
            sampleInfo = OrderedDict()
            lcVariablesSet = OrderedDict()
            
        if vNo == None:
            vNo = [1, 1]
        
        firstV = None
        integrate = False
        
        for eIndex, e in enumerate(lc.e): 
            if  isinstance(e, V):
                continue # Already processed in the previous Concept 
            
            if isinstance(e, (Concept,  LcElement, tuple)): 
                # Look one step ahead in the parsed logical constraint and get variables names (if present) after the current concept
                if eIndex + 1 < len(lc.e) and isinstance(lc.e[eIndex+1], V):
                    variable = lc.e[eIndex+1]
                else:
                    if isinstance(e, LogicalConstrain):
                        variable = V(name="_lc" + str(vNo[1]))
                        vNo[1] += 1
                    elif isinstance(e, tuple) and isinstance(e[0], CandidateSelection):
                        e[0].CandidateSelectionVariable = e[1]
                        e = e[0]
                        
                        variable = V(name="_cs" + str(vNo[1]))
                        vNo[1] += 1
                    else:
                        if firstV == None:
                            variable = V(name="_x" + str(vNo[0]) )
                            if not isinstance(lc, CandidateSelection):
                                firstV = variable.name
                            vNo[0] += 1
                        else:
                            variable = V(name="_x" + str(vNo[0]), v = (firstV,))
                            vNo[0] += 1
                    
                if variable.name:
                    variableName = variable.name
                else:
                    variableName = "V" + str(vNo[0])
                    vNo[0] += 1
                    
                if variableName in lcVariables:
                    newvVariableName = "_x" + str(vNo[0])
                    vNo[0] += 1
                    
                    lcVariablesDns[newvVariableName] = lcVariablesDns[variableName]
                    if None in lcVariablesDns:
                        pass
  
                    lcVariables[newvVariableName] = lcVariables[variableName]

                elif isinstance(e, (Concept, tuple)): # -- Concept 
                    # -- Get dataNodes candidates 
                    dnsList = getCandidates(dn, e, variable, lcVariablesDns, lc, self.myLogger, integrate = integrate)
                    lcVariablesDns[variableName] = dnsList
                                
                    if isinstance(lc, CandidateSelection):
                        continue
                    
                    # -- Get ILP variables from collected DataNodes for the given element of logical constraint
                    
                    conceptName = e[0].name
                    vDns = [] # Stores ILP variables
                    if sample:
                        sampleInfoForVariable = []
                    xPkey = '<' + conceptName + ">" + key
                    
                    for dns in dnsList:
                        _vDns = []
                        if sample:
                            _sampleInfoForVariable = []
                            
                        for _dn in dns:
                            if not _dn:
                                vDn = None
                                _vDns.append(vDn)
                                continue

                            if isinstance(e[0], EnumConcept) and e[2] == None: # Multiclass concept
                                eList = e[0].enum
                                for i, _ in enumerate(eList):
                                    eT = (e[0].name, i, i)
                                    if sample:
                                        vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss = loss, sample=sample)
                                        
                                        _sampleInfoForVariable.append(vDnSampleInfo)
                                    else:
                                        vDn = self.getMLResult(_dn, xPkey, eT, p, loss = loss, sample=sample)
                                    
                                    if lc.__str__() == "fixedL":
                                        vDn = self.fixedLSupport(_dn, conceptName, vDn, i, m)
                                        
                                    _vDns.append(vDn)
                            elif isinstance(e[0], EnumConcept) and e[2] != None: # Multiclass concept label
                                eT = (e[0].name, e[2], e[2])
                                
                                if sample:
                                    vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss = loss, sample=sample)                                    
                                    _sampleInfoForVariable.append(vDnSampleInfo)
                                else:
                                    vDn = self.getMLResult(_dn, xPkey, eT, p, loss = loss, sample=sample)
                                
                                if lc.__str__() == "fixedL":
                                    self.fixedLSupport(_dn, conceptName, vDn, e[2], m)
                                    
                                vDn = _vDns.append(vDn)
                            else: # Binary concept
                                eT = (conceptName, 1, 0)
                                if sample:
                                    vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss = loss, sample=sample)

                                    _sampleInfoForVariable.append(vDnSampleInfo)
                                else:
                                    vDn = self.getMLResult(_dn, xPkey, eT, p, loss = loss, sample=sample)
                                
                                if lc.__str__() == "fixedL":
                                    self.fixedLSupport(_dn, conceptName, vDn, 1, m)
                                        
                                vDn = _vDns.append(vDn)
                        
                        if len(_vDns) == 0:
                            #continue
                            pass
                        
                        vDns.append(_vDns)
                        
                        if sample:
                            sampleInfoForVariable.append(_sampleInfoForVariable)
                        
                    # -- Store ILP variables
                    
                    if None in lcVariablesDns:
                        pass
                    
                    if vDns and loss and not sample:
                        vDnsList = [v[0] for v in vDns]
                        try:
                            tStack = torch.stack(vDnsList, dim=0)
                            tsqueezed = torch.squeeze(tStack, dim=0)
                            if not len(tsqueezed.shape):
                                tsqueezed = torch.unsqueeze(tsqueezed, 0)
                            lcVariables[variableName] = [[tStack]]
                        except TypeError:
                            for v in vDns:
                                if v[0] != None and torch.is_tensor(v[0]):
                                    v[0] = torch.unsqueeze(v[0], 0)
                                                                    
                            lcVariables[variableName] = vDns
                    else:
                        lcVariables[variableName] = vDns
                    
                    if sample:
                        sampleInfo[variableName] = sampleInfoForVariable
                
                if isinstance(e, LcElement):

                    if isinstance(e, CandidateSelection): # -- nested LogicalConstrain - process recursively 
                        lcVariablesDnsNew = self.__constructLogicalConstrains(e, booleanProcessor, m, dn, p, key = key, 
                                                                     lcVariablesDns = lcVariablesDns, headLC = False, loss = loss, sample = sample, vNo=vNo, verify=verify)
                         
                        lcVariablesDns = lcVariablesDnsNew
                        vDns = None
                        if lcVariablesDns:
                            length_of_list = len(next(iter(lcVariablesDns.values())))

                            if sample:
                                vDns = [[torch.ones(p, device=self.current_device, requires_grad=False, dtype=torch.bool)] for _ in range(length_of_list)]
                            elif loss:
                                vDns = [[torch.zeros(length_of_list, device=self.current_device, requires_grad=True, dtype=torch.float64)]]
                                vDns = self.__addLossTovDns(loss, vDns)
                            else:
                                vDns = [[1] for _ in range(length_of_list)]
                                   
                    if isinstance(e, LogicalConstrain): # -- nested LogicalConstrain - process recursively 
                        self.myLogger.info('Processing Nested %s(%s) - %s'%(e.lcName, e, e.strEs()))

                        if sample:
                            vDns, sampleInfoLC, lcVariablesLC = self.__constructLogicalConstrains(e, booleanProcessor, m, dn, p, key = key, 
                                                                                   lcVariablesDns = lcVariablesDns, headLC = False, loss = loss, sample = sample, vNo=vNo, verify=verify)
                            sampleInfo = {**sampleInfo, **sampleInfoLC} # sampleInfo|sampleInfoLC in python 9
                            
                            lcVariablesSet = {**lcVariablesSet, **lcVariablesLC}
                        else:
                            vDns = self.__constructLogicalConstrains(e, booleanProcessor, m, dn, p, key = key, 
                                                                     lcVariablesDns = lcVariablesDns, headLC = False, loss = loss, sample = sample, vNo=vNo, verify=verify)
                            
                            vDns = self.__addLossTovDns(loss, vDns)
                        
                                                    
                    if vDns == None:
                        self.myLogger.warning('Not found data for %s(%s) nested Logical Constraint required to build %s(%s) - skipping it'%(e.lcName,e,lc.lcName,lc))
                        return None
                        
                    countValid = sum(1 for sublist in vDns if sublist and any(elem is not None for elem in sublist))
                    self.myLogger.info('Size of candidate list returned by %s(%s) nested Logical Constraint is %i of which %i is not None'%(e.lcName,e,len(vDns),countValid))
                    lcVariables[variableName] = vDns   
            # Int - limit 
            elif isinstance(e, int): 
                if eIndex == 0:
                    pass # if this lc using it
                else:
                    pass # error!
            elif isinstance(e, str): 
                if eIndex == 2:
                    pass # if this lc using it
                else:
                    pass # error!
            else:
                self.myLogger.error('Logical Constraint %s has incorrect element %s'%(lc,e))
                return None
            
        if isinstance(lc, CandidateSelection):
            return lc(lcVariablesDns, keys=lc.CandidateSelectionVariable)
        elif sample:
            lcVariablesSet[lc] = lcVariables
            return lc(m, booleanProcessor, lcVariables, headConstrain = headLC, integrate = integrate), sampleInfo, lcVariablesSet
        elif verify and headLC:
            return lc(m, booleanProcessor, lcVariables, headConstrain = headLC, integrate = integrate), lcVariables
        else:
            if loss:
                slpitT = False
                for v in lcVariables:
                    if lcVariables[v] and len(lcVariables[v]) > 1:
                        slpitT = True
                        break
                    
                if slpitT:
                    for v in lcVariables:
                        if lcVariables[v] and len(lcVariables[v]) > 1:
                            continue
                         
                        lcVSplitted = torch.split(lcVariables[v][0][0], 1)
                        lcVariables[v] = []
                        
                        for s in lcVSplitted:
                            lcVariables[v].append([s]) 
                    
            return lc(m, booleanProcessor, lcVariables, headConstrain = headLC, integrate = integrate)
    
    # ---------------
                
    # -- Main method of the solver - creating ILP constraints plus objective, invoking the ILP solver and returning the result of the ILP solver classification  
    def calculateILPSelection(self, dn, *conceptsRelations, key = ("local" , "softmax"), fun=None, epsilon = 0.00001, minimizeObjective = False, ignorePinLCs = False):
        if self.ilpSolver == None:
            self.myLogger.warning('ILP solver not provided - returning')
            self.myLoggerTime.warning('ILP solver not provided - returning')
            
            return 
        
        self.current_device = dn.current_device
        
        self.myLogger.info('Calculating ILP Inference ')
        self.myLoggerTime.info('Calculating ILP Inference ')
        start = perf_counter()

        gurobiEnv = Env("", empty=True)
        gurobiEnv.setParam('OutputFlag', 0)
        gurobiEnv.start()  
        
        # Check if existing ILP model can be reuse without recreating ILP constraints
        try:
            # Find count of instance in each concept 
            ilpVarCount = self.countLCVariables(dn, *conceptsRelations)

            reusingModel = False
            if self.reuse_model:
                # Find it there is saved ILP model with this count of instances per concept
                for modelDec in self.model:
                    if ilpVarCount == modelDec[0]:
                        m = modelDec[1]
                        x = modelDec[2]
                        reusingModel = True
                        break
                    
            if not reusingModel:
                # If not reusing the model or if the right model was yet saved - create new Gurabi model
                if dn.gurobiModel == None:
                    m = Model("decideOnClassificationResult" + str(start), gurobiEnv)
                    dn.gurobiModel = m
                else:
                    m = dn.gurobiModel
                    m.reset()
                    mConstr = m.getConstrs()
                    m.remove(mConstr)
                    m.update()
                    
                m.params.outputflag = 0
                x = OrderedDict()
                
            # Handle ILP Variables for concepts and objective
            Q = self.createILPVariables(m, x, dn, *conceptsRelations, key=key, fun=fun, epsilon = epsilon)
            
            endVariableInit = perf_counter()
            elapsedVariablesInMs = (endVariableInit - start) *1000
            self.myLoggerTime.info('ILP Variables Init - time: %ims'%(elapsedVariablesInMs))

            if not reusingModel:
                # Add constraints based on ontology and graph definition
                self.addOntologyConstrains(m, dn, *conceptsRelations)
                self.addGraphConstrains(m, dn, *conceptsRelations)
                # Create constraint for multiclass exclusivity 
                self.addMulticlassExclusivity(conceptsRelations, dn, m)
            else:
                self.myLoggerTime.info('Reusing ILP Model - LCs already present in the model')

            endGraphAndOntologyConstraints = perf_counter()
            elapsedGandOConstraintsInMs = (endGraphAndOntologyConstraints - endVariableInit) * 1000
            self.myLoggerTime.info('ILP Graph and Ontology Constraints - time: %ims'%(elapsedGandOConstraintsInMs))
            
            # ILP Model objective setup
            if Q is None:
                Q = 0
                self.myLogger.error("No data provided to create any ILP variable - not ILP result returned")
                self.myLoggerTime.error("No data provided to create any ILP variable - not ILP result returned")
                
            if minimizeObjective:
                m.setObjective(Q, GRB.MINIMIZE)
            else:
                m.setObjective(Q, GRB.MAXIMIZE) # -------- Default

            m.update()
            
            # Collect head logical constraints
            _lcP = {}
            _lcP[100] = []
            pUsed = False
            for graph in self.myGraph:
                for _, lc in graph.logicalConstrains.items():
                    if lc.headLC:     
                        if not ignorePinLCs:
                            lcP = lc.p
                        else:
                            lcP = 100   
                                            
                        if lcP not in _lcP:
                            _lcP[lcP] = []
                            pUsed = True # Found p different then default 100
                        
                        _lcP[lcP].append(lc) # Keep constraint with the same p in the list 
            
            # Sort constraints according to their p
            lcP = OrderedDict(sorted(_lcP.items(), key=lambda t: t[0], reverse = True))
            for p in lcP:
                self.myLogger.info('Found %i logical constraints with p %i - %s\n'%(len(lcP[p]),p,lcP[p]))
                self.myLoggerTime.info('Starting ILP interference - Found %i logical constraints'%(len(lcP[p])))
            
            # Search through set of logical constraints for subset satisfying and the max/min calculated objective value
            lcRun = {} # Keeps information about subsequent model runs
            ps = [] # List with processed p 
            
            #  -----------  Run ILP solver for each set of lcs based on their p
            for p in lcP:
                ps.append(p)
                
                if pUsed:
                    mP = m.copy() # Copy model for this run                    
                    xP = {}
                    lckey = "/ILP/xP"
                   
                    pStart= perf_counter()
                    
                    for _x in x:
                        # Map variables to the new copy model
                        xP[_x] = mP.getVarByName(x[_x].VarName)
                        
                        rootConcept = dn.findRootConceptOrRelation(_x[0])
                        
                        dns = dn.findDatanodes(select = ((rootConcept,), ("instanceID", _x[2])))  
                        
                        if dns:
                            if _x[1].startswith('Not'):
                                xPkey = '<' + _x[0].name + '>/ILP/notxP'
                            else:
                                xPkey = '<' + _x[0].name + '>/ILP/xP'
    
                            if xPkey not in dns[0].attributes:
                                dns[0].attributes[xPkey] = {}
                                
                            if p not in dns[0].attributes[xPkey]:
                                xkey = '<' + _x[0].name + '>/ILP/x'
                                if xkey not in dns[0].attributes:
                                    continue
                                
                                xLen = len(dns[0].attributes[xkey])
                                dns[0].attributes[xPkey][p] = [None] * xLen
                                
                            dns[0].attributes[xPkey][p][_x[3]] = mP.getVarByName(x[_x].VarName)
                            
                            pEnd = perf_counter()
                            self.myLoggerTime.info('ILP Model init for p %i - time: %ims'%(p, (pEnd - pStart)*1000))
                else:
                    mP = m
                    xP = x
                    
                    lckey = "/ILP/x"
            
                # Prepare set with logical constraints for this run
                lcs = []
                for _p in lcP:
                    lcs.extend(lcP[_p])

                    if _p == p:
                        break     
    
                # ----------- Add LC constraints to the ILP model
                
                endLogicalConstraintsPrep = perf_counter()
                elapsedLogicalConstraintsPrepInMs = (endLogicalConstraintsPrep - endGraphAndOntologyConstraints)*1000
                self.myLoggerTime.info('ILP Logical Constraints Preprocessing - time: %ims'%(elapsedLogicalConstraintsPrepInMs))
                
                if pUsed or not reusingModel:
                    self.addLogicalConstrains(mP, dn, lcs, p, key = lckey) # <--- LC constraints
                    
                    # Save model
                    if self.reuse_model:
                        self.model.append((ilpVarCount, mP, xP))
                        import sys
                        memoryUsage = sys.getsizeof(mP)
                        # Convert bytes to kilobytes
                        memoryUsage_kB = memoryUsage / 1024
                        self.myLoggerTime.info(f'ILP Logical Constraints Preprocessing - memory use by saved Gurobi models: {memoryUsage_kB:.2f} kB')


                self.myLogger.info('Optimizing model for LCs with probabilities %s with %i ILP variables and %i ILP constraints'%(p,mP.NumVars,mP.NumConstrs))
                self.myLoggerTime.info('Optimizing model for LCs with probabilities %s with %i ILP variables and %i ILP constraints'%(p,mP.NumVars,mP.NumConstrs))

                endLogicalConstraints = perf_counter()
                elapsedLogicalConstraintsInMs = (endLogicalConstraints - endLogicalConstraintsPrep) *1000
                self.myLoggerTime.info('ILP Logical Constraints - time: %ims'%(elapsedLogicalConstraintsInMs))
            
                #mP.update()
                #mP.display() 
                startOptimize = perf_counter()

                # ----------- Run ILP model - Find solution 
                mP.optimize()
                mP.update()
                                
                endOptimize = perf_counter()
                elapsedOptimizeInMs = (endOptimize - startOptimize) * 1000
    
                # ----------- Check model run result
                solved = False
                objValue = None
                if mP.status == GRB.Status.OPTIMAL:
                    self.myLogger.info('%s solution was found in %ims for p - %i with optimal value: %.2f'
                                       %('Min' if minimizeObjective else 'Max', elapsedOptimizeInMs, p, mP.ObjVal))
                    
                    self.myLoggerTime.info('%s solution was found in %ims for p - %i with optimal value: %.2f'
                                       %('Min' if minimizeObjective else 'Max', elapsedOptimizeInMs, p, mP.ObjVal))
                    solved = True
                    objValue = mP.ObjVal
                elif mP.status == GRB.Status.INFEASIBLE:
                    self.myLogger.error('Model was proven to be infeasible for p - %i.'%(p))
                    self.myLoggerTime.error('Model was proven to be infeasible for p - %i.'%(p))
                elif mP.status == GRB.Status.INF_OR_UNBD:
                    self.myLogger.error('Model was proven to be infeasible or unbound for p - %i.'%(p))
                    self.myLoggerTime.error('Model was proven to be infeasible or unbound for p - %i.'%(p))
                elif mP.status == GRB.Status.UNBOUNDED:
                    self.myLogger.error('Model was proven to be unbound.')
                    self.myLoggerTime.error('Model was proven to be unbound.')
                else:
                    self.myLogger.error('Optimal solution not was found for p - %i - error code %i'%(p,mP.status))
                    self.myLoggerTime.error('Optimal solution not was found for p - %i - error code %i'%(p,mP.status))
                 
                # Print ILP model to log file if model is not solved or logger level is DEBUG 
                # -  check if Gurobi is version 9.1.1 or less; newer versions hang on mP.display() 
                gurobiVersion = gurobipy.gurobi.version()
                if (gurobiVersion[0] < 9) or \
                   ((gurobiVersion[0] < 10) and (gurobiVersion[1] < 1)) or \
                   ((gurobiVersion[0] < 10) and (gurobiVersion[1] < 2) and (gurobiVersion[2] < 2)):
                    if (not solved or self.myLogger.level <= logging.INFO) and self.myLogger.filter(""):
                        import sys
                        so = sys.stdout 
                        logFileName = self.myLogger.handlers[0].baseFilename
                        log = open(logFileName, "a")
                        sys.stdout = log
                        mP.display() 
                        sys.stdout = so
                    
                if (not solved):
                    mP.computeIIS()
                    mP.write("logs/gurabiInfeasible.ilp")

                #if (self.myLogger.level <= logging.INFO):# and self.myLogger.filter(""):
                #    mP.write("logs/gurabiInfeasible.lp")

                # Keep result of the model run    
                lcRun[p] = {'p':p, 'solved':solved, 'objValue':objValue, 'lcs':lcs, 'mP':mP, 'xP':xP, 'elapsedOptimize':elapsedOptimizeInMs}

            # ----------- Select model run with the max/min objective value 
            maxP = None
            for p in lcRun:
                if lcRun[p]['objValue']:
                    if maxP:
                        if minimizeObjective and lcRun[maxP]['objValue'] >= lcRun[p]['objValue']:
                            maxP = p
                        elif not minimizeObjective and lcRun[maxP]['objValue'] <= lcRun[p]['objValue']:
                            maxP = p
                    else:
                        maxP = p
               
            # ----------- If solution found model - return best result          
            if maxP:
                self.myLogger.info('Best  solution found for p - %i'%(maxP))
                #self.myLoggerTime.info('Best  solution found for p - %i'%(maxP))
                
                lcRun[maxP]['mP'].update()
                xVars = list(lcRun[maxP]['xP'].values())
                xVarsIndex = 0
                
                for c in conceptsRelations:
                    if c[2] is None:
                        index = 0
                    else:
                        index = c[2] # multiclass
                         
                    c_root = dn.findRootConceptOrRelation(c[0])
                    c_root_dns = dn.findDatanodes(select = c_root)
                    
                    ILPkey = '<' + c[0].name + '>/ILP'
                    
                    xkey = ILPkey + '/x'
                    xPkey = ILPkey + '/xP'
                    xNotPkey = ILPkey + '/notxP'
                    
                    ilpTensor = None
                    if getDnSkeletonMode() and "variableSet" in dn.attributes:
                        ilpKeyInVariableSet = c_root.name + "/<" + c[0].name +">" + "/ILP"
                        
                        if ilpKeyInVariableSet in dn.attributes["variableSet"]:
                            ilpTensor = dn.attributes["variableSet"][ilpKeyInVariableSet]
                        else:
                            ilpTensor = torch.zeros([len(c_root_dns), c[3]], dtype=torch.float, device=self.current_device)
                   
                    for i, cDn in enumerate(c_root_dns):
                        dnAtt = cDn.getAttributes()
                        
                        if xkey not in dnAtt and xPkey not in dnAtt:
                            if xVars[xVarsIndex] == None or not reusingModel:
                                
                                if ilpTensor is not None:
                                   ilpTensor[i][index] = float("nan")
                                else:
                                    if ILPkey not in dnAtt:
                                        dnAtt[ILPkey] = torch.empty(c[3], dtype=torch.float)
                                    
                                    dnAtt[ILPkey][index] = float("nan")
                                
                                # Update index for x variables 
                                if c[2] is None:
                                    xVarsIndex +=2 # skip Not variable
                                else:
                                    xVarsIndex +=1
                                    
                                continue 
                            else:
                                if pUsed: 
                                    dnAtt[xPkey] = {}
                                    dnAtt[xPkey][maxP] = [None] * c[3]
                                    dnAtt[xPkey][maxP][index] = xVars[xVarsIndex]
                                else:
                                    dnAtt[xkey] = [None] * c[3]
                                    dnAtt[xkey][index] = xVars[xVarsIndex]
                        
                        if pUsed: 
                            if dnAtt[xPkey][maxP][index] == None:
                                dnAtt[xPkey][maxP][index] = xVars[xVarsIndex]
                        else:
                            if dnAtt[xkey][index] == None:
                                dnAtt[xkey][index] = xVars[xVarsIndex]
                                    
                        # Update index for x variables           
                        if c[2] is None:
                            xVarsIndex +=2 # skip Not variable
                        else:
                            xVarsIndex +=1
                            
                        #  Get solution    
                        if pUsed:
                            solution = dnAtt[xPkey][maxP][index].X
                        else:
                            solution = dnAtt[xkey][index].X
                            
                        if solution == 0:
                            solution = 0
                        elif solution == 1: 
                            solution = 1

                        if ilpTensor is None and ILPkey not in dnAtt:
                            dnAtt[ILPkey] = torch.empty(c[3], dtype=torch.float)
                        
                        if xkey not in dnAtt:
                            dnAtt[xkey] = torch.empty(c[3], dtype=torch.float)
                       
                        if ilpTensor is not None:
                           ilpTensor[i][index] = solution
                        else:
                            dnAtt[ILPkey][index] = solution
                            
                        if pUsed: # Set main ILP variable x based on max P ILP variable x
                            dnAtt[xkey][index] = dnAtt[xPkey][maxP][index]
                            if xNotPkey in dnAtt: # do this for not x as well
                                dnAtt[xNotPkey][index] = dnAtt[xNotPkey][maxP][index]

                        
                        if solution == 1:
                            self.myLogger.info('\"%s\" is \"%s\"'%(cDn, c[1]))
                            #self.myLoggerTime.info('\"%s\" is \"%s\"'%(cDn,c[1]))
                    
                    if ilpTensor is not None:
                        dn.attributes["variableSet"][ilpKeyInVariableSet] = ilpTensor

            else:
                pass
                                       
        except Exception as inst:
            self.myLogger.error('Error returning solutions -  %s'%(inst))
            self.myLoggerTime.error('Error returning solutions -  %s'%(inst))
            
            raise
           
        endResultPrep = perf_counter()
        elapsedResultPrepInMs = (endResultPrep - endOptimize) *1000
        self.myLoggerTime.info('ILP Preparing Return Results - time: %ims'%(elapsedResultPrepInMs))
            
        self.myLogger.info('')

        end = perf_counter()
        elapsedInS = end - start
        if elapsedInS > 1:
            self.myLogger.info('End ILP Inference - internal total time: %fs'%(elapsedInS))
            self.myLoggerTime.info('End ILP Inference - internal total time: %fs'%(elapsedInS))
        else:
            elapsedInMs = elapsedInS *1000
            self.myLogger.info('End ILP Inference - internal total time: %ims'%(elapsedInMs))
            self.myLoggerTime.info('End ILP Inference - internal total time: %ims'%(elapsedInMs))
            
        self.myLogger.info('')
        
        # ----------- Return
        return

    def eliminateDuplicateSamples(self, lcVariables, sampleSize):
        variablesSamples = [lcVariables[v][1] for v in lcVariables]
                    
        variablesSamplesT = torch.stack(variablesSamples)
        
        uniqueSampleIndex = OrderedDict()
        
        for i in range(sampleSize):
            currentS = variablesSamplesT[:,i]
            
            currentSHash = hash(currentS.cpu().detach().numpy().tobytes())
            
            if currentSHash in uniqueSampleIndex:
                continue
                
                # For testing  - not used now
                if torch.equal(currentS, variablesSamplesT[:,uniqueSampleIndex[currentSHash]]):
                    continue
                else:
                    raise Exception("HashWrong")
            else:
                uniqueSampleIndex[currentSHash] = i
            
        va = list(uniqueSampleIndex.values())
            
        newSampleSize = len(va)
    
        indices = torch.tensor(va, device = self.current_device)
        #x = torch.arange(6).view(2,3)
        Vs = torch.index_select(variablesSamplesT, dim=1, index=indices)
        
        return newSampleSize, indices, Vs

    def calulateSampleLossForVariable(self, currentLcName, lcVariables, lcSuccesses, sampleSize, eliminateDuplicateSamples, replace_mul=False):
        lcSampleSize = sampleSize
        if eliminateDuplicateSamples:
            lcSampleSize, indices, Vs = self.eliminateDuplicateSamples(lcVariables, sampleSize)

        # Calculate loss value
        if eliminateDuplicateSamples: 
            lossTensor = torch.index_select(lcSuccesses, dim=0, index=indices)
        else:
            # lossTensor = torch.clone(lcSuccesses)
            if replace_mul:
                lossTensor= torch.zeros(lcSuccesses.shape).to(self.current_device)
            else:
                lossTensor = torch.ones(lcSuccesses.shape).to(self.current_device)

            #lossTensor = countSuccesses.div_(len(lossList))
            
        for i, v in enumerate(lcVariables):
            currentV = lcVariables[v]
            
            if eliminateDuplicateSamples:
                P = currentV[0][:lcSampleSize] # Tensor with the current variable p (v[0])
            else:
                P = currentV[0]
            oneMinusP = torch.sub(torch.ones(lcSampleSize, device=self.current_device), P) # Tensor with the current variable 1-p
            
            if eliminateDuplicateSamples:
                S = Vs[i, :] #currentV[1] # Sample for the current Variable
            else:
                S = currentV[1]
                
            if isinstance(S, list):
                continue
            
            notS = torch.sub(torch.ones(lcSampleSize, device=self.current_device), S.float()) # Negation of Sample
            
            pS = torch.mul(P, S) # Tensor with p multiply by True variable sample
            oneMinusPS = torch.mul(oneMinusP, notS) # Tensor with 1-p multiply by False variable sample
            
            cLoss = torch.add(pS, oneMinusPS) # Sum of p and 1-p tensors
                                            
            # Multiply the loss
            if replace_mul:
                lossTensor = lossTensor + torch.log(cLoss)
            else:
                lossTensor.mul_(cLoss)
        
        # if replace_mul:
        #     lossTensor = torch.exp(lossTensor)
        return lossTensor, lcSampleSize
            
    def generateSemanticSample(self, rootDn, conceptsRelations):
        sampleSize = -1
        
        # Collect master concepts with length of multiclass set
        masterConcepts = OrderedDict()
        productConcepts = []
        productSize = 1
        productArgs = []
        for currentConceptRelation in conceptsRelations:
            currentConceptName = self.getConceptName(currentConceptRelation)
            
            if currentConceptName not in masterConcepts:
                masterConcepts[currentConceptName] = OrderedDict()
                
                masterConcepts[currentConceptName]['e'] = []
                masterConcepts[currentConceptName]['e'].append(currentConceptRelation)
                
                
                rootConcept = rootDn.findRootConceptOrRelation(currentConceptName)
                dns = rootDn.findDatanodes(select = rootConcept)
                masterConcepts[currentConceptName]["dns"] = dns
                
                conceptRange = 2
                if self.conceptIsMultiClass(currentConceptRelation):
                    conceptRange = currentConceptRelation[3]
                masterConcepts[currentConceptName]["range"] = [x for x in range(conceptRange)]
                
                masterConcepts[currentConceptName]['xkey'] = '<' + currentConceptName + '>/sample'
                 
                if self.isConceptFixed(currentConceptName):
                    continue
                
                for _ in dns:
                    productArgs.append(masterConcepts[currentConceptName]["range"])
                    productSize *= conceptRange
                    
                productConcepts.append(currentConceptName)
            else:
                masterConcepts[currentConceptName]['e'].append(currentConceptRelation)

        # Init sample 
        for mConcept in masterConcepts:
            mConceptInfo = masterConcepts[mConcept]
            
            if len(mConceptInfo["e"]) == 1: # Binary concept
                mConceptInfo["binary"] = True
            else:
                mConceptInfo["binary"] = False
            
            for dn in mConceptInfo['dns']:
                if mConceptInfo['xkey'] not in dn.getAttributes():
                    dn.getAttributes()[mConceptInfo['xkey']] = OrderedDict()
                                                                    
                dn.getAttributes()[mConceptInfo['xkey']][sampleSize] = OrderedDict()
                
                for i, e in enumerate(mConceptInfo["e"]):
                    isFiexd = self.isVariableFixed(dn, mConcept, e)
                    
                    if mConceptInfo["binary"]:
                        i = 1
                        
                    if isFiexd != None:
                        if isFiexd == 1:
                            dn.getAttributes()[mConceptInfo['xkey']][sampleSize][i] = torch.ones(productSize, device = self.current_device)
                            continue
                        
                    dn.getAttributes()[mConceptInfo['xkey']][sampleSize][i] = torch.zeros(productSize, device = self.current_device)
             
        from itertools import product
        for j, p in enumerate(product(*productArgs, repeat=1)):
            index = 0
            if mConceptInfo["binary"]:
                index = 1
            for pConcept in productConcepts:
                pConceptInfo = masterConcepts[pConcept]

                for dn in pConceptInfo['dns']:
                    try:
                        dn.getAttributes()[pConceptInfo['xkey']][sampleSize][p[index]][j] = 1
                    except:
                        pass
                    index +=1
        
        return productSize
        
    # -- Calculated loss values for logical constraints
    def calculateLcLoss(self, dn, tnorm='L', sample = False, sampleSize = 0, sampleGlobalLoss = False, conceptsRelations = None):
        start = perf_counter()

        m = None 
        p = 0
        
        self.current_device = dn.current_device

        if sample: 
            p = sampleSize
            if sampleSize == -1: # Semantic Sample
                sampleSize = self.generateSemanticSample(dn, conceptsRelations)
            if sampleSize < -1: 
                raise Exception("Sample size is not incorrect - %i"%(sampleSize))
           
            myBooleanMethods = self.myLcLossSampleBooleanMethods
            myBooleanMethods.sampleSize = sampleSize
                    
            self.myLogger.info('Calculating sample loss with sample size: %i'%(p))
            self.myLoggerTime.info('Calculating sample loss with sample size: %i'%(p))
        else:
            myBooleanMethods = self.myLcLossBooleanMethods
            self.myLcLossBooleanMethods.setTNorm(tnorm)
            
            self.myLogger.info('Calculating loss ')
            self.myLoggerTime.info('Calculating loss ')

        myBooleanMethods.current_device = dn.current_device
        
        key = "/local/softmax"
        
        lcCounter = 0 # Count processed lcs
        lcLosses = {}
        for graph in self.myGraph: # Loop through graphs
            for _, lc in graph.logicalConstrains.items(): # loop trough lcs in the graph
                startLC = perf_counter_ns()

                if not lc.headLC or not lc.active: # Process only active and head lcs
                    continue
                    
                if type(lc) is fixedL: # Skip fixedL lc
                    continue
                    
                lcCounter +=  1
                self.myLogger.info('\n')
                self.myLogger.info('Processing %s(%s) - %s'%(lc.lcName, lc, lc.strEs()))
                
                lcName = lc.lcName
                    
                lcLosses[lcName] = {}
                current_lcLosses = lcLosses[lcName]
                
                # Calculate loss for the given lc
                if sample:
                    # lossList will contain boolean results for lc evaluation for the given sample element
                    # sampleInfo - will contain list of variable exiting in the given lc with their sample and probabilities
                    lossList, sampleInfo, inputLc = self.__constructLogicalConstrains(lc, myBooleanMethods, m, dn, p, key = key, headLC = True, loss = True, sample = sample)
                    current_lcLosses['lossList'] = lossList
                    current_lcLosses['sampleInfo'] = sampleInfo
                    current_lcLosses['input'] = inputLc
                    current_lcLosses['lossRate'] = []
                    for li in lossList:
                        liList = []
                        for l in li:
                            if l != None:
                                liList.append(torch.sum(l)/l.shape[0])
                            else:
                                liList.append(None)
                        
                        current_lcLosses['lossRate'].append(liList)
                else:
                    # lossList will contain float result for lc loss calculation
                    lossList = self.__constructLogicalConstrains(lc, myBooleanMethods, m, dn, p, key = key, headLC = True, loss = True, sample = sample)
                    current_lcLosses['lossList'] = lossList
                             
                endLC = perf_counter_ns()
                elapsedInNsLC = endLC - startLC
                elapsedInMsLC = elapsedInNsLC/1000000
                current_lcLosses['elapsedInMsLC'] = elapsedInMsLC
                            
        if not sample: # Loss value
            for currentLcName in lcLosses:
                startLC = perf_counter_ns()

                current_lcLosses = lcLosses[currentLcName]
                lossList = current_lcLosses['lossList']
                lossTensor = None

                seperateTensorsUsed = False
                if lossList and lossList[0]:
                    if torch.is_tensor(lossList[0][0]) and (len(lossList[0][0].shape) == 0 or len(lossList[0][0].shape) == 1 and lossList[0][0].shape[0] == 1):
                        seperateTensorsUsed = True
                        
                    if seperateTensorsUsed:
                        lossTensor = torch.zeros(len(lossList))#, requires_grad=True) # Entry lcs
                        for i, l in enumerate(lossList):
                            lossTensor[i] = float("nan")
                            for entry in l:
                                if entry is not None:
                                    if lossTensor[i] != lossTensor[i]:
                                        lossTensor[i] = entry
                                    else:
                                        lossTensor[i] += entry.item()
                    else:
                        for entry in lossList[0]:
                            if entry is not None:
                                if lossTensor == None:
                                    lossTensor = entry
                                else:
                                    lossTensor += entry
                    
                current_lcLosses['lossTensor'] = lossTensor
                if lossTensor != None and torch.is_tensor(lossTensor):
                    current_lcLosses['loss'] = torch.nansum(lossTensor).item()
                else:
                    current_lcLosses['loss'] = None

                endLC = perf_counter_ns()
                elapsedInNsLC = endLC - startLC
                elapsedInMsLC = elapsedInNsLC/1000000
                current_lcLosses['elapsedInMsLC'] += elapsedInMsLC

                if lossList is not None:
                    self.myLoggerTime.info('Processing time for %s with %i entries is: %ims'%(currentLcName, len(lossList),  current_lcLosses['elapsedInMsLC']))
                else:
                    self.myLoggerTime.info('Processing time for %s with %i entries is: %ims'%(currentLcName, 0,  current_lcLosses['elapsedInMsLC']))

                [h.flush() for h in self.myLoggerTime.handlers]
            
        else: # -----------Sample
            globalSuccesses = torch.ones(sampleSize, device = self.current_device)
            
            for currentLcName in lcLosses:
                startLC = perf_counter_ns()

                current_lcLosses = lcLosses[currentLcName]
                lossList = current_lcLosses['lossList']
                
                sampleInfo = current_lcLosses['sampleInfo']
                    
                successesList = [] # Entry lcs successes
                lcSuccesses = torch.ones(sampleSize, device = self.current_device) # Consolidated successes for all the entry lcs
                lcVariables = OrderedDict() # Unique variables used in all the entry lcs
                countSuccesses = torch.zeros(sampleSize, device = self.current_device)
                oneT = torch.ones(sampleSize, device = self.current_device)
                
                # Prepare data
                if len(lossList) == 1:
                    for currentFailures in lossList[0]:
                        if currentFailures is None:
                            successesList.append(None)
                            continue
                        
                        currentSuccesses = torch.sub(oneT, currentFailures.float())
                        successesList.append(currentSuccesses)
                            
                        lcSuccesses.mul_(currentSuccesses)
                        globalSuccesses.mul_(currentSuccesses)
                        countSuccesses.add_(currentSuccesses)
                        
                    # Collect lc variable
                    for k in sampleInfo.keys():
                        for c in sampleInfo[k]:
                            if not c:
                                continue
                            c = c[0]
                            if len(c) > 2:                                    
                                if c[2] not in lcVariables:
                                    lcVariables[c[2]] = c
                else:
                    for i, l in enumerate(lossList):
                        for currentFailures in l:
                            if currentFailures is None:
                                successesList.append(None)
                                continue
                            
                            currentSuccesses = torch.sub(oneT, currentFailures.float())
                            successesList.append(currentSuccesses)
                                
                            lcSuccesses.mul_(currentSuccesses)
                            globalSuccesses.mul_(currentSuccesses)
                            countSuccesses.add_(currentSuccesses)
                            
                        # Collect lc variable
                        for k in sampleInfo.keys():
                            for c in sampleInfo[k][i]:
                                if len(c) > 2:                                        
                                    if c[2] not in lcVariables:
                                        lcVariables[c[2]] = c
                
                current_lcLosses['successesList'] = successesList
                current_lcLosses['lcSuccesses'] = lcSuccesses
                current_lcLosses['lcVariables'] = lcVariables
                current_lcLosses['countSuccesses'] = countSuccesses

                endLC = perf_counter_ns()
                elapsedInNsLC = endLC - startLC
                elapsedInMsLC = elapsedInNsLC/1000000
                current_lcLosses['elapsedInMsLC'] += elapsedInMsLC

            lcLosses["globalSuccesses"] = globalSuccesses
            lcLosses["globalSuccessCounter"] = torch.nansum(globalSuccesses).item()
            self.myLoggerTime.info('Global success counter is %i '%(lcLosses["globalSuccessCounter"]))
            
            for currentLcName in lcLosses:
                if currentLcName in ["globalSuccessCounter", "globalSuccesses"]:
                    continue
                
                startLC = perf_counter_ns()

                current_lcLosses = lcLosses[currentLcName]
                lossList = current_lcLosses['lossList']
                
                successesList = current_lcLosses['successesList']
                lcSuccesses = current_lcLosses['lcSuccesses']
                lcVariables = current_lcLosses['lcVariables']
                        
                #lcSuccessesSum = torch.sum(lcSuccesses).item()
                                
                eliminateDuplicateSamples = False # Eliminate duplicate samples
                
                # --- Calculate sample loss for lc variables
                current_lcLosses['lossTensor'] = []
                current_lcLosses['lcSuccesses'] = []
                current_lcLosses['lcVariables'] = []
                current_lcLosses['loss'] = []
                
                # Per each lc entry separately
                if lc.sampleEntries:
                    current_lcLosses['lcSuccesses'] = successesList
                    
                    for i, l in enumerate(lossList):
                        currentLcVariables = OrderedDict()
                        for k in sampleInfo.keys():
                            for c in sampleInfo[k][i]:
                                if len(c) > 2:                                        
                                    if c[2] not in currentLcVariables:
                                        currentLcVariables[c[2]] = c
                                        
                        
                        usedLcSuccesses = successesList[i]
                        if sampleGlobalLoss:
                            usedLcSuccesses = globalSuccesses
                        currentLossTensor, _ = self.calulateSampleLossForVariable(currentLcVariables, usedLcSuccesses, sampleSize, eliminateDuplicateSamples)
                        
                        current_lcLosses['lossTensor'].append(currentLossTensor)
                        current_lcLosses['lcVariables'].append(currentLcVariables)
    
                        currentLoss = torch.nansum(currentLossTensor).item() # Sum of losses across sample for x|=alfa
                        current_lcLosses['loss'].append(currentLoss)
    
                else: # Regular calculation for all lc entries at once
                    usedLcSuccesses = lcSuccesses
                    if sampleGlobalLoss:
                        usedLcSuccesses = globalSuccesses
                    lossTensor, lcSampleSize = self.calulateSampleLossForVariable(currentLcName, lcVariables, usedLcSuccesses, sampleSize, eliminateDuplicateSamples)
                    current_lcLosses['loss'].append(torch.nansum(lossTensor).item()) # Sum of losses across sample for x|=alfa
    
                    current_lcLosses['lossTensor'].append(lossTensor)
                    current_lcLosses['lcSuccesses'].append(lcSuccesses)
                    current_lcLosses['lcVariables'].append(lcVariables)
            
                # Calculate processing time
                endLC = perf_counter_ns()
                elapsedInNsLC = endLC - startLC
                elapsedInMsLC = elapsedInNsLC/1000000
                current_lcLosses['elapsedInMsLC'] += elapsedInMsLC

                if lc.sampleEntries:
                    self.myLoggerTime.info('Processing time for %s with %i entries and %i variables is: %ims'
                                           %(currentLcName, len(lossList), len(lcVariables), current_lcLosses['elapsedInMsLC']))
                if eliminateDuplicateSamples: 
                    self.myLoggerTime.info('Processing time for %s with %i entries, %i variables and %i unique samples is: %ims'
                                           %(currentLcName, len(lossList), len(lcVariables), lcSampleSize, current_lcLosses['elapsedInMsLC']))
                else:
                    self.myLoggerTime.info('Processing time for %s with %i entries and %i variables is: %ims'
                                           %(currentLcName, len(lossList), len(lcVariables), current_lcLosses['elapsedInMsLC']))
        
        self.myLogger.info('')

        self.myLogger.info('Processed %i logical constraints'%(lcCounter))
        self.myLoggerTime.info('Processed %i logical constraints'%(lcCounter))
            
        end = perf_counter()
        elapsedInS = end - start
        
        if elapsedInS > 1:
            self.myLogger.info('End of Loss Calculation - total internl time: %fs'%(elapsedInS))
            self.myLoggerTime.info('End of Loss Calculation - total internl time: %fs'%(elapsedInS))
        else:
            elapsedInMs = (end - start) *1000
            self.myLogger.info('End of Loss Calculation - total internl time: %ims'%(elapsedInMs))
            self.myLoggerTime.info('End of Loss Calculation - total internl time: %ims'%(elapsedInMs))
            
        self.myLogger.info('')

        [h.flush() for h in self.myLoggerTime.handlers]
            
        return lcLosses
    
    # -- Calculated loss values for logical constraints
    def verifyResultsLC(self, dn, key = "/argmax"):
        start = perf_counter()

        m = None 
        p = 0
        
        myBooleanMethods = self.booleanMethodsCalculator
        myBooleanMethods.current_device = dn.current_device

        self.myLogger.info('Calculating verify ')
        self.myLoggerTime.info('Calculating verify ')
        
        lcCounter = 0 # Count processed lcs
        lcVerifyResult = OrderedDict()
        for graph in self.myGraph: # Loop through graphs
            for _, lc in graph.logicalConstrains.items(): # loop trough lcs in the graph
                startLC = perf_counter_ns()

                if not lc.headLC or not lc.active: # Process only active and head lcs
                    continue
                    
                if type(lc) is fixedL: # Skip fixedL lc
                    continue
                    
                lcCounter +=  1
                self.myLogger.info('\n')
                self.myLogger.info('Processing %s(%s) - %s'%(lc.lcName, lc, lc.strEs()))
                
                lcName = lc.lcName
                    
                lcVerifyResult[lcName] = {}
                current_verifyResult = lcVerifyResult[lcName]
                
                # verifyList will contain result for the lc verification
                verifyList, lcVariables = self.__constructLogicalConstrains(lc, myBooleanMethods, m, dn, p, key = key, headLC = True, verify=True)
                current_verifyResult['verifyList'] = verifyList
                
                verifyListLen = 0
                verifyListSatisfied = 0
                for vl in verifyList:
                    verifyListLen += len(vl)
                    verifyListSatisfied += sum(vl)
                
                if verifyListLen:
                    current_verifyResult['satisfied'] = (verifyListSatisfied / verifyListLen) * 100
                else:
                    current_verifyResult['satisfied'] = 0 # float("nan")
                    
                # If  this if logical constraints
                if type(lc) is ifL or type(lc) is forAllL: # if LC
                    firstKey = next(iter(lcVariables)) # antecedent variable name in the given if LC
                    firstLcV = lcVariables[firstKey]   # values of antecedent 
                    
                    ifVerifyList = []
                    
                    ifVerifyListLen = 0
                    ifVerifyListSatisfied = 0
                    
                    # Loop through the collected verification 
                    for i, v in enumerate(verifyList):
                        ifVi = []
                        # Check if the number of elements in verify list is the same as number of elements in antecedent variables
                        if len(v) != len(firstLcV[i]):
                            continue # should be exception
                        
                        # Select these calculated verification which have antecedent true
                        for j, w in enumerate(v):
                            if torch.is_tensor(firstLcV[i][j]):
                                currentAntecedent = firstLcV[i][j].item()
                            else: 
                                currentAntecedent = firstLcV[i][j] 
                                
                            if currentAntecedent == 1: # antecedent true
                                ifVi.append(w)
                                    
                        ifVerifyList.append(ifVi)
                        
                        ifVerifyListLen += len(ifVi)
                        ifVerifyListSatisfied += sum(ifVi)
                    
                    current_verifyResult['ifVerifyList'] = ifVerifyList
                    if ifVerifyListLen:
                        current_verifyResult['ifSatisfied'] = (ifVerifyListSatisfied / ifVerifyListLen) *100
                    else:
                        current_verifyResult['ifSatisfied'] = 0 #float("nan")

                endLC = perf_counter_ns()
                elapsedInNsLC = endLC - startLC
                elapsedInMsLC = elapsedInNsLC/1000000
                current_verifyResult['elapsedInMsLC'] = elapsedInMsLC
        
        self.myLogger.info('Processed %i logical constraints'%(lcCounter))
        self.myLoggerTime.info('Processed %i logical constraints'%(lcCounter))
            
        end = perf_counter()
        elapsedInS = end - start
        
        if elapsedInS > 1:
            self.myLogger.info('End of Verify Calculation - total time: %fs'%(elapsedInS))
            self.myLoggerTime.info('End of Verify Calculation - total time: %fs'%(elapsedInS))
        else:
            elapsedInMs = (end - start) *1000
            self.myLogger.info('End of Verify Calculation - total time: %ims'%(elapsedInMs))
            self.myLoggerTime.info('End of Verify Calculation - total time: %ims'%(elapsedInMs))
            
        self.myLogger.info('')
        self.myLoggerTime.info('')

        [h.flush() for h in self.myLoggerTime.handlers]
        return lcVerifyResult