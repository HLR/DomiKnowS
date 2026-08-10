import math
from time import perf_counter, perf_counter_ns
from collections import OrderedDict
import logging
import os

from colorama import Fore, Style

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
from domiknows.solver.booleanMethodsCalculator import booleanMethodsCalculator
from domiknows.solver.circuitBooleanMethods import circuitBooleanMethods
from domiknows.solver.circuitLossCalculator import CircuitLossCalculator

from domiknows.graph import LcElement, LogicalConstrain, V, fixedL, ifL, forAllL
from domiknows.graph import CandidateSelection
from domiknows.utils import _default_log_dir
from domiknows.utils import getReuseModel
from domiknows.utils import getDnSkeletonMode

from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor

class gurobiILPOntSolver(ilpOntSolver):
    ilpSolver = 'Gurobi'

    def __init__(self, graph, ontologiesTuple, _ilpConfig, reuse_model=False) -> None:
        super().__init__(graph, ontologiesTuple, _ilpConfig)
        self.myIlpBooleanProcessor = gurobiILPBooleanProcessor()
        self.myLcLossBooleanMethods = lcLossBooleanMethods()
        self.myLcLossSampleBooleanMethods = lcLossSampleBooleanMethods()
        self.booleanMethodsCalculator = booleanMethodsCalculator()
        self.myCircuitBooleanMethods = circuitBooleanMethods()
        self.constraintConstructor = LogicalConstraintConstructor(self.myLogger)
        self.circuitLossCalculator = CircuitLossCalculator(self)

        self.logical_constraints = {}
        for g in graph:
            self.logical_constraints = {**self.logical_constraints, **g.logicalConstrains}

        self.reuse_model = reuse_model
        if getReuseModel():
            self.reuse_model = True
            
        self.model = collections.deque([], 20)

    def configureCircuitBackend(self, *, backend=None, max_nodes=None, size_limit_action=None):
        """Reconfigure the exact circuit backend and discard incompatible cache."""
        current = self.myCircuitBooleanMethods
        backend = current.requested_backend if backend is None else backend
        max_nodes = current.max_nodes if max_nodes is None else max_nodes
        size_limit_action = (
            current.size_limit_action
            if size_limit_action is None
            else size_limit_action
        )
        requested = (backend, max_nodes, size_limit_action)
        active = (
            current.requested_backend,
            current.max_nodes,
            current.size_limit_action,
        )
        if requested != active:
            self.myCircuitBooleanMethods = circuitBooleanMethods(
                backend=backend,
                max_nodes=max_nodes,
                size_limit_action=size_limit_action,
            )
            self.circuitLossCalculator._compile_cache.clear()

    def calculateCircuitLoss(
        self,
        dn,
        *,
        backend=None,
        max_nodes=None,
        size_limit_action=None,
        aggregation=None,
    ):
        """Calculate exact ``-log(WMC)`` losses for active head constraints."""
        self.configureCircuitBackend(
            backend=backend,
            max_nodes=max_nodes,
            size_limit_action=size_limit_action,
        )
        return self.circuitLossCalculator.calculateCircuitLoss(
            dn, aggregation=aggregation)
        
    def set_logical_constraints(self, new_logical_constraints):
        self.logical_constraints = new_logical_constraints
        
    def get_logical_constraints(self, ):
        return self.logical_constraints
    
    def reset_logical_constraints(self, ):
        self.logical_constraints = self.myGraph[0].logicalConstrains ### Can myGraph really be multiple graphs?
        
    def getConcept(self, concept):
        return self.constraintConstructor.getConcept(concept)
    
    def getConceptName(self, concept):
        return self.constraintConstructor.getConceptName(concept)
    
    def conceptIsBinary(self, concept):
        return self.constraintConstructor.conceptIsBinary(concept)
    
    def conceptIsMultiClass(self, concept):
        return self.constraintConstructor.conceptIsMultiClass(concept)
    
    def valueToBeSkipped(self, x):
        return self.constraintConstructor.valueToBeSkipped(x)
    
    def getDatanodesForConcept(self, rootDn, currentName, conceptToDNSCash=None):
        return self.constraintConstructor.getDatanodesForConcept(rootDn, currentName, conceptToDNSCash)
    
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
            value = torch.zeros(2, dtype=torch.float) if self.conceptIsMultiClass(conceptRelation) else valueI.squeeze(0)
            value[0] = 1 - valueI[labelIndex] if self.conceptIsMultiClass(conceptRelation) else value[0]
            value[1] = valueI[labelIndex] if self.conceptIsMultiClass(conceptRelation) else value[1]
            
        # If softmax process probability through function and apply epsilon
        if "softmax" in key and value is not None and not torch.isnan(value[0]).item() and epsilon is not None:
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
                        Q = currentProbability[1].detach() * xNew
                    else:
                        Q += currentProbability[1].detach() * xNew       
                            
                    if self.conceptIsBinary(currentConceptRelation): 
                        Q += currentProbability[0].detach() * xNotNew    
                
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
            key = "/ILP/xP"
        
        for lc in lcs:   
            startLC = perf_counter_ns()
            m.update()
            startNumConstrs = m.NumConstrs
            
            if lc.active:
                self.myLogger.info('Processing %r - %s'%(lc, lc.strEs()))
            else:
                self.myLogger.debug('Skipping not active Logical Constraint %r - %s'%(lc,  [str(e) for e in lc.e]))
                continue

            lcRepr = f'{lc.__class__.__name__} {lc.strEs()}'
            
            # Use the constraint constructor
            self.constraintConstructor.current_device = self.current_device
            self.constraintConstructor.myGraph = self.myGraph
            result, _ = self.constraintConstructor.constructLogicalConstrains(
                lc, self.myIlpBooleanProcessor, m, dn, p, key=key, headLC=True)
            
            m.update()
            endNumConstrs = m.NumConstrs
            newNumConstrs = endNumConstrs - startNumConstrs
            endLC = perf_counter_ns()
            elapsedInNsLC = endLC - startLC
            elapsedInMsLC = elapsedInNsLC/1000000
            
            if result != None and isinstance(result, list):
                if newNumConstrs:
                    self.myLogger.info('Successfully added Logical Constraint %r - created %i new ILP constraint\n'%(lc, newNumConstrs))
                    self.myLoggerTime.info('Processing time for %r is: %ims - created %i new ILP constraint'%(lc, elapsedInMsLC, newNumConstrs))
                else:
                    self.myLogger.info('Finished processing Logical Constraint %r - no new ILP constraint was created for it\n'%(lc))
                    self.myLoggerTime.info('Processing time for %r is: %ims - no new ILP constraint was created'%(lc, elapsedInMsLC))
            else:
                self.myLogger.error('Failed to add Logical Constraint %r\n'%(lc))
                self.myLoggerTime.error('Failed to add Logical Constraint %r'%(lc))
                
    # ---------------
                
    # -- Main method of the solver - creating ILP constraints plus objective, invoking the ILP solver and returning the result of the ILP solver classification
    def calculateILPSelection(self, dn, *conceptsRelations, key = ("local" , "softmax"), fun=None, epsilon = 0.00001, minimizeObjective = False, ignorePinLCs = False):
        """Run ordinary ILP inference and populate the DataNode in place.

        The signature and in-place return contract of this public entry point
        intentionally remain unchanged.  Hypothesis-aware inference uses the
        private variant below so it can compare detached candidate solutions
        before writing exactly one winner.
        """
        self._calculateILPSelection(
            dn,
            *conceptsRelations,
            key=key,
            fun=fun,
            epsilon=epsilon,
            minimizeObjective=minimizeObjective,
            ignorePinLCs=ignorePinLCs,
        )

    def _calculateILPSelection(
        self,
        dn,
        *conceptsRelations,
        key=("local", "softmax"),
        fun=None,
        epsilon=0.00001,
        minimizeObjective=False,
        ignorePinLCs=False,
        extraLogicalConstraints=(),
        populate=True,
        forceFreshModel=False,
        raiseOnInfeasible=True,
    ):
        """Internal ILP run used by executable-constraint hypotheses.

        ``extraLogicalConstraints`` are hard constraints added to every
        priority run.  When ``populate`` is false the method returns a
        detached objective/assignment dictionary and leaves final ``/ILP``
        tensors untouched.  Temporary Gurobi variable handles are still
        created on the DataNode because the logical-constraint compiler uses
        them; callers must clear those handles between fresh models.
        """
        if self.ilpSolver == None:
            self.myLogger.warning('ILP solver not provided - returning')
            self.myLoggerTime.warning('ILP solver not provided - returning')
            
            return None
        
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
            if self.reuse_model and not forceFreshModel:
                # Find it there is saved ILP model with this count of instances per concept
                for modelDec in self.model:
                    if ilpVarCount == modelDec[0]:
                        m = modelDec[1]
                        x = modelDec[2]
                        reusingModel = True
                        break
                    
            if not reusingModel:
                # If not reusing the model or if the right model was yet saved - create new Gurabi model
                if forceFreshModel:
                    m = Model("hypothesisClassificationResult" + str(start), gurobiEnv)
                elif dn.gurobiModel == None:
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
            dn.setActiveExecutableLCs() # Set active executive LCs in the data node if executive LC datanode set
            _lcP = {}
            _lcP[100] = list(extraLogicalConstraints)
            pUsed = False
            for graph in self.myGraph:
                for _, lc in graph.logicalConstrains.items():
                    if lc.headLC and lc.active: # Process only active and head lcs
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
                self.myLogger.info('Found %i active logical constraints with p %i - %s\n'%(len(lcP[p]),p,lcP[p]))
                self.myLoggerTime.info('Starting ILP interference - Found %i active logical constraints'%(len(lcP[p])))

            # Search through set of logical constraints for subset satisfying and the max/min calculated objective value
            lcRun = {} # Keeps information about subsequent model runs
            ps = [] # List with processed p 
            
            #  -----------  Run ILP solver for each p
            for p in lcP:
                self.processILPModelForP(
                    p,
                    lcP,
                    m,
                    x,
                    dn,
                    pUsed,
                    reusingModel,
                    ilpVarCount,
                    minimizeObjective,
                    lcRun,
                    cacheModel=not forceFreshModel,
                )

            endOptimize = perf_counter()

            # ----------- Select model run with the max/min objective value 
            maxP = None
            for p in lcRun:
                if lcRun[p]['solved']:
                    if maxP is not None:
                        if minimizeObjective and lcRun[maxP]['objValue'] >= lcRun[p]['objValue']:
                            maxP = p
                        elif not minimizeObjective and lcRun[maxP]['objValue'] <= lcRun[p]['objValue']:
                            maxP = p
                    else:
                        maxP = p
               
            selectionResult = None

            # ----------- If solution found model - return best result
            if maxP is not None:
                self.myLogger.info('Best  solution found for p - %i'%(maxP))
                #self.myLoggerTime.info('Best  solution found for p - %i'%(maxP))
                
                selectedModel = lcRun[maxP]['mP']
                selectedVariables = lcRun[maxP]['xP']
                selectedModel.update()
                detachedValues = self._snapshotILPVariables(selectedVariables)
                selectionResult = {
                    'objective': lcRun[maxP]['objValue'],
                    'values': detachedValues,
                    'variables': selectedVariables,
                    'priority': maxP,
                }

                if populate:
                    self.populateILPSelection(
                        dn,
                        conceptsRelations,
                        detachedValues,
                        variables=selectedVariables,
                    )
                    
            else:
                end = perf_counter()
                elapsedInS = end - start
                if elapsedInS > 1:
                    self.myLogger.info('ILP Model Infeasible no solution found - internal total time: %fs'%(elapsedInS))
                    self.myLoggerTime.info('ILP Model Infeasible no solution found - internal total time: %fs'%(elapsedInS))
                else:
                    elapsedInMs = elapsedInS *1000
                    self.myLogger.info('ILP Model Infeasible no solution found - internal total time: %ims'%(elapsedInMs))
                    self.myLoggerTime.info('ILP Model Infeasible no solution found - internal total time: %ims'%(elapsedInMs))

                self.myLogger.info('')

                # Raise exception if no solution found
                if raiseOnInfeasible:
                    raise Exception('ILP Model Infeasible - no solution found')
                                       
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
        return selectionResult

    @staticmethod
    def _snapshotILPVariables(variables):
        """Detach a solved Gurobi variable mapping into plain numeric values."""
        values = {}
        for variable_key, variable in variables.items():
            if variable is None:
                values[variable_key] = None
                continue

            value = variable.X
            if value == 0:
                value = 0
            elif value == 1:
                value = 1
            values[variable_key] = value

        return values

    def populateILPSelection(self, dn, conceptsRelations, values, *, variables=None):
        """Write a detached ILP assignment into the standard DataNode outputs.

        ``values`` uses the same tuple keys as ``createILPVariables``.  Keeping
        this writer independent of a live Gurobi model lets AnswerSolver retain
        the best hypothesis assignment after later candidate models have been
        constructed and disposed.
        """
        solutions_to_log = set()
        initialized_outputs = set()
        use_skeleton = getDnSkeletonMode() and "variableSet" in dn.attributes
        output_device = getattr(self, 'current_device', dn.current_device)

        for concept_relation in conceptsRelations:
            index = 0 if concept_relation[2] is None else concept_relation[2]
            concept = self.getConcept(concept_relation)
            concept_name = concept_relation[0].name
            label = concept_relation[1]
            label_count = concept_relation[3]

            root_concept = dn.findRootConceptOrRelation(concept_relation[0])
            root_dns = dn.findDatanodes(select=root_concept)
            ilp_key = f'<{concept_name}>/ILP'
            x_key = ilp_key + '/x'
            not_x_key = ilp_key + '/notx'
            output_identity = (root_concept.name, concept_name)

            ilp_tensor = None
            skeleton_key = None
            if use_skeleton:
                skeleton_key = root_concept.name + "/" + ilp_key
                if output_identity not in initialized_outputs:
                    dn.attributes["variableSet"][skeleton_key] = torch.full(
                        (len(root_dns), label_count),
                        float("nan"),
                        dtype=torch.float,
                        device=output_device,
                    )
                ilp_tensor = dn.attributes["variableSet"][skeleton_key]

            for dn_index, concept_dn in enumerate(root_dns):
                attributes = concept_dn.getAttributes()
                if not use_skeleton and output_identity not in initialized_outputs:
                    attributes[ilp_key] = torch.full(
                        (label_count,),
                        float("nan"),
                        dtype=torch.float,
                        device=output_device,
                    )

                variable_key = (
                    concept,
                    label,
                    concept_dn.getInstanceID(),
                    index,
                )
                solution = values.get(variable_key)
                if solution is not None:
                    if ilp_tensor is not None:
                        ilp_tensor[dn_index][index] = solution
                    else:
                        attributes[ilp_key][index] = solution

                    if solution == 1 and not label.startswith("not_"):
                        solutions_to_log.add((concept_dn, label))

                if variables is not None:
                    attributes.setdefault(x_key, [None] * label_count)
                    attributes[x_key][index] = variables.get(variable_key)

                    if concept_relation[2] is None:
                        not_variable_key = (
                            concept,
                            'Not_' + label,
                            concept_dn.getInstanceID(),
                            index,
                        )
                        attributes.setdefault(not_x_key, [None] * label_count)
                        attributes[not_x_key][index] = variables.get(not_variable_key)

            initialized_outputs.add(output_identity)

            if ilp_tensor is not None:
                dn.attributes["variableSet"][skeleton_key] = ilp_tensor

        if solutions_to_log:
            self.log_sorted_solutions(solutions_to_log)

    def log_sorted_solutions(self, solutions_to_log):
        """
        Remove duplicates, sort solutions by datanode name and concept, then log them
        in alphabetical order with red colour (console only) for items with multiple solutions.
        """
        if not solutions_to_log:
            return
        
        # Group solutions by datanode to count how many solutions each has
        datanode_counts = {}
        for dn, concept in solutions_to_log:
            dn_str = str(dn)
            if dn_str not in datanode_counts:
                datanode_counts[dn_str] = 0
            datanode_counts[dn_str] += 1
        
        for dn, concept in sorted(solutions_to_log, key=lambda x: (str(x[0]), x[1])):
            dn_str = str(dn)
            # Only use red color if this datanode has more than one solution
            if datanode_counts[dn_str] > 1:
                self.myLogger.info('%s"%s" is "%s"%s', Fore.RED, dn_str, concept, Style.RESET_ALL)
            else:
                self.myLogger.info('"%s" is "%s"', dn_str, concept)
            
    def processILPModelForP(
        self,
        p,
        lcP,
        m,
        x,
        dn,
        pUsed,
        reusingModel,
        ilpVarCount,
        minimizeObjective,
        lcRun,
        cacheModel=True,
    ):
        ps = []
        ps.append(p)
        
        startLogicalConstraintsPrep = perf_counter()

        if pUsed:
            mP = m.copy()  # Copy model for this run
            xP = {}
            lckey = "/ILP/xP"
            
            pStart = perf_counter()
            
            for _x in x:
                # Map variables to the new copy model
                xP[_x] = mP.getVarByName(x[_x].VarName)
                
                rootConcept = dn.findRootConceptOrRelation(_x[0])
                
                dns = dn.findDatanodes(select=((rootConcept,), ("instanceID", _x[2])))
                
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
                    self.myLoggerTime.info('ILP Model init for p %i - time: %ims' % (p, (pEnd - pStart) * 1000))
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
        elapsedLogicalConstraintsPrepInMs = (endLogicalConstraintsPrep - startLogicalConstraintsPrep) * 1000
        self.myLoggerTime.info('ILP Logical Constraints Preprocessing - time: %ims' % (elapsedLogicalConstraintsPrepInMs))
        
        if pUsed or not reusingModel:
            self.addLogicalConstrains(mP, dn, lcs, p, key=lckey)  # <--- LC constraints
            
            # Save model
            if self.reuse_model and cacheModel:
                self.model.append((ilpVarCount, mP, xP))
                import sys
                memoryUsage = sys.getsizeof(mP)
                # Convert bytes to kilobytes
                memoryUsage_kB = memoryUsage / 1024
                self.myLoggerTime.info(f'ILP Logical Constraints Preprocessing - memory use by saved Gurobi models: {memoryUsage_kB:.2f} kB')
        
        self.myLogger.info('Optimizing model for LCs with probabilities %s with %i ILP variables and %i ILP constraints' % (p, mP.NumVars, mP.NumConstrs))
        self.myLoggerTime.info('Optimizing model for LCs with probabilities %s with %i ILP variables and %i ILP constraints' % (p, mP.NumVars, mP.NumConstrs))
        
        endLogicalConstraints = perf_counter()
        elapsedLogicalConstraintsInMs = (endLogicalConstraints - endLogicalConstraintsPrep) * 1000
        self.myLoggerTime.info('ILP Logical Constraints - time: %ims' % (elapsedLogicalConstraintsInMs))
        
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
            self.myLogger.info('%s solution was found in %ims for p - %i with optimal value: %.2f' % ('Min' if minimizeObjective else 'Max', elapsedOptimizeInMs, p, mP.ObjVal))
            self.myLoggerTime.info('%s solution was found in %ims for p - %i with optimal value: %.2f' % ('Min' if minimizeObjective else 'Max', elapsedOptimizeInMs, p, mP.ObjVal))
            solved = True
            objValue = mP.ObjVal
        elif mP.status == GRB.Status.INFEASIBLE:
            self.myLogger.error('Model was proven to be infeasible for p - %i.' % (p))
            self.myLoggerTime.error('Model was proven to be infeasible for p - %i.' % (p))
        elif mP.status == GRB.Status.INF_OR_UNBD:
            self.myLogger.error('Model was proven to be infeasible or unbound for p - %i.' % (p))
            self.myLoggerTime.error('Model was proven to be infeasible or unbound for p - %i.' % (p))
        elif mP.status == GRB.Status.UNBOUNDED:
            self.myLogger.error('Model was proven to be unbound.')
            self.myLoggerTime.error('Model was proven to be unbound.')
        else:
            self.myLogger.error('Optimal solution not was found for p - %i - error code %i' % (p, mP.status))
            self.myLoggerTime.error('Optimal solution not was found for p - %i - error code %i' % (p, mP.status))
        
         # ----------- Write model to file if logging level is INFO
        _log_dir = _default_log_dir()
        os.makedirs(_log_dir, exist_ok=True)

        if self.myLogger.level <= logging.INFO:
            model_path = os.path.join(_log_dir, "GurobiModel.lp")
            if os.path.exists(model_path):
                os.remove(model_path)
            mP.write(model_path) # Write model to file

        # ----------- Write infeasible model to file if model was proven to be infeasible or solution when found
        infeasible_path = os.path.join(_log_dir, "GurobiInfeasible.ilp")
        sol_path = os.path.join(_log_dir, "GurobiSolution.sol")
        json_path = os.path.join(_log_dir, "GurobiSolution.json")

        if not solved:
            # Remove solution files if they exist from previous runs
            if os.path.exists(sol_path):
                os.remove(sol_path)
            if os.path.exists(json_path):
                os.remove(json_path)
            mP.computeIIS()
            if os.path.exists(infeasible_path):
                os.remove(infeasible_path)
            mP.write(infeasible_path)
        elif self.myLogger.level <= logging.INFO:
            # Remove infeasible file and solution files if they exist from previous runs
            if os.path.exists(infeasible_path):
                os.remove(infeasible_path)
            if os.path.exists(sol_path):
                os.remove(sol_path)
            if os.path.exists(json_path):
                os.remove(json_path)
            mP.write(sol_path) # Write solution to file
            mP.write(json_path) # Write solution to file in json format extended
        
        # Keep result of the model run
        lcRun[p] = {'p': p, 'solved': solved, 'objValue': objValue, 'lcs': lcs, 'mP': mP, 'xP': xP, 'elapsedOptimize': elapsedOptimizeInMs}
