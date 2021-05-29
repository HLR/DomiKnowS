from datetime import datetime
from collections import OrderedDict
from collections.abc import Mapping
import logging
# ontology
from owlready2 import And, Or, Not, FunctionalProperty, InverseFunctionalProperty, ReflexiveProperty, SymmetricProperty, AsymmetricProperty, IrreflexiveProperty, TransitiveProperty

# numpy
import numpy as np

# pytorch
import torch

# Gurobi
from gurobipy import GRB, Model

from regr.graph.concept import Concept
from regr.solver.ilpOntSolver import ilpOntSolver
from regr.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from regr.solver.lcLossBooleanMethods import lcLossBooleanMethods
from regr.graph import LogicalConstrain, V

class gurobiILPOntSolver(ilpOntSolver):
    ilpSolver = 'Gurobi'

    def __init__(self, graph, ontologiesTuple, _ilpConfig) -> None:
        super().__init__(graph, ontologiesTuple, _ilpConfig)
        self.myIlpBooleanProcessor = gurobiILPBooleanProcessor()
        self.myLcLossBooleanMethods = lcLossBooleanMethods()
        
    def valueToBeSkipped(self, x):
        return ( 
                x != x or  # nan 
                abs(x) == float('inf')  # inf 
                ) 
    
    # Get Ground Truth for provided concept
    def __getLabel(self, dn, conceptRelation, fun=None, epsilon = None):
        value = dn.getAttribute(conceptRelation, 'label')
        
        return value
        
    # Get and calculate probability for provided concept
    def __getProbability(self, dn, conceptRelation, fun=None, epsilon = 0.00001):
        if not dn:
            valueI = None
        else:
            valueI = dn.getAttribute(conceptRelation, "local" , "softmax")
                    
        if valueI is None: # No probability value - return negative probability 
            return [float("nan"), float("nan")]
        
        if conceptRelation[2] is not None:
            value = torch.empty(2, dtype=torch.float)
            
            value[0] = 1 - valueI[conceptRelation[2]]
            value[1] = valueI[conceptRelation[2]]
        else:
            value = valueI

        # Process probability through function and apply epsilon
        if epsilon is not None:
            if value[0] > 1-epsilon:
                value[0] = 1-epsilon
            elif value[1] > 1-epsilon:
                value[1] = 1-epsilon
                
            if value[0] < epsilon:
                value[0] = epsilon
            elif value[1] < epsilon:
                value[1] = epsilon
               
            # Apply fun on probabilities 
            if fun is not None:
                value = fun(value)
            
        return value # Return probability
    
    def createILPVariables(self, m, rootDn, *conceptsRelations, dnFun = None, fun=None, epsilon = 0.00001):
        x = {}
        Q = None
        
        # Create ILP variables 
        for _conceptRelation in conceptsRelations: 
            rootConcept = rootDn.findRootConceptOrRelation(_conceptRelation[0])
            dns = rootDn.findDatanodes(select = rootConcept)
                        
            for dn in dns:
                currentProbability = dnFun(dn, _conceptRelation, fun=fun, epsilon=epsilon)
                
                if currentProbability == None or (torch.is_tensor(currentProbability) and currentProbability.dim() == 0) or len(currentProbability) < 2:
                    self.myLogger.warning("Probability not provided for variable concept %s in dataNode %s - skipping it"%(_conceptRelation[0].name,dn.getInstanceID()))

                    continue
                
                # Check if probability is NaN or if and has to be skipped
                if self.valueToBeSkipped(currentProbability[1]):
                    self.myLogger.info("Probability is %f for concept %s and dataNode %s - skipping it"%(currentProbability[1],_conceptRelation[1],dn.getInstanceID()))
                    continue
    
                # Create variable
                xNew = m.addVar(vtype=GRB.BINARY,name="x_%s_is_%s"%(dn.getInstanceID(), _conceptRelation[1])) 
                xkey = '<' + _conceptRelation[0].name + '>/ILP/x'
                
                if xkey not in dn.attributes:
                    dn.attributes[xkey] = [None] * _conceptRelation[3]
                    
                if _conceptRelation[2] is not None:
                    dn.attributes[xkey][_conceptRelation[2]] = xNew
                    x[_conceptRelation[0], _conceptRelation[1], dn.getInstanceID(), _conceptRelation[2]] = xNew
                else:
                    dn.attributes[xkey][0] = xNew
                    x[_conceptRelation[0], _conceptRelation[1], dn.getInstanceID(), 0] = xNew

                Q += currentProbability[1] * xNew       
    
                # Check if probability is NaN or if and has to be created based on positive value
                if self.valueToBeSkipped(currentProbability[0]):
                    currentProbability[0] = 1 - currentProbability[1]
                    self.myLogger.info("No ILP negative variable for concept %s and dataNode %s - created based on positive value %f"%(dn.getInstanceID(), _conceptRelation[0].name, currentProbability[1]))
    
                # Create negative variable for binary concept
                if _conceptRelation[2] is None: # ilpOntSolver.__negVarTrashhold:
                    xNotNew = m.addVar(vtype=GRB.BINARY,name="x_%s_is_not_%s"%(dn.getInstanceID(),  _conceptRelation[1]))
                    notxkey = '<' + _conceptRelation[0].name + '>/ILP/notx'
                
                    if notxkey not in dn.attributes:
                        dn.attributes[notxkey] = [None] * _conceptRelation[3]
                    
                    if _conceptRelation[2] is not None:
                        dn.attributes[notxkey][_conceptRelation[2]] = xNotNew
                        x[_conceptRelation[0], 'Not_'+_conceptRelation[1], dn.getInstanceID(), _conceptRelation[2]] = xNotNew
                    else:
                        dn.attributes[notxkey][0] = xNotNew
                        x[_conceptRelation[0], 'Not_'+_conceptRelation[1], dn.getInstanceID(), 0] = xNotNew
                                        
                    Q += currentProbability[0] * xNotNew    

                else:
                    self.myLogger.info("No ILP negative variable for concept %s and dataNode %s created"%( _conceptRelation[1], dn.getInstanceID()))

        m.update()

        if len(x):
            self.myLogger.info("Created %i ILP variables"%(len(x)))
        else:
            self.myLogger.warning("No ILP variables created")
            
        return Q, x     
    
    def addGraphConstrains(self, m, rootDn, *conceptsRelations):
        # Add constrain based on probability 
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
                self.myLogger.debug("Disjoint constrain between variable %s is  %s and variable %s is not - %s == %i"%(dn.getInstanceID(),_conceptRelation[1],dn.getInstanceID(),'Not_'+_conceptRelation[1],1))

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
                        
                    self.myIlpBooleanProcessor.countVar(m, dn.getAttribute(cxkey)[0], dn.getAttribute(dxkey)[0], onlyConstrains = True,  limitOp = '<=', limit = 1, logicMethodName = "atMostL")
                        
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
                        
            for arg_id, rel in enumerate(concept[0].has_a()): 
                
                if  not rel.auto_constraint:
                    continue
                
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
                            
            self.myLogger.info("Created - doman/range constraints for concepts %s"%(concept[2]))
                    
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
                
            for d in currentConcept.disjoints():
                disjointConcept = d.entities[1]._name
                    
                if currentConcept._name == disjointConcept:
                    disjointConcept = d.entities[0]._name
                        
                    if currentConcept._name == disjointConcept:
                        continue
                        
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
        
    def addLogicalConstrains(self, m, dn, lcs, p):
        self.myLogger.info('Starting method')
        
        key = "/ILP/xP" # to get ILP variable from datanodes
        
        for lc in lcs:   
            
            if lc.active:
                self.myLogger.info('Processing Logical Constrain %s(%s) - %s'%(lc.lcName, lc, [str(e) for e in lc.e]))
            else:
                self.myLogger.info('Skipping not active Logical Constrain %s(%s) - %s'%(lc.lcName, lc, [str(e) for e in lc.e]))
                continue

            result = self.__constructLogicalConstrains(lc, self.myIlpBooleanProcessor, m, dn, p, key = key,  lcVariablesDns = {}, headLC = True)
            
            if result != None and isinstance(result, list):
                self.myLogger.info('Successfully added Logical Constrain %s'%(lc.lcName))
            else:
                self.myLogger.error('Failed to add Logical Constrain %s'%(lc.lcName))

    def __constructLogicalConstrains(self, lc, booleanProcesor, m, dn, p, key = "", lcVariablesDns = {}, headLC = False):
        lcVariables = {}
        vNo = 0
        firstV = True
        
        for eIndex, e in enumerate(lc.e): 
            if  isinstance(e, V):
                continue # already processed in the previous Concept 
            
            if isinstance(e, (Concept,  LogicalConstrain, tuple)): 
                # Look one step ahead in the parsed logical constrain and get variables names (if present) after the current concept
                if eIndex + 1 < len(lc.e) and isinstance(lc.e[eIndex+1], V):
                    variable = lc.e[eIndex+1]
                else:
                    if isinstance(e, LogicalConstrain):
                        variable = V(name="_lc" + str(vNo))
                        vNo += 1
                    else:
                        if firstV:
                            variable = V(name="_x" )
                            firstV = False
                        else:
                            variable = V(name="_x" + str(vNo), v = ("_x",))
                            vNo += 1
                    
                if variable.name:
                    variableName = variable.name
                else:
                    variableName = "V" + str(eIndex)
                    
                if variableName in lcVariables:
                    newvVariableName = "_x" + str(vNo)
                    vNo += 1
                    
                    lcVariablesDns[newvVariableName] = lcVariablesDns[variableName]
                    if None in lcVariablesDns:
                        pass
  
                    lcVariables[newvVariableName] = lcVariables[variableName]

                elif isinstance(e, (Concept, tuple)): # -- Concept 
                    if isinstance(e, Concept):
                        conceptName = e.name
                    else:
                        conceptName = e[0].name
                        
                    xPkey = '<' + conceptName + ">" + key

                    dnsList = [] # Stores lists of dataNode for each corresponding dataNode 
                    vDns = [] # Stores ILP variables
                    
                    if variable.v == None:
                        if variable.name == None:
                            self.myLogger.error('The element %s of logical constrain %s has no name for variable'%(conceptName, lc.lcName))
                            return None
                                                 
                        rootConcept = dn.findRootConceptOrRelation(conceptName)
                        _dns = dn.findDatanodes(select = rootConcept)
                        dnsList = [[dn] for dn in _dns]
                    else:
                        if len(variable.v) == 0:
                            self.myLogger.error('The element %s of logical constrain %s has no empty part v of the variable'%(conceptName, lc.lcName))
                            return None
                          
                        path = variable.v
  
                        paths = []
                        lo = None
                        
                        if isinstance(path[0], str) and len(path) == 1:
                            paths.append(path)
                        elif isinstance(path[0], str) and not isinstance(path[1], tuple):
                            paths.append(path)
                        else:
                            for i, vE in enumerate(variable.v):
                                if i == 0 and isinstance(vE, str):
                                    lo = vE 
                                    continue
                                
                                paths.append(vE)
                                
                        _dnsList = []
                        for i, v in enumerate(paths):
                            _dnsList.append([])
                            referredVariableName = v[0] # Get name of the referred variable already defined in the logical constrain from the v part 
                        
                            if referredVariableName not in lcVariablesDns:
                                self.myLogger.error('The element %s of logical constrain %s has v referring to undefined variable %s'%(conceptName, lc.lcName, referredVariableName))
                                return None
                           
                            referredDns = lcVariablesDns[referredVariableName] # Get Datanodes for referred variables already defined in the logical constrain
                            for rDn in referredDns:
                                eDns = []
                                for _rDn in rDn:
                                    _eDns = _rDn.getEdgeDataNode(v[1:]) # Get Datanodes for the edge defined by the path part of the v
                                    
                                    if _eDns and _eDns[0]:
                                        eDns.extend(_eDns)
                                    else:
                                        self.myLogger.error('The element %s of logical constrain %s has v referring to not valid path %s'%(conceptName, lc.lcName, v[1:]))
                                        eDns.extend([None])
                                        
                                _dnsList[i].append(eDns)
                                
                        dnsList = _dnsList[0]
                            
                        for l in _dnsList[1:]:
                            # Intersection - use lo if defined to determine if different set operation
                            _d = [x if x in l else [None] for x in dnsList]
                            dnsList = _d
                            
                    # Get ILP variables from collected Datanodes for the given element of logical constrain
                    for dns in dnsList:
                        _vDns = []
                        for _dn in dns:
                            if _dn == None:
                                _vDns.append(None)
                                continue
                            
                            if _dn.ontologyNode.name == conceptName:
                                _vDns.append(1)
                                continue
                            
                            if xPkey not in _dn.attributes:
                                _vDns.append(None)
                                continue
                            
                            if isinstance(e, Concept):
                                ilpVs = _dn.getAttribute(xPkey) # Get ILP variable for the concept 
                                
                                if isinstance(ilpVs, Mapping) and p not in ilpVs:
                                    _vDns.append(None)
                                    continue
                                
                                vDn = ilpVs[p]
                            else:
                                if p == 0:
                                    try:
                                        vDn = _dn.getAttribute(xPkey)[e[1]] # Get ILP variable for the concept 
                                    except IndexError: 
                                        vDn = None
                                else:
                                    vDn = _dn.getAttribute(xPkey)[p][e[2]] # Get ILP variable for the concept 
                        
                            if torch.is_tensor(vDn):
                                vDn = vDn.item()  
                                                              
                            _vDns.append(vDn)
                        
                        vDns.append(_vDns)
                        
                    lcVariablesDns[variable.name] = dnsList
                    
                    if None in lcVariablesDns:
                        pass
                    
                    lcVariables[variableName] = vDns
                
                elif isinstance(e, LogicalConstrain): # LogicalConstrain - process recursively 
                    self.myLogger.info('Processing Logical Constrain %s(%s) - %s'%(e.lcName, e, [str(e1) for e1 in e.e]))
                    vDns = self.__constructLogicalConstrains(e, booleanProcesor, m, dn, p, key = key, lcVariablesDns = lcVariablesDns, headLC = False)
                    
                    if vDns == None:
                        self.myLogger.warning('Not found data for %s(%s) nested logical Constrain required to build Logical Constrain %s(%s) - skipping this constrain'%(e.lcName,e,lc.lcName,lc))
                        return None
                        
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
                self.myLogger.error('Logical Constrain %s has incorrect element %s'%(lc,e))
                return None
        
        return lc(m, booleanProcesor, lcVariables, headConstrain = headLC)
    
    # ---------------
                
    # -- Main method of the solver - creating ILP constraints plus objective, invoking the ILP solver and returning the result of the ILP solver classification  
    def calculateILPSelection(self, dn, *conceptsRelations, fun=None, epsilon = 0.00001, minimizeObjective = False, ignorePinLCs = False):
        if self.ilpSolver == None:
            self.myLogger.warning('ILP solver not provided - returning')
            return 
        
        start = datetime.now()
        
        try:
            # Create a new Gurobi model
            self.myIlpBooleanProcessor.resetCaches()
            m = Model("decideOnClassificationResult" + str(start))
            m.params.outputflag = 0
            
            # Create ILP Variables for concepts and objective
            Q, x = self.createILPVariables(m, dn, *conceptsRelations, dnFun = self.__getProbability, fun=fun, epsilon = epsilon)
            
            # Add constraints based on ontology and graph definition
            self.addOntologyConstrains(m, dn, *conceptsRelations)
            self.addGraphConstrains(m, dn, *conceptsRelations)
        
            # ILP Model objective setup
            if Q is None:
                Q = 0
                self.myLogger.error("No data provided to create any ILP varibale - not ILP result reurn")
                
            if minimizeObjective:
                m.setObjective(Q, GRB.MINIMIZE)
            else:
                m.setObjective(Q, GRB.MAXIMIZE) # -------- Default

            m.update()
            
            # Collect head logical constraints
            _lcP = {}
            _lcP[100] = []
            for graph in self.myGraph:
                for _, lc in graph.logicalConstrains.items():
                    if lc.headLC:     
                        if not ignorePinLCs:
                            lcP = lc.p
                        else:
                            lcP = 100   
                                            
                        if lcP not in _lcP:
                            _lcP[lcP] = []
                        
                        _lcP[lcP].append(lc) # Keep constrain with the same p in the list 
            
            # Sort constraints according to their p
            lcP = OrderedDict(sorted(_lcP.items(), key=lambda t: t[0], reverse = True))
            for p in lcP:
                self.myLogger.info('Found %i logical constraints with p %i - %s'%(len(lcP[p]),p,lcP[p]))

            # Search through set of logical constraints for subset satisfying and the mmax/min calculated objective value
            lcRun = {} # Keeps information about subsequent model runs
            ps = [] # List with processed p 
            for p in lcP:
                ps.append(p)
                mP = m.copy() # Copy model for this run
                
                # Map variables to the new copy model
                xP = {}
                for _x in x:
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
                                    
                # Prepare set with logical constraints for this run
                lcs = []
                for _p in lcP:
                    lcs.extend(lcP[_p])

                    if _p == p:
                        break     
    
                # Add LC constraints to the copy model
                self.addLogicalConstrains(mP, dn, lcs, p)
                self.myLogger.info('Optimizing model for logical constraints with probabilities %s with %i variables and %i constraints'%(p,mP.NumVars,mP.NumConstrs))

                startOptimize = datetime.now()

                # Run ILP model - Find solution 
                mP.optimize()
                mP.update()
                
                #mP.display()    
                
                endOptimize = datetime.now()
                elapsedOptimize = endOptimize - startOptimize
    
                # check model run result
                solved = False
                objValue = None
                if mP.status == GRB.Status.OPTIMAL:
                    self.myLogger.info('%s optimal solution was found for p - %i with value %f - solver time: %ims'%('Min' if minimizeObjective else 'Max', p, mP.ObjVal,elapsedOptimize.microseconds/1000))
                    solved = True
                    objValue = mP.ObjVal
                elif mP.status == GRB.Status.INFEASIBLE:
                    self.myLogger.warning('Model was proven to be infeasible for p - %i.'%(p))
                elif mP.status == GRB.Status.INF_OR_UNBD:
                    self.myLogger.warning('Model was proven to be infeasible or unbound for p - %i.'%(p))
                elif mP.status == GRB.Status.UNBOUNDED:
                    self.myLogger.warning('Model was proven to be unbound.')
                else:
                    self.myLogger.warning('Optimal solution not was found for p - %i - error code %i'%(p,mP.status))
                 
                # Print ILP model to log file if model is not solved or logger level is DEBUG
                if not solved or logging.DEBUG >= self.myLogger.level:
                    import sys
                    so = sys.stdout 
                    logFileName = self.myLogger.handlers[0].baseFilename
                    log = open(logFileName, "a")
                    sys.stdout = log
                    mP.display() 
                    sys.stdout = so

                # Keep result of the model run    
                lcRun[p] = {'p':p, 'solved':solved, 'objValue':objValue, 'lcs':lcs, 'mP':mP, 'xP':xP, 'elapsedOptimize':elapsedOptimize.microseconds/1000}

            # Select model run with the max/min objective value 
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
               
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # If found model - return best result          
            if maxP:
                self.myLogger.info('Best  solution found for p - %i'%(maxP))
                
                lcRun[maxP]['mP'].update()
                
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
                   
                    for cDn in c_root_dns:
                        dnAtt = cDn.getAttributes()
                        if xPkey not in dnAtt:
                            dnAtt[ILPkey] = torch.tensor([float("nan")], device=device) 
                            self.myLogger.info('Not returning solutions for %s in %sit is nan'%(c[1], cDn))
                            continue
                        
                        solution = dnAtt[xPkey][maxP][index].X
                        if solution == 0:
                            solution = 0
                        elif solution == 1: 
                            solution = 1

                        if ILPkey not in dnAtt:
                            dnAtt[ILPkey] = torch.empty(c[3], dtype=torch.float)
                        
                        if xkey not in dnAtt:
                            dnAtt[xkey] = torch.empty(c[3], dtype=torch.float)
                       
                        dnAtt[ILPkey][index] = solution
                        dnAtt[xkey][index] = dnAtt[xPkey][maxP][index]
                        if xNotPkey in dnAtt:
                            dnAtt[xNotPkey][index] = dnAtt[xNotPkey][maxP][index]

                        ILPV = dnAtt[ILPkey][index]
                        if ILPV == 1:
                            self.myLogger.info('\"%s\" is \"%s\"'%(cDn,c[1]))
                        
            else:
                pass
                                       
        except Exception as inst:
            self.myLogger.error('Error returning solutions -  %s'%(inst))
            raise
           
        end = datetime.now()
        elapsed = end - start
        self.myLogger.info('')
        self.myLogger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # Return
        return
                
    # -- Calculated loss values for logical constraints
    def calculateLcLoss(self, dn):
        m = None 
                
        p = 0
        key = "/local/softmax"
        #key = ""
        
        lcLosses = {}
        for graph in self.myGraph:
            for _, lc in graph.logicalConstrains.items():
                if not lc.headLC:
                    continue
                    
                self.myLogger.info('Processing Logical Constrain %s(%s) - %s'%(lc.lcName, lc, [str(e) for e in lc.e]))
                lossList = self.__constructLogicalConstrains(lc, self.myLcLossBooleanMethods, m, dn, p, key = key, lcVariablesDns = {}, headLC = True)
                
                if not lossList:
                    continue
                
                lossTensor = torch.zeros(len(lossList))
                
                for i, l in enumerate(lossList):
                    l = l[0]
                    if l is not None:
                        lossTensor[i] = l
                    else:
                        lossTensor[i] = float("nan")
               
                lcLosses[lc.lcName] = {}
                
                lcLosses[lc.lcName]['lossTensor'] = lossTensor
                
        return lcLosses