from time import process_time
from collections import OrderedDict
import logging
# ontology
#from owlready2 import And, Or, Not, FunctionalProperty, InverseFunctionalProperty, ReflexiveProperty, SymmetricProperty, AsymmetricProperty, IrreflexiveProperty, TransitiveProperty

# pytorch
import torch

# Gurobi
from gurobipy import GRB, Model, Var

from regr.graph.concept import Concept, EnumConcept
from regr.solver.ilpOntSolver import ilpOntSolver
from regr.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from regr.solver.lcLossBooleanMethods import lcLossBooleanMethods
from regr.solver.lcLossSampleBooleanMethods import lcLossSampleBooleanMethods

from regr.graph import LogicalConstrain, V, fixedL
class gurobiILPOntSolver(ilpOntSolver):
    ilpSolver = 'Gurobi'

    def __init__(self, graph, ontologiesTuple, _ilpConfig, reuse_model=False) -> None:
        super().__init__(graph, ontologiesTuple, _ilpConfig)
        self.myIlpBooleanProcessor = gurobiILPBooleanProcessor()
        self.myLcLossBooleanMethods = lcLossBooleanMethods()
        self.myLcLossSampleBooleanMethods = lcLossSampleBooleanMethods()

        self.reuse_model = reuse_model
        self.model = None
        
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
    
    def createILPVariables(self, m, x, rootDn, *conceptsRelations, dnFun = None, fun=None, epsilon = 0.00001):
        Q = None
        
        # Create ILP variables 
        for _conceptRelation in conceptsRelations: 
            rootConcept = rootDn.findRootConceptOrRelation(_conceptRelation[0])
            dns = rootDn.findDatanodes(select = rootConcept)
            xkey = '<' + _conceptRelation[0].name + '>/ILP/x'  
       
            for dn in dns:
                currentProbability = dnFun(dn, _conceptRelation, fun=fun, epsilon=epsilon)
                
                if currentProbability == None or (torch.is_tensor(currentProbability) and currentProbability.dim() == 0) or len(currentProbability) < 2:
                    self.myLogger.warning("Probability not provided for variable concept %s in dataNode %s - skipping it"%(_conceptRelation[0].name,dn.getInstanceID()))

                    continue
                
                # Check if probability is NaN or if and has to be skipped
                if self.valueToBeSkipped(currentProbability[1]):
                    self.myLogger.info("Probability is %f for concept %s and dataNode %s - skipping it"%(currentProbability[1],_conceptRelation[1],dn.getInstanceID()))
                    continue
    
                xNew = None
                if _conceptRelation[2] is not None:
                    if (_conceptRelation[0], _conceptRelation[1], dn.getInstanceID(), _conceptRelation[2]) in x:
                        xNew = x[_conceptRelation[0], _conceptRelation[1], dn.getInstanceID(), _conceptRelation[2]]
                else:
                    if (_conceptRelation[0], _conceptRelation[1], dn.getInstanceID(), 0) in x:
                        xNew = x[_conceptRelation[0], _conceptRelation[1], dn.getInstanceID(), 0] = xNew
                
                if xkey not in dn.attributes:
                    dn.attributes[xkey] = [None] * _conceptRelation[3]
                        
                if xNew is None:
                    # Create variable
                    xVarName = "%s_%s_is_%s"%(dn.getOntologyNode(), dn.getInstanceID(), _conceptRelation[1])
                    xNew = m.addVar(vtype=GRB.BINARY,name=xVarName) 
                    
                    if _conceptRelation[2] is not None:
                        x[_conceptRelation[0], _conceptRelation[1], dn.getInstanceID(), _conceptRelation[2]] = xNew
                    else:
                        x[_conceptRelation[0], _conceptRelation[1], dn.getInstanceID(), 0] = xNew

                if _conceptRelation[2] is not None:
                    dn.attributes[xkey][_conceptRelation[2]] = xNew
                else:
                    dn.attributes[xkey][0] = xNew
                    
                Q += currentProbability[1] * xNew       
    
                # Check if probability is NaN or if and has to be created based on positive value
                if self.valueToBeSkipped(currentProbability[0]):
                    currentProbability[0] = 1 - currentProbability[1]
                    self.myLogger.info("No ILP negative variable for concept %s and dataNode %s - created based on positive value %f"
                                       %(dn.getInstanceID(), _conceptRelation[0].name, currentProbability[1]))
    
                # Create negative variable for binary concept
                if _conceptRelation[2] is None: # ilpOntSolver.__negVarTrashhold:
                    xNotNew  = None
                    
                    if _conceptRelation[2] is not None:
                        if (_conceptRelation[0], 'Not_'+_conceptRelation[1], dn.getInstanceID(), _conceptRelation[2]) in x:
                            xNotNew= x[_conceptRelation[0], 'Not_'+_conceptRelation[1], dn.getInstanceID(), _conceptRelation[2]]
                    else:
                        if (_conceptRelation[0], 'Not_'+_conceptRelation[1], dn.getInstanceID(), 0) in x:
                            xNotNew = x[_conceptRelation[0], 'Not_'+_conceptRelation[1], dn.getInstanceID(), 0]
                    
                    notxkey = '<' + _conceptRelation[0].name + '>/ILP/notx'
                
                    if notxkey not in dn.attributes:
                        dn.attributes[notxkey] = [None] * _conceptRelation[3]
                        
                    if xNotNew is None:
                        xNotNew = m.addVar(vtype=GRB.BINARY,name="x_%s_is_not_%s"%(dn.getInstanceID(),  _conceptRelation[1]))
                        if _conceptRelation[2] is not None:
                            x[_conceptRelation[0], 'Not_'+_conceptRelation[1], dn.getInstanceID(), _conceptRelation[2]] = xNotNew
                        else:
                            x[_conceptRelation[0], 'Not_'+_conceptRelation[1], dn.getInstanceID(), 0] = xNotNew
                        
                    if _conceptRelation[2] is not None:
                        dn.attributes[notxkey][_conceptRelation[2]] = xNotNew
                    else:
                        dn.attributes[notxkey][0] = xNotNew
                                        
                    Q += currentProbability[0] * xNotNew    
                
            if _conceptRelation[2] is not None:
                self.myLogger.info("No creating ILP negative variables for multiclass concept %s"%( _conceptRelation[1]))
                
        # Create constraint for multiclass exclusivity 
        if _conceptRelation[2] is not None:
            m.update()

            for dn in dns:
                var = []
                for _conceptRelation in conceptsRelations: 
                    currentV = dn.attributes[xkey][_conceptRelation[2]]
                    
                    if currentV:
                        var.append(currentV)
                
                     
                self.myIlpBooleanProcessor.countVar(m, *var, onlyConstrains = True, limitOp = '==', limit = 1, logicMethodName = "MultiClass")

        m.update()

        if len(x):
            self.myLogger.info("Created %i ILP variables"%(len(x)))
        else:
            self.myLogger.warning("No ILP variables created")
            
        return Q
    
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
        
    def addLogicalConstrains(self, m, dn, lcs, p):
        self.myLogger.info('Starting method')
        
        key = "/ILP/xP" # to get ILP variable from datanodes
        
        for lc in lcs:   
            
            if lc.active:
                self.myLogger.info('Processing Logical Constrain %s(%s) - %s'%(lc.lcName, lc, lc.strEs()))
            else:
                self.myLogger.info('Skipping not active Logical Constrain %s(%s) - %s'%(lc.lcName, lc, [str(e) for e in lc.e]))
                continue

            result = self.__constructLogicalConstrains(lc, self.myIlpBooleanProcessor, m, dn, p, key = key,  lcVariablesDns = {}, headLC = True)
            
            if result != None and isinstance(result, list):
                self.myLogger.info('Successfully added Logical Constrain %s'%(lc.lcName))
            else:
                self.myLogger.error('Failed to add Logical Constrain %s'%(lc.lcName))

    def isVariableFixed(self, dn, conceptName, e):
        
        for graph in self.myGraph: # Loop through graphs
            for _, lc in graph.logicalConstrains.items(): # loop trough lcs in the graph
                if not lc.headLC or not lc.active: # Process only active and head lcs
                    continue
                    
                if type(lc) is not fixedL: # Skip not fixedL lc
                    continue
                
                lcN = lc
                # TO be finished
                
        # For spec
        if "fixed" in dn.getAttributes() and dn.getAttributes()["fixed"].item() == 1:
            vDnLabel = self.__getLabel(dn, conceptName).item()

            if vDnLabel == e[1]:
                return 1
            else:
                return 0
            
        return None
        
    def getMLResult(self, dn, conceptName, xPkey, e, p, loss = False, sample = False):
        if dn == None:
            raise Exception("No datanode provided")
                        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sampleKey = '<' + conceptName + ">/sample" 
        if sample and sampleKey not in dn.getAttributes():
            dn.getAttributes()[sampleKey] = {}
        
        if dn.ontologyNode.name == conceptName:
            if not sample:
                return 1
            else:
                sampleSize = p
                
                if sampleSize not in dn.getAttributes()[sampleKey]: 
                    dn.getAttributes()[sampleKey][sampleSize] = {}
                    
                xVarName = "%s_%s_is_%s"%(dn.getOntologyNode(), dn.getInstanceID(), e[1])

                dn.getAttributes()[sampleKey][sampleSize][e[1]] = torch.ones(sampleSize, dtype=torch.bool, device = device)
                xP = torch.ones(sampleSize, device = device)
                
                return (dn.getAttributes()[sampleKey][sampleSize][e[1]], (xP, dn.getAttributes()[sampleKey][sampleSize][e[1]], xVarName))
        
        if xPkey not in dn.attributes:
            if not sample:
                return None
            else:   
                return ([None], (None, [None]))
        
        if loss: # Loss calculation
            try:
                vDn = dn.getAttribute(xPkey)[e[1]] # Get value for the concept 
            except IndexError: 
                vDn = None
        else: # If ILP inference
            vDn = dn.getAttribute(xPkey)[p][e[2]] # Get ILP variable for the concept 
    
        if torch.is_tensor(vDn) and (len(vDn.shape) == 0 or len(vDn.shape) == 1 and vDn.shape[0] == 1):
            vDn = vDn.item()  
             
        if not sample:
            return vDn # Return here if not sample
        
        # --- Generate sample 
        sampleSize = p

        xVarName = "%s_%s_is_%s"%(dn.getOntologyNode(), dn.getInstanceID(), e[1])
        
        isFiexd = self.isVariableFixed(dn, conceptName, e)
        
        if isFiexd != None:
            if isFiexd == 1:
                xP = torch.ones(sampleSize, device = device)
            else:
                xP = torch.zeros(sampleSize, device = device)
        else:
            xV = dn.getAttribute(xPkey)
            xEp = dn.getAttribute(xPkey).expand(sampleSize, len(xV))
            xP = xEp[:,e[1]]
                
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
        
    def __constructLogicalConstrains(self, lc, booleanProcesor, m, dn, p, key = None, lcVariablesDns = None, headLC = False, loss = False, sample = False):
        if key == None:
            key = ""
        if lcVariablesDns == None:
            lcVariablesDns = {}
            
        lcVariables = {}
        if sample:
            sampleInfo = {}
        vNo = 0
        firstV = True
        
        for eIndex, e in enumerate(lc.e): 
            if  isinstance(e, V):
                continue # Already processed in the previous Concept 
            
            if isinstance(e, (Concept,  LogicalConstrain, tuple)): 
                # Look one step ahead in the parsed logical constraint and get variables names (if present) after the current concept
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
                    conceptName = e[0].name
                        
                    # -- Collect dataNode for the logical constraint (path)
                    
                    dnsList = [] # Stores lists of dataNodes for each corresponding dataNode 
                    
                    if variable.v == None: # No path - just concept
                        if variable.name == None:
                            self.myLogger.error('The element %s of logical constraint %s has no name for variable'%(conceptName, lc.lcName))
                            return None
                                                 
                        rootConcept = dn.findRootConceptOrRelation(conceptName)
                        _dns = dn.findDatanodes(select = rootConcept)
                        dnsList = [[dn] for dn in _dns]
                    else: # Path specified
                        if len(variable.v) == 0:
                            self.myLogger.error('The element %s of logical constraint %s has empty part v of the variable'%(conceptName, lc.lcName))
                            return None
                          
                        # -- Prepare paths
                        path = variable.v
                        paths = []
                        
                        if isinstance(path[0], str) and len(path) == 1:
                            paths.append(path)
                        elif isinstance(path[0], str) and not isinstance(path[1], tuple):
                            paths.append(path)
                        else:
                            for i, vE in enumerate(variable.v):
                                if i == 0 and isinstance(vE, str):
                                    continue
                                
                                paths.append(vE)
                                
                        # -- Process  paths
                        dnsListForPaths = []
                        for i, v in enumerate(paths):
                            dnsListForPaths.append([])
                            
                            # Get name of the referred variable 
                            referredVariableName = v[0] 
                        
                            if referredVariableName not in lcVariablesDns: # Not yet defined - it has to be the current lc element dataNodes list
                                rootConcept = dn.findRootConceptOrRelation(conceptName)
                                _dns = dn.findDatanodes(select = rootConcept)
                                referredDns = [[dn] for dn in _dns]
                            else: # already defined in the logical constraint from the v part 
                                referredDns = lcVariablesDns[referredVariableName] # Get DataNodes for referred variables already defined in the logical constraint
                                
                            # Get variables from dataNodes selected  based on referredVariableName
                            for rDn in referredDns:
                                eDns = []
                                
                                for _rDn in rDn:
                                    if _rDn is None:
                                        continue
                                    
                                    # -- Get DataNodes for the edge defined by the path part of the v
                                    _eDns = _rDn.getEdgeDataNode(v[1:]) 
                                    
                                    if _eDns and _eDns[0]:
                                        eDns.extend(_eDns)
                                    else:
                                        vNames = [v if isinstance(v, str) else v.name for v in v[1:]]
                                        if lc.__str__() != "fixedL":
                                            self.myLogger.info('The graph node %s has no path %s requested by logical constraint %s for concept %s '%
                                                               (_rDn, vNames, lc.lcName, conceptName))
                                        eDns.extend([None])
                                        
                                dnsListForPaths[i].append(eDns)
                           
                        # ----------- Fix this - TODO: use all the list -----
                        dnsList = dnsListForPaths[0]
                           
                        # -- Combine the collected lists of dataNodes based on paths 
                        for l in dnsListForPaths[1:]:
                            # --- Assume Intersection - TODO: in future use lo if defined to determine if different  operation
                            _d = []
                            for i in range(len(l)):
                                di = []
                                for x in dnsList[i]:
                                    if x in l[i]:
                                        di.append(x)
                                        
                                if not di:
                                    di = [None]
                                    
                                _d.append(di)
                                
                            dnsList = _d
                            
                    # -- Get ILP variables from collected DataNodes for the given element of logical constraint
                    
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
                                    eT = (e[0], i, i)
                                    if sample:
                                        vDn, vDnSampleInfo = self.getMLResult(_dn, conceptName, xPkey, eT, p, loss = loss, sample=sample)
                                        
                                        _sampleInfoForVariable.append(vDnSampleInfo)
                                    else:
                                        vDn = self.getMLResult(_dn, conceptName, xPkey, eT, p, loss = loss, sample=sample)
                                    
                                    if lc.__str__() == "fixedL":
                                        vDn = self.fixedLSupport(_dn, conceptName, vDn, i, m)
                                        
                                    _vDns.append(vDn)
                            elif isinstance(e[0], EnumConcept) and e[2] != None: # Multiclass concept label
                                eT = (e[0], e[2], e[2])
                                
                                if sample:
                                    vDn, vDnSampleInfo = self.getMLResult(_dn, conceptName, xPkey, eT, p, loss = loss, sample=sample)                                    
                                    _sampleInfoForVariable.append(vDnSampleInfo)
                                else:
                                    vDn = self.getMLResult(_dn, conceptName, xPkey, eT, p, loss = loss, sample=sample)
                                
                                if lc.__str__() == "fixedL":
                                    self.fixedLSupport(_dn, conceptName, vDn, e[2], m)
                                    
                                vDn = _vDns.append(vDn)
                            else: # Binary concept
                                eT = (conceptName, 1, 0)
                                if sample:
                                    vDn, vDnSampleInfo = self.getMLResult(_dn, conceptName, xPkey, eT, p, loss = loss, sample=sample)

                                    _sampleInfoForVariable.append(vDnSampleInfo)
                                else:
                                    vDn = self.getMLResult(_dn, conceptName, xPkey, eT, p, loss = loss, sample=sample)
                                
                                if lc.__str__() == "fixedL":
                                    self.fixedLSupport(_dn, conceptName, vDn, 1, m)
                                        
                                vDn = _vDns.append(vDn)
                        
                        vDns.append(_vDns)
                        if sample:
                            sampleInfoForVariable.append(_sampleInfoForVariable)
                        
                    # -- Store dataNodes and ILP variables
                    
                    lcVariablesDns[variableName] = dnsList
                    
                    if None in lcVariablesDns:
                        pass
                    
                    lcVariables[variableName] = vDns
                    
                    if sample:
                        sampleInfo[variableName] = sampleInfoForVariable
                
                elif isinstance(e, LogicalConstrain): # -- nested LogicalConstrain - process recursively 
                    self.myLogger.info('Processing Nested Logical Constrain %s(%s) - %s'%(e.lcName, e, e.strEs()))
                    if sample:
                        vDns, sampleInfoLC = self.__constructLogicalConstrains(e, booleanProcesor, m, dn, p, key = key, 
                                                                               lcVariablesDns = lcVariablesDns, headLC = False, loss = loss, sample = sample)
                        sampleInfo = {**sampleInfo, **sampleInfoLC} # sampleInfo|sampleInfoLC in python 9
                    else:
                        vDns = self.__constructLogicalConstrains(e, booleanProcesor, m, dn, p, key = key, 
                                                                 lcVariablesDns = lcVariablesDns, headLC = False, loss = loss, sample = sample)
                    
                    if vDns == None:
                        self.myLogger.warning('Not found data for %s(%s) nested logical Constrain required to build Logical Constrain %s(%s) - skipping this constraint'%
                                              (e.lcName,e,lc.lcName,lc))
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
        if sample:
            return lc(m, booleanProcesor, lcVariables, headConstrain = headLC), sampleInfo
        else:
            return lc(m, booleanProcesor, lcVariables, headConstrain = headLC)
    
    # ---------------
                
    # -- Main method of the solver - creating ILP constraints plus objective, invoking the ILP solver and returning the result of the ILP solver classification  
    def calculateILPSelection(self, dn, *conceptsRelations, fun=None, epsilon = 0.00001, minimizeObjective = False, ignorePinLCs = False):
        if self.ilpSolver == None:
            self.myLogger.warning('ILP solver not provided - returning')
            self.myLoggerTime.warning('ILP solver not provided - returning')
            
            return 
        
        start = process_time() # timer()
        
        try:
            if self.reuse_model and self.model:
                m = self.model['m']
                x = self.model['x']
            else:
                # Create a new Gurobi model
                self.myIlpBooleanProcessor.resetCaches()
                m = Model("decideOnClassificationResult" + str(start))
                m.params.outputflag = 0
                x = {}
                
            # Create ILP Variables for concepts and objective
            Q = self.createILPVariables(m, x, dn, *conceptsRelations, dnFun = self.__getProbability, fun=fun, epsilon = epsilon)
                
            if self.model is None:
                # Add constraints based on ontology and graph definition
                self.addOntologyConstrains(m, dn, *conceptsRelations)
                self.addGraphConstrains(m, dn, *conceptsRelations)
                
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
                            pUsed = True
                        
                        _lcP[lcP].append(lc) # Keep constraint with the same p in the list 
            
            # Sort constraints according to their p
            lcP = OrderedDict(sorted(_lcP.items(), key=lambda t: t[0], reverse = True))
            for p in lcP:
                self.myLogger.info('Found %i logical constraints with p %i - %s'%(len(lcP[p]),p,lcP[p]))
                self.myLoggerTime.info('Starting ILP inferencing - Found %i logical constraints'%(len(lcP[p])))
                
            # Search through set of logical constraints for subset satisfying and the mmax/min calculated objective value
            lcRun = {} # Keeps information about subsequent model runs
            ps = [] # List with processed p 
            for p in lcP:
                ps.append(p)
                
                if pUsed:
                    mP = m.copy() # Copy model for this run                    
                    xP = {}
                else:
                    mP = m
                    xP = x
                    
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
                    
                # Prepare set with logical constraints for this run
                lcs = []
                for _p in lcP:
                    lcs.extend(lcP[_p])

                    if _p == p:
                        break     
    
                # Add LC constraints to the copy model
                
                if pUsed or self.model is None:
                    self.addLogicalConstrains(mP, dn, lcs, p)
                    
                    if self.reuse_model:
                        self.model = {}
                        self.model['m'] = mP
                        self.model['x'] = xP
                
                    
                self.myLogger.info('Optimizing model for lCs with probabilities %s with %i ILP variables and %i ILP constraints'%(p,mP.NumVars,mP.NumConstrs))
                self.myLoggerTime.info('Optimizing model for lCs with probabilities %s with %i ILP variables and %i ILP constraints'%(p,mP.NumVars,mP.NumConstrs))

                startOptimize = process_time() # timer()

                # Run ILP model - Find solution 
                mP.optimize()
                mP.update()
                
                #mP.display()    
                
                endOptimize = process_time() # timer()
                elapsedOptimizeInMs = (endOptimize - startOptimize) * 1000
    
                # Check model run result
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
                if (not solved or self.myLogger.level <= logging.INFO) and self.myLogger.filter(""):
                    import sys
                    so = sys.stdout 
                    logFileName = self.myLogger.handlers[0].baseFilename
                    log = open(logFileName, "a")
                    sys.stdout = log
                    mP.display() 
                    sys.stdout = so

                # Keep result of the model run    
                lcRun[p] = {'p':p, 'solved':solved, 'objValue':objValue, 'lcs':lcs, 'mP':mP, 'xP':xP, 'elapsedOptimize':elapsedOptimizeInMs}

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
                #self.myLoggerTime.info('Best  solution found for p - %i'%(maxP))
                
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
                            #self.myLoggerTime.info('\"%s\" is \"%s\"'%(cDn,c[1]))
            else:
                pass
                                       
        except Exception as inst:
            self.myLogger.error('Error returning solutions -  %s'%(inst))
            self.myLoggerTime.error('Error returning solutions -  %s'%(inst))
            
            raise
           
        end = process_time() # timer()
        elapsedInS = end - start
        
        self.myLogger.info('')
        
        self.myLogger.info('End ILP Inferencing - elapsed time: %is'%(elapsedInS))
        self.myLoggerTime.info('End ILP Inferencing - elapsed time: %is'%(elapsedInS))
        self.myLogger.info('')
        self.myLoggerTime.info('')
        
        # Return
        return

    # -- Calculated loss values for logical constraints
    def calculateLcLoss(self, dn, tnorm='L', sample = False, sampleSize = 0, sampleGlobalLoss = False):
        start = process_time() # timer()

        m = None 
        p = 0
        
        if sample: 
            if sampleSize <= 0: 
                raise Exception("Sample size is not incorrect - %i"%(sampleSize))
            p = sampleSize
            
            myBooleanMethods = self.myLcLossSampleBooleanMethods
                    
            self.myLogger.info('Calculating sample loss with sample size: %i'%(p))
            self.myLoggerTime.info('Calculating sample loss with sample size: %i'%(p))
        else:
            myBooleanMethods = self.myLcLossBooleanMethods
            self.myLcLossBooleanMethods.setTNorm(tnorm)
            
            self.myLogger.info('Calculating loss ')
            self.myLoggerTime.info('Calculating loss ')

        key = "/local/softmax"
        
        lcCounter = 0 # Count processed lcs
        lcLosses = {}
        for graph in self.myGraph: # Loop through graphs
            for _, lc in graph.logicalConstrains.items(): # loop trough lcs in the graph
                startLC = process_time() # timer()

                if not lc.headLC or not lc.active: # Process only active and head lcs
                    continue
                    
                if type(lc) is fixedL: # Skip fixedL lc
                    continue
                    
                lcCounter +=  1
                self.myLogger.info('Processing Logical Constrain %s(%s) - %s'%(lc.lcName, lc, lc.strEs()))
                
                # Calculate loss for the given lc
                if sample:
                    # lossList will contain boolean results for lc evaluation for the given sample element
                    # sampleInfo - will contain list of variable exiting in the given lc with their sample and probabilities
                    lossList, sampleInfo = self.__constructLogicalConstrains(lc, myBooleanMethods, m, dn, p, key = key, lcVariablesDns = {}, headLC = True, loss = True, sample = sample)
                else:
                    # lossList will contain float result for lc loss calculation
                    lossList = self.__constructLogicalConstrains(lc, myBooleanMethods, m, dn, p, key = key, lcVariablesDns = {}, headLC = True, loss = True, sample = sample)
                
                if not lossList:
                    continue
                
                lcName = lc.lcName
                    
                lcLosses[lcName] = {}
                current_lcLosses = lcLosses[lcName]
                
                if not sample: # Loss value
                    lossTensor = torch.zeros(len(lossList))#, requires_grad=True) # Entry lcs
                    for i, l in enumerate(lossList):
                        lossTensor[i] = float("nan")
                        for entry in l:
                            if entry is not None:
                                if lossTensor[i] != lossTensor[i]:
                                    lossTensor[i] = entry
                                else:
                                    lossTensor[i] += entry

                    current_lcLosses['lossTensor'] = lossTensor
                    current_lcLosses['loss'] = torch.nansum(lossTensor).item()
                    
                else: # -----------Sample
                    # Prepare data
                    currentDevice = "cpu"
                    if lossList[0] != None and lossList[0][0] != None:
                        currentDevice = lossList[0][0].device
                        
                    successesList = [] # Entry lcs successes
                    sampleInfoFiltered = []
                    lcSuccesses = torch.ones(sampleSize, device = currentDevice) # Consolidated successes for all the entry lcs
                    lcVariables = {} # Unique variables used in all the entry lcs
                    countSuccesses = torch.zeros(sampleSize, device = currentDevice)
                    oneT = torch.ones(sampleSize, device = currentDevice)
                    for i, l in enumerate(lossList):
                        for currentFailures in l:
                            if currentFailures is None:
                                successesList.append(None)
                                continue
                            
                            currentSuccesses = torch.sub(oneT, currentFailures.float())
                            successesList.append(currentSuccesses)
                                
                            lcSuccesses =  lcSuccesses.mul_(currentSuccesses)
                            countSuccesses = countSuccesses.add_(currentSuccesses)
                            
                            currentSampleInfo = []
                            for k in sampleInfo.keys():
                                for c in sampleInfo[k][i]:
                                    if len(c) > 2:
                                        currentSampleInfo.append(c)
                                        
                                        if c[2] not in lcVariables:
                                            lcVariables[c[2]] = c
                            
                            sampleInfoFiltered.append(currentSampleInfo)
                        
                    newSampleSize = sampleSize
                    # Eliminate duplicate samples
                    eliminateDuplicateSamples = False
                    if eliminateDuplicateSamples:
                        variablesSamples = [lcVariables[v][1] for v in lcVariables]
                        
                        variablesSamplesT = torch.stack(variablesSamples)
                        
                        uniqueSampleIndex = []
                        
                        for i in range(sampleSize):
                            currentS = variablesSamplesT[:,i]
                            
                            isUnique = True
                            for index in uniqueSampleIndex:
                                indexSample = variablesSamplesT[:,index]
                                if torch.equal(indexSample, currentS):
                                    isUnique = False
                                    continue
                                
                            if not isUnique:
                                continue
                            
                            uniqueSampleIndex.append(i)
                            
                        newSampleSize = len(uniqueSampleIndex)
                    
                        indices = torch.tensor(uniqueSampleIndex, device = currentDevice)
                        #x = torch.arange(6).view(2,3)
                        Vs = torch.index_select(variablesSamplesT, dim=1, index=indices)
                    
                    # Calculate loss value
                    if eliminateDuplicateSamples: 
                        lossTensor = torch.index_select(lcSuccesses, dim=0, index=indices)
                    else:
                        lossTensor = torch.clone(lcSuccesses)
                    #lossTensor = countSuccesses.div_(len(lossList))
                    for i, v in enumerate(lcVariables):
                        currentV = lcVariables[v]
                        
                        if eliminateDuplicateSamples:
                            P = currentV[0][:newSampleSize] # Tensor with the current variable p (v[0])
                        else:
                            P = currentV[0]
                        oneMinusP = torch.sub(torch.ones(newSampleSize, device=P.device), P) # Tensor with the current variable 1-p
                        
                        if eliminateDuplicateSamples:
                            S = Vs[i, :] #currentV[1] # Sample for the current Variable
                        else:
                            S = currentV[1]
                        notS = torch.sub(torch.ones(newSampleSize, device=P.device), S.float()) # Negation of Sample
                        
                        pS = torch.mul(P, S) # Tensor with p multiply by True variable sample
                        oneMinusPS = torch.mul(oneMinusP, notS) # Tensor with 1-p multiply by False variable sample
                        
                        cLoss = pS.add(oneMinusPS) # Sum of p and 1-p tensors
                                                        
                        # Multiply the loss
                        lossTensor.mul_(cLoss)
                        
                    current_lcLosses['loss'] = torch.nansum(lossTensor).item() # Sum of losses across sample for x|=alfa

                    current_lcLosses['lossTensor'] = lossTensor
                    current_lcLosses['lcSuccesses'] = lcSuccesses
                    current_lcLosses['lcVariables'] = lcVariables
                
                    endLC = process_time() # timer()
                    elapsedInSLC = endLC - startLC
                    
                    if eliminateDuplicateSamples: 
                        self.myLoggerTime.info('Processing time for Lc %s with %i entries, %i variables and %i unique samples is: %is'
                                               %(lcName, len(lossList), len(lcVariables), newSampleSize, elapsedInSLC))
                    else:
                        self.myLoggerTime.info('Processing time for Lc %s with %i entries and %i variables is: %is'
                                               %(lcName, len(lossList), len(lcVariables), elapsedInSLC))
        
        self.myLogger.info('')

        self.myLogger.info('Processed %i logical constraints'%(lcCounter))
        self.myLoggerTime.info('Processed %i logical constraints'%(lcCounter))
              
        if sample:
            if sampleGlobalLoss: # Calculate global sample loss  
                globalLossT = torch.zeros(sampleSize, device = currentDevice)
                globalSuccesses = torch.ones(sampleSize, device = currentDevice)
                for lc in lcLosses:
                    currentLC = lcLosses[lc]
                    
                    globalSuccesses = globalSuccesses.mul_(currentLC['lcSuccesses']) # Multiply to find common successes
                    globalLossT = globalLossT.add_(currentLC['lossTensor']) 
                        
                globalLossFiltered = torch.mul(globalSuccesses, globalLossT) # Select loss for common successes
    
                globalLoss = torch.nansum(globalLossFiltered).item()
                lcLosses['globalLoss'] = globalLoss
                
                globalSuccessesCounter = torch.nansum(globalSuccesses).item()
                self.myLoggerTime.info('Count of global Successes is: %f'%(globalSuccessesCounter))
                
                lossSum = torch.nansum(globalLossT).item()
                self.myLoggerTime.info('Sum of lc losses: %f'%(lossSum))
    
                self.myLogger.info('Calculated sample global loss: %f'%(lcLosses['globalLoss']))
                self.myLoggerTime.info('Calculated sample global loss: %f'%(lcLosses['globalLoss']))
            elif False: # TODO
                globalLossT = torch.zeros(sampleSize, device = currentDevice)
                for lc in lcLosses:
                    globalLossT = globalLossT.add_(lcLosses[lc]['lossTensor']) 
                
                lossSum = torch.nansum(globalLossT).item()
                self.myLoggerTime.info('Sum of lc losses: %f'%(lossSum))
            
        end = process_time() # timer()
        elapsedInS = end - start
        
        self.myLogger.info('End Loss Calculation - elapsed time: %is'%(elapsedInS))
        self.myLoggerTime.info('End Loss Calculation - elapsed time: %is'%(elapsedInS))
        self.myLogger.info('')
        self.myLoggerTime.info('')

        [h.flush() for h in self.myLoggerTime.handlers]
            
        return lcLosses