from itertools import product
from datetime import datetime
from collections import OrderedDict
from collections.abc import Mapping
from torch import tensor

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
from torch import tensor

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
               
    def __getProbability(self, dn, conceptRelation):
        if not dn:
            currentProbability = [1, 0]
        else:
            currentProbability = dn.getAttribute(conceptRelation)
            
        if currentProbability == None:
            currentProbability = [1,0]
            
        return currentProbability
    
    def createILPVariables(self, m, rootDn, *conceptsRelations):
        x = {}
        Q = None
        
        # Create ILP variables 
        for _conceptRelation in conceptsRelations: 
            rootConcept = rootDn.findRootConceptOrRelation(_conceptRelation)
            dns = rootDn.findDatanodes(select = rootConcept)
            
            conceptRelation = _conceptRelation.name
            
            for dn in dns:
                currentProbability = self.__getProbability(dn, conceptRelation)
                
                # Check if probability is NaN or if and has to be skipped
                if self.valueToBeSkipped(currentProbability[1]):
                    self.myLogger.info("Probability is %f for variable concept %s and dataNode %s - skipping it"%(currentProbability[1],conceptRelation,dn.getInstanceID()))
                    continue
    
                # Create variable
                x[conceptRelation, dn.getInstanceID()] = m.addVar(vtype=GRB.BINARY,name="x_%s_is_%s"%(dn.getInstanceID(), conceptRelation)) 
                xkey = '<' + conceptRelation + '>/ILP/x'
                
                dn.attributes[xkey] = x[conceptRelation, dn.getInstanceID()]    
                
                Q += currentProbability[1] * dn.attributes[xkey]       
    
                # Check if probability is NaN or if and has to be created based on positive value
                if self.valueToBeSkipped(currentProbability[0]):
                    currentProbability[0] = 1 - currentProbability[1]
                    self.myLogger.info("No ILP negative variable for concept %s and dataNode %s - created based on positive value %f"%(dn.getInstanceID(), conceptRelation, currentProbability[0]))
    
                # Create negative variable
                if True: # ilpOntSolver.__negVarTrashhold:
                    x['Not_'+conceptRelation, dn.getInstanceID()] = m.addVar(vtype=GRB.BINARY,name="x_%s_is_not_%s"%(dn.getInstanceID(), conceptRelation))
                    notxkey = '<' + conceptRelation + '>/ILP/notx'
                
                    dn.attributes[notxkey] = x['Not_'+conceptRelation, dn.getInstanceID()]  
                    
                    Q += currentProbability[0] * dn.attributes[notxkey]     

                else:
                    self.myLogger.info("No ILP negative variable for concept %s and dataNode %s created"%(conceptRelation, dn.getInstanceID()))

        m.update()

        if len(x):
            self.myLogger.info("Created %i ILP variables"%(len(x)))
        else:
            self.myLogger.warning("No ILP variables created")
            
        return Q, x     
    
    def addGraphConstrains(self, m, rootDn, *conceptsRelations):
        # Add constrain based on probability 
        for _conceptRelation in conceptsRelations: 
            rootConcept = rootDn.findRootConceptOrRelation(_conceptRelation)
            dns = rootDn.findDatanodes(select = rootConcept)
            
            conceptRelation = _conceptRelation.name

            xkey = '<' + conceptRelation + '>/ILP/x'
            notxkey = '<' + conceptRelation + '>/ILP/notx'
            
            for dn in dns:
                # Add constraints forcing decision between variable and negative variables 
                if notxkey in dn.attributes:
                    currentConstrLinExpr = dn.getAttribute(xkey) + dn.getAttribute(notxkey) # x[conceptName, token] + x['Not_'+conceptName, token]
                    
                    m.addConstr(currentConstrLinExpr == 1, name='c_%s_%sselfDisjoint'%(conceptRelation, 'Not_'+conceptRelation))
                    self.myLogger.debug("Disjoint constrain between variable \"token %s is concept %s\" and variable \"token %s is concept - %s\" == %i"%(dn.getInstanceID(),conceptRelation,dn.getInstanceID(),'Not_'+conceptRelation,1))

        m.update()
        
        # Create subclass constrains
        for concept in conceptsRelations:
            rootConcept = rootDn.findRootConceptOrRelation(concept)
            dns = rootDn.findDatanodes(select = rootConcept)
            
            for rel in concept.is_a():
                # A is_a B : if(A, B) : A(x) <= B(x)
                
                sxkey = '<' + rel.src.name + '>/ILP/x'
                dxkey = '<' + rel.dst.name + '>/ILP/x'

                for dn in dns:
                     
                    if sxkey not in dn.attributes: # subclass (A)
                        continue
                    
                    if dxkey not in dn.attributes: # superclass (B)
                        continue
                                                                    
                    self.myIlpBooleanProcessor.ifVar(m, dn.getAttribute(sxkey), dn.getAttribute(dxkey), onlyConstrains = True)
                    self.myLogger.info("Created - subclass - constrains between concept \"%s\" and concepts %s"%(rel.src.name,rel.dst.name))

        # Create disjoint constraints
        foundDisjoint = dict() # To eliminate duplicates
        for concept in conceptsRelations:
            rootConcept = rootDn.findRootConceptOrRelation(concept)
            dns = rootDn.findDatanodes(select = rootConcept)
               
            for rel in concept.not_a():
                conceptName = concept.name
                
                if rel.dst not in conceptsRelations:
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
                        
                    self.myIlpBooleanProcessor.nandVar(m, dn.getAttribute(cxkey), dn.getAttribute(dxkey), onlyConstrains = True)
                        
                if not (conceptName in foundDisjoint):
                    foundDisjoint[conceptName] = {disjointConcept}
                else:
                    foundDisjoint[conceptName].add(disjointConcept)
                           
            if concept.name in foundDisjoint:
                self.myLogger.info("Created - disjoint - constrains between concept \"%s\" and concepts %s"%(conceptName,foundDisjoint[conceptName]))
            
        m.update()
        
    def addLogicalConstrains(self, m, dn, lcs, p):
        self.myLogger.info('Starting method')
        
        key = "/ILP/xP"
        
        for lc in lcs:   
            self.myLogger.info('Processing Logical Constrain %s(%s) - %s'%(lc.lcName, lc, [str(e) for e in lc.e]))
            result = self.__constructLogicalConstrains(lc, self.myIlpBooleanProcessor, m, dn, p, key = key,  headLC = True)
            
            if result != None and isinstance(result, list):
                self.myLogger.info('Successfully added Logical Constrain %s'%(lc.lcName))
            else:
                self.myLogger.error('Failed to add Logical Constrain %s'%(lc.lcName))

    def __constructLogicalConstrains(self, lc, booleanProcesor, m, dn, p, key = "", lcVariablesDns = {}, headLC = False):
        resultVariableNames = []
        lcVariables = {}
        lcNo = 0
        
        for eIndex, e in enumerate(lc.e): 
            if isinstance(e, Concept) or isinstance(e, LogicalConstrain): 
                # Look one step ahead in the parsed logical constrain and get variables names (if present) after the current concept
                if eIndex + 1 < len(lc.e):
                    if isinstance(lc.e[eIndex+1], V):
                        variable = lc.e[eIndex+1]
                    else:
                        if isinstance(e, LogicalConstrain):
                            variable = V(name="lc" + str(lcNo))
                            lcNo =+ 1
                        else:
                            self.myLogger.error('The element of logical constrain %s after %s is of type %s but should be variable of type %s'%(lc.lcName, e, type(lc.e[eIndex+1]), V))
                            return None
                else:
                    if isinstance(e, LogicalConstrain):
                        variable = V(name="lc" + str(lcNo))
                        lcNo =+ 1
                    else:
                        self.myLogger.error('The element %s of logical constrain %s has no variable'%(e, lc.lcName))
                        return None
                    
                if variable.name:
                    variableName = variable.name
                else:
                    variableName = "V" + str(eIndex)
                            
                # -- Concept 
                if isinstance(e, Concept):
                    conceptName = e.name
                    xPkey = '<' + conceptName + ">" + key

                    dnsList = [] # Stores lists of dataNode for each corresponding dataNode 
                    vDns = [] # Stores ILP variables
                    
                    if variable.v == None:
                        if variable.name == None:
                            self.myLogger.error('The element %s of logical constrain %s has no name for variable'%(e, lc.lcName))
                            return None
                                                 
                        rootConcept = dn.findRootConceptOrRelation(conceptName)
                        _dns = dn.findDatanodes(select = rootConcept)
                        dnsList = [[dn] for dn in _dns]
                    else:
                        if len(variable.v) == 0:
                            self.myLogger.error('The element %s of logical constrain %s has no empty part v of the variable'%(e, lc.lcName))
                            return None
                            
                        referredVariableName = variable.v[0] # Get name of the referred variable already defined in the logical constrain from the v part 
                    
                        if referredVariableName not in lcVariablesDns:
                            self.myLogger.error('The element %s of logical constrain %s has v referring to undefined variable %s'%(e, lc.lcName, referredVariableName))
                            return None
                       
                        referredDns = lcVariablesDns[referredVariableName] # Get Datanodes for referred variables already defined in the logical constrain
                        for rDn in referredDns:
                            eDns = []
                            for _rDn in rDn:
                                _eDns = _rDn.getEdgeDataNode(variable.v[1:]) # Get Datanodes for the edge defined by the path part of the v
                                
                                if _eDns:
                                    eDns.extend(_eDns)
                                    
                            dnsList.append(eDns)
                                
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
                            
                            ilpVs = _dn.getAttribute(xPkey) # Get ILP variable for the concept 
                            
                            if isinstance(ilpVs, Mapping) and p not in ilpVs:
                                _vDns.append(None)
                                continue
                            
                            vDn = ilpVs[p]
                        
                            if torch.is_tensor(vDn):
                                vDn = vDn.item()  
                                                              
                            _vDns.append(vDn)
                        
                        vDns.append(_vDns)
                        
                    resultVariableNames.append(variableName)
                    lcVariablesDns[variable.name] = dnsList
                    lcVariables[variableName] = vDns
                # LogicalConstrain - process recursively 
                elif isinstance(e, LogicalConstrain):
                    self.myLogger.info('Processing Logical Constrain %s(%s) - %s'%(e.lcName, e, [str(e1) for e1 in e.e]))
                    vDns = self.__constructLogicalConstrains(e, booleanProcesor, m, dn, p, key = key, lcVariablesDns = lcVariablesDns, headLC = False)
                    
                    if vDns == None:
                        self.myLogger.warning('Not found data for %s(%s) nested logical Constrain required to build Logical Constrain %s(%s) - skipping this constrain'%(e.lcName,e,lc.lcName,lc))
                        return None
                        
                    resultVariableNames.append(variableName)
                    lcVariables[variableName] = vDns   
                                     
            # Tuple with named variable 
            elif isinstance(e, tuple): 
                if eIndex == 0:
                    pass
                else:
                    pass # Already processed 
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
        
        return lc(m, booleanProcesor, lcVariables, resultVariableNames=resultVariableNames, headConstrain = headLC)
                
    # -- Main method of the solver - creating ILP constrains and objective and invoking ILP solver, returning the result of the ILP solver classification  
    def calculateILPSelection(self, dn, *conceptsRelations, fun=None, epsilon = 0.00001, minimizeObjective = False):
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
            Q, x = self.createILPVariables(m, dn, *conceptsRelations)
            
            # Add constrains based on graph definition
            self.addGraphConstrains(m, dn, *conceptsRelations)
                        
            # ILP Model objective setup
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
                        if lc.p not in _lcP:
                            _lcP[lc.p] = []
                        
                        _lcP[lc.p].append(lc) # Keep constrain with the same p in the list 
            
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
                    
                    if _x[0].startswith('Not_'):
                        rootConcept = dn.findRootConceptOrRelation(_x[0][4:])
                    else:
                        rootConcept = dn.findRootConceptOrRelation(_x[0])

                    dns = dn.findDatanodes(select = ((rootConcept,), ("instanceID", _x[1])))  
                    
                    if dns:
                        if _x[0].startswith('Not'):
                            xPkey = '<' + _x[0] + '>/ILP/notxP'
                        else:
                            xPkey = '<' + _x[0] + '>/ILP/xP'

                        if xPkey not in dns[0].attributes:
                            dns[0].attributes[xPkey] = {}
                            
                        dns[0].attributes[xPkey][p] = mP.getVarByName(x[_x].VarName)
                                    
                # Prepare set with logical constraints for this run
                lcs = []
                for _p in lcP:
                    lcs.extend(lcP[_p])

                    if _p == p:
                        break     
    
                # Add constraints to the copy model
                self.addLogicalConstrains(mP, dn, lcs, p)
                self.myLogger.info('Optimizing model for logical constraints with probabilities %s with %i variables and %i constrains'%(p,mP.NumVars,mP.NumConstrs))

                startOptimize = datetime.now()

                # Run ILP model - Find solution 
                mP.optimize()
                mP.update()
                
                print(mP.display())    
                   
                endOptimize = datetime.now()
                elapsedOptimize = endOptimize - startOptimize
    
                # check model run result
                solved = False
                objValue = None
                if mP.status == GRB.Status.OPTIMAL:
                    self.myLogger.info('%s optimal solution was found with value %f - solver time: %ims'%('Min' if minimizeObjective else 'Max', mP.ObjVal,elapsedOptimize.microseconds/1000))
                    solved = True
                    objValue = mP.ObjVal
                elif mP.status == GRB.Status.INFEASIBLE:
                    self.myLogger.warning('Model was proven to be infeasible.')
                elif mP.status == GRB.Status.INF_OR_UNBD:
                    self.myLogger.warning('Model was proven to be infeasible or unbound.')
                elif mP.status == GRB.Status.UNBOUNDED:
                    self.myLogger.warning('Model was proven to be unbound.')
                else:
                    self.myLogger.warning('Optimal solution not was found - error code %i'%(mP.status))
                 
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

                for c in conceptsRelations:
                    c_root = dn.findRootConceptOrRelation(c)
                    c_root_dns = dn.findDatanodes(select = c_root)
                    
                    ILPkey = '<' + c.name + '>/ILP'
                    xkey = ILPkey + '/x'
                    xPkey = ILPkey + '/xP'
                    xNotPkey = ILPkey + '/notxP'
                   
                    for dn in c_root_dns:
                        dnAtt = dn.getAttributes()
                        if xPkey not in dnAtt:
                            dnAtt[ILPkey] = torch.tensor([float("nan")], device=device) 
                            continue
                        
                        maxPVar = dnAtt[xPkey][maxP]
                        solution = maxPVar.X
                        dnAtt[ILPkey] = torch.tensor([solution], device=device)
                        dnAtt[xkey] = maxPVar
                        
                        del dnAtt[xPkey]
                        
                        if xNotPkey in dnAtt:
                            del dnAtt[xNotPkey]
                            
                        ILPV = dnAtt[ILPkey].item()
                        if ILPV == 1:
                            self.myLogger.info('\"%s\" is \"%s\"'%(dn,c))

                 
                # self.__collectILPSelectionResults(dn, lcRun[maxP]['mP'], lcRun[maxP]['xP'])
                # Get ILP result from  maxP x 
            else:
                pass
                                       
        except:
            self.myLogger.error('Error returning solutions')
            raise
           
        end = datetime.now()
        elapsed = end - start
        self.myLogger.info('')
        self.myLogger.info('End - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        # Return
        return
    
    # -- Main method of the solver - creating ILP constrains and objective and invoking ILP solver, returning the result of the ILP solver classification  
    def verifySelection(self, phrase, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None, minimizeObjective = False, hardConstrains = []):
        tokenResult, relationResult, tripleRelationResult = self.calculateILPSelection(phrase, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation, minimizeObjective, hardConstrains)
        concepts = [k for k in graphResultsForPhraseToken.keys()]
        self.__checkIfContainNegativeProbability(concepts, tokenResult, relationResult, tripleRelationResult)

        if graphResultsForPhraseToken:
            for key in graphResultsForPhraseToken:
                if key not in tokenResult:
                    return False
                
                if not np.array_equal(graphResultsForPhraseToken[key], tokenResult[key]):
                    return False
                
        if graphResultsForPhraseRelation:
            for key in graphResultsForPhraseRelation:
                if key not in relationResult:
                    return False
                
                if not np.array_equal(graphResultsForPhraseRelation[key], relationResult[key]):
                    return False
                
        if graphResultsForPhraseTripleRelation:
            for key in graphResultsForPhraseTripleRelation:
                if key not in tripleRelationResult:
                    return False
                
                if not np.array_equal(graphResultsForPhraseTripleRelation[key], tripleRelationResult[key]):
                    return False
                
        return True
    
    # -- Main method of the solver - creating ILP constrains and objective and invoking ILP solver, returning the result of the ILP solver classification  
    def verifySelectionLC(self, phrase, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None, minimizeObjective = False, hardConstrains = []):
        if not graphResultsForPhraseToken:
            self.myLogger.warning('graphResultsForPhraseToken is None or empty - returning False')
            return False
            
        concepts = [k for k in graphResultsForPhraseToken.keys()]
        relations = [k for k in graphResultsForPhraseRelation.keys()]
        
        graphResultsForPhraseToken1 = next(iter(graphResultsForPhraseToken.values()))
        tokens = [i for i ,_ in enumerate(graphResultsForPhraseToken1)]
        
        self.__checkIfContainNegativeProbability(concepts, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation)

        hardConstrainsN = concepts + relations
        
        hardConstrains = {}
        for c in hardConstrainsN:
            if c in graphResultsForPhraseToken:
                hardConstrains[c] = graphResultsForPhraseToken[c]
            elif c in graphResultsForPhraseRelation:
                hardConstrains[c] = graphResultsForPhraseRelation[c]
            elif c in graphResultsForPhraseTripleRelation:
                hardConstrains[c] = graphResultsForPhraseTripleRelation[c]
            else:
                pass
            
        m = None 
        x = {}
        y = {}
        z = {} 
        for graph in self.myGraph:
            for _, lc in graph.logicalConstrains.items():
                if not lc.headLC:
                    continue
                    
                self.myLogger.info('Processing Logical Constrain %s(%s) - %s'%(lc.lcName, lc, [str(e) for e in lc.e]))
                self.__constructLogicalConstrains(lc, self.myIlpBooleanProcessor, m, concepts, tokens, x, y, z, hardConstrains=hardConstrains, headLC = True)
                
    # -- Calculated values for logical constrains
    def calculateLcLoss(self, dn):
        m = None 
                
        p = 1
        
        lcLosses = {}
        for graph in self.myGraph:
            for _, lc in graph.logicalConstrains.items():
                if not lc.headLC:
                    continue
                    
                self.myLogger.info('Processing Logical Constrain %s(%s) - %s'%(lc.lcName, lc, [str(e) for e in lc.e]))
                lossList = self.__constructLogicalConstrains(lc, self.myLcLossBooleanMethods, m, dn, p, key = "", lcVariablesDns = {}, headLC = True)
                
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