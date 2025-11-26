import math
from collections import OrderedDict
import torch

from domiknows.graph.concept import Concept, EnumConcept
from domiknows.graph import LcElement, LogicalConstrain, V
from domiknows.graph import CandidateSelection
from domiknows.graph.candidates import getCandidates


class LogicalConstraintConstructor:
    """
    Helper class for constructing logical constraints.
    
    This class handles the construction of logical constraints by processing
    concepts, variables, and nested constraints. It's independent of the ILP
    solver and can be used by various functionalities (ILP inference, loss
    calculation, verification, etc.).
    """
    
    def __init__(self, logger):
        """
        Initialize the constraint constructor.
        
        Args:
            logger: Logger instance for debugging/info messages
        """
        self.myLogger = logger
        self.current_device = None
        
    def getConcept(self, concept):
        return concept[0]
    
    def getConceptName(self, concept):
        return concept[0].name
    
    def conceptIsBinary(self, concept):
        return concept[2] is None
    
    def conceptIsMultiClass(self, concept):
        return concept[2] is not None
    
    def valueToBeSkipped(self, x):
        """Check if value is NaN or Inf and should be skipped"""
        return math.isnan(x) or math.isinf(x)
    
    def getLabel(self, dn, conceptRelation):
        """Get Ground Truth for provided concept"""
        value = dn.getAttribute(conceptRelation, 'label')
        return value
    
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
    
    def getMLResult(self, dn, xPkey, e, p, loss=False, sample=False):
        """
        Get ML result for a datanode and concept.
        
        Args:
            dn: Datanode
            xPkey: Key for accessing predictions
            e: Concept tuple (concept_name, label, index)
            p: Sample size (for sampling) or priority (for ILP)
            loss: Whether calculating loss
            sample: Whether generating samples
            
        Returns:
            For ILP: ILP variable
            For loss without sample: Tensor value
            For loss with sample: Tuple of (sample, (probability, sample, variable_name))
        """
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

                dn.getAttributes()[sampleKey][sampleSize][e[1]] = torch.ones(sampleSize, dtype=torch.bool, device=self.current_device)
                xP = torch.ones(sampleSize, device=self.current_device)
                
                return (dn.getAttributes()[sampleKey][sampleSize][e[1]], (xP, dn.getAttributes()[sampleKey][sampleSize][e[1]], xVarName))
        
        if dn.getAttribute(xPkey) == None:
            if not sample:
                return None
            else:   
                return ([None], (None, [None]))
        
        if not loss:
            if "xP" in xPkey:
                vDn = dn.getAttribute(xPkey)[p][e[2]]
            elif "local/argmax" in xPkey:
                vDn = dn.getAttribute(xPkey)[e[1]]
            else:
                vDn = dn.getAttribute(xPkey)[e[2]]
                
            return vDn
        
        # Loss calculation
        isFiexd = self.isVariableFixed(dn, conceptName, e)

        if isFiexd != None:
            if isFiexd == 1:
                vDn = torch.tensor(1.0, device=self.current_device, requires_grad=True)
            else:
                vDn = torch.tensor(0.0, device=self.current_device, requires_grad=True)
        else:
            try:
                vDn = dn.getAttribute(xPkey)[e[1]]
            except IndexError: 
                vDn = None
    
        if not sample:
            return vDn
        
        if torch.is_tensor(vDn) and (len(vDn.shape) == 0 or len(vDn.shape) == 1 and vDn.shape[0] == 1):
            vDn = vDn.item()  
             
        sampleSize = p

        xVarName = "%s_%s_is_%s"%(e[0], dn.getInstanceID(), e[1])
                
        usedSampleSize = sampleSize
        if sampleSize == -1:
            usedSampleSize = dn.getAttributes()[sampleKey][-1][e[1]].shape[0]
        if isFiexd != None:  
            if isFiexd == 1:
                xP = torch.ones(usedSampleSize, device=self.current_device, requires_grad=True)
            else:
                xP = torch.zeros(usedSampleSize, device=self.current_device, requires_grad=True)
        else:
            xV = dn.getAttribute(xPkey)
            xEp = dn.getAttribute(xPkey).expand(usedSampleSize, len(xV.squeeze(0)))
            xP = xEp[:,e[1]]
          
        if sampleSize > -1: 
            if sampleSize not in dn.getAttributes()[sampleKey]: 
                dn.getAttributes()[sampleKey][sampleSize] = {}
                
            if e[1] not in dn.getAttributes()[sampleKey][sampleSize]:
                if vDn == None or vDn != vDn:
                    dn.getAttributes()[sampleKey][sampleSize][e[1]] = [None]
                else:
                    dn.getAttributes()[sampleKey][sampleSize][e[1]] = torch.bernoulli(xP)
            
        return (dn.getAttributes()[sampleKey][sampleSize][e[1]], (xP, dn.getAttributes()[sampleKey][sampleSize][e[1]], xVarName))
    
    def isVariableFixed(self, dn, conceptName, e):
        """Check if a variable is fixed by fixedL constraint"""
        fixedAttribute = None
        fixedValue = None
        
        if not hasattr(self, 'myGraph'):
            return None
            
        from domiknows.graph import fixedL
        
        for graph in self.myGraph:
            for _, lc in graph.logicalConstrains.items():
                if not lc.headLC or not lc.active:
                    continue
                    
                if type(lc) is not fixedL:
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
                      
        if fixedAttribute not in dn.getAttributes():
            return None
        
        attributeValue = dn.getAttribute(fixedAttribute).item()
        
        if attributeValue in fixedValue:
            pass
        elif (True in fixedValue) and attributeValue == 1:
            pass
        elif (False in fixedValue) and attributeValue == 0:
            pass
        else:
            return None
       
        vDnLabel = self.getLabel(dn, conceptName).item()

        if vDnLabel == e[2]:
            return 1
        else:
            return 0
    
    def fixedLSupport(self, _dn, conceptName, vDn, i, m):
        """Support for fixed constraints"""
        from gurobipy import Var
        
        vDnLabel = self.getLabel(_dn, conceptName).item()

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
    
    def addLossTovDns(self, loss, vDns):
        """Add loss tensor to vDns"""
        if loss and vDns:
            vDnsList = [v[0] for v in vDns]
            
            updatedVDns = []
            try:
                if len(vDnsList) > 1:
                    tStack = torch.stack(vDnsList, dim=1)
                else:
                    tStack = vDnsList[0]
                tsqueezed = torch.squeeze(tStack, dim=0)

            except IndexError:
                tsqueezed = torch.stack(vDnsList, dim=0)
        
            if not len(tsqueezed.shape):
                tsqueezed = torch.unsqueeze(tsqueezed, 0)
                
            tList = [tsqueezed]
            updatedVDns.append(tList)
            
            return updatedVDns
        else:
            return vDns
    
    def eliminate_duplicate_columns(self, data_dict, rows_to_consider, data_dict_target):
        """Eliminates columns that have identical elements across specified rows."""
        if not rows_to_consider or not data_dict or not data_dict_target:
            return data_dict_target
        
        first_row = list(data_dict.values())[0]
        num_columns = len(first_row)
        
        columns_to_keep = []
        
        for col_idx in range(num_columns):
            column_values = []
            for row_name in rows_to_consider:
                if row_name in data_dict:
                    if col_idx >= len(data_dict[row_name]):
                        continue
                    column_values.append(data_dict[row_name][col_idx])
            
            unique_values = set(str(val) for val in column_values)
            if len(unique_values) < len(column_values):
                pass
            else:
                columns_to_keep.append(col_idx)
        
        result = OrderedDict()
        for row_name, row_data in data_dict_target.items():
            if len(row_data) == 1:
                try:
                    original_tensor = row_data[0][0]
                    filtered_tensor = original_tensor[columns_to_keep]
                    result[row_name] = [[filtered_tensor]]
                except (TypeError, IndexError):
                    result[row_name] = [row_data[i] for i in columns_to_keep]
            else:
                result[row_name] = [row_data[i] for i in columns_to_keep]
        
        return result
    
    def constructLogicalConstrains(self, lc, booleanProcessor, m, dn, p, key=None, 
                                   lcVariablesDns=None, lcVariables=None, headLC=False, 
                                   loss=False, sample=False, vNo=None, verify=False):
        """
        Construct logical constraints by processing concepts and variables.
        
        Args:
            lc: Logical constraint to construct
            booleanProcessor: Boolean processor for constraint operations
            m: Model (ILP model or None for loss/verify)
            dn: Root datanode
            p: Sample size (for sampling) or priority (for ILP)
            key: Key for accessing predictions
            lcVariablesDns: Dictionary mapping variable names to datanodes
            lcVariables: Dictionary mapping variable names to values/variables
            headLC: Whether this is a head constraint
            loss: Whether calculating loss
            sample: Whether generating samples
            vNo: Variable numbering counter [concept_counter, lc_counter]
            verify: Whether verifying constraints
            
        Returns:
            For sample=True: (result, sampleInfo, lcVariablesSet, lcVariables)
            For verify=True and headLC=True: (result, lcVariables)
            Otherwise: (result, lcVariables)
        """
        if key == None:
            key = ""
            
        lcRepr = f'{lc.__class__.__name__} {lc.strEs()}'

        if lcVariablesDns == None:
            lcVariablesDns = OrderedDict()

        if lcVariables == None:
            lcVariables = OrderedDict()
            
        usedVariablesNames = set()

        if sample:
            sampleInfo = OrderedDict()
            lcVariablesSet = OrderedDict()
            
        if vNo == None:
            vNo = [1, 1]
        
        firstV = None
        integrate = False
        newVariables = {}
        
        for eIndex, e in enumerate(lc.e):
            if isinstance(e, V):
                continue
            
            if isinstance(e, (Concept, LcElement, tuple)): 
                # Look ahead for variable names
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
                            variable = V(name="_x" + str(vNo[0]))
                            if not isinstance(lc, CandidateSelection):
                                firstV = variable.name
                            vNo[0] += 1
                        else:
                            variable = V(name="_x" + str(vNo[0]), v=(firstV,))
                            vNo[0] += 1
                    
                if variable.name:
                    variableName = variable.name
                else:
                    variableName = "V" + str(vNo[0])
                    vNo[0] += 1
                    
                if variableName in lcVariables:
                    newVariableName = "_x" + str(vNo[0])
                    vNo[0] += 1

                    lcVariablesDns[newVariableName] = lcVariablesDns[variableName]
                    lcVariables[newVariableName] = lcVariables[variableName]
                    usedVariablesNames.add(variableName)

                elif isinstance(e, (Concept, tuple)):
                    # Get dataNode candidates 
                    dnsList, referedVariables = getCandidates(dn, e, variable, lcVariablesDns, lc, self.myLogger, integrate=integrate)
                    lcVariablesDns[variableName] = dnsList
                                
                    if isinstance(lc, CandidateSelection):
                        continue
                    
                    if len(referedVariables) == 1:
                        referedVariable = referedVariables.pop()
                        
                        if referedVariable.startswith('p'):
                            if referedVariable not in newVariables:
                                newVariables[referedVariable] = set()
                            newVariables[referedVariable].add(variableName)

                    # Get ILP variables/values from collected DataNodes
                    conceptName = e[0].name
                    vDns = []
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

                            if isinstance(e[0], EnumConcept) and e[2] == None:
                                eList = e[0].enum
                                for i, _ in enumerate(eList):
                                    eT = (e[0].name, i, i)
                                    if sample:
                                        vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample)
                                        _sampleInfoForVariable.append(vDnSampleInfo)
                                    else:
                                        vDn = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample)
                                    
                                    if lc.__str__() == "fixedL":
                                        vDn = self.fixedLSupport(_dn, conceptName, vDn, i, m)
                                        
                                    _vDns.append(vDn)
                            elif isinstance(e[0], EnumConcept) and e[2] != None:
                                eT = (e[0].name, e[2], e[2])
                                
                                if sample:
                                    vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample)                                    
                                    _sampleInfoForVariable.append(vDnSampleInfo)
                                else:
                                    vDn = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample)
                                
                                if lc.__str__() == "fixedL":
                                    self.fixedLSupport(_dn, conceptName, vDn, e[2], m)
                                    
                                vDn = _vDns.append(vDn)
                            else:
                                eT = (conceptName, 1, 0)
                                if sample:
                                    vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample)
                                    _sampleInfoForVariable.append(vDnSampleInfo)
                                else:
                                    vDn = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample)
                                
                                if lc.__str__() == "fixedL":
                                    self.fixedLSupport(_dn, conceptName, vDn, 1, m)
                                        
                                vDn = _vDns.append(vDn)
                        
                        vDns.append(_vDns)
                        
                        if sample:
                            sampleInfoForVariable.append(_sampleInfoForVariable)
                        
                    # Store values/variables
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
                        
                    usedVariablesNames.add(variableName)
                
                if isinstance(e, LcElement):

                    if isinstance(e, CandidateSelection):
                        lcVariablesDnsNew = self.constructLogicalConstrains(
                            e, booleanProcessor, m, dn, p, key=key, 
                            lcVariablesDns=lcVariablesDns, lcVariables=lcVariables, 
                            headLC=False, loss=loss, sample=sample, vNo=vNo, verify=verify)
                         
                        lcVariablesDns = lcVariablesDnsNew
                        vDns = None
                        if lcVariablesDns:
                            length_of_list = len(next(iter(lcVariablesDns.values())))

                            if sample:
                                vDns = [[torch.ones(p, device=self.current_device, requires_grad=False, dtype=torch.bool)] for _ in range(length_of_list)]
                            elif loss:
                                vDns = [[torch.zeros(length_of_list, device=self.current_device, requires_grad=True, dtype=torch.float64)]]
                                vDns = self.addLossTovDns(loss, vDns)
                            else:
                                vDns = [[1] for _ in range(length_of_list)]
                                   
                    if isinstance(e, LogicalConstrain):
                        self.myLogger.info('Processing Nested %r - %s'%(e, e.strEs()))

                        if sample:
                            vDns, sampleInfoLC, lcVariablesLC, lcVariableUpdated = self.constructLogicalConstrains(
                                e, booleanProcessor, m, dn, p, key=key, 
                                lcVariablesDns=lcVariablesDns, lcVariables=lcVariables, 
                                headLC=False, loss=loss, sample=sample, vNo=vNo, verify=verify)
                            sampleInfo = {**sampleInfo, **sampleInfoLC}
                            lcVariablesSet = {**lcVariablesSet, **lcVariablesLC}
                            lcVariables = lcVariableUpdated 
                        else:
                            vDns, lcVariableUpdated = self.constructLogicalConstrains(
                                e, booleanProcessor, m, dn, p, key=key, 
                                lcVariablesDns=lcVariablesDns, lcVariables=lcVariables,
                                headLC=False, loss=loss, sample=sample, vNo=vNo, verify=verify)
                            
                            vDns = self.addLossTovDns(loss, vDns)
                            lcVariables = lcVariableUpdated

                    if vDns == None:
                        self.myLogger.warning('Not found data for %s(%s) nested Logical Constraint required to build %s(%s) - skipping it'%(e.lcName,e,lc.lcName,lc))
                        return None
                        
                    countValid = sum(1 for sublist in vDns if sublist and any(elem is not None for elem in sublist))
                    self.myLogger.info('Size of candidate list returned by %s(%s) nested Logical Constraint is %i of which %i is not None'%(e.lcName,e,len(vDns),countValid))
                    lcVariables[variableName] = vDns   
                    usedVariablesNames.add(variableName)    
            elif isinstance(e, (int, str)):
                pass
            else:
                self.myLogger.error('Logical Constraint %s has incorrect element %s'%(lc,e))
                return None

        for referedVariable in newVariables:
            refVarSet = newVariables[referedVariable]
            refVarSet.add(referedVariable)  
            lcVariables = self.eliminate_duplicate_columns(lcVariablesDns, refVarSet, lcVariables)

        useLcVariables = {k: v for k, v in lcVariables.items() if k in usedVariablesNames}

        if isinstance(lc, CandidateSelection):
            return lc(lcVariablesDns, keys=lc.CandidateSelectionVariable)
        elif sample:
            lcVariablesSet[lc] = useLcVariables
            return lc(m, booleanProcessor, useLcVariables, headConstrain=headLC, integrate=integrate), sampleInfo, lcVariablesSet, lcVariables
        elif verify and headLC:
            return lc(m, booleanProcessor, useLcVariables, headConstrain=headLC, integrate=integrate), lcVariables
        else:
            if loss:
                slpitT = False
                for v in useLcVariables:
                    if useLcVariables[v] and len(useLcVariables[v]) > 1:
                        slpitT = True
                        break
                    
                if slpitT:
                    for v in useLcVariables:
                        if useLcVariables[v] and len(useLcVariables[v]) > 1:
                            continue
                         
                        lcVSplitted = torch.split(useLcVariables[v][0][0], 1)
                        useLcVariables[v] = []
                        
                        for s in lcVSplitted:
                            useLcVariables[v].append([s]) 
                    
            return lc(m, booleanProcessor, useLcVariables, headConstrain=headLC, integrate=integrate), lcVariables