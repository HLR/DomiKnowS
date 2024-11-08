from functools import reduce
from collections import OrderedDict

from domiknows.graph import LcElement
from domiknows.graph.logicalConstrain import eqL

import torch

class CandidateSelection(LcElement):
    def __init__(self, *e,  name = None):
        super().__init__(*e, name = name)
        
        originalCall  = type(self).__call__
        originalCallName = originalCall.__name__
        if "newCall" in originalCall.__name__:
            return
        
        def newCall (self, *input, keys = None):
            if input and isinstance(input[0], OrderedDict):
                candidatesDict = input[0]
                                    
                return originalCall(self, list(candidatesDict.values()), keys=keys)
            else:
                return self, input
            
        type(self).__call__ = newCall  
        
    def __call__(self, *candidates, keys=None): 
        pass 

# object creating Cartesian product of all candidates
class combinationC(CandidateSelection):
    def __init__(self, *e, name = None):
        super().__init__(*e, name = name)
        
    def __call__(self, candidates_list, keys=None):
        from  itertools import product
        
        # Create the Cartesian product of all candidates
        cartesian_product = list(product(*candidates_list))
        
        # Extract lists of first elements, second elements, etc.
        extracted_elements = list(zip(*cartesian_product))
              
        # Create a dictionary using the provided keys and the extracted lists
        assert keys is not None, "Keys must be provided for the combinationC candidate selection"
        result_dict = dict(zip(keys, extracted_elements))
        
        return result_dict

def intersection_of_lists(lists):
    # Find the intersection of n lists while preserving order
    if not lists:
        return []
    # Find the common elements
    common_elements = set(lists[0])
    for lst in lists[1:]:
        common_elements.intersection_update(lst)
    # Preserve the order of the common elements based on the first list
    ordered_common_elements = [elem for elem in lists[0] if elem in common_elements]
    return ordered_common_elements

def findDatanodesForRootConcept(dn, rootConcept):
    if dn.myBuilder != None and "DataNodesConcepts" in dn.myBuilder:
        if rootConcept.name in dn.myBuilder["DataNodesConcepts"]:
            return dn.myBuilder["DataNodesConcepts"][rootConcept.name]

    dns = dn.findDatanodes(select = rootConcept)
    
    if dn.myBuilder != None:
        dn.myBuilder["DataNodesConcepts"][rootConcept.name] = dns
        
    return dns
    
def getCandidates(dn, e, variable, lcVariablesDns, lc, logger, integrate = False):
    conceptName = e[0].name
                        
    # -- Collect dataNode for the logical constraint (path)
    
    dnsList = [] # Stores lists of dataNodes for each corresponding dataNode 
    pathsCount = 0
    if variable.v == None: # No path - just concept
        if variable.name == None:
            logger.error('The element %s of logical constraint %s has no name for variable'%(conceptName, lc.lcName))
            return None
        
        # Check if we already found this variable
        if variable.name in lcVariablesDns:
            dnsList = lcVariablesDns[variable.name]  
        else:
            rootConcept = dn.findRootConceptOrRelation(conceptName)
            rootDns = findDatanodesForRootConcept(dn, rootConcept)
            dnsList = [[rDn] for rDn in rootDns]
    else: # Path specified
        from domiknows.graph.logicalConstrain import eqL
        if not isinstance(variable.v, eqL):
            if len(variable.v) == 0:
                logger.error('The element %s of logical constraint %s has empty part v of the variable'%(conceptName, lc.lcName))
                return None
            
        # -- Prepare paths
        path = variable.v
        if not isinstance(path, tuple):
            if isinstance(path, str):
                path = (path,)
        
        if not isinstance(path, (tuple, eqL)):
            raise TypeError("Path must be a tuple or eqL")
        
        paths = []
        
        if isinstance(path, eqL):
            paths.append(path)
        elif isinstance(path[0], str) and len(path) == 1:
            paths.append(path)
        elif isinstance(path[0], str) and not isinstance(path[1], tuple):
            paths.append(path)
        else: # If many paths
            for i, vE in enumerate(path):
                if i == 0 and isinstance(vE, str):
                    continue
                
                paths.append(vE)
                
        pathsCount = len(paths)
        eqLPaths = []
        
        # -- Process  paths
        dnsListForPaths = []
        eqLPaths = []
        for i, v in enumerate(paths):
            dnsListForPaths.append([])
            
            # Get name of the referred variable 
            if isinstance(path, eqL):
                referredVariableName = None
            else:
                referredVariableName = v[0] 
        
            if referredVariableName not in lcVariablesDns: # Not yet defined - it has to be the current lc element dataNodes list
                rootConcept = dn.findRootConceptOrRelation(conceptName)
                rootDns = findDatanodesForRootConcept(dn, rootConcept)
                referredDns = [[rDn] for rDn in rootDns]
                integrate = True
                new_iterate = True
            else: # Already defined in the logical constraint from the v part 
                new_iterate = False
                referredDns = lcVariablesDns[referredVariableName] # Get DataNodes for referred variables already defined in the logical constraint
                
            # Get variables from dataNodes selected  based on referredVariableName
            for indexDn, listOfDataNodes in enumerate(referredDns):
                eDns = [] 
                
                for currentReferredDataNode in listOfDataNodes:
                    if currentReferredDataNode is None:
                        continue
                    
                    # -- Get DataNodes for the edge defined by the path part of the v
                    if isinstance(path, eqL):
                        currentEDns = getEdgeDataNode(currentReferredDataNode, v, indexDn, lcVariablesDns)
                        currentEDnsNew = [currentEDn for currentEDn in currentEDns if currentEDn is not None]
                        if currentEDnsNew and len(currentEDnsNew):
                            eDns.extend(currentEDnsNew)
                        else:
                            pass
                    else:
                        currentEDns = getEdgeDataNode(currentReferredDataNode, v[1:], indexDn, lcVariablesDns) 
                        # currentEDnsNew = [currentEDn for currentEDn in currentEDns if currentEDn is not None]
                        if currentEDns is not None and len(currentEDns):
                            eDns.extend(currentEDns)
                        elif lc.__str__() != "fixedL":
                            # Log info that there is no candidates for the path 
                            vNames = [v if isinstance(v, str) else v.name for v in v[1:]] # collect names on the path
                            logger.info('%s has no path %s requested by %s for concept %s'%(currentReferredDataNode, vNames, lc.lcName, conceptName))
                                
                if not eDns and not new_iterate:
                    eDns = [None] # None - to keep track of candidates
                    
                if not new_iterate:
                    dnsListForPaths[i].append(eDns)
                elif len(eDns):
                    dnsListForPaths[i].append(eDns)
                    
            
            if isinstance(v, eqL) or (isinstance(path, tuple) and any(isinstance(elem, eqL) for elem in v)):
                eqLPaths.append(False)
            else:
                eqLPaths.append(True)
                
        # -- Compress candidates and print log about the path candidates
        for ip, dnsPathList in enumerate(dnsListForPaths):
            dnsPathListCompressed = [[elem for elem in sublist if elem is not None] or [None] for sublist in dnsPathList]
            if len(dnsPathListCompressed) == len(dnsPathList):
                dnsListForPaths[ip] = dnsPathListCompressed
            else:
                pass
            
            if len(dnsListForPaths) > 1:
                countValid = sum(1 for sublist in dnsListForPaths[ip] if sublist and any(elem is not None for elem in sublist))
                if eqLPaths[ip]:
                    logger.info('path %i involving eqL has collected %i candidates of which %i is not None %s'%(ip,len(dnsListForPaths[ip]), countValid, dnsListForPaths[ip]))
                else:
                    logger.info('path %i has collected %i candidates of which %i is not None %s'%(ip,len(dnsListForPaths[ip]), countValid, dnsListForPaths[ip]))
            
        # -- Select a single dns list or Combine the collected lists of dataNodes based on paths 
        dnsList = [] # candidates to be returned
        if pathsCount == 1: # Single path
            dnsList = dnsListForPaths[0]
        else:
            # --- Assume Intersection - TODO: in future use lo if defined to determine if different operation 
            noOfCandidatesSubsets = len(dnsListForPaths[0])
            for i in range(noOfCandidatesSubsets):
                # Collect subsets with the same index from each path
                try:
                    listsOfSubsetsForIndex = [dnsListForPaths[item][i] for item in range(pathsCount)]
                except IndexError as ei:
                    continue
                
                # Calculate the intersection
                se = intersection_of_lists(listsOfSubsetsForIndex)
                dnsList.append(se)
                
    # Returns candidates  
    dnsListCompressed = [[elem for elem in sublist if elem is not None] or [None] for sublist in dnsList] # compress
    if len(dnsListCompressed) == len(dnsList):
        dnsList = dnsListCompressed
    else:
        pass
    
    countValidC = sum(1 for sublist in dnsList if sublist and any(elem is not None for elem in sublist))
    if pathsCount > 1:
        logger.info('intersection of path resulted in %i candidates for %s of which %i is not None - %s'%(len(dnsList),conceptName,countValidC,dnsList))
    else:
        logger.info('collected %i candidates for %s of which %i is not None - %s'%(len(dnsList),conceptName,countValidC,dnsList))
  
    return dnsList

# Find DataNodes starting from the given DataNode following provided path
#     path can contain eqL statement selecting DataNodes from the DataNodes collecting on the path
def getEdgeDataNode(dn, path, currentIndexDN, lcVariablesDns):
    # Path is empty
    
    if isinstance(path, eqL):
        path = [path]
    if len(path) == 0:
        return [dn]

    # Path has single element
    if (not isinstance(path[0], eqL)) and len(path) == 1:
        relDns = dn.getDnsForRelation(path[0])
                
        if relDns is None or len(relDns) == 0 or relDns[0] is None:
            return [None]
        
        return relDns
            
    # Path has at least 2 elements - will perform recursion

    if isinstance(path[0], eqL): # check if eqL
        if isinstance(path[0].e[0], tuple):
            path0 = path[0].e[0][0]
        else:
            path0 = path[0].e[0]
    else:
        path0 = path[0]

    relDns = None         
    if dn.isRelation(path0):
        relDns = dn.getDnsForRelation(path0)            
    elif isinstance(path0, str):
        relDns = dn.getDnsForRelation(path0)
    else: # if not relation then has to be attribute in eql
        path0Dns = relDns
        attributeName = path[0].e[1]
        
        relDns = []
        if attributeName == "instanceID":
            
            for pDns in path0Dns:
                attributeValue = pDns.getInstanceID()
                referredCandidateID = path[0].e[2]
                
                if referredCandidateID in lcVariablesDns and len(lcVariablesDns[referredCandidateID]) - 1 >= currentIndexDN:
                    currentReferredCandidates = lcVariablesDns[referredCandidateID][currentIndexDN]
                    for currentReferredCandidate in currentReferredCandidates:
                        currentReferredCandidateID = currentReferredCandidate.getInstanceID()
                        
                        if currentReferredCandidateID == attributeValue:
                            if dn not in relDns:
                                relDns.append(dn)
                    
                # Check if it is a valid relation link with not empty set of connected datanodes      
                if relDns is None or len(relDns) == 0 or relDns[0] is None:
                    return [None]
                else:
                    return relDns
        else:
            relDns = []
            attributeName = path[0].e[1]
            
            attributeValue = dn.getAttribute(attributeName)
                
            if torch.is_tensor(attributeValue) and attributeValue.ndimension() == 0:
                attributeValue = attributeValue.item()
                
            requiredValue = path[0].e[2]
                
            if attributeValue in requiredValue:
                # return [dn]
                relDns.append(dn)
            elif (True in  requiredValue ) and attributeValue == 1:
                # return [dn]
                relDns.append(dn)
            elif (False in  requiredValue ) and attributeValue == 0:
                attributeValue = False
            else:
                # return [None]
                # relDns.append(None)
                pass
    
    # Check if it is a valid relation link  with not empty set of connected datanodes      
    if relDns is None or len(relDns) == 0 or relDns[0] is None:
        return [None]
        
    # if eqL then filter DataNode  
    if isinstance(path[0], eqL):
        _cDns = []
        for cDn in relDns:
            if isinstance(path[0].e[1], str):
                path0e1 = path[0].e[1]
            else:
                path0e1 = path[0].e[1].name
                
            if path0e1 in cDn.attributes or ("rootDataNode" in cDn.attributes and (path0.name + "/" + path0e1) in cDn.attributes["rootDataNode"].attributes["propertySet"]):
                if torch.is_tensor(cDn.getAttribute(path0e1)):
                    if cDn.getAttribute(path0e1).item() in path[0].e[2]:
                        _cDns.append(cDn)
                else:
                    if cDn.getAttribute(path0e1) in path[0].e[2]:
                        _cDns.append(cDn) 
                
        relDns = _cDns
    
    # recursion
    rDNS = []
    for cDn in relDns:
        rDn = getEdgeDataNode(cDn, path[1:], currentIndexDN, lcVariablesDns)
        
        if rDn:
            rDNS.extend(rDn)
            
    if rDNS:
        return rDNS
    else:
        return [None]