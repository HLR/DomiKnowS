from functools import reduce
from collections import OrderedDict

from domiknows.graph import LcElement

class CandidateSelection(LcElement):
    def __init__(self, *e,  name = None):
        super().__init__(*e, name = name)
        
        originalCall  = type(self).__call__
        originalCallName = originalCall.__name__
        if "newCall" in originalCall.__name__:
            return
        
        def newCall (self, *input, keys = None):
            if input and isinstance(input[0], OrderedDict):
                candiatesDict = input[0]
                candidates = []
                
                for cKey in candiatesDict:
                    flat_list = [item for sublist in candiatesDict[cKey] for item in sublist]
                    candidates.append(flat_list)
                
                return originalCall(self, *candidates, keys=keys)
            else:
                return self, input
        type(self).__call__ = newCall  
        
    def __call__(self, *candidates, keys=None): 
        pass 

# object creating Cartesian product of all candidates
class combinationC(CandidateSelection):
    def __init__(self, *e, name = None):
        super().__init__(*e, name = name)
        
    def __call__(self, *candidates_list, keys=None):
        from  itertools import product
        
        # Create the Cartesian product of all candidates
        cartesian_product = list(product(*candidates_list))
        
        # Extract lists of first elements, second elements, etc.
        extracted_elements = list(zip(*cartesian_product))
        
        # Wrap each element in the extracted elements lists in a separate list
        wrapped_elements = [[element] for sublist in extracted_elements for element in sublist]
        
        # Group the wrapped elements based on their original sublist in extracted_elements
        grouped_elements = [list(map(lambda x: [x], sublist)) for sublist in extracted_elements]  
              
        # Create a dictionary using the provided keys and the grouped elements
        result_dict = dict(zip(keys, grouped_elements))
        
        return result_dict

def getCandidates(dn, e, variable, lcVariablesDns, lc, logger):
    conceptName = e[0].name
                        
    # -- Collect dataNode for the logical constraint (path)
    
    dnsList = [] # Stores lists of dataNodes for each corresponding dataNode 
    
    if variable.v == None: # No path - just concept
        if variable.name == None:
            logger.error('The element %s of logical constraint %s has no name for variable'%(conceptName, lc.lcName))
            return None
                                    
        rootConcept = dn.findRootConceptOrRelation(conceptName)
        _dns = dn.findDatanodes(select = rootConcept)
        dnsList = [[dn] for dn in _dns]
    else: # Path specified
        from domiknows.graph.logicalConstrain import eqL
        if not isinstance(variable.v, eqL):
            if len(variable.v) == 0:
                logger.error('The element %s of logical constraint %s has empty part v of the variable'%(conceptName, lc.lcName))
                return None
            
        # -- Prepare paths
        path = variable.v
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
        
        # -- Process  paths
        dnsListForPaths = []
        for i, v in enumerate(paths):
            dnsListForPaths.append([])
            
            # Get name of the referred variable 
            if isinstance(path, eqL):
                referredVariableName = None
            else:
                referredVariableName = v[0] 
        
            if referredVariableName not in lcVariablesDns: # Not yet defined - it has to be the current lc element dataNodes list
                rootConcept = dn.findRootConceptOrRelation(conceptName)
                _dns = dn.findDatanodes(select = rootConcept)
                referredDns = [[dn] for dn in _dns]
                integrate = True
            else: # already defined in the logical constraint from the v part 
                referredDns = lcVariablesDns[referredVariableName] # Get DataNodes for referred variables already defined in the logical constraint
                
            # Get variables from dataNodes selected  based on referredVariableName
            for listOfDataNodes in referredDns:
                eDns = [] 
                
                for currentReferredDataNode in listOfDataNodes:
                    if currentReferredDataNode is None:
                        continue
                    
                    # -- Get DataNodes for the edge defined by the path part of the v
                    if isinstance(path, eqL):
                        _eDns = currentReferredDataNode.getEdgeDataNode(v) 
                    else:
                        _eDns = currentReferredDataNode.getEdgeDataNode(v[1:]) 
                    
                    if _eDns and _eDns[0]:
                        eDns.extend(_eDns)
                    elif not isinstance(path, eqL):
                        vNames = [v if isinstance(v, str) else v.name for v in v[1:]]
                        if lc.__str__() != "fixedL":
                            logger.info('%s has no path %s requested by %s for concept %s'%(currentReferredDataNode, vNames, lc.lcName, conceptName))
                if not eDns:
                    eDns = [None] # None - to keep track of candidates
                    
                dnsListForPaths[i].append(eDns)
            
        # -- Select a single dns list or Combine the collected lists of dataNodes based on paths 
        dnsList = []
        newIntersection = True
        if newIntersection:
            if pathsCount == 1: # Single path
                dnsList = dnsListForPaths[0]
            else:
                # --- Assume Intersection - TODO: in future use lo if defined to determine if different operation                                
                for i in range(len(dnsListForPaths[0])):
                    try:
                        se = [set(dnsListForPaths[item][i]) for item in range(pathsCount)]
                    except IndexError as ei:
                        continue
                    # Go through all the sets and calculate the intersection
                    dnsListR = reduce(set.intersection, se)
                    dnsList.append(list(dnsListR))
        else:
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
                
    return dnsList