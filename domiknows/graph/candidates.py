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
        
    def strEs(self):
        return "combinationC(%s)"%(",".join([str(e) for e in self.e]))  
            
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
    if isinstance(rootConcept, str):
        print(f"Warning: rootConcept {rootConcept} is a string, expected a Concept or Relation object.")
    
    # Check if rootConcept is a string
    concept_name = rootConcept if isinstance(rootConcept, str) else rootConcept.name
    
    if dn.myBuilder != None and "DataNodesConcepts" in dn.myBuilder:
        if concept_name in dn.myBuilder["DataNodesConcepts"]:
            return dn.myBuilder["DataNodesConcepts"][concept_name]

    dns = dn.findDatanodes(select = rootConcept)
    
    if dn.myBuilder != None:
        dn.myBuilder["DataNodesConcepts"][concept_name] = dns
        
    return dns
    
def getCandidates(dn, e, variable, lcVariablesDns, lc, logger, integrate=False):
    """
    Get candidates for a constraint variable.
    
    Returns:
        tuple: (dnsList, referredVariableNames, expansionInfo)
            - dnsList: List of candidate DataNode lists
            - referredVariableNames: List of variable names referenced in path
            - expansionInfo: None if no expansion, or dict with:
                - 'mapping': list of (orig_group_idx, item_idx) tuples
                - 'expanded_vars': list of variable names that were expanded
    """
    indent = "  " * 0
    log = False
    def _log(msg):
        if not log:
            return
        if logger:
            logger.info(f"{indent}[getEdgeDataNode] {msg}")
        else:
            print(f"{indent}[getEdgeDataNode] {msg}")
            
    conceptName = e[0].name

    def _dn_repr(d):
        if d is None:
            return "None"
        return f"{d.getOntologyNode().name}#{d.getInstanceID()}"

    def _dns_list_repr(dns_list):
        if not dns_list:
            return "[]"
        parts = []
        for i, group in enumerate(dns_list[:3]):
            if group:
                items = [_dn_repr(d) for d in group[:3]]
                if len(group) > 3:
                    items.append(f"...+{len(group)-3}")
                parts.append(f"[{','.join(items)}]")
            else:
                parts.append("[]")
        if len(dns_list) > 3:
            parts.append(f"...+{len(dns_list)-3} groups")
        return f"({len(dns_list)} groups): [{', '.join(parts)}]"

    _log(f"=== getCandidates START for {conceptName}, variable={variable.name}, path={variable.v} ===")
    _log(f"  Current lcVariablesDns keys: {list(lcVariablesDns.keys())}")
    for k, v in lcVariablesDns.items():
        _log(f"    {k}: {len(v)} groups, sizes={[len(g) for g in v[:5]]}{'...' if len(v) > 5 else ''}")

    # -- Collect dataNode for the logical constraint (path)
    referredVariableNames = []
    expansionInfo = None  # Will be set if expansion occurs

    dnsList = []  # Stores lists of dataNodes for each corresponding dataNode
    pathsCount = 0
    if variable.v == None:  # No path - just concept
        if variable.name == None:
            logger.error('The element %s of logical constraint %s has no name for variable' % (conceptName, lc.lcName))
            return None, None, None

        # Check if we already found this variable
        if variable.name in lcVariablesDns:
            dnsList = lcVariablesDns[variable.name]
            _log(f"  No path, reusing existing variable '{variable.name}': {_dns_list_repr(dnsList)}")
        else:
            rootConcept = dn.findRootConceptOrRelation(conceptName)
            rootDns = findDatanodesForRootConcept(dn, rootConcept)
            dnsList = [[rDn] for rDn in rootDns]
            _log(f"  No path, created fresh candidates: {_dns_list_repr(dnsList)}")
    else:  # Path specified
        from domiknows.graph.logicalConstrain import eqL
        if not isinstance(variable.v, eqL):
            if len(variable.v) == 0:
                logger.error('The element %s of logical constraint %s has empty part v of the variable' % (conceptName, lc.lcName))
                return None, None, None

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
        else:  # If many paths
            for i, vE in enumerate(path):
                if i == 0 and isinstance(vE, str):
                    continue
                paths.append(vE)

        pathsCount = len(paths)
        logger.info(f"  Parsed {pathsCount} path(s): {paths}")
        eqLPaths = []

        # -- Process paths
        dnsListForPaths = []
        eqLPaths = []
        for i, v in enumerate(paths):
            dnsListForPaths.append([])

            # Get name of the referred variable
            if isinstance(path, eqL):
                referredVariableName = None
            else:
                referredVariableName = v[0]

            _log(f"  PATH {i}: referredVariableName='{referredVariableName}', path_spec={v}")

            if referredVariableName not in lcVariablesDns:  # Not yet defined
                rootConcept = dn.findRootConceptOrRelation(conceptName)
                rootDns = findDatanodesForRootConcept(dn, rootConcept)
                referredDns = [[rDn] for rDn in rootDns]
                integrate = True
                new_iterate = True
                _log(f"    Variable '{referredVariableName}' NOT in lcVariablesDns, using root: {_dns_list_repr(referredDns)}")
            else:  # Already defined in the logical constraint
                new_iterate = False
                referredDns = lcVariablesDns[referredVariableName]
                referredVariableNames.append(referredVariableName)
                _log(f"    Variable '{referredVariableName}' found in lcVariablesDns: {_dns_list_repr(referredDns)}")

            # Check if expansion is needed: source has multiple items per group
            needs_expansion = (
                not new_iterate and
                any(len(group) > 1 for group in referredDns if group)
            )

            if needs_expansion:   
                _log(f"    *** EXPANSION MODE ACTIVATED *** (source has multi-item groups)")
                
                # Collect results per (group, item) to maintain tuple correspondence
                expansion_results = []  # List of (orig_group_idx, item_idx, results)

                for indexDn, listOfDataNodes in enumerate(referredDns):
                    for itemIdx, currentReferredDataNode in enumerate(listOfDataNodes):
                        if currentReferredDataNode is None:
                            expansion_results.append((indexDn, itemIdx, [None]))
                            continue

                        if isinstance(path, eqL):
                            currentEDns = getEdgeDataNode(currentReferredDataNode, v, indexDn, lcVariablesDns, logger)
                        else:
                            currentEDns = getEdgeDataNode(currentReferredDataNode, v[1:], indexDn, lcVariablesDns, logger)

                        if currentEDns is not None and len(currentEDns):
                            filtered = [ee for ee in currentEDns if ee is not None]
                            expansion_results.append((indexDn, itemIdx, filtered if filtered else [None]))
                        else:
                            expansion_results.append((indexDn, itemIdx, [None]))

                _log(f"    Expansion: {len(referredDns)} groups → {len(expansion_results)} expanded groups")

                # Store expanded results for this path
                for orig_group_idx, item_idx, results in expansion_results:
                    dnsListForPaths[i].append(results)

                # Build expansion mapping for caller to use on lcVariables
                expansion_mapping = [(orig_group_idx, item_idx) for orig_group_idx, item_idx, _ in expansion_results]
                expanded_var_names = list(lcVariablesDns.keys())
                
                # Expand all previously defined variables in lcVariablesDns
                _log(f"    Expanding prior variables to match new structure...")
                for var_name in expanded_var_names:
                    old_structure = lcVariablesDns[var_name]
                    old_total = sum(len(g) for g in old_structure)
                    new_structure = []
                    for orig_group_idx, item_idx in expansion_mapping:
                        if orig_group_idx < len(old_structure):
                            old_group = old_structure[orig_group_idx]
                            if item_idx < len(old_group):
                                new_structure.append([old_group[item_idx]])
                            else:
                                new_structure.append([None])
                        else:
                            new_structure.append([None])
                    lcVariablesDns[var_name] = new_structure
                    new_total = sum(len(g) for g in new_structure)
                    _log(f"      {var_name}: {len(old_structure)} groups ({old_total} items) → {len(new_structure)} groups ({new_total} items)")

                # Return expansion info for caller to apply to lcVariables
                expansionInfo = {
                    'mapping': expansion_mapping,
                    'expanded_vars': expanded_var_names
                }

            else:
                _log(f"    Standard mode (no expansion needed)")
                _log(f"    Iterating over {len(referredDns)} groups (outer loop = group index)")
                
                for indexDn, listOfDataNodes in enumerate(referredDns):
                    eDns = []

                    for currentReferredDataNode in listOfDataNodes:
                        if currentReferredDataNode is None:
                            continue

                        if isinstance(path, eqL):
                            currentEDns = getEdgeDataNode(currentReferredDataNode, v, indexDn, lcVariablesDns, logger)
                            currentEDnsNew = [currentEDn for currentEDn in currentEDns if currentEDn is not None]
                            if currentEDnsNew and len(currentEDnsNew):
                                eDns.extend(currentEDnsNew)
                        else:
                            currentEDns = getEdgeDataNode(currentReferredDataNode, v[1:], indexDn, lcVariablesDns, logger)
                            if currentEDns is not None and len(currentEDns):
                                eDns.extend(currentEDns)
                            elif lc.__str__() != "fixedL":
                                vNames = [vv if isinstance(vv, str) else vv.name for vv in v[1:]]
                                _log('%s has no path %s requested by %s for concept %s' % (currentReferredDataNode, vNames, lc.lcName, conceptName))

                    if not eDns and not new_iterate:
                        eDns = [None]

                    if not new_iterate:
                        dnsListForPaths[i].append(eDns)
                    elif len(eDns):
                        dnsListForPaths[i].append(eDns)
                    
                    _log(f"      Group[{indexDn}] ACCUMULATED {len(eDns)} candidates (from {len(listOfDataNodes)} items)")

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

            total_items = sum(len(sublist) for sublist in dnsListForPaths[ip])
            countValid = sum(1 for sublist in dnsListForPaths[ip] if sublist and any(elem is not None for elem in sublist))
            _log(f"  PATH {ip} result: {len(dnsListForPaths[ip])} groups, {total_items} total items, {countValid} non-None groups")

        # -- Select a single dns list or Combine the collected lists of dataNodes based on paths
        dnsList = []  # candidates to be returned
        if pathsCount == 1:  # Single path
            dnsList = dnsListForPaths[0]
        else:
            # --- Assume Intersection
            noOfCandidatesSubsets = len(dnsListForPaths[0])
            for i in range(noOfCandidatesSubsets):
                try:
                    listsOfSubsetsForIndex = [dnsListForPaths[item][i] for item in range(pathsCount)]
                except IndexError as ei:
                    continue

                se = intersection_of_lists(listsOfSubsetsForIndex)
                dnsList.append(se)

    # Returns candidates
    dnsListCompressed = [[elem for elem in sublist if elem is not None] or [None] for sublist in dnsList]
    if len(dnsListCompressed) == len(dnsList):
        dnsList = dnsListCompressed
    else:
        pass

    countValidC = sum(1 for sublist in dnsList if sublist and any(elem is not None for elem in sublist))
    total_candidates = sum(len(sublist) for sublist in dnsList)

    _log(f"=== getCandidates END for {conceptName} ===")
    _log(f"  RESULT: {len(dnsList)} groups, {total_candidates} total candidates, {countValidC} non-None groups")
    _log(f"  Structure: {_dns_list_repr(dnsList)}")
    
    # Log final state of lcVariablesDns
    _log(f"  Updated lcVariablesDns:")
    for k, v in lcVariablesDns.items():
        _log(f"    {k}: {len(v)} groups, sizes={[len(g) for g in v[:5]]}{'...' if len(v) > 5 else ''}")

    if expansionInfo:
        _log(f"  EXPANSION INFO: {len(expansionInfo['mapping'])} mappings, expanded vars: {expansionInfo['expanded_vars']}")

    if total_candidates > 1000:
        logger.warning(f"  ⚠️ CANDIDATE EXPLOSION: {total_candidates} candidates for {conceptName}!")

    return dnsList, referredVariableNames, expansionInfo

# Find DataNodes starting from the given DataNode following provided path
#     path can contain eqL statement selecting DataNodes from the DataNodes collecting on the path
def getEdgeDataNode(dn, path, currentIndexDN, lcVariablesDns, logger=None, depth=0):
    """
    Find DataNodes from the DataNodes collecting on the path.
    Enhanced with diagnostic logging.
    
    Args:
        dn: Starting DataNode
        path: Path specification (tuple or eqL)
        currentIndexDN: Current index in the outer loop (group index)
        lcVariablesDns: Dictionary of variable name -> list of DataNode lists
        logger: Optional logger for diagnostics
        depth: Recursion depth for indentation
    """
    indent = "  " * depth
    log = False
    def _log(msg):
        if not log:
            return
        if logger:
            logger.debug(f"{indent}[getEdgeDataNode] {msg}")
        else:
            print(f"{indent}[getEdgeDataNode] {msg}")
    
    def _dn_repr(d):
        """Compact representation of DataNode"""
        if d is None:
            return "None"
        return f"{d.getOntologyNode().name}#{d.getInstanceID()}"
    
    def _dns_repr(dns):
        """Compact representation of DataNode list"""
        if not dns:
            return "[]"
        return f"[{', '.join(_dn_repr(d) for d in dns[:5])}{'...' if len(dns) > 5 else ''}]({len(dns)})"
    
    # Path is empty
    if isinstance(path, eqL):
        path = [path]
    if len(path) == 0:
        _log(f"Empty path, returning dn={_dn_repr(dn)}")
        return [dn]

    _log(f"Processing path={path}, from dn={_dn_repr(dn)}, currentIndexDN={currentIndexDN}")

    # Path has single element
    if (not isinstance(path[0], eqL)) and len(path) == 1:
        relDns = dn.getDnsForRelation(path[0])
        _log(f"Single element path '{path[0]}' -> {_dns_repr(relDns)}")
                
        if relDns is None or len(relDns) == 0 or relDns[0] is None:
            return [None]
        
        return relDns
            
    # Path has at least 2 elements - will perform recursion
    if isinstance(path[0], eqL):
        if isinstance(path[0].e[0], tuple):
            path0 = path[0].e[0][0]
        else:
            path0 = path[0].e[0]
    else:
        path0 = path[0]

    _log(f"Multi-element path, first element path0={path0}")

    relDns = None         
    if dn.isRelation(path0):
        relDns = dn.getDnsForRelation(path0)
        _log(f"path0 is relation, getDnsForRelation -> {_dns_repr(relDns)}")
    elif isinstance(path0, str):
        relDns = dn.getDnsForRelation(path0)
        _log(f"path0 is string '{path0}', getDnsForRelation -> {_dns_repr(relDns)}")
    else:
        # Handle attribute in eqL
        path0Dns = relDns
        attributeName = path[0].e[1]
        _log(f"path0 is eqL attribute check: {attributeName}")
        
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
                    
                if relDns is None or len(relDns) == 0 or relDns[0] is None:
                    return [None]
                else:
                    return relDns
        else:
            attributeValue = dn.getAttribute(attributeName)
                
            if torch.is_tensor(attributeValue) and attributeValue.ndimension() == 0:
                attributeValue = attributeValue.item()
                
            requiredValue = path[0].e[2]
                
            if attributeValue in requiredValue:
                relDns.append(dn)
            elif (True in requiredValue) and attributeValue == 1:
                relDns.append(dn)
            elif (False in requiredValue) and attributeValue == 0:
                attributeValue = False
    
    # Check if it is a valid relation link
    if relDns is None or len(relDns) == 0 or relDns[0] is None:
        _log(f"No valid relation links found, returning [None]")
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
        _log(f"After eqL filter: {_dns_repr(relDns)}")
    
    # recursion
    _log(f"Recursing through {len(relDns)} relation DataNodes with remaining path {path[1:]}")
    rDNS = []
    for cDn in relDns:
        rDn = getEdgeDataNode(cDn, path[1:], currentIndexDN, lcVariablesDns, logger, depth + 1)
        
        if rDn:
            rDNS.extend(rDn)
            
    _log(f"Recursion result: {_dns_repr(rDNS)}")
    
    if rDNS:
        return rDNS
    else:
        return [None]