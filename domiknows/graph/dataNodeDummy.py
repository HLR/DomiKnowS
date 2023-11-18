from . import DataNode
import torch
from domiknows.graph import fixedL, ifL

dataSizeInit = 5

def findConcept(conceptName, usedGraph):
    subGraph_keys = [key for key in usedGraph._objs]
    for subGraphKey in subGraph_keys:
        subGraph = usedGraph._objs[subGraphKey]
       
        for conceptNameItem in subGraph.concepts:
            if conceptName == conceptNameItem:
                concept = subGraph.concepts[conceptNameItem]
               
                return concept
            
    return None 

def findConceptInfo(usedGraph, concept):
    conceptInfo = {
        'concept': concept,
        'relation': bool(concept.has_a()),
        'relationAttrs': {rel.name: findConcept(rel.dst.name, usedGraph) for _, rel in enumerate(concept.has_a())},
        'contains': [contain.dst for contain in concept._out.get('contains', [])],
        'containedIn': [contain.src for contain in concept._in.get('contains', [])],
        'is_a': [contain.dst for contain in concept._out.get('is_a', [])]
    }

    if not conceptInfo['containedIn'] and not conceptInfo['is_a'] and not conceptInfo['relation']:
        conceptInfo['root'] = True
    else:
        conceptInfo['root'] = False
        
    return conceptInfo

def addDatanodes(concept, conceptInfos, datanodes, allDns, level=1):
    currentConceptInfo = conceptInfos[concept.name]
    instanceID = currentConceptInfo.get('count', 0)

    for dn in datanodes:
        dns = []
        for _ in range(dataSizeInit * level):
            newDN = DataNode(instanceID = instanceID, ontologyNode = currentConceptInfo['concept'])
            dn.addChildDataNode(newDN)
            dns.append(newDN)
            instanceID += 1
        
        for contain in currentConceptInfo['contains']:
            addDatanodes(contain, conceptInfos, dns, allDns, level = level+1)
            
        currentConceptInfo['count'] = currentConceptInfo.get('count', 0) + dataSizeInit * level
        currentConceptInfo.setdefault('dns', {}).setdefault(dn.ontologyNode.name, []).extend(dns)
        currentConceptInfo['dns'].setdefault('all', []).extend(dns)
        
        allDns.extend(dns)

def createDummyDataNode(graph):
    rootDataNode = None # this will be the created dummy Datanode graph root 
    rootConcept = None # find the root concept
    
    conceptInfos = {} # info collected about all concepts
    allDns = []       # all created datanodes
    
    # Collect concepts info from main graph
    for currentConceptKey, currentConcept in graph.concepts.items():
        conceptInfo = findConceptInfo(graph, currentConcept)
        conceptInfos[currentConceptKey] = conceptInfo
        if conceptInfo['root']:
            rootConcept = currentConceptKey 
        
    # Collect concepts info from subgraph
    for subGraphKey, subGraph in graph.subgraphs.items():
        for currentConceptKey, currentConcept in subGraph.concepts.items():
            conceptInfo = findConceptInfo(subGraph, currentConcept)
            conceptInfos[currentConceptKey] = conceptInfo
            if conceptInfo['root']:
                rootConcept = currentConceptKey 
                  
    if rootConcept:
        # Add root datanodes
        rootConceptInfo = conceptInfos[rootConcept]
        rootDataNode = DataNode(instanceID = 1, ontologyNode = rootConceptInfo['concept'])
        rootDataNode.attributes["variableSet"] = {}
        
        # Add children datanodes
        for contain in rootConceptInfo['contains']:
            addDatanodes(contain, conceptInfos, [rootDataNode], allDns)
            
        # Add relations
        for currentConceptKey in conceptInfos:
            relationConceptInfo = conceptInfos[currentConceptKey]
            relationDns = []
            
            if relationConceptInfo['relation'] and not relationConceptInfo['is_a']:
                for d, attr in enumerate(relationConceptInfo['relationAttrs']):
                    attrConceptInfo = conceptInfos[relationConceptInfo['relationAttrs'][attr].name]
                    
                    instanceID = relationConceptInfo.get('count', 0)
                   
                    for i in range(attrConceptInfo['count']):
                        if d == 0:
                            newDN = DataNode(instanceID = instanceID, ontologyNode = relationConceptInfo['concept'])
                            relationDns.append(newDN)
                            instanceID += 1
                        else:
                            if i < len(relationDns):
                                newDN = relationDns[i]
                            else:
                                break
                            
                        newDN.addRelationLink(attr, attrConceptInfo["dns"]["all"][i])
                        
                    relationConceptInfo['count'] = relationConceptInfo.get('count', 0) + instanceID

                allDns.extend(relationDns)       
                relationConceptInfo.setdefault('dns', {})['all'] = relationDns
                    
        # Add probabilities
        for currentConceptKey in conceptInfos:
            conceptInfo = conceptInfos[currentConceptKey]
            if conceptInfo['is_a']:
                conceptRootConceptInfo = conceptInfos[conceptInfo['is_a'][0].name]
                
                if 'count' not in conceptRootConceptInfo:
                    continue
                
                m = conceptRootConceptInfo['count']
                random_tensor = torch.rand(m, 1, device=rootDataNode.current_device)
                final_tensor = torch.cat((1 - random_tensor, random_tensor), dim=1)
                rootDataNode.attributes["variableSet"][conceptRootConceptInfo['concept'].name +'/<' + conceptInfo['concept'].name + '>'] = final_tensor
                continue
        
    # Iterate over the data nodes in "allDns" and add the "rootDataNode" attribute to them
    for dn in allDns:
        if dn == rootDataNode:
            continue
        dn.attributes["rootDataNode"] = rootDataNode
                
    return rootDataNode

def ifConstrainSatisfactionMsg(current_lcSatisfaction, constInputIndex, InstantiationsAndSatisfactionMsg, InstantiationsAndSatisfactionMsgs):
    currentLc = current_lcSatisfaction['lc']

    ifResult = not current_lcSatisfaction['lcResult'][constInputIndex][0].item()

    ifKey = next(reversed(current_lcSatisfaction['input']))
    ifLc = current_lcSatisfaction['input'][ifKey]

    # Create an iterator for the keys of ifLc
    key_iterator = iter(ifLc)

    # Get the first key and its corresponding value
    ifPromiseKey = next(key_iterator)
    ifPromise = ifLc[ifPromiseKey][constInputIndex][0].item()

    # Get the next key and its corresponding value
    ifConclusionKey = next(key_iterator)
    ifConclusion = ifLc[ifConclusionKey][constInputIndex][0].item()

    if ifResult:
        InstantiationsAndSatisfactionMsg = f'IfL premise is {ifPromise} and its conclusion is {ifConclusion}\n' + InstantiationsAndSatisfactionMsg
    else:
        InstantiationsAndSatisfactionMsg = f'When in IfL the premise is True, the conclusion should also be True\n' + InstantiationsAndSatisfactionMsg

    if ifConclusionKey.startswith("_lc"):
        InstantiationsAndSatisfactionMsg = f'{currentLc.e[-1]}(**) -> {ifConclusion}\n' + InstantiationsAndSatisfactionMsg
    else:
        InstantiationsAndSatisfactionMsg = f'{currentLc.e[-1][0]}(\'{ifConclusionKey}\') -> {ifConclusion}\n' + InstantiationsAndSatisfactionMsg

    if ifPromiseKey.startswith("_lc"):
        InstantiationsAndSatisfactionMsg = f'{currentLc.e[0]}(**) -> {ifPromise}\n' + InstantiationsAndSatisfactionMsg
    else:
        InstantiationsAndSatisfactionMsg = f'{currentLc.e[0][0]}(\'{ifPromiseKey}\') -> {ifPromise}\n' + InstantiationsAndSatisfactionMsg

    if ifResult:
        InstantiationsAndSatisfactionMsg = f'The constraint is satisfied because:\n' + InstantiationsAndSatisfactionMsg
    else:
        InstantiationsAndSatisfactionMsg = f'The constraint is Not satisfied because:\n' + InstantiationsAndSatisfactionMsg

    InstantiationsAndSatisfactionMsgs.append(InstantiationsAndSatisfactionMsg)
    
    return 
    if ifConclusionKey.startswith("_lc"):
        if isinstance(currentLc.e[-1], ifL):
            ifConstrainSatisfactionMsg(current_lcSatisfaction, constInputIndex, "", InstantiationsAndSatisfactionMsgs)

def instanceReportOfConstraints(dn):
    m = None     
    sampleSize = 1
    p = sampleSize
   
    key = "/local/softmax"
    dn.inferLocal()
    
    mySolver, _ = dn.getILPSolver(conceptsRelations = dn.collectConceptsAndRelations())
    mySolver.current_device = dn.current_device
    mySolver.myLcLossSampleBooleanMethods.sampleSize = sampleSize        
    mySolver.myLcLossSampleBooleanMethods.current_device = dn.current_device

    lcCounter = 0 # Count processed lcs
    lcSatisfaction = {}
    for graph in mySolver.myGraph: # Loop through graphs
        for _, lc in graph.logicalConstrains.items(): # loop trough lcs in the graph

            if not lc.headLC or not lc.active: # Process only active and head lcs
                continue
                
            if type(lc) is fixedL: # Skip fixedL lc
                continue
                
            lcCounter +=  1
            
            lcName = lc.lcName
                
            lcSatisfaction[lcName] = {}
            current_lcSatisfaction = lcSatisfaction[lcName]
            current_lcSatisfaction['lc'] = lc
            
            lcResult, lcInput, inputLc = \
                mySolver.constructLogicalConstrains(lc, mySolver.myLcLossSampleBooleanMethods, m, dn, p, key = key, headLC = True, loss = True, sample = True)
            current_lcSatisfaction['lcResult'] = lcResult
            current_lcSatisfaction['lcInput'] = lcInput
            current_lcSatisfaction['input'] = inputLc

    for lc in lcSatisfaction:
        current_lcSatisfaction = lcSatisfaction[lc]
        InstantiationsAndSatisfactionMsgs = []
        noConstInput = len(current_lcSatisfaction['lcResult'])
        for constInputIndex in range(noConstInput):
            '''
            Instantiations and satisfaction:
                For 
                `x` being Pair 3, connecting Phrase 1 and Phrase 5 to each other. 
                work_for('x') == 1
                person('x', arg1) == 0 (phrase1)
                organization('x', arg2) ==1 (phrase 5)
                The constraint is Not satisfied because:
                work_for('x') --> True
                andL(**) --> False 
                When in IfL the premise is True, the conclusion should also be True. 
            '''
            
            ifConstrainSatisfactionMsg(current_lcSatisfaction, constInputIndex, "", InstantiationsAndSatisfactionMsgs)
                    
        current_lcSatisfaction['InstantiationsAndSatisfactionMsgs'] = InstantiationsAndSatisfactionMsgs
        
    return lcSatisfaction