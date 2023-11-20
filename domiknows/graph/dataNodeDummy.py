from . import DataNode
import torch
from domiknows.graph import fixedL, ifL, V

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

def lcConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, lcResult, lcTestIndex, lcSatisfactionMsg, headLc):
    currentLc = next(inputIterator)
    currentLcInput = current_lcSatisfaction['input'][currentLc]

    if lcResult:
        lcSatisfactionMsg += f'{currentLc}{headLc} is satisfied (True) because:\n'
    else:
        lcSatisfactionMsg += f'{currentLc}{headLc} is Not satisfied (False) because:\n'
        
    # Create an iterator for the operands of currentLc
    operands_iterator = iter(currentLcInput)
    index = 0
    for operand in operands_iterator:
        # Get the first key and its corresponding value
        currentOperand = currentLcInput[operand][lcTestIndex][0].item()
         
        if isinstance(currentLc.e[index], V):
            if len(currentLc.e) < index+1:
                continue
            index += 1
            
        if operand.startswith("_lc"):
            lcSatisfactionMsg += f'\t{currentLc.e[index]}(**) -> {currentOperand}\n'
       
            operandResult = currentLcInput[currentOperand][lcTestIndex][0].item()
            if isinstance(currentLc.e[1], ifL):
                lcSatisfactionMsg = ifConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, operandResult, lcTestIndex, lcSatisfactionMsg, "(**)")
            else:
                lcSatisfactionMsg = lcConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, operandResult, lcTestIndex, lcSatisfactionMsg, "(**)")
        elif operand.startswith("_"): # anonymous - no variable in the lc
            lcSatisfactionMsg += f'\t{currentLc.e[index][1]} -> {currentOperand}\n'
        else:
            if len(currentLc.e) > index+1 and isinstance(currentLc.e[index+1], V) and currentLc.e[index+1].v != None:
                currentV = currentLc.e[index+1].v
                lcSatisfactionMsg += f'\t{currentLc.e[index][1]}(\'{currentV[0]}\', {currentV[1]}) -> {currentOperand}\n'
            else:
                lcSatisfactionMsg += f'\t{currentLc.e[index][1]}(\'{operand}\') -> {currentOperand}\n'
                
        index += 1
            
    return lcSatisfactionMsg

def ifConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, ifResult, lcTestIndex, lcSatisfactionMsg, headLc):
    currentLc = next(inputIterator)
    currentLcInput = current_lcSatisfaction['input'][currentLc]

    if ifResult:
        lcSatisfactionMsg += f'{currentLc}{headLc} is satisfied (True) because:\n' 
    else:
        lcSatisfactionMsg += f'{currentLc}{headLc} is Not satisfied (False) because:\n'
        
    # Create an iterator for the keys of currentLc
    operands_iterator = iter(currentLcInput)

    # Get the first key and its corresponding value
    ifPromiseKey = next(operands_iterator)
    ifPromise = currentLcInput[ifPromiseKey][lcTestIndex][0].item()

    # Get the next key and its corresponding value
    ifConclusionKey = next(operands_iterator)
    ifConclusion = currentLcInput[ifConclusionKey][lcTestIndex][0].item()
    
    # Create an iterator for the operands of currentLc
    operands_iterator = iter(currentLcInput)
    index = 0
    promiseIndex = None
    conclusionIndex = None
    for operand in operands_iterator:
        # Get the first key and its corresponding value
        currentOperand = currentLcInput[operand][lcTestIndex][0].item()
        
        if isinstance(currentLc.e[index], V):
            if len(currentLc.e) < index+1:
                continue
            index += 1
            
        if operand.startswith("_lc"):
            lcSatisfactionMsg += f'\t{currentLc.e[index]}(**) -> {currentOperand}\n'
        elif operand.startswith("_"): # anonymous - no variable in the lc
            lcSatisfactionMsg += f'\t{currentLc.e[index][1]} -> {currentOperand}\n'
        else:
            if len(currentLc.e) > index+1 and isinstance(currentLc.e[index+1], V) and currentLc.e[index+1].v != None:
                currentV = currentLc.e[index+1].v
                lcSatisfactionMsg += f'\t{currentLc.e[index][1]}(\'{currentV[0]}\', {currentV[1]}) -> {currentOperand}\n'
            else:
                lcSatisfactionMsg += f'\t{currentLc.e[index][1]}(\'{operand}\') -> {currentOperand}\n'
                
        if promiseIndex == None:
            promiseIndex = index
        else:
            conclusionIndex = index
            
        index += 1
        
    if ifResult:
        lcSatisfactionMsg += f'\t{currentLc} premise is {ifPromise} and its conclusion is {ifConclusion}\n' 
    else:
        lcSatisfactionMsg += f'\tWhen in {currentLc} the premise is True, the conclusion should also be True\n'

    if ifPromiseKey.startswith("_lc"):
        promiseResult = currentLcInput[ifPromiseKey][lcTestIndex][0].item()
        if isinstance(currentLc.e[promiseIndex], ifL):
            lcSatisfactionMsg = ifConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, promiseResult, lcTestIndex, lcSatisfactionMsg, "(**)")
        else:
            lcSatisfactionMsg = lcConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, promiseResult, lcTestIndex, lcSatisfactionMsg, "(**)")
            
    if ifConclusionKey.startswith("_lc"):
        conclusionResult = currentLcInput[ifConclusionKey][lcTestIndex][0].item()
        if isinstance(currentLc.e[conclusionIndex], ifL):
            lcSatisfactionMsg = ifConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, conclusionResult, lcTestIndex, lcSatisfactionMsg, "(**)")
        else:
            lcSatisfactionMsg = lcConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, conclusionResult, lcTestIndex, lcSatisfactionMsg, "(**)")
    
    return lcSatisfactionMsg

def satisfactioReportOfConstraints(dn):
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
            
            lcResult, lcInput, inputLc = \
                mySolver.constructLogicalConstrains(lc, mySolver.myLcLossSampleBooleanMethods, m, dn, p, key = key, headLC = True, loss = True, sample = True)
            current_lcSatisfaction['lcResult'] = lcResult
            current_lcSatisfaction['lcInput'] = lcInput
            current_lcSatisfaction['input'] = inputLc

    for lc in lcSatisfaction:
        current_lcSatisfaction = lcSatisfaction[lc]
        lcSatisfactionMsgs = []
        lenOfLcTests = len(current_lcSatisfaction['lcResult'])
        
        for lcTestIndex in range(lenOfLcTests):
            inputIterator = reversed(current_lcSatisfaction['input'])

            currentLc = next(reversed(current_lcSatisfaction['input']))
            lcResult = not current_lcSatisfaction['lcResult'][lcTestIndex][0].item()
            lcSatisfactionMsg = ''
            if isinstance(currentLc, ifL):
                lcSatisfactionMsg = ifConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, lcResult, lcTestIndex, lcSatisfactionMsg, "")
            else:
                lcSatisfactionMsg = lcConstrainSatisfactionMsg(current_lcSatisfaction, inputIterator, lcResult, lcTestIndex, lcSatisfactionMsg, "")
            
            lcSatisfactionMsgs.append(lcSatisfactionMsg)

        current_lcSatisfaction['lcSatisfactionMsgs'] = lcSatisfactionMsgs
        
    return lcSatisfaction