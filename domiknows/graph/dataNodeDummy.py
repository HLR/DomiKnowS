from . import DataNode
import torch
import traceback
from domiknows.graph import LogicalConstrain, fixedL, ifL, V
from domiknows.graph import EnumConcept

dataSizeInit = 5

def findConcept(conceptName, usedGraph):
    subGraph_keys = [key for key in usedGraph.subgraphs]
    for subGraphKey in subGraph_keys:
        subGraph = usedGraph.subgraphs[subGraphKey]
       
        for conceptNameItem in subGraph.concepts:
            if conceptName == conceptNameItem:
                concept = subGraph.concepts[conceptNameItem]
               
                return concept
            
    return None 

def findConceptInfo(usedGraph, concept):
    conceptInfo = {
        'concept': concept,
        'multiplicity': len(concept.enum) if isinstance(concept, EnumConcept) else 2,
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
                    if not relationConceptInfo['relationAttrs'][attr]:
                        continue
                        
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
                    m = 1
                else:
                    m = conceptRootConceptInfo['count']
                    
                # Assuming m, multiplicity, and rootDataNode.current_device are already defined
                random_tensor = torch.rand(m, conceptInfo['multiplicity'], device=rootDataNode.current_device)

                # Normalize the tensor so that the sum of each row equals 1
                final_tensor = random_tensor / random_tensor.sum(dim=1, keepdim=True)
                
                if 'count' not in conceptRootConceptInfo:
                    rootDataNode.attributes['<' + conceptInfo['concept'].name + '>'] = final_tensor
                else:
                    rootDataNode.attributes["variableSet"][conceptRootConceptInfo['concept'].name + '/<' + conceptInfo['concept'].name + '>'] = final_tensor
                continue
        
    # Iterate over the data nodes in "allDns" and add the "rootDataNode" attribute to them
    for dn in allDns:
        if dn == rootDataNode:
            continue
        dn.attributes["rootDataNode"] = rootDataNode
        
    if not rootDataNode.attributes.get("variableSet"):
        # Remove "variableSet" from rootDataNode.attributes
        rootDataNode.attributes.pop("variableSet", None)  # None is the default value if "variableSet" is not found
          
    return rootDataNode

def construct_ls_path_string(value):
    path_elements = []
    if isinstance(value[1], tuple):
        for subpath in value:
            # Convert each subpath element to a string
            subpath_elements = ["\'" + subpath[0] + "\'"] + [str(e) for e in subpath[1:]]
            path_elements.append("(" + ", ".join(subpath_elements) + ")")
    else:
        # Convert each value element to a string
        value_elements = ["\'" + value[0] + "\'"] + [str(e) for e in value[1:]]
        path_elements.append(", ".join(value_elements))
        
    return "".join(path_elements)

def lcConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, currentLc, lcResult, lcTestIndex, lcSatisfactionMsg, headLc):
    nestedLc = []
    for _, e in enumerate(currentLc.e):
        if isinstance(e, LogicalConstrain):
            currentNestedLc = next(lcIterator)
            nestedLc.append(currentNestedLc)
     
    nestedLcIterator = reversed(nestedLc)  
    
    currentLcInput = lcSatisfactionTest['lcs'][currentLc]

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
       
            # call nested lc satisfaction
            operandResult = currentOperand
            if isinstance(currentLc.e[index], ifL):
                lcSatisfactionMsg = ifConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, next(nestedLcIterator),operandResult, lcTestIndex, lcSatisfactionMsg, "(**)")
            else:
                lcSatisfactionMsg = lcConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, next(nestedLcIterator), operandResult, lcTestIndex, lcSatisfactionMsg, "(**)")
        elif operand.startswith("_"): # anonymous - no variable in the lc
            lcSatisfactionMsg += f'\t{currentLc.e[index][1]} -> {currentOperand}\n'
        else:
            if len(currentLc.e) > index + 1 and isinstance(currentLc.e[index + 1], V) and currentLc.e[index + 1].v is not None:
                currentV = currentLc.e[index + 1].v
                path = construct_ls_path_string(currentV)
                lcSatisfactionMsg += f'\t{currentLc.e[index][1]}({path}) -> {currentOperand}\n'
            else:
                lcSatisfactionMsg += f'\t{currentLc.e[index][1]}(\'{operand}\') -> {currentOperand}\n'
                
        index += 1
            
    return lcSatisfactionMsg

def ifConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, currentLc, ifResult, lcTestIndex, lcSatisfactionMsg, headLc):
    currentLcInput = lcSatisfactionTest['lcs'][currentLc]

    nestedLc = []
    for _, e in enumerate(currentLc.e):
        if isinstance(e, LogicalConstrain):
            currentNestedLc = next(lcIterator)
            nestedLc.append(currentNestedLc)
     
    nestedLcIterator = reversed(nestedLc)    
           
    if ifResult:
        lcSatisfactionMsg += f'{currentLc}{headLc} is satisfied (True) because:\n' 
    else:
        lcSatisfactionMsg += f'{currentLc}{headLc} is Not satisfied (False) because:\n'
        
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
                currentV = currentLc.e[index + 1].v
                path = construct_ls_path_string(currentV)
                lcSatisfactionMsg += f'\t{currentLc.e[index][1]}({path}) -> {currentOperand}\n'
            else:
                lcSatisfactionMsg += f'\t{currentLc.e[index][1]}(\'{operand}\') -> {currentOperand}\n'
                
        if promiseIndex == None:
            promiseIndex = index
            promiseResult = currentOperand
        else:
            conclusionIndex = index
            conclusionResult = currentOperand
            
        index += 1
        
    if ifResult:
        lcSatisfactionMsg += f'\t{currentLc} premise is {promiseResult} and its conclusion is {conclusionResult}\n' 
    else:
        lcSatisfactionMsg += f'\tWhen in {currentLc} the premise is True, the conclusion should also be True\n'

    if isinstance(currentLc.e[promiseIndex], LogicalConstrain):
        if isinstance(currentLc.e[promiseIndex], ifL):
            lcSatisfactionMsg = ifConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, next(nestedLcIterator), promiseResult, lcTestIndex, lcSatisfactionMsg, "(**)")
        else:
            lcSatisfactionMsg = lcConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, next(nestedLcIterator), promiseResult, lcTestIndex, lcSatisfactionMsg, "(**)")
            
    if isinstance(currentLc.e[conclusionIndex], LogicalConstrain):
        if isinstance(currentLc.e[conclusionIndex], ifL):
            lcSatisfactionMsg = ifConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, next(nestedLcIterator), conclusionResult, lcTestIndex, lcSatisfactionMsg, "(**)")
        else:
            lcSatisfactionMsg = lcConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, next(nestedLcIterator), conclusionResult, lcTestIndex, lcSatisfactionMsg, "(**)")
    
    return lcSatisfactionMsg

def satisfactionReportOfConstraints(dn):
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
        # Skip executable LCs — they are question-specific hypothesis
        # checks, not domain constraints, and are handled separately.
        for _, lc in graph.logicalConstrains.items():

            if not lc.headLC or not lc.active: # Process only active and head lcs
                continue
                
            if type(lc) is fixedL: # Skip fixedL lc
                continue
                
            lcCounter +=  1
            
            lcName = lc.lcName
                
            lcSatisfaction[lcName] = {}
            lcSatisfactionTest = lcSatisfaction[lcName]
            
            mySolver.constraintConstructor.current_device = dn.current_device
            mySolver.constraintConstructor.myGraph = mySolver.myGraph
            lcResult, lcVariables, inputLc, *extra = \
                mySolver.constraintConstructor.constructLogicalConstrains(lc, mySolver.myLcLossSampleBooleanMethods, m, dn, p, key = key, headLC = True, loss = True, sample = True)
            lcSatisfactionTest['lcResult'] = lcResult
            lcSatisfactionTest['lcVariables'] = lcVariables
            lcSatisfactionTest['lcs'] = inputLc

    for lcName in lcSatisfaction:
        lcSatisfactionTest = lcSatisfaction[lcName]
        lcSatisfactionMsgs = {}
        lcSatisfactionMsgs["Satisfied"] = []
        lcSatisfactionMsgs["NotSatisfied"] = []
        lenOfLcTests = len(lcSatisfactionTest['lcResult'])
        
        for lcTestIndex in range(lenOfLcTests):
            lcIterator = reversed(lcSatisfactionTest['lcs'])

            currentLc = next(lcIterator)
            if lcSatisfactionTest['lcResult'][lcTestIndex][0] == None:
                continue
            lcResult = not lcSatisfactionTest['lcResult'][lcTestIndex][0].item()
            lcSatisfactionMsg = ''
            if isinstance(currentLc, ifL):
                lcSatisfactionMsg = ifConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, currentLc, lcResult, lcTestIndex, lcSatisfactionMsg, "")
            else:
                lcSatisfactionMsg = lcConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, currentLc, lcResult, lcTestIndex, lcSatisfactionMsg, "")
            
            if lcResult:
                lcSatisfactionMsgs["Satisfied"].append(lcSatisfactionMsg)
            else:
                lcSatisfactionMsgs["NotSatisfied"].append(lcSatisfactionMsg)

        lcSatisfactionTest['lcSatisfactionMsgs'] = lcSatisfactionMsgs

    return lcSatisfaction


def _describe_graph_structure(graph):
    """Build a human-readable summary of the graph's concept/relation structure.

    This is included in error messages so the LLM can see what the graph
    actually contains and compare it with what the constraint expects.
    """
    lines = []

    # Collect from main graph + subgraphs
    # Guard against property descriptors (e.g. if Graph class was passed instead of instance)
    try:
        raw_concepts = graph.concepts
        if isinstance(raw_concepts, property):
            return "(graph.concepts is a property descriptor — not a Graph instance)"
        all_concepts = dict(raw_concepts)
    except TypeError:
        return "(graph.concepts is not iterable — not a Graph instance)"

    try:
        subgraphs = graph.subgraphs if not isinstance(graph.subgraphs, property) else {}
    except Exception:
        subgraphs = {}

    for _sk, sg in subgraphs.items():
        try:
            all_concepts.update(sg.concepts)
        except Exception:
            pass

    root_concepts = []
    child_concepts = {}   # parent_name -> [child_name, ...]
    enum_concepts = {}     # concept_name -> [value, ...]
    relation_concepts = {} # concept_name -> {attr_name: dst_concept_name}
    is_a_map = {}          # child_name -> parent_name

    for cname, concept in all_concepts.items():
        # is_a
        for rel in concept._out.get('is_a', []):
            is_a_map[cname] = rel.dst.name

        # contains
        children = [rel.dst.name for rel in concept._out.get('contains', [])]
        if children:
            child_concepts[cname] = children

        # has_a (relations)
        has_a_rels = concept._out.get('has_a', [])
        if has_a_rels:
            attrs = {}
            for rel in has_a_rels:
                attrs[rel.name] = rel.dst.name if hasattr(rel, 'dst') and rel.dst else '?'
            relation_concepts[cname] = attrs

        # EnumConcept
        if isinstance(concept, EnumConcept):
            enum_concepts[cname] = list(concept.enum)

        # Root detection
        contained_in = concept._in.get('contains', [])
        has_is_a = concept._out.get('is_a', [])
        if not contained_in and not has_is_a and not has_a_rels:
            root_concepts.append(cname)

    if root_concepts:
        lines.append(f"  Root concepts: {', '.join(root_concepts)}")

    if child_concepts:
        for parent, children in child_concepts.items():
            lines.append(f"  {parent} contains: {', '.join(children)}")

    if is_a_map:
        for child, parent in is_a_map.items():
            lines.append(f"  {child}.is_a({parent})")

    if enum_concepts:
        for cname, values in enum_concepts.items():
            lines.append(f"  EnumConcept {cname}: [{', '.join(values)}]")

    if relation_concepts:
        for cname, attrs in relation_concepts.items():
            attr_str = ', '.join(f"{k}->{v}" for k, v in attrs.items())
            lines.append(f"  Relation {cname}.has_a({attr_str})")

    return "\n".join(lines) if lines else "  (empty graph)"


def _describe_constraint(elc):
    """Build a human-readable description of an executable constraint."""
    try:
        inner = elc.innerLC
        desc = f"{type(inner).__name__}"
        if hasattr(inner, 'strEs'):
            desc += inner.strEs()

        # For queryL, add subclass info
        if hasattr(inner, 'concept') and hasattr(inner, '_subclass_names'):
            desc += f" over concept '{inner.concept.name}'"
            if inner._subclass_names:
                desc += f" with subclasses [{', '.join(inner._subclass_names)}]"

        # List referenced concepts
        try:
            concepts = inner.getLcConcepts()
            if concepts:
                desc += f" | references concepts: {{{', '.join(sorted(concepts))}}}"
        except Exception:
            pass

        return desc
    except Exception:
        return repr(elc)


def verifyExecutableConstraints(graph):
    """Verify executable constraints are structurally valid using dummy data.

    Creates a dummy DataNode with random probabilities, then runs AnswerSolver
    on each executable constraint.  Exceptions indicate structural errors in
    the constraint definition (wrong concepts, broken nesting, etc.).

    A ``None`` return from AnswerSolver means all hypotheses were infeasible,
    which can happen legitimately with random data — this is logged as a
    warning but **not** treated as an error.

    Error messages are designed to be actionable for an LLM that generated
    the graph code: they include the graph structure, constraint descriptions,
    and tracebacks so the LLM can identify what to fix.

    Args:
        graph: A DomiKnowS Graph instance that has ``executableLCs``.

    Returns:
        tuple: ``(all_ok, messages)`` where *all_ok* is ``True`` when no
        structural errors were found and *messages* is a list of error /
        warning strings.
    """
    try:
        from domiknows.solver.answerModule import AnswerSolver
    except ImportError as e:
        # Gurobi or other solver dependency missing — skip verification
        return True, [f"Skipped executable constraint verification (missing dependency: {e})"]

    if not graph.executableLCs:
        return True, []

    graph_desc = _describe_graph_structure(graph)

    # --- Create dummy DataNode ---
    try:
        dn = createDummyDataNode(graph)
    except Exception as e:
        tb = traceback.format_exc()
        return False, [
            f"Failed to create dummy DataNode for graph verification.\n"
            f"  This usually means the graph structure has an issue — e.g. a relation's "
            f"has_a() target concept is missing, a concept hierarchy is broken, or "
            f"an EnumConcept is misconfigured.\n"
            f"  Error: {type(e).__name__}: {e}\n"
            f"  Graph structure:\n{graph_desc}\n"
            f"  Traceback:\n{tb}"
        ]

    if dn is None:
        return True, ["Skipped: could not create dummy DataNode (no root concept found)"]

    try:
        dn.inferLocal()
    except Exception as e:
        tb = traceback.format_exc()
        return False, [
            f"Failed to run inferLocal() on dummy DataNode.\n"
            f"  This usually means concept probabilities could not be computed — "
            f"check that all concepts have valid is_a/contains/has_a structure.\n"
            f"  Error: {type(e).__name__}: {e}\n"
            f"  Graph structure:\n{graph_desc}\n"
            f"  Traceback:\n{tb}"
        ]

    # --- Create AnswerSolver ---
    try:
        answer_solver = AnswerSolver(graph)
    except Exception as e:
        tb = traceback.format_exc()
        return False, [
            f"Failed to create AnswerSolver for constraint verification.\n"
            f"  Error: {type(e).__name__}: {e}\n"
            f"  Graph structure:\n{graph_desc}\n"
            f"  Traceback:\n{tb}"
        ]

    # --- Test each executable constraint ---
    errors = []
    warnings = []

    for elc_name, elc in graph.executableLCs.items():
        constraint_desc = _describe_constraint(elc)

        # Ensure the constraint is active for verification
        prev_active = elc.active
        prev_inner_active = elc.innerLC.active
        elc.active = True
        elc.innerLC.active = True

        try:
            result = answer_solver.answer(f"execute({elc_name})", dn)
            if result is None:
                warnings.append(
                    f"{elc_name}: all hypotheses infeasible on dummy data "
                    f"(may be normal with random probabilities)\n"
                    f"  Constraint: {constraint_desc}"
                )
        except Exception as e:
            tb = traceback.format_exc()
            errors.append(
                f"{elc_name}: constraint verification failed.\n"
                f"  Constraint: {constraint_desc}\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  This means the constraint references concepts or relations "
                f"that don't match the graph structure. Check that all concept "
                f"names in the constraint exist in the graph and have the "
                f"correct is_a/contains/has_a relationships.\n"
                f"  Graph structure:\n{graph_desc}\n"
                f"  Traceback:\n{tb}"
            )
        finally:
            # Restore original active state
            elc.active = prev_active
            elc.innerLC.active = prev_inner_active

    all_ok = len(errors) == 0
    messages = errors + warnings
    return all_ok, messages