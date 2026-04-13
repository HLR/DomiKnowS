from collections import namedtuple
from difflib import get_close_matches

from domiknows.graph import V, CandidateSelection, Concept, LogicalConstrain

def checkLcCorrectness(graph):
    """
    Check that all concepts used in a logical constraint exist in the graph.
    Raises an exception with detailed information if any concept is missing.
    
    Args:
        graph: The graph containing the concepts
        
    """
     # --- Process logical constraints variable syntax (only for unprocessed LCs)
    for lc_name, lc in _iter_all_lcs(graph=graph):
        if not lc.active or not lc.headLC:
            continue
        
        # Skip if this LC has already been processed
        if lc_name in graph._processed_lcs:
            continue

        # collect VarMaps - store info about lc variable syntax if used in this logical constraint
        varMapsList = collectVarMaps(lc, [])
        
        # process variable syntax - translate it to the path syntax
        if varMapsList:
            handleVarsPath(lc, varMapsList[0])
        
        # Mark this LC as processed for VarMaps
        graph._processed_lcs.add(lc_name)
                
    # --- Check if the logical constrains are correct ---
    
    lc_info = {}
    LcInfo = namedtuple('CurrentLcInfo', ['foundVariables', 'usedVariables', 'headLcName'])
    
    allConceptNames = graph.getAllConceptNames()
    
    for lc_name, lc in _iter_all_lcs(graph=graph):
        if not lc.active or not lc.headLC:
            continue
        
        # Validate all concepts - will throw immediately on first missing concept
        # with detailed location information
        validateConceptsInLogicalConstraint(graph, allConceptNames,  lc, lc_name)
    
    # --- Gather information about variables used and defined in the logical constrains and 
    #     report errors if some of them are not defined and used or defined more than once
    for lc_name, lc in _iter_all_lcs(graph=graph):
        if not lc.active or not lc.headLC:
            continue
            
        # find variable defined in the logical constrain - report error if some of them are defined more than once
        found_variables = find_lc_variable(lc, headLc=lc.name)

        # find all variables used in the logical constrain - report error if some of them are not defined
        # gather paths defined in the logical constrain per variable
        used_variables = check_if_all_used_variables_are_defined(lc, found_variables, headLc=lc.name, graph=graph)
        
        # save information about variables used and defined in the logical constrain
        current_lc_info = LcInfo(found_variables, used_variables, lc.name)
        lc_info[lc_name] = current_lc_info
    
    # --- Check if the paths defined in the logical constrains are correct
    for lc_name, lc in _iter_all_lcs(graph=graph):
        if not lc.active or not lc.headLC:
            continue
        
        # current logical constrain info and variables found and used in the current logical constrain
        current_lc_info = lc_info[lc_name]
        usedVariables = current_lc_info.usedVariables
        foundVariables = current_lc_info.foundVariables
        headLcName = current_lc_info.headLcName
        
        # loop over all variables used in the logical constrain
        for variableName, pathInfos in usedVariables.items():
            # get information about the variable in the found variables record
            variableConcept = foundVariables[variableName][2][0]
            
            # get the root parent of the variable concept
            variableConceptParent = graph.findRootConceptOrRelation(variableConcept)
            
            # loop over all paths defined using the variable as starting point
            for pathInfo in pathInfos:
                path = pathInfo[3]
                resultConcept = pathInfo[2]
                
                if isinstance(path[0], tuple): # this path is a combination of paths 
                    for subpath in path: 
                        if len(subpath) < 1:  
                            continue  # skip this subpath it is empty
                            
                        check_path(graph, subpath, resultConcept, variableConceptParent, headLcName, foundVariables, variableName)
                        
                else: # this path is a single path
                    if len(path) < 1:
                        continue # skip this path it is empty
                        
                    check_path(graph, path, resultConcept, variableConceptParent, headLcName, foundVariables, variableName)
        
    # --- Validate queryL constraints have proper multiclass concepts ---
    for lc_name, lc in _iter_all_lcs(graph=graph):
        if not lc.active or not lc.headLC:
            continue
        
        validate_queryL_constraints(graph, lc, headLc=lc.name)
        
    # --- Validate counting constraints have proper counting syntax and cardinality ---
    for lc_name, lc in _iter_all_lcs(graph=graph):
        if not lc.active or not lc.headLC:
            continue
        _validate_counting_constraints(lc, lc_name)
        
        
    # --- Validate that all relations used in the constraints exist in the graph and are corretly used ---
    for lc_name, lc in _iter_all_lcs(graph=graph):
        if not lc.active or not lc.headLC:
            continue
        _validate_relations_in_constraints(graph, allConceptNames, lc, lc_name)
           
def getConceptsFromLogicalConstraint(lc, concept_names=None):
        """
        Extract all concept names used in a logical constraint.
        
        This method recursively traverses the logical constraint structure
        to find all concepts (including those in nested constraints).
        
        Args:
            lc: Logical constraint to extract concepts from
            concept_names: Set to accumulate concept names (used for recursion)
            
        Returns:
            List of unique concept names used in the constraint
        """
        # Track if this is the top-level call
        is_top_level = (concept_names is None)
        
        if concept_names is None:
            concept_names = set()
        
        if lc is None:
            return list(concept_names) if is_top_level else concept_names
        
        # Process all elements in the constraint
        if hasattr(lc, 'e') and lc.e:
            for e in lc.e:
                # Skip variables
                if isinstance(e, V):
                    continue
                
                # Handle Concept or tuple (concept, label, index)
                if isinstance(e, Concept):
                    concept_names.add(e.name)
                elif isinstance(e, tuple) and len(e) > 0:
                    if isinstance(e[0], Concept):
                        concept_names.add(e[0].name)
                    elif isinstance(e[0], CandidateSelection):
                        # CandidateSelection might reference concepts
                        getConceptsFromLogicalConstraint(e[0], concept_names)
                
                # Handle nested logical constraints
                if isinstance(e, LogicalConstrain):
                    getConceptsFromLogicalConstraint(e, concept_names)
                elif isinstance(e, CandidateSelection):
                    getConceptsFromLogicalConstraint(e, concept_names)
        
        # Also check innerLC attribute (common in constraint wrappers)
        if hasattr(lc, 'innerLC') and lc.innerLC is not None:
            getConceptsFromLogicalConstraint(lc.innerLC, concept_names)
        
        # Only convert to list at top level
        return list(concept_names) if is_top_level else concept_names
    
def validateConceptsInLogicalConstraint(graph, allConceptNames, lc, lc_name, path="root"):
    """
    Recursively validate that all concepts used in a logical constraint exist in the graph.
    Throws exception immediately upon finding the first missing concept with detailed location info.
    
    Args:
        graph: The graph containing the concepts
        lc: Logical constraint to validate
        lc_name: Name of the logical constraint (for error reporting)
        path: Current path in the constraint tree (for error location tracking)
        
    Raises:
        ValueError: When a concept is not found in the graph, with detailed location info
    """
    if lc is None:
        return
    
    # Get constraint type for better error messages
    constraint_type = type(lc).__name__
    
    # Process all elements in the constraint
    if hasattr(lc, 'e') and lc.e:
        for idx, e in enumerate(lc.e):
            current_path = f"{path}.e[{idx}]"
            
            # Skip variables
            if isinstance(e, V):
                continue
            
            # Handle Concept
            if isinstance(e, Concept):
                if e.name not in allConceptNames:
                    _raise_missing_concept_error(
                        graph=graph,
                        lc_name=lc_name,
                        concept_name=e.name,
                        constraint_type=constraint_type,
                        location=current_path,
                        element_type="Concept"
                    )
            
            # Handle tuple (concept, label, index)
            elif isinstance(e, tuple) and len(e) > 0:
                if isinstance(e[0], Concept):
                    if e[0].name not in allConceptNames:
                        _raise_missing_concept_error(
                            graph=graph,
                            lc_name=lc_name,
                            concept_name=e[0].name,
                            constraint_type=constraint_type,
                            location=f"{current_path}[0]",
                            element_type="Concept (in tuple)"
                        )
                            
                elif isinstance(e[0], CandidateSelection):
                    # Recursively validate CandidateSelection
                    validateConceptsInLogicalConstraint(
                        allConceptNames=allConceptNames,
                        graph=graph,
                        lc=e[0], 
                        lc_name=lc_name, 
                        path=f"{current_path}[0]"
                    )
            
            # Handle nested logical constraints
            if isinstance(e, LogicalConstrain):
                validateConceptsInLogicalConstraint(
                    allConceptNames=allConceptNames,
                    graph=graph,
                    lc=e, 
                    lc_name=lc_name, 
                    path=current_path
                )
            elif isinstance(e, CandidateSelection):
                validateConceptsInLogicalConstraint(
                    allConceptNames=allConceptNames,
                    graph=graph,
                    lc=e, 
                    lc_name=lc_name, 
                    path=current_path
                )
    
    # Check innerLC attribute (common in constraint wrappers)
    if hasattr(lc, 'innerLC') and lc.innerLC is not None:
        validateConceptsInLogicalConstraint(
            allConceptNames=allConceptNames,
            graph=graph,
            lc=lc.innerLC, 
            lc_name=lc_name, 
            path=f"{path}.innerLC"
        )
    
    # Check common logical operators
    for op_name in ['andL', 'orL', 'nandL', 'norL', 'notL']:
        if hasattr(lc, op_name):
            op_list = getattr(lc, op_name)
            if op_list:
                for idx, sub_lc in enumerate(op_list):
                    validateConceptsInLogicalConstraint(
                        allConceptNames=allConceptNames,
                        graph=graph,
                        lc=sub_lc, 
                        lc_name=lc_name, 
                        path=f"{path}.{op_name}[{idx}]"
                    )

def _raise_missing_concept_error(graph, lc_name, concept_name, constraint_type, 
                                  location, element_type):
    """
    Raise a detailed error about a missing concept.
    
    Args:
        graph: The graph containing the concepts
        lc_name: Name of the logical constraint
        concept_name: Name of the missing concept
        constraint_type: Type of the constraint where concept was found
        location: Path to the location in the constraint tree
        element_type: Type of element containing the concept
    """
    # Get list of available concepts for helpful error message
    available_concepts = list(graph.concepts.keys())
    
    # Find similar concept names (simple string matching)
    suggestions = [c for c in available_concepts if concept_name.lower() in c.lower() 
                   or c.lower() in concept_name.lower()]
    
    error_msg = (
        f"Missing concept in logical constraint '{lc_name}':\n"
        f"  Concept: '{concept_name}' (not found in graph)\n"
        f"  Location: {location}\n"
        f"  Constraint type: {constraint_type}\n"
        f"  Element type: {element_type}\n"
    )
    
    if suggestions:
        error_msg += f"  Did you mean: {', '.join(suggestions[:5])}?\n"
    
    error_msg += f"\nAvailable concepts ({len(available_concepts)}): {', '.join(available_concepts[:10])}"
    if len(available_concepts) > 10:
        error_msg += f", ... and {len(available_concepts) - 10} more"
    
    raise ValueError(error_msg)


def _raise_undefined_variable(variable_name, lc_context, lc_path, found_variables):
    """Raise a clear error when a variable used in a path was never defined in the constraint.

    Includes 'did you mean' suggestions based on edit distance against already-defined
    variable names so that common typos (e.g. 'e1' vs 'el1') are caught early.
    """
    defined_vars = sorted(v for v in found_variables if isinstance(v, str))
    close = get_close_matches(variable_name, defined_vars, n=3, cutoff=0.5)

    msg = f"Variable '{variable_name}' used in {lc_context} is not defined."

    if lc_path is not None:
        msg += f"\n  Used in path: {lc_path}"

    msg += (
        f"\n  You must first introduce '{variable_name}' as a direct argument "
        f"(without a path) before referencing it inside a path."
    )

    if close:
        msg += f"\n  Did you mean: {', '.join(repr(c) for c in close)}?"
    elif defined_vars:
        msg += f"\n  Variables defined in this constraint: {', '.join(defined_vars)}"

    raise ValueError(msg)


def _iter_all_lcs(graph):
        """Helper to iterate over all logical constraints including executable ones.
        
        graph: The graph containing logical constraints and executable logical constraints.
        Yields:
            tuple: (lc_name, lc) pairs for both regular and executable logical constraints.
        """
        yield from graph.logicalConstrains.items()
        for key, elc in graph.executableLCs.items():
            yield (key, elc.innerLC)

def _format_lc_context(headLc, lc):
    """Format LC context text and include nested LC only when different from head LC."""
    if headLc is None:
        return str(lc)

    if getattr(lc, 'name', None) == headLc:
        return str(headLc)

    return f"{headLc} {lc}"
            
def find_lc_variable(lc, found_variables=None, headLc=None):
    '''Finds all variables defined in a logical constraint and reports errors for duplicates.

    This method traverses through the elements of a logical constraint to find all the variables 
    that have been defined. It checks for incorrect cardinality definitions, multiple definitions of the 
    same variable, and variables that are not associated with any concept among other things.
    
    Args:
    lc (LogicalConstrain): The logical constraint to be processed.
    found_variables (dict, optional): Dictionary to store found variables. The key is the variable name, 
                                        and the value is a tuple containing the logical constraint, variable name, 
                                        and the concept associated with the variable.
                                        Defaults to None.
    headLc (str, optional): The name of the parent logical constraint. Defaults to None.
    
    Returns:
    dict: A dictionary containing all found variables.
    
    Raises:
    Exception: If there are issues with the variable definitions or cardinality.
    '''
    if lc.cardinalityException:
        if lc.name != headLc:
            exceptionStr1 = f"{lc.typeName} {headLc} has incorrect cardinality definition in nested {lc} logical operator- " 
        else:
            exceptionStr1 = f"{lc.typeName} {headLc} has incorrect cardinality definition - "
        
        exceptionStr2 = f"integer {lc.cardinalityException} has to be last element in the same Logical operator for counting or existing logical operators!"
        raise ValueError(f"{exceptionStr1} {exceptionStr2}")

    if found_variables is None:
        found_variables = {}

    from domiknows.graph import V, LogicalConstrain, LcElement

    lc_context = _format_lc_context(headLc, lc)

    e_before = None
    for e in lc.e:
        # checking if element is a variable
        if isinstance(e, V) and e and e.name:
            variable_name = e.name
            if e_before:
                if variable_name not in found_variables:
                    variable_info = (lc, variable_name, e_before)
                    found_variables[variable_name] = variable_info
            else:
                exceptionStr = f"In logical constraint {lc_context} variable {variable_name} is not associated with any concept"
                raise ValueError(exceptionStr)
            
        # checking for extra variable:
        elif e and isinstance(e, tuple) and e[0] == 'extraV':
            predicate = lc.e[0][1]
            exceptionStr1 = f"Logical constraint {lc_context}: Each predicate can only have one new variable definition. For the predicate {predicate}, you have used both {e[1]} and {e[2]} as new variables."
            exceptionStr2 = f"Either wrap both under on variable, if you intended to initialize {e[1]} based on another value, then the second argument should be a path=(...)."
            raise ValueError(f"{exceptionStr1} {exceptionStr2}")
        # checking if element is a tuple 
        elif isinstance(e, tuple) and e and isinstance(e[0], LcElement) and not isinstance(e[0], LogicalConstrain):
            find_lc_variable(e[0], found_variables=found_variables, headLc=headLc)
            current_lc_element = e[0]
            current_lc_element_concepts = [c for c in current_lc_element.e if isinstance(c, tuple) and not isinstance(c, V)]

            if len(current_lc_element_concepts) != len(e[1]):
                raise ValueError(f"Logical constraint {lc_context} has incorrect definition of combination {e} - number of variables does not match number of concepts in combination")

            if len(e) >= 2 and isinstance(e[1], tuple):
                for v in e[1]:
                    if not isinstance(v, str):
                        raise ValueError(f"Logical constraint {lc_context} has incorrect definition of combination {e} - all variables should be strings")

                for index, v in enumerate(e[1]):
                    variable_name = v
                    variable_info = (lc, variable_name, current_lc_element_concepts[index])
                    found_variables[variable_name] = variable_info

        # Checking if element is a LogicalConstrain
        elif isinstance(e, LogicalConstrain):
            find_lc_variable(e, found_variables=found_variables, headLc=headLc)

        e_before = e

    return found_variables

def check_if_all_used_variables_are_defined(lc, found_variables, used_variables=None, headLc=None, graph=None):
    '''Checks if all variables used in a logical constraint are properly defined.

    This method traverses through the elements of a logical constraint to identify all the variables 
    that are used but not defined. It also handles variable names in different types of paths.
    
    Args:
    lc (LogicalConstrain): The logical constraint to be processed.
    found_variables (dict): Dictionary containing all variables that have been defined.
                            The key is the variable name and the value is a tuple containing information
                            about the variable.
    used_variables (dict, optional): Dictionary to store variables that are used. The key is the variable name,
                                        and the value is a list of tuples, each containing the logical constraint,
                                        variable name, the type of the element that uses it, and the path to the variable.
                                        Defaults to None.
    headLc (str, optional): The name of the parent logical constraint. Defaults to None.
    
    Returns:
    dict: A dictionary containing all used variables.
    
    Raises:
    Exception: If there are variables that are used but not defined, or if there are errors in the path definitions.
    '''
    from .logicalConstrain import eqL
    
    if used_variables is None:
        used_variables = {}

    from domiknows.graph import V, LogicalConstrain

    lc_context = _format_lc_context(headLc, lc)

    def _is_implicit_relation_var(variable_name):
        """Return relation concept name for implicit vars like relationName_1, else None."""
        if graph is None or not isinstance(variable_name, str):
            return None

        parts = variable_name.rsplit('_', 1)
        if len(parts) != 2 or not parts[1].isdigit():
            return None

        relation_name = parts[0]
        relation_concept = graph.findConcept(relation_name)
        if relation_concept is None:
            return None

        return relation_name

    def _find_variable_definition_in_lc(current_lc, variable_name):
        """Best-effort scan for a variable definition inside the current LC tree.

        This helps native binary syntax where path checks may encounter a variable
        before the standard definition map was populated for a nested structure.
        """
        from domiknows.graph import V, LogicalConstrain

        if not hasattr(current_lc, 'e') or not current_lc.e:
            return None

        prev = None
        for item in current_lc.e:
            if isinstance(item, V) and item.name == variable_name and prev is not None:
                return (current_lc, variable_name, prev)
            if isinstance(item, LogicalConstrain):
                nested = _find_variable_definition_in_lc(item, variable_name)
                if nested is not None:
                    return nested
            prev = item

        return None

    def handle_variable_name(lc_variable_name, lcPath):
        lcElementType = lc.e[i-1]

        if lc_variable_name not in found_variables:
            implicit_relation_name = _is_implicit_relation_var(lc_variable_name)

            if implicit_relation_name:
                relation_concept = graph.findConcept(implicit_relation_name)
                found_variables[lc_variable_name] = (lc, lc_variable_name, (relation_concept, relation_concept.name, None, 1))
            else:
                inferred_definition = _find_variable_definition_in_lc(lc, lc_variable_name)
                if inferred_definition is not None:
                    found_variables[lc_variable_name] = inferred_definition
                else:
                    # Native binary syntax can introduce path-root variables in
                    # nested fragments before explicit variable-defining atoms.
                    # Fall back to the canonical object concept when available.
                    fallback_object = None
                    if graph is not None:
                        fallback_object = graph.findConcept("obj") or graph.findConcept("object")

                    if fallback_object is not None:
                        found_variables[lc_variable_name] = (
                            lc,
                            lc_variable_name,
                            (fallback_object, fallback_object.name, None, 1),
                        )
                    else:
                        _raise_undefined_variable(lc_variable_name, lc_context, lcPath, found_variables)

        if lc_variable_name not in used_variables:
            used_variables[lc_variable_name] = []

        variable_info = (lc, lc_variable_name, lcElementType, lcPath)
        used_variables[lc_variable_name].append(variable_info)

    for i, e in enumerate(lc.e):
        if isinstance(e, V) and e.v: # has path
            if isinstance(e.v, eqL):
                continue
            elif isinstance(e.v, str):
                handle_variable_name(e.v, e.v)
            elif isinstance(e.v, tuple):
                if isinstance(e.v[0], str): # single path
                    handle_variable_name(e.v[0], e.v)
                elif isinstance(e.v[0], tuple): # path union
                    for t in e.v:
                        if isinstance(t[0], str):
                            handle_variable_name(t[0], t)
                        else:
                            raise ValueError(f"Path {t} found in {lc_context} is not correct")
                else:
                    raise ValueError(f"Path {e} found in {lc_context} is not correct")
            else:
                raise ValueError(f"Path {e} found in {lc_context} is not correct")
        elif isinstance(e, LogicalConstrain):
            check_if_all_used_variables_are_defined(e, found_variables, used_variables=used_variables, headLc=headLc, graph=graph)

    return used_variables

def getPathStr(path):
    '''Converts a path of concepts and relations to a string representation.

    This method iterates over a given path, which can include instances of the Relation and Concept classes,
    and constructs a string representation of the path.
    
    Args:
    path (list): A list of path elements which can be instances of Relation or Concept classes.
                    The first element in the list is not processed, and the list should be non-empty.
    
    Returns:
    str: A string representation of the path, excluding the first element.
    '''
    from .concept import Concept
    from .relation import Relation
    pathStr = ""
    for pathElement in path[1:]:
        if isinstance(pathElement, (Relation,)):
            if pathElement.var_name:
                pathStr += pathElement.var_name + " "
            else:
                pathStr += pathElement.name + " "
        elif isinstance(pathElement, (Concept,)):
            pathStr += pathElement.var_name + " "
        else:
            pathStr += str(pathElement)
            
    return pathStr.strip()
            
def check_path(graph, path, resultConcept, variableConceptParent, lc_name, foundVariables, variableName):
    '''Checks the validity of a path in terms of relations and concepts.

    This function checks the validity of a given path, including ensuring that each relation
    or concept in the path has the correct type. It raises exceptions with informative error messages 
    if the path is not valid.

    Args:
    path (list): The path to check, starting from the source concept.
    resultConcept (tuple): The expected end concept of the path.
    variableConceptParent (Concept): The parent concept for the source of the path.
    lc_name (str): The name of the logical constraint where the path is defined.
    foundVariables (dict): Dictionary of found variables in the scope.
    variableName (str): The name of the variable being checked.

    Raises:
    Exception: Various types of exceptions are raised for different kinds of path invalidity.
    '''
    from .relation import IsA, HasA, Relation
    from .logicalConstrain import eqL
    from .concept import Concept

    def _is_ancestor_of(ancestor_name, descendant_concept):
        """Check if ancestor_name is an ancestor of descendant_concept via is_a chain.

        In an andL, a variable like 'x' can satisfy multiple predicates
        simultaneously â€” e.g. brown('x') says x has color=brown, while
        right_of('z', 'x') says x is an object.  The relation endpoint
        (object) is an ancestor of the variable's declared concept (brown â†’
        color â†’ â€¦ â†’ object) through the containment hierarchy, so the path
        is valid even though the types don't match directly.
        """
        current = descendant_concept
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if current.name == ancestor_name:
                return True
            # Walk up is_a
            parents = current._out.get('is_a', [])
            if parents:
                current = parents[0].dst
            else:
                # Walk up contains (child â†’ parent)
                containers = current._in.get('contains', [])
                if containers:
                    current = containers[0].src
                else:
                    break
        return False

    requiredLeftConcept = variableConceptParent.name # path element has to be relation with this type to the left
    requiredEndOfPathConcept = resultConcept[1] # path has to result in this concept
    requiredEndOfPathConceptRoot = graph.findRootConceptOrRelation(resultConcept[0]).name
    expectedRightConcept =  None
    lastPathElement = False
    pathStr = getPathStr(path)
    pathVariable = path[0]
    pathPart = path[0]
        
    if len(path) == 1:
        if requiredLeftConcept == requiredEndOfPathConceptRoot:
            return
        else:
            exceptionStr1 = f"The variable {pathVariable}, defined in the path for {lc_name} is not valid. The concept of {pathVariable} is a of type {requiredLeftConcept},"
            exceptionStr2 = f"but the required concept by the logical constraint element is {requiredEndOfPathConceptRoot}."
            exceptionStr3 = f"The variable used inside the path should match its type with {requiredEndOfPathConceptRoot}."
            raise ValueError(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
        
    for pathIndex, pathElement in enumerate(path[1:], start=1):   
        if isinstance(pathElement, (eqL,)):
            continue
        if isinstance(pathElement, (str,)): # It is a string check if we have corresponding relation in the graph
            if pathElement in graph.varNameReversedMap:
                pathElement = graph.varNameReversedMap[pathElement]
            else:
                exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid."
                exceptionStr2 = f"The required source type after {pathPart} is a {requiredLeftConcept},"
                exceptionStr3 = f"but the used variable {pathElement} is a string which is not a valid name of a graph relationship."
                raise ValueError(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
            
        if pathIndex < len(path) - 1:
            expectedRightConcept = pathElement.dst
            expectedRightConceptRoot = graph.findRootConceptOrRelation(expectedRightConcept).name
        else:
            expectedRightConcept = requiredEndOfPathConcept
            lastPathElement = True
            expectedRightConceptRoot = requiredEndOfPathConceptRoot                
        
        if isinstance(pathElement, (HasA, IsA, Relation)):
            pathElementSrc = pathElement.src.name
            pathElementDst = pathElement.dst.name
            pathElementVarName = pathElement.var_name if pathElement.var_name else ""

            # In an andL, a variable may satisfy multiple predicates.
            # E.g. brown('x'), right_of('z', 'x') â€” x is both a "brown"
            # (color attribute) and an "object" (relation endpoint).
            # The relation src/dst may be an ancestor of the variable's
            # declared concept, which is valid.
            srcCompatible = (
                requiredLeftConcept == pathElementSrc
                or _is_ancestor_of(pathElementSrc, variableConceptParent)
            )
            dstCompatible = (
                expectedRightConceptRoot == pathElementDst
                or _is_ancestor_of(pathElementDst, resultConcept[0])
            )

            # Check if there is a problem with reversed usage of the current path element - it has to be possible to reverse the order to fix it
            if (not srcCompatible and requiredLeftConcept == pathElementDst
                    and (expectedRightConceptRoot == pathElementSrc or _is_ancestor_of(pathElementSrc, resultConcept[0]))):
                exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid"
                exceptionStr2 = f"The relation {pathElementVarName} is from a {pathElementSrc} to a {pathElementDst}, but you have used it from a {pathElementDst} to a {pathElementSrc}."
                if not pathElement.is_reversed:
                    exceptionStr3 = f"You can change '{pathElement.var_name}' to '{pathElement.var_name}.reversed' to go from {pathElementDst} to the {pathElementSrc}, which is what is required here."
                else:
                    exceptionStr3 = f"You can change  '{pathElement.var_name}.reversed' to '{pathElement.var_name}' to go from {pathElementSrc} to the {pathElementDst}, which is what is required here."
                    f"You can use without the .reversed property to change the direction."
                raise ValueError(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
            # Check if the current path element is correctly connected to the left (source) - has matching type
            elif not srcCompatible:
                exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid."
                exceptionStr2 = f"The required source type after {pathPart} is a {requiredLeftConcept},"
                exceptionStr3 = f"but the used variable {pathElementVarName} is a relationship defined between a {pathElementSrc} and a {pathElementDst}, which is not correctly used here."
                raise ValueError(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
            # Check if the current path element is correctly connected to the right (destination) - has matching type
            elif not dstCompatible:
                exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid."
                if lastPathElement: # if this is the last element it has to match the concept in which this path is embedded
                    exceptionStr2 = f"The required destination type after {pathPart} is a {expectedRightConcept}."
                else: # if this it intermediary path element that if is expected that it will match next path element source type
                    exceptionStr2 = f"The expected destination type after {pathPart} is a {expectedRightConcept}."
                exceptionStr3 = f"The used variable {pathElementVarName} is a relationship defined between a {pathElementSrc} and a {pathElementDst}, which is not correctly used here."
                raise ValueError(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
            
            # Move along the path with the requiredLeftConcept and pathVariable
            requiredLeftConcept = pathElementDst
            pathPart += " " + pathElementVarName
        else:
            if isinstance(pathElement, (Concept,)):
                exceptionStr1 = f"You have used the notion {expectedRightConcept}(path=('{pathVariable}', {pathStr})) which is incorrect."
                exceptionStr2 = f"{pathElement} is a concept and cannot be used as part of the path."
                exceptionStr3 = f"- If you meant that '{pathVariable}' should be of type {expectedRightConcept}: {expectedRightConcept}(path=('{pathVariable}'))"
                exceptionStr4 = f"- If you meant another entity 'y' should be of type {expectedRightConcept} which is somehow related to '{pathVariable}': {expectedRightConcept}(path=('x', edge1, edge2, ...))"
                exceptionStr5 = f"where edge1, edge2, ... are relations that connect '{pathVariable}' to 'y'."
                raise ValueError(f"{exceptionStr1} {exceptionStr2} {exceptionStr3} {exceptionStr4} {exceptionStr5}")
            else: # all other types not allowed in path
                pathElementType = type(pathElement)
                exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, after {pathPart} is not valid."
                exceptionStr2 = f"The used variable {pathElement} is a {pathElementType}, path element can be only relation or eqL logical constraint used to filter candidates in the path."
                raise ValueError(f"{exceptionStr1} {exceptionStr2}")

def are_keys_new(given_dict, dict_list):
    """
    Check if all keys in 'given_dict' are not present in any dictionary within 'dict_list'.

    This method iterates over each key in 'given_dict' and checks if it exists in any of the dictionaries
    contained within 'dict_list'. If a key from 'given_dict' is found in any dictionary in 'dict_list',
    the method returns False, indicating that not all keys are new. Otherwise, it returns True,
    indicating all keys in 'given_dict' are new (i.e., not present in any dictionary in 'dict_list').

    Parameters:
    given_dict (dict): A dictionary whose keys are to be checked.
    dict_list (list of dict): A list of dictionaries against which the keys of 'given_dict' are to be checked.

    Returns:
    bool: True if all keys in 'given_dict' are new, False otherwise.
    """
    for key in given_dict:
        for d in dict_list:
            if key in d:
                return False
    return True

def collectVarMaps(lc, varMapsList):
    """
    Collects variable mappings (VarMaps) from a logical constraint (lc) and updates the list of collected VarMaps.

    This method recursively traverses the elements of the logical constraint 'lc' to identify and process VarMaps.
    It differentiates between the definition of new variables and the usage of existing ones. For new variables, 
    it clones the current VarMap, adds the name of the logical constraint, and appends it to 'varMapsList'. For 
    existing variables, it updates the path variable in the current VarMap to match the one used in their 
    definition. The method modifies 'lc' by removing VarMaps that define new variables.

    Parameters:
    lc (LogicalConstrain): The logical constraint from which VarMaps are to be collected.
    varMapsList (list): A list that accumulates VarMaps. This list collects only the definitions of variables.

    Returns:
    list: The updated list of variable mappings (VarMaps) after processing 'lc'.
    
    Note:
    - The method assumes the existence of specific types and structures within 'lc', such as 'VarMaps' tuples.
    - The method is recursive and alters the structure of 'lc' by removing defining VarMaps.
    """
    from domiknows.graph import LogicalConstrain
    import copy
    
    newE = []
    # collect VarMaps from lc
    for e in lc.e:
        if isinstance(e, LogicalConstrain):
            collectVarMaps(e, varMapsList) # recursive
            newE.append(e)
        # check if VarMap
        elif isinstance(e, tuple) and e[0] == 'VarMaps':
            currentVarMap = e[1]
            
            # check if variables in the current VarMap are new, have not been found already
            # If they are new it means it it their definition in the lc
            if are_keys_new(currentVarMap, varMapsList):
                cloned_CurrentVarMaps = copy.deepcopy(currentVarMap)
                # add info about the lc to varMap  - this is the lc in which this variables are defined
                cloned_CurrentVarMaps["lcName"] = lc.name 
                
                # Add this VarMaps to the collected VarMapss in the varMapsList
                # This list collect only definition of variables 
                # The current varMap will be removed from the current lc
                varMapsList.append(cloned_CurrentVarMaps)
            else:
                # variables in VarMaps are already found 
                # It means that this is the usage of these variables in the lc
                for variableName in currentVarMap:
                    # Find previous definition in varMapsList of the variable in the current VarMap
                    definedVarible = next((d.get(variableName, None) for d in varMapsList if d.get(variableName, None) is not None), None)

                    if definedVarible is not None:
                        variable = currentVarMap[variableName]
                    
                        # Update the path variable in the current varMap to the one used in variable definition
                        if isinstance(variable, tuple) and len(variable) > 1 and isinstance(variable[1], tuple):
                            new_inner_tuple = (definedVarible[1][0],) + variable[1][1:]
                            variable = (variable[0], new_inner_tuple) + variable[2:]
                            currentVarMap[variableName] = variable

                # keep this varMap in the lc - it will be used when processing the lc variable syntax
                newE.append(e)
        else:
            # it is not varMap - keep it in lc
            newE.append(e)
            
    lc.e = newE # Update logical constraint element - defining VarMaps are removed
    return varMapsList
                
def handleVarsPath(lc, varMaps):
    """
    Processes and updates the variable paths in a logical constraint (lc) based on the mappings provided in varMaps.

    This method iterates through the elements of 'lc' and performs various transformations based on the type
    of each element and the presence of variable mappings in 'varMaps'. The method handles nested logical 
    constraints recursively, updates variables already in V form, and modifies variable paths using mappings 
    from 'varMaps'. Additionally, it removes all 'VarMaps' elements from 'lc'.

    Parameters:
    lc (LogicalConstrain): The logical constraint to be processed.
    varMaps (dict): A dictionary containing mappings of variable names to their respective V instances or paths.

    Note:
    - The method assumes a specific structure of 'lc' and 'varMaps', with 'lc' containing elements like 
        LogicalConstrain, V, Concept, and tuples with 'VarMaps'.
    - It employs a flag 'needsVariableUpdate' to track if the next element requires variable path updates.
    - The method is recursive for nested logical constraints and alters the structure of 'lc'.
    """

    from domiknows.graph import V, LogicalConstrain, Concept
    
    # process logical constraint variable syntax is used - translate it to the path V syntax of logical constraints 
    newE = []
    needsVariableUpdate = False # flag set by lc element concept if VarMap is present in the current lc
    for i, e in enumerate(lc.e):
        if isinstance(e, LogicalConstrain):
            handleVarsPath(e, varMaps) # recursive
        elif isinstance(e, V) and e.name in varMaps: # check if the current lc element is already in the V  form and its name is in varMaps
            # this is the variable syntax for a single variable usage
            # replace the V with the mapping for this found in varMap
            e = varMaps[e.name]
        elif needsVariableUpdate: # ths flag was set by the previous lc element
            # The next lc element should be ('VarMaps', {...}) created by
            # Concept.processLCArgs for shorthand binary syntax relation('a','b').
            if i + 1 < len(lc.e) and isinstance(lc.e[i+1], tuple) and lc.e[i+1][0] == 'VarMaps':
                usedVarMap = lc.e[i+1][1]
                usedVarValues = [v for v in usedVarMap.values() if isinstance(v, V) and v.v is not None]

                if len(usedVarValues) >= 2:
                    firstUsedV = usedVarValues[0].v
                    secondUsedV = usedVarValues[1].v

                    if (
                        isinstance(firstUsedV, tuple) and len(firstUsedV) >= 2 and hasattr(firstUsedV[1], 'reversed') and
                        isinstance(secondUsedV, tuple) and len(secondUsedV) >= 2 and hasattr(secondUsedV[1], 'reversed')
                    ):
                        # Build relation-path pair expected by downstream processor:
                        # ((relVar, arg1.reversed), (relVar, arg2.reversed))
                        firstNewPath = (firstUsedV[0], firstUsedV[1].reversed)
                        secondNewPath = (secondUsedV[0], secondUsedV[1].reversed)
                        path = (firstNewPath, secondNewPath)
                        e = V(name="", v=path)

            needsVariableUpdate = False
        elif isinstance(e, tuple) and isinstance(e[0], Concept) and len(lc.e) > i+2:
            if isinstance(lc.e[i+2], tuple) and lc.e[i+2][0] == 'VarMaps':
                # set the flag if the next element needs to be updated because the variable syntax is used
                needsVariableUpdate = True
        
        if isinstance(e, tuple) and e[0] == 'VarMaps':
            pass # removed all VarMaps from lc
        else:
            newE.append(e)
                
    lc.e = newE #Update logical constraint element - all VarMaps are removed

def validate_queryL_constraints(graph, lc, headLc=None):
    """
    Validates queryL constraints to ensure they have a proper multiclass concept.
    
    A queryL constraint requires its first argument (the concept) to be either:
    1. An EnumConcept with explicit values
    2. A Concept that has subclasses defined via is_a() relationships
    
    This method recursively checks all queryL constraints within a logical constraint,
    including nested ones.
    
    Args:
        lc (LogicalConstrain): The logical constraint to validate.
        headLc (str, optional): The name of the parent logical constraint for error messages.
                            Defaults to None.
    
    Raises:
        Exception: If a queryL constraint has a concept without subclasses.
    """
    from .logicalConstrain import queryL, LogicalConstrain
    from .concept import EnumConcept, Concept
    
    if headLc is None:
        headLc = lc.name
    
    # Check if this constraint itself is a queryL
    if isinstance(lc, queryL):
        concept = lc.concept
        concept_name = concept.name if hasattr(concept, 'name') else str(concept)
        
        # Check if it's an EnumConcept
        if isinstance(concept, EnumConcept):
            if not hasattr(concept, 'enum') or not concept.enum:
                exceptionStr1 = f"queryL constraint in {headLc} has invalid EnumConcept '{concept_name}'."
                exceptionStr2 = f"EnumConcept must have non-empty 'enum' values defined."
                raise ValueError(f"{exceptionStr1} {exceptionStr2}")
            # Valid EnumConcept
            return
        
        # Check if it's a regular Concept with is_a subclasses
        if isinstance(concept, Concept):
            # Check for incoming is_a relations (subclasses pointing to this concept)
            has_subclasses = False
            
            if hasattr(concept, '_in') and 'is_a' in concept._in:
                is_a_relations = concept._in.get('is_a', [])
                if is_a_relations and len(is_a_relations) > 0:
                    has_subclasses = True
            
            if not has_subclasses:
                exceptionStr1 = f"queryL constraint in {headLc} has concept '{concept_name}' without subclasses."
                exceptionStr2 = f"The concept used in queryL must be a multiclass concept with subclasses defined via is_a()."
                exceptionStr3 = f"Example: metal.is_a({concept_name}), rubber.is_a({concept_name})"
                exceptionStr4 = f"Alternatively, use EnumConcept: {concept_name} = EnumConcept('{concept_name}', values=['value1', 'value2'])"
                raise ValueError(f"{exceptionStr1} {exceptionStr2} {exceptionStr3} {exceptionStr4}")
            
            # Valid - concept has subclasses
            return
        
        # Neither EnumConcept nor Concept
        exceptionStr1 = f"queryL constraint in {headLc} has invalid concept type: {type(concept)}."
        exceptionStr2 = f"The first argument to queryL must be a Concept with is_a subclasses or an EnumConcept."
        raise ValueError(f"{exceptionStr1} {exceptionStr2}")
    
    # Recursively check nested logical constraints
    for e in lc.e:
        if isinstance(e, LogicalConstrain):
            validate_queryL_constraints(graph, e, headLc=headLc)
        elif isinstance(e, tuple) and len(e) > 0 and isinstance(e[0], LogicalConstrain):
            validate_queryL_constraints(graph, e[0], headLc=headLc)

def _validate_counting_constraints(lc, lc_name):
        """
        Validate that counting constraints (atLeastL, atMostL, exactL, etc.) have:
        1. At least one element to count
        2. Valid limit values (positive integers where applicable)
        
        Raises:
            Exception: If validation fails, with detailed error message
        """
        from .logicalConstrain import (LogicalConstrain, atLeastL, atMostL, exactL,
                                        atLeastAL, atMostAL, exactAL, existsL, existsAL,
                                        greaterL, greaterEqL, lessL, lessEqL)
        
        counting_types = (atLeastL, atMostL, exactL, atLeastAL, atMostAL, exactAL, 
                            existsL, existsAL, greaterL, greaterEqL, lessL, lessEqL)
        
        if isinstance(lc, counting_types):
            # Check if constraint has elements to count
            if not lc.e or len(lc.e) == 0:
                raise ValueError(
                    f"Counting constraint '{lc_name}' ({type(lc).__name__}) has no elements to count"
                )
            
            # Check if limit is valid (should be positive integer for most counting constraints)
            if hasattr(lc, 'fixedLimit') and lc.fixedLimit is not None:
                limit = lc.fixedLimit
            elif lc.e and isinstance(lc.e[-1], int):
                limit = lc.e[-1]
            else:
                limit = 1  # default
            
            if isinstance(limit, int) and limit < 0:
                raise ValueError(
                    f"Counting constraint '{lc_name}' ({type(lc).__name__}) has negative limit: {limit}"
                )
        
        # Recursively check nested constraints
        if hasattr(lc, 'e'):
            for e in lc.e:
                if isinstance(e, LogicalConstrain):
                    _validate_counting_constraints(e, lc_name)
                elif isinstance(e, tuple) and len(e) > 0 and isinstance(e[0], LogicalConstrain):
                    _validate_counting_constraints(e[0], lc_name)

def _validate_relations_in_constraints(graph, allConceptNames, lc, lc_name):
        """
        Validate that relations referenced in constraints:
        1. Exist in the graph
        2. Have proper has_a structure (at least 2 destinations)
        3. Are not malformed
        
        Raises:
            Exception: If validation fails, with detailed error message
        """
        from .logicalConstrain import LogicalConstrain
        from .relation import Relation
        from .concept import Concept
        
        # Check elements in the constraint
        if hasattr(lc, 'e'):
            for e in lc.e:
                # Check if element is a relation concept
                if isinstance(e, tuple) and len(e) > 0 and isinstance(e[0], Concept):
                    concept = e[0]
                    
                    # If this concept is supposed to be a relation (has has_a relations)
                    if concept.has_a():
                        has_a_relations = concept.has_a()
                        
                        # Validate has_a has at least 2 destinations
                        if len(has_a_relations) < 2:
                            raise ValueError(
                                f"Relation concept '{concept.name}' in '{lc_name}' has only {len(has_a_relations)} destination(s), but has_a requires at least 2"
                            )
                        
                        # Check if relation destinations are in the graph
                        for rel in has_a_relations:
                            dest_name = rel.dst.name if hasattr(rel.dst, 'name') else str(rel.dst)
                            if dest_name not in allConceptNames:
                                raise ValueError(
                                    f"Relation '{concept.name}' in '{lc_name}' references destination concept '{dest_name}' which is not in the graph"
                                )
                
                # Recursively check nested constraints
                if isinstance(e, LogicalConstrain):
                    _validate_relations_in_constraints(graph, allConceptNames, e, lc_name)
                elif isinstance(e, tuple) and len(e) > 0 and isinstance(e[0], LogicalConstrain):
                    _validate_relations_in_constraints(graph, allConceptNames, e[0], lc_name)
