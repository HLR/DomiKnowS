from collections import OrderedDict, namedtuple
from itertools import chain
import inspect
from distutils.dep_util import newer

if __package__ is None or __package__ == '':
    from base import BaseGraphTree
    from property import Property
else:
    from .base import BaseGraphTree
    from .property import Property


@BaseGraphTree.localize_namespace
class Graph(BaseGraphTree):
    """
    Represents a graph structure, extending from BaseGraphTree.
    
    Class Attributes:
        varNameReversedMap (dict): Class-level variable to store a reversed mapping for variable names.
        
    Instance Attributes:
        _concepts (OrderedDict): Stores the concepts in an ordered dictionary.
        _logicalConstrains (OrderedDict): Stores the logical constraints in an ordered dictionary.
        _relations (OrderedDict): Stores the relations in an ordered dictionary.
        _batch (NoneType): Currently not set, reserved for batch operations.
        cacheRootConcepts (dict): Cache for root concepts, initialized as an empty dictionary.
        auto_constraint (Any): Specifies whether to automatically create constraints.
        reuse_model (bool): Flag to indicate whether to reuse an existing model.
        ontology (tuple or Graph.Ontology): The ontology associated with the graph.
    """
    varNameReversedMap = {}  # Class variable
        
    def __init__(self, name=None, ontology=None, iri=None, local=None, auto_constraint=None, reuse_model=False):
        '''Initializes an instance of the graph class.
    
        Args:
        name (str, optional): The name of the object. Defaults to None.
        ontology (Graph.Ontology or str, optional): The ontology associated with the object. If provided as a string, it will be paired with the 'local' parameter. Defaults to None.
        iri (str, optional): The IRI (Internationalized Resource Identifier) of the ontology. Used when 'ontology' is None. Defaults to None.
        local (str, optional): The local namespace. Paired with either 'iri' or 'ontology' if provided as a string. Defaults to None.
        auto_constraint (Any, optional): Specifies whether to automatically create constraints. The type and use are determined by the class. Defaults to None.
        reuse_model (bool, optional): Flag to indicate whether to reuse an existing model. Defaults to False.
    
        Attributes:
        _concepts (OrderedDict): Stores the concepts in an ordered dictionary.
        _logicalConstrains (OrderedDict): Stores the logical constraints in an ordered dictionary.
        _relations (OrderedDict): Stores the relations in an ordered dictionary.
        _batch (NoneType): Currently not set, reserved for batch operations.
        cacheRootConcepts (dict): Cache for root concepts, initialized as an empty dictionary.
        '''
        BaseGraphTree.__init__(self, name)
        if ontology is None:
            self.ontology = (iri, local)
        elif isinstance(ontology, Graph.Ontology):
            self.ontology = ontology
        elif isinstance(ontology, str):
            self.ontology = (ontology, local)
        self.auto_constraint = auto_constraint
        self.reuse_model = reuse_model
        self._concepts = OrderedDict()
        self._logicalConstrains = OrderedDict()
        self._relations = OrderedDict()
        self._batch = None
        self.cacheRootConcepts = {}

    def __iter__(self):
        yield from BaseGraphTree.__iter__(self)
        yield from self._concepts
        yield from self._relations

      
    def findRootConceptOrRelation(self, relationConcept):
        # If the result is already in cache, return it
        if relationConcept in self.cacheRootConcepts:
            return self.cacheRootConcepts[relationConcept]

        try:
            isAs = relationConcept.is_a()
        except (AttributeError, TypeError):
            isAs = []
        
        for _isA in isAs:
            parentRelationConcept = _isA.dst
            
            # Recursive call, but result is cached if previously computed
            root = self.findRootConceptOrRelation(parentRelationConcept)
            
            # Store result in cache
            self.cacheRootConcepts[relationConcept] = root
            return root

        # If the provided concept or relation is root (has no parents)
        # Store result in cache
        self.cacheRootConcepts[relationConcept] = relationConcept
        
        return relationConcept
    
    def findConcept(self, conceptName):
        '''Finds the root concept or relation for a given concept or relation.
    
        This method performs a recursive search to identify the root concept or relation.
        If a result has been previously computed, it retrieves the result from cache to avoid redundant computation.
        
        Args:
        relationConcept (Any): The concept or relation for which the root is to be found. The type depends on the implementation.
    
        Returns:
        Any: The root concept or relation.
    
        Raises:
        AttributeError, TypeError: If the attribute 'is_a' is not available or if the type is incorrect.
        '''
        subGraph_keys = [key for key in self._objs]
        for subGraphKey in subGraph_keys:
            subGraph = self._objs[subGraphKey]
           
            for conceptNameItem in subGraph.concepts:
                if conceptName == conceptNameItem:
                    concept = subGraph.concepts[conceptNameItem]
                   
                    return concept
            
        return None 

    def findConceptInfo(self, concept):
        '''Finds and returns various information related to a given concept.
    
        This method compiles a dictionary containing different attributes and relations of the concept. 
        It looks for the 'has_a', 'contains', and 'is_a' relationships and also identifies if the concept is a root.
        
        Args:
        concept (Any): The concept for which information is to be found. The type depends on the implementation.
        
        Returns:
        OrderedDict: A dictionary containing the following keys:
            - 'concept': The original concept.
            - 'relation': Boolean indicating if the concept has a 'has_a' relationship.
            - 'has_a': List of 'has_a' relations.
            - 'relationAttrs': An ordered dictionary of relation attributes.
            - 'contains': List of concepts that the original concept contains.
            - 'containedIn': List of concepts that contain the original concept.
            - 'is_a': List of concepts that the original concept is a type of.
            - 'root': Boolean indicating if the concept is a root concept.
        '''
        has_a = concept.has_a()
        
        if not has_a:
            is_a = [contain.dst for contain in concept._out.get('is_a', [])]
            
            for i in is_a:
                has_a = i.has_a()
                
                if has_a:
                    break
        
        conceptInfo = OrderedDict([
            ('concept', concept),
            ('relation', bool(concept.has_a())),
            ('has_a', concept.has_a()),
            ('relationAttrs', OrderedDict((rel, self.findConcept(rel.dst.name)) for _, rel in enumerate(has_a))),
            ('contains', [contain.dst for contain in concept._out.get('contains', [])]),
            ('containedIn', [contain.src for contain in concept._in.get('contains', [])]),
            ('is_a', [contain.dst for contain in concept._out.get('is_a', [])])
        ])
    
        if not conceptInfo['containedIn'] and not conceptInfo['is_a'] and not conceptInfo['relation']:
            conceptInfo['root'] = True
        else:
            conceptInfo['root'] = False
            
        return conceptInfo
        
    # find all variables defined in the logical constrain and report error if some of them are defined more than once
    def find_lc_variable(self, lc, found_variables=None, headLc=None):
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
            raise Exception(f"{exceptionStr1} {exceptionStr2}")

        if found_variables is None:
            found_variables = {}

        from domiknows.graph import V, LogicalConstrain, LcElement

        e_before = None
        for e in lc.e:
            # checking if element is a variable
            if isinstance(e, V) and e and e.name:
                variable_name = e.name
                if e_before:
                    if variable_name in found_variables:
                        raise Exception(f"In logical constraint {headLc} {lc} variable {variable_name} already defined in {found_variables[variable_name][0]} and associated with concept {found_variables[variable_name][2][1]}")

                    variable_info = (lc, variable_name, e_before)
                    found_variables[variable_name] = variable_info
                else:
                    raise Exception(f"In logical constraint {headLc} {lc} variable {variable_name} is not associated with any concept")

            # checking for extra variable:
            elif e and isinstance(e, tuple) and e[0] == 'extraV':
                predicate = lc.e[0][1]
                exceptionStr1 = f"Logical constraint {headLc} {lc}: Each predicate can only have one new variable definition. For the predicate {predicate}, you have used both {e[1]} and {e[2]} as new variables."
                exceptionStr2 = f"Either wrap both under on variable, if you intended to initialize {e[1]} based on another value, then the second argument should be a path=(...)."
                raise Exception(f"{exceptionStr1} {exceptionStr2}")
            # checking if element is a tuple 
            elif isinstance(e, tuple) and e and isinstance(e[0], LcElement) and not isinstance(e[0], LogicalConstrain):
                self.find_lc_variable(e[0], found_variables=found_variables, headLc=headLc)
                current_lc_element = e[0]
                current_lc_element_concepts = [c for c in current_lc_element.e if isinstance(c, tuple) and not isinstance(c, V)]

                if len(current_lc_element_concepts) != len(e[1]):
                    raise Exception(f"Logical constraint {headLc} {lc} has incorrect definition of combination {e} - number of variables does not match number of concepts in combination")

                if len(e) >= 2 and isinstance(e[1], tuple):
                    for v in e[1]:
                        if not isinstance(v, str):
                            raise Exception(f"Logical constraint {headLc} {lc} has incorrect definition of combination {e} - all variables should be strings")

                    for index, v in enumerate(e[1]):
                        variable_name = v
                        variable_info = (lc, variable_name, current_lc_element_concepts[index])
                        found_variables[variable_name] = variable_info

            # Checking if element is a LogicalConstrain
            elif isinstance(e, LogicalConstrain):
                self.find_lc_variable(e, found_variables=found_variables, headLc=headLc)

            e_before = e

        return found_variables

    def check_if_all_used_variables_are_defined(self, lc, found_variables, used_variables=None, headLc=None):
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

        def handle_variable_name(lc_variable_name, lcPath):
            if lc_variable_name not in found_variables:
                raise Exception(f"Variable {lc_variable_name} found in {headLc} {lc} is not defined")

            if lc_variable_name not in used_variables:
                used_variables[lc_variable_name] = []

            lcElementType = lc.e[i-1]
            variable_info = (lc, lc_variable_name, lcElementType, lcPath)
            used_variables[lc_variable_name].append(variable_info)

        for i, e in enumerate(lc.e):
            if isinstance(e, V) and e.v: # has path
                if isinstance(e.v, eqL):
                    continue
                elif isinstance(e.v, str):
                    handle_variable_name(e.v, e.v)
                elif isinstance(e, tuple):
                    if isinstance(e.v[0], str): # single path
                        handle_variable_name(e.v[0], e.v)
                    elif isinstance(e.v[0], tuple): # path union
                        for t in e.v:
                            if isinstance(t[0], str):
                                handle_variable_name(t[0], t)
                            else:
                                raise Exception(f"Path {t} found in {headLc} {lc} is not correct")
                    else:
                        raise Exception(f"Path {e} found in {headLc} {lc} is not correct")
                else:
                    raise Exception(f"Path {e} found in {headLc} {lc} is not correct")
            elif isinstance(e, LogicalConstrain):
                self.check_if_all_used_variables_are_defined(e, found_variables, used_variables=used_variables, headLc=headLc)

        return used_variables

    def getPathStr(self, path):
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
                
    def check_path(self, path, resultConcept, variableConceptParent, lc_name, foundVariables, variableName):
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
        
        requiredLeftConcept = variableConceptParent.name # path element has to be relation with this type to the left
        requiredEndOfPathConcept = resultConcept[1] # path has to result in this concept
        requiredEndOfPathConceptRoot =  self.findRootConceptOrRelation(resultConcept[0]).name  
        expectedRightConcept =  None
        lastPathElement = False
        pathStr = self.getPathStr(path)
        pathVariable = path[0]
        pathPart = path[0]
            
        if len(path) == 1:
            if requiredLeftConcept == requiredEndOfPathConceptRoot:
                return
            else:
                exceptionStr1 = f"The variable {pathVariable}, defined in the path for {lc_name} is not valid. The concept of {pathVariable} is a of type {requiredLeftConcept},"
                exceptionStr2 = f"but the required concept by the logical constraint element is {requiredEndOfPathConceptRoot}."
                exceptionStr3 = f"The variable used inside the path should match its type with {requiredEndOfPathConceptRoot}."
                raise Exception(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
            
        for pathIndex, pathElement in enumerate(path[1:], start=1):   
            if isinstance(pathElement, (eqL,)):
                continue
            if isinstance(pathElement, (str,)): # It is a string check if we have corresponding relation in the graph
                if pathElement in self.varNameReversedMap:
                    pathElement = self.varNameReversedMap[pathElement]
                else:
                    exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid."
                    exceptionStr2 = f"The required source type after {pathPart} is a {requiredLeftConcept},"
                    exceptionStr3 = f"but the used variable {pathElement} is a string which is not a valid name of a graph relationship."
                    raise Exception(f"{exceptionStr1} {exceptionStr2}")
                
            if pathIndex < len(path) - 1:
                expectedRightConcept = pathElement.dst.name
                expectedRightConceptRoot = expectedRightConcept
            else:
                expectedRightConcept = requiredEndOfPathConcept
                lastPathElement = True
                expectedRightConceptRoot = requiredEndOfPathConceptRoot                
            
            if isinstance(pathElement, (HasA, IsA, Relation)):
                pathElementSrc = pathElement.src.name
                pathElementDst = pathElement.dst.name
                pathElementVarName = pathElement.var_name if pathElement.var_name else ""

                # Check if there is a problem with reversed usage of the current path element - it has to be possible to reverse the order to fix it
                if requiredLeftConcept == pathElementDst and expectedRightConceptRoot == pathElementSrc:                    
                    exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid"
                    exceptionStr2 = f"The relation {pathElementVarName} is from a {pathElementSrc} to a {pathElementDst}, but you have used it from a {pathElementDst} to a {pathElementSrc}."
                    if not pathElement.is_reversed:
                        exceptionStr3 = f"You can change '{pathElement.var_name}' to '{pathElement.var_name}.reversed' to go from {pathElementDst} to the {pathElementSrc}, which is what is required here."
                    else:
                        exceptionStr3 = f"You can change  '{pathElement.var_name}.reversed' to '{pathElement.var_name}' to go from {pathElementSrc} to the {pathElementDst}, which is what is required here."
                        f"You can use without the .reversed property to change the direction."
                    raise Exception(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
                # Check if the current path element is correctly connected to the left (source) - has matching type
                elif requiredLeftConcept != pathElementSrc:
                    exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid."
                    exceptionStr2 = f"The required source type after {pathPart} is a {requiredLeftConcept},"
                    exceptionStr3 = f"but the used variable {pathElementVarName} is a relationship defined between a {pathElementSrc} and a {pathElementDst}, which is not correctly used here."
                    raise Exception(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
                # Check if the current path element is correctly connected to the right (destination) - has matching type
                elif expectedRightConceptRoot != pathElementDst: 
                    exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid."
                    if lastPathElement: # if this is the last element it has to match the concept in which this path is embedded
                        exceptionStr2 = f"The required destination type after {pathPart} is a {expectedRightConcept}."
                    else: # if this it intermediary path element that if is expected that it will match next path element source type
                        exceptionStr2 = f"The expected destination type after {pathPart} is a {expectedRightConcept}."
                    exceptionStr3 = f"The used variable {pathElementVarName} is a relationship defined between a {pathElementSrc} and a {pathElementDst}, which is not correctly used here."
                    raise Exception(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
                
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
                    raise Exception(f"{exceptionStr1} {exceptionStr2} {exceptionStr3} {exceptionStr4} {exceptionStr5}")
                else: # all other types not allowed in path
                    pathElementType = type(pathElement)
                    exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, after {pathPart} is not valid."
                    exceptionStr2 = f"The used variable {pathElement} is a {pathElementType}, path element can be only relation or eqL logical constraint used to filter candidates in the path."
                    raise Exception(f"{exceptionStr1} {exceptionStr2}")


    def are_keys_new(self, given_dict, dict_list):
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

    def collectVarMaps(self, lc, varMapsList):
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
                self.collectVarMaps(e, varMapsList) # recursive
                newE.append(e)
            # check if VarMap
            elif isinstance(e, tuple) and e[0] == 'VarMaps':
                currentVarMap = e[1]
                
                # check if variables in the current VarMap are new, have not been found already
                # If they are new it means it it their definition in the lc
                if self.are_keys_new(currentVarMap, varMapsList):
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
                        definedVaribleList = [d.get(variableName, None) for d in varMapsList]
                        
                        if definedVaribleList:
                            definedVarible = definedVaribleList[0]
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
                   
    def handleVarsPath(self, lc, varMaps):
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
                self.handleVarsPath(e, varMaps) # recursive
            elif isinstance(e, V) and e.name in varMaps: # check if the current lc element is already in the V  form and its name is in varMaps
                # this is the variable syntax for a single variable usage
                # replace the V with the mapping for this found in varMap
                e = varMaps[e.name]
            elif needsVariableUpdate: # ths flag was set by the previous lc element
                usedVarMap = lc.e[i+1][1] # get varMaps to use
                usedVarMapIt= iter(usedVarMap.items())
                firstUsedV = next(usedVarMapIt)[1][1]
                secondUsedV = next(usedVarMapIt)[1][1]

                # create new paths based on variables from varMaps
                firstNewPath = firstUsedV + (firstUsedV[1].reversed,)
                secondNewPath = secondUsedV + (secondUsedV[1].reversed,)
                
                path = (firstNewPath, secondNewPath)  
                
                updated_e = V(name="", v=path)
                e = updated_e
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
                
    def __exit__(self, exc_type, exc_value, traceback):
        '''Handles the exiting logic for the context manager.
    
        This method performs clean-up operations like mapping variable names and 
        validating logical constraints. It is automatically called when exiting the 
        'with' block of the context manager.
    
        Args:
        exc_type (type): The type of the exception that caused the context manager to exit.
        exc_value (Exception): The instance of the exception that caused the context manager to exit.
        traceback (traceback): A traceback object encapsulating the call stack at the point 
                               where the exception originally occurred.
    
        Side-effects:
        Modifies internal state to include variable name mappings and validates logical constraints.
        '''
        super().__exit__(exc_type, exc_value, traceback)
            
        #image_name = "./image" + self.name
        #self.visualize(image_name, open_image=True)
        
        # Get the current frame and then go one level back
        frame = inspect.currentframe().f_back

        from . import Concept
        from .relation import IsA, HasA
    
        # --- Iterate through all local variables in that frame
        for var_name, var_value in frame.f_locals.items():
            # Check if any of them are instances of the Concept class or Relation subclass
            if isinstance(var_value, (Concept, HasA, IsA)):
                # If they are, and their var_name attribute is not already set,
                # set it to the name of the variable they are stored in.
                if var_value.var_name is None:
                    var_value.var_name = var_name
                    self.varNameReversedMap[var_value.name] = var_value
                    if isinstance(var_value, (HasA,)):
                        var_value.reversed.var_name = var_name + ".reversed"
                        self.varNameReversedMap[var_value.reversed.name] = var_value.reversed

        # --- Process logical constraints variable syntax
        for lc_name, lc in self.logicalConstrains.items():
            if not lc.active or not lc.headLC:
                continue

            # collect VarMaps - store info about lc variable syntax if used in this logical constraint
            varMapsList = self.collectVarMaps(lc, [])
            
            # process variable syntax - translate it to the path syntax
            if varMapsList:
                self.handleVarsPath(lc, varMapsList[0])
                    
        # --- Check if the logical constrains are correct ---
        
        lc_info = {}
        LcInfo = namedtuple('CurrentLcInfo', ['foundVariables', 'usedVariables', 'headLcName'])
        
        # --- Gather information about variables used and defined in the logical constrains and 
        #     report errors if some of them are not defined and used or defined more than once
        for lc_name, lc in self.logicalConstrains.items():
            if not lc.active or not lc.headLC:
                continue
                
            # find variable defined in the logical constrain - report error if some of them are defined more than once
            found_variables = self.find_lc_variable(lc, headLc=lc.name)

            # find all variables used in the logical constrain - report error if some of them are not defined
            # gather paths defined in the logical constrain per variable
            used_variables = self.check_if_all_used_variables_are_defined(lc, found_variables, headLc=lc.name)
            
            # save information about variables used and defined in the logical constrain
            current_lc_info = LcInfo(found_variables, used_variables, lc.name)
            lc_info[lc_name] = current_lc_info
        
        # --- Check if the paths defined in the logical constrains are correct
        for lc_name, lc in self.logicalConstrains.items():
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
                #variableConceptWhat = variableConcept.what()
                
                # get the root parent of the variable concept
                variableConceptParent = self.findRootConceptOrRelation(variableConcept)
                
                # loop over all paths defined using the variable as starting point
                for pathInfo in pathInfos:
                    path = pathInfo[3]
                    resultConcept = pathInfo[2]
                    
                    if isinstance(path[0], tuple): # this path is a combination of paths 
                        for subpath in path: 
                            if len(subpath) < 1:  
                                continue  # skip this subpath it is empty
                                
                            self.check_path(subpath, resultConcept, variableConceptParent, headLcName, foundVariables, variableName)
                            
                    else: # this path is a single path
                        if len(path) < 1:
                            continue # skip this path it is empty
                            
                        self.check_path(path, resultConcept, variableConceptParent, headLcName, foundVariables, variableName)
                       
    @property
    def ontology(self):
        '''Gets the ontology associated with the object.
    
        Returns:
        Graph.Ontology: The ontology associated with the object.
        '''
        return self._ontology

    @ontology.setter
    def ontology(self, ontology):
        '''Sets the ontology for the object.
        
        Args:
        ontology (Graph.Ontology|str|tuple): The ontology to set. This can be either 
                                             an instance of Graph.Ontology, a string, 
                                             or a tuple containing two elements.
    
        Raises:
        TypeError: If the provided ontology is not of the correct type.
        '''
        if isinstance(ontology, Graph.Ontology):
            self._ontology = ontology
        elif isinstance(ontology, str):
            self._ontology = Graph.Ontology(ontology)
        elif isinstance(ontology, tuple) and len(ontology) == 2:
            self._ontology = Graph.Ontology(*ontology)
        else:
            raise TypeError("Invalid ontology type.")

    @property
    def batch(self):
        '''Gets the current batch.
        
        Returns:
        Any: The current batch.
        '''
        return self._batch
    
    @batch.setter
    def batch(self, value):
        '''Sets the batch.
    
        Args:
        value (Any): The batch value to set.
        '''
        self._batch = value
    
    @property
    def auto_constraint(self):
        '''Determines whether to automatically enforce constraints.
    
        If the auto_constraint is not set, it will defer to the `sup` attribute.
    
        Returns:
        bool: True if constraints should be automatically enforced, False otherwise.
        '''
        if self._auto_constraint is None and self.sup:
            return self.sup.auto_constraint
        return self._auto_constraint or False  # if None, return False instead
    
    @auto_constraint.setter
    def auto_constraint(self, value):
        '''Sets whether to automatically enforce constraints.
        
        Args:
        value (bool): The value to set for automatic constraint enforcement.
        '''
        self._auto_constraint = value
    
    def get_properties(self, *tests):
        '''Finds and returns properties that meet the given conditions.
    
        Args:
        tests (callable): Variable-length argument list of test conditions each property must pass.
    
        Returns:
        list: A list of properties that meet all the given test conditions.

        '''
        def func(node):
            if isinstance(node, Property):
                for test in tests:
                    if not test(node):
                        return None
                else:
                    return node
            return None
        return list(self.traversal_apply(func))
    
    def get_sensors(self, *tests):
        '''Finds and returns sensors that meet the given conditions.
    
        Args:
        tests (callable): Variable-length argument list of test conditions each sensor must pass.
    
        Returns:
        list: A list of sensors that meet all the given test conditions.
        '''
        return list(chain(*(prop.find(*tests) for prop in self.get_properties())))

    def get_apply(self, name):
        '''Finds and returns the concept or relation with the given name.
    
        Args:
        name (str): The name of the concept or relation to find.
    
        Returns:
        Object: The concept or relation with the specified name, or the result of BaseGraphTree.get_apply if not found.
        '''
        if name in self.concepts:
            return self.concepts[name]
        if name in self.relations:
            return self.relations[name]
        return BaseGraphTree.get_apply(self, name)

    def set_apply(self, name, sub):
        '''Sets the `Concept` or `Relation` for a given name in the object.

        If the sub is a `Graph`, it delegates to the parent class `BaseGraphTree.set_apply`.
        If it is a `Concept` or `Relation`, it adds it to the respective dictionary.
    
        Args:
        name (str): The name to set for the `Concept` or `Relation`.
        sub (Graph|Concept|Relation): The object to set for the name. This can be an instance of `Graph`, 
                                      `Concept`, or `Relation`.
    
        TODO:
        1. Handle cases where a concept has the same name as a subgraph.
        2. Handle other types for the 'sub' parameter.
    
        '''
        if __package__ is None or __package__ == '':
            from concept import Concept
            from relation import Relation
        else:
            from .concept import Concept
            from .relation import Relation
        # TODO: what if a concept has same name with a subgraph?
        if isinstance(sub, Graph):
            BaseGraphTree.set_apply(self, name, sub)
        elif isinstance(sub, Concept):
            self.concepts[name] = sub
            if sub.get_batch():
                self.batch = sub
        elif isinstance(sub, Relation):
            self.relations[name] = sub
        else:
            # TODO: what are other cases
            pass

    def visualize(self, filename, open_image=False):
        '''Visualizes the graph using Graphviz.

        Creates a directed graph and populates it with nodes for concepts and
        edges for relations. It also recursively adds subgraphs. Finally, it
        either saves the graph to a file or returns the graph object depending on
        the `filename` parameter.
    
        Args:
        filename (str|None): The name of the file where the graph will be saved.
                             If None, the graph object is returned instead.
        open_image (bool): Whether to open the image after rendering.
                          Defaults to False.
    
        Returns:
        graphviz.Digraph|None: The graph object if filename is None, otherwise None.
        '''
        import graphviz
        concept_graph = graphviz.Digraph(name=f"{self.name}")
        concept_graph.attr(label=f"Graph: {self.name}") 

        for concept_name, concept in self.concepts.items():
            concept_graph.node(concept_name)

        for subgraph_name, subgraph in self.subgraphs.items():
            sub_graph_viz = subgraph.visualize(filename=None)
            sub_graph_viz.name = 'cluster_' + sub_graph_viz.name
            concept_graph.subgraph(sub_graph_viz)

        for relation_name, relation in self.relations.items():
            if not relation.name.endswith('reversed') and not ('not_a' in relation.name):
                # add case for HasA
                concept_graph.edge(relation.src.name, relation.dst.name, label=relation.__class__.__name__)

        if filename is not None:
            concept_graph.render(filename, format='png', view=open_image)
        else:
            return concept_graph

    @property
    def subgraphs(self):
        """Ordered dictionary containing the subgraphs.

        Returns:
            OrderedDict: An ordered dictionary of the subgraphs.
        """
        return OrderedDict(self)

    @property
    def concepts(self):
        """Getter for the concepts.

        Returns:
            dict: Dictionary of concepts.
        """
        return self._concepts

    @property
    def logicalConstrains(self):
        """Getter for the logical constraints.

        Returns:
            dict: Dictionary of logical constraints.
        """
        return self._logicalConstrains

    @property
    def logicalConstrainsRecursive(self):
        """Generator function that yields logical constraints recursively.

        This method goes through all nodes and yields the logical constraints 
        if the node is an instance of Graph.

        Yields:
            tuple: A tuple containing key-value pairs of logical constraints.
        """
        def func(node):
            if isinstance(node, Graph):
                yield from node.logicalConstrains.items()
        yield from self.traversal_apply(func)

    @property
    def relations(self):
        """Getter for the relations.

        Returns:
            dict: Dictionary of relations.
        """
        return self._relations

    def what(self):
        """Method to get the summary of the graph tree.

        This method provides a dictionary containing the base graph tree and its concepts.

        Returns:
            dict: Dictionary containing 'concepts' and the base graph tree.
        """
        wht = BaseGraphTree.what(self)
        wht['concepts'] = dict(self.concepts)
        return wht
    
    def print_predicates(self,):
        """Generate and return a list of predicates with their variables.

        This method goes through the subgraphs and concepts to generate a list of predicates,
        determining the variables they use based on their relations and superclass structure.

        Returns:
            list: A list of strings, each representing a predicate and its variables.
        """
        predicate_list = dict()
        variable_list = ['x', 'y', 'z', 'a', 'b', 'r', 'c', 'l', 'i']
        concepts = list(self.concepts.values())
        for subgraph in self.subgraphs.values():
            concepts.extend(list(subgraph.concepts.values()))
        for concept in concepts:
            predicate_name = concept.name
            variables = variable_list[0]
            check = False
            if concept._out:
                if 'has_a' in concept._out:
                    num_relations = len(concept._out['has_a'])
                    variables = ", ".join(variable_list[:num_relations])
                    check = True
            if not check:
                new_node = concept
                while 'is_a' in new_node._out:
                    if new_node._out['is_a'][0].dst.name in predicate_list:
                        variables = predicate_list[new_node._out['is_a'][0].dst.name][1]
                        break
                    else:
                        if 'has_a' in new_node.sup._out:
                            num_relations = len(new_node.sup._out['has_a'])
                            variables = ", ".join(variable_list[:num_relations])
                            break
                    new_node = new_node._out['is_a'][0].dst
            if hasattr(concept, 'enum'):
                ### there should be another variable to indicate the type
                variables = f"{variables}, t"
                predicate_list[predicate_name] = [predicate_name, variables, concept.enum]
            else:
                predicate_list[predicate_name] = [predicate_name, variables]
        final_list = []
        for predicate in predicate_list.values():
            if len(predicate) == 3:
                final_list.append(f"{predicate[0]}({predicate[1]}) --> 't' stand for the possible types which should be selected from this list: {predicate[2]}")
            else:
                final_list.append(f"{predicate[0]}({predicate[1]})")
        return final_list

    Ontology = namedtuple('Ontology', ['iri', 'local'], defaults=[None])
    """Namedtuple for Ontology data structure.

    Attributes:
        iri (str): The IRI of the ontology. Default is None.
        local (str): The local identification of the ontology. Default is None.
    """