from collections import OrderedDict, namedtuple
from itertools import chain
import inspect

if __package__ is None or __package__ == '':
    from base import BaseGraphTree
    from property import Property
else:
    from .base import BaseGraphTree
    from .property import Property


@BaseGraphTree.localize_namespace
class Graph(BaseGraphTree):
    def __init__(self, name=None, ontology=None, iri=None, local=None, auto_constraint=None, reuse_model=False):
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
    
    from collections import namedtuple

    # find all variables defined in the logical constrain and report error if some of them are defined more than once
    def find_lc_variable(self, lc, found_variables=None, headLc=None):
        if lc.cardinalityException:
            if lc.name != headLc:
                raise Exception(f"{lc.typeName} {headLc} has incorrect cardinality definition in nested constraint {lc.name} - integer {lc.cardinalityException} has to be last element in the {lc.typeName}")
            else:
                raise Exception(f"{lc.typeName} {headLc} has incorrect cardinality definition - integer {lc.cardinalityException} has to be last element in the {lc.typeName}")
            
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
        if used_variables is None:
            used_variables = {}

        from domiknows.graph import V, LogicalConstrain

        def handle_variable_name(lc_variable_name):
            if lc_variable_name not in found_variables:
                raise Exception(f"Variable {lc_variable_name} found in {headLc} {lc} is not defined")

            if lc_variable_name not in used_variables:
                used_variables[lc_variable_name] = []

            lcElementType = lc.e[i-1]
            lcPath = e.v
            variable_info = (lc, lc_variable_name, lcElementType, lcPath)
            used_variables[lc_variable_name].append(variable_info)

        for i, e in enumerate(lc.e):
            if isinstance(e, V) and e.v: # has path
                if isinstance(e.v, str):
                    handle_variable_name(e.v)
                elif isinstance(e, tuple):
                    if isinstance(e.v[0], str): # single path
                        handle_variable_name(e.v[0])
                    elif isinstance(e.v[0], tuple): # path union
                        for t in e.v:
                            if isinstance(t[0], str):
                                handle_variable_name(t[0])
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
                pathStr += pathElement
               
        return pathStr.strip()
                
    def check_path(self, path, resultConcept, variableConceptParent, lc_name, foundVariables, variableName):
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
            
        for p, pathElement in enumerate(path[1:]):
            pathIndex = p+1
        
            if pathIndex < len(path)-1:
                expectedRightConcept = path[pathIndex].dst.name
                expectedRightConceptRoot = expectedRightConcept
            else:
                expectedRightConcept = requiredEndOfPathConcept
                lastPathElement = True
                expectedRightConceptRoot = requiredEndOfPathConceptRoot                
            
            if isinstance(pathElement, (HasA, IsA, Relation)):
                pathElementSrc = pathElement.src.name
                pathElementDst = pathElement.dst.name
                if pathElement.var_name != None:
                    pathElementVarName = pathElement.var_name
                else:
                    pathElementVarName =""
                
                # Check if there is a problem with reversed usage of the current path element - it has to be possible to reverse the order to fix it
                if requiredLeftConcept == pathElementDst and expectedRightConceptRoot == pathElementSrc:                    
                    exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid"
                    exceptionStr2 = f"The relation {pathElementVarName} is from a {pathElementSrc} to a {pathElementDst}, but you have used it from a {pathElementDst} to a {pathElementSrc}."
                    if not pathElement.is_reversed:
                        exceptionStr3 = f"You can use the .reversed property to change the direction."
                    else:
                        exceptionStr3 = f"You can use without the .reversed property to change the direction."
                    raise Exception(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
                # Check if the current path element is correctly connected to the left (source) - has matching type
                elif requiredLeftConcept != pathElementSrc:
                    exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid."
                    exceptionStr2 = f"The required source type in this place of the path is a {requiredLeftConcept},"
                    exceptionStr3 = f"but the used variable {pathElementVarName} is a relationship defined between a {pathElementSrc} and a {pathElementDst}, which is not correctly used here."
                    raise Exception(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
                # Check if the current path element is correctly connected to the right (destination) - has matching type
                elif expectedRightConceptRoot != pathElementDst: 
                    exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid."
                    if lastPathElement: # if this is the last element it has to match the concept in which this path is embedded
                        exceptionStr2 = f"The required destination type of the last element of the path is a {expectedRightConcept}."
                    else: # if this it intermediary path element that if is expected that it will match next path element source type
                        exceptionStr2 = f"The expected destination type in this place of the path is a {expectedRightConcept}."
                    exceptionStr3 = f"The used variable {pathElementVarName} is a relationship defined between a {pathElementSrc} and a {pathElementDst}, which is not correctly used here."
                    raise Exception(f"{exceptionStr1} {exceptionStr2} {exceptionStr3}")
                
                # Move along the path with the requiredLeftConcept and pathVariable
                requiredLeftConcept = pathElementDst
                pathVariable = pathElementVarName
            elif isinstance(pathElement, (eqL,)):
                pass # it is not check for now
            else:
                exceptionStr1 = f"The Path '{pathStr}' from the variable {pathVariable}, defined in {lc_name} is not valid."
                if isinstance(pathElement, (Concept,)):
                    exceptionStr2 = f"The used variable {pathElement} is a concept, path element can be only relation or eqL logical constraint used to filter candidates in the path."
                else:
                    pathElementType = type(pathElement)
                    exceptionStr2 = f"The used variable {pathElement} is a {pathElementType}, path element can be only relation or eqL logical constraint used to filter candidates in the path."
                raise Exception(f"{exceptionStr1} {exceptionStr2}")

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        
        #image_name = "./image" + self.name
        #self.visualize(image_name, open_image=True)
        
        # Get the current frame and then go one level back
        frame = inspect.currentframe().f_back

        from . import Concept
        from .relation import IsA, HasA
        
        # Iterate through all local variables in that frame
        for var_name, var_value in frame.f_locals.items():
            # Check if any of them are instances of the Concept class or Relation subclass
            if isinstance(var_value, (Concept, HasA, IsA)):
                # If they are, and their var_name attribute is not already set,
                # set it to the name of the variable they are stored in.
                if var_value.var_name is None:
                    var_value.var_name = var_name
                    if isinstance(var_value, (HasA,)):
                        var_value.reversed.var_name = var_name + ".reversed"
                    
        lc_info = {}
        LcInfo = namedtuple('CurrentLcInfo', ['foundVariables', 'usedVariables', 'headLcName'])

        # --- Check if the logical constrains are correct ---
        
        # --- Gather information about variables used and defined in the logical constrains and 
        #     report errors if some of them are not defined and used or defined more than once
        for lc_name, lc in self.logicalConstrains.items():
            if not lc.headLC:
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
            if not lc.headLC:
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
                            if len(subpath) < 2:  
                                continue  # skip this subpath as it has only the starting point variable
                                
                            self.check_path(subpath, resultConcept, variableConceptParent, headLcName, foundVariables, variableName)
                            
                    else: # this path is a single path
                        if len(path) < 2:
                            continue # skip this path as it has only the starting point variable
                            
                        self.check_path(path, resultConcept, variableConceptParent, headLcName, foundVariables, variableName)
                       
    @property
    def ontology(self):
        return self._ontology
    
    @property
    def batch(self):
        return self._batch
    
    @batch.setter
    def batch(self, value):
        self._batch = value

    @property
    def auto_constraint(self):
        if self._auto_constraint is None and self.sup:
            return self.sup.auto_constraint
        return self._auto_constraint or False  # if None, return False instead

    @auto_constraint.setter
    def auto_constraint(self, value):
        self._auto_constraint = value

    @ontology.setter
    def ontology(self, ontology):
        if isinstance(ontology, Graph.Ontology):
            self._ontology = ontology
        elif isinstance(ontology, str):
            self._ontology = Graph.Ontology(ontology)
        elif isinstance(ontology, tuple) and len(ontology) == 2:
            self._ontology = Graph.Ontology(*ontology)

    def get_properties(self, *tests):
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
        return list(chain(*(prop.find(*tests) for prop in self.get_properties())))

    def get_apply(self, name):
        if name in self.concepts:
            return self.concepts[name]
        if name in self.relations:
            return self.relations[name]
        return BaseGraphTree.get_apply(self, name)

    def set_apply(self, name, sub):
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
        return OrderedDict(self)

    @property
    def concepts(self):
        return self._concepts

    @property
    def logicalConstrains(self):
        return self._logicalConstrains

    @property
    def logicalConstrainsRecursive(self):
        def func(node):
            if isinstance(node, Graph):
                yield from node.logicalConstrains.items()
        yield from self.traversal_apply(func)

    @property
    def relations(self):
        return self._relations

    def what(self):
        wht = BaseGraphTree.what(self)
        wht['concepts'] = dict(self.concepts)
        return wht
    
    def print_predicates(self,):
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