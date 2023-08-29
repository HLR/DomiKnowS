from collections import OrderedDict, namedtuple
from itertools import chain

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

    def __iter__(self):
        yield from BaseGraphTree.__iter__(self)
        yield from self._concepts
        yield from self._relations


    # find the root parent of the provided concept or relation
    def findRootConceptOrRelation(self, relationConcept):
        try:
            isAs = relationConcept.is_a()
        except (AttributeError, TypeError):
            isAs = []
        
        for _isA in isAs:
            parentRelationConcept = _isA.dst
            
            return self.findRootConceptOrRelation(parentRelationConcept)
        
        # If the provided concept or relation is root (has not parents)
        return relationConcept 
    
    from collections import namedtuple

    # find all variables defined in the logical constrain and report error if some of them are defined more than once
    def find_lc_variable(self, lc, found_variables=None):
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
                        raise Exception(f"Variable {variable_name} already defined in {found_variables[variable_name][0]} and associated with concept {found_variables[variable_name][2][1]}")

                    variable_info = (lc.name, variable_name, e_before)
                    found_variables[variable_name] = variable_info
                else:
                    raise Exception(f"Variable {variable_name} is not associated with any concept")

            # checking if element is a tuple 
            elif isinstance(e, tuple) and e and isinstance(e[0], LcElement) and not isinstance(e[0], LogicalConstrain):
                self.find_lc_variable(e[0], found_variables=found_variables)
                current_lc_element = e[0]
                current_lc_element_concepts = [c for c in current_lc_element.e if isinstance(c, tuple) and not isinstance(c, V)]

                if len(current_lc_element_concepts) != len(e[1]):
                    raise Exception(f"Logical Constrain {lc.name} has incorrect definition of combination {e} - number of variables does not match number of concepts in combination")

                if len(e) >= 2 and isinstance(e[1], tuple):
                    for v in e[1]:
                        if not isinstance(v, str):
                            raise Exception(f"Logical Constrain {lc.name} has incorrect definition of combination {e} - all variables should be strings")

                    for index, v in enumerate(e[1]):
                        variable_name = v
                        variable_info = (lc.name, variable_name, current_lc_element_concepts[index])
                        found_variables[variable_name] = variable_info

            # Checking if element is a LogicalConstrain
            elif isinstance(e, LogicalConstrain):
                self.find_lc_variable(e, found_variables=found_variables)

            e_before = e

        return found_variables

    def check_if_all_used_variables_are_defined(self, lc, found_variables, used_variables=None):
        if used_variables is None:
            used_variables = {}

        from domiknows.graph import V, LogicalConstrain

        def handle_variable_name(variable_name):
            if variable_name not in found_variables:
                raise Exception(f"Variable {variable_name} found in {lc.name} is not defined")

            if variable_name not in used_variables:
                used_variables[variable_name] = []

            variable_info = (lc.name, variable_name, e.v)
            used_variables[variable_name].append(variable_info)

        for e in lc.e:
            if isinstance(e, V) and e.v:
                if isinstance(e.v, str):
                    handle_variable_name(e.v)
                elif isinstance(e, tuple):
                    if isinstance(e.v[0], str):
                        handle_variable_name(e.v[0])
                    elif isinstance(e.v[0], tuple):
                        for t in e.v:
                            if isinstance(t[0], str):
                                handle_variable_name(t[0])
                            else:
                                raise Exception(f"Path {t} found in {lc.name} is not correct")
                    else:
                        raise Exception(f"Path {e} found in {lc.name} is not correct")
                else:
                    raise Exception(f"Path {e} found in {lc.name} is not correct")
            elif isinstance(e, LogicalConstrain):
                self.check_if_all_used_variables_are_defined(e, found_variables, used_variables=used_variables)

        return used_variables

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        lc_info = {}
        LcInfo = namedtuple('CurrentLcInfo', ['foundVariables', 'usedVariables'])

        # --- Check if the logical constrains are correct ---

        from .relation import IsA, HasA
        
        # --- Gather information about variables used and defined in the logical constrains and 
        #     report errors if some of them are not defined and used or defined more than once
        for lc_name, lc in self.logicalConstrains.items():
            if not lc.headLC:
                continue

            # find variable defined in the logical constrain - report error if some of them are defined more than once
            found_variables = self.find_lc_variable(lc)

            # find all variables used in the logical constrain - report error if some of them are not defined
            # gather paths defined in the logical constrain per variable
            used_variables = self.check_if_all_used_variables_are_defined(lc, found_variables)
            
            # save information about variables used and defined in the logical constrain
            current_lc_info = LcInfo(found_variables, used_variables)
            lc_info[lc_name] = current_lc_info

        
        # --- Check if the paths defined in the logical constrains are correct
        for lc_name, lc in self.logicalConstrains.items():
            if not lc.headLC:
                continue
            
            # current logical constrain info and variables found and used in the current logical constrain
            current_lc_info = lc_info[lc_name]
            usedVariables = current_lc_info.usedVariables
            foundVariables = current_lc_info.foundVariables
            
            # loop over all variables used in the logical constrain
            for variableName, pathInfos in usedVariables.items():
                # get information about the variable in the found variables record
                variableConcept = foundVariables[variableName][2][0]
                #variableConceptWhat = variableConcept.what()
                # get the root parent of the variable concept
                variableConceptParent = self.findRootConceptOrRelation(variableConcept)
                
                # loop over all paths defined using the variable as starting point
                for pathInfo in pathInfos:
                    path = pathInfo[2]
                   
                    if isinstance(path[0], tuple): # this path is a combination of paths 
                        for subpath in path: 
                            if len(subpath) < 2:  
                                continue  # skip this subpath as it has only the starting point variable
                                
                            # loop over all elements in the subpath         
                            for pathElement in subpath[1:]: 
                                pathElementSrc = pathElement.src
                               
                                # check if the parent of the variable concept is the same as the source of the path element
                                if variableConceptParent.name != pathElementSrc.name:
                                    raise Exception(f"Path {path} found in {lc.name} is not correct for element {pathElement}")
                       
                    else: # this path is a single path
                        if len(path) < 2:
                            continue # skip this path as it has only the starting point variable
                            
                        # loop over all elements in the path
                        for pathElement in path[1:]:
                            if isinstance(pathElement, (HasA, IsA)):
                                pathElementSrc = pathElement.src
                               
                                # check if the parent of the variable concept is the same as the source of the path element  
                                if variableConceptParent.name != pathElementSrc.name:
                                    raise Exception(f"Path {path} found in {lc.name} is not correct for element {pathElement}")
                       
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

    Ontology = namedtuple('Ontology', ['iri', 'local'], defaults=[None])