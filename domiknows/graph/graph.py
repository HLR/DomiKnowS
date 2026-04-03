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
        self.executableLCsLabels = {}
        self._executableLCs = OrderedDict()
        self.varContext = None # None before calling `with graph...`, dictionary after
        self.constraint = None  # Will hold the constraint concept
        self._processed_lcs = set()  # Track which LCs have been processed


    def __iter__(self):
        yield from BaseGraphTree.__iter__(self)
        yield from self._concepts
        yield from self._relations

    def __enter__(self):
        parent_obj = super().__enter__()

        if self.constraint is None:
            # Check if a constraint concept already exists in the graph
            existing_constraint = self.findConcept("constraint")
            if existing_constraint is not None:
                self.constraint = existing_constraint
            else:
                from . import Concept
                constraint = Concept(name="constraint")
                self.constraint = constraint

        return self  # Return self (the current graph), not parent_obj

    def _populate_var_context(self, frame):
        """Populate varContext from a given frame's local variables.
        
        This method captures local variables from the provided frame and updates
        varContext and varNameReversedMap for Concept and Relation instances.
        
        Args:
            frame: A Python frame object whose f_locals will be inspected.
            
        Side-effects:
            - Initializes self.varContext if None
            - Populates self.varContext with frame's local variables (only on first call)
            - Sets var_name attribute on Concept/Relation instances
            - Updates self.varNameReversedMap with name mappings
        """
        from . import Concept
        from .relation import IsA, HasA
        
        first_population = self.varContext is None or not self.varContext
        if first_population:
            self.varContext = {}
        
        for var_name, var_value in frame.f_locals.items():
            # Only update varContext on first population to preserve original graph variables
            if first_population:
                self.varContext[var_name] = var_value
            
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
                        
    def __exit__(self, exc_type, exc_value, traceback):
        '''Handles the exiting logic for the context manager.
    
        This method performs clean-up operations like mapping variable names and 
        validating logical constraints. It is automatically called when exiting the 
        'with' block of the context manager.

        Supports repeat calls - tracks which logical constraints have already been
        processed to avoid reprocessing VarMaps. Also preserves varContext from the
        first call to maintain original graph definition variables.
    
        Args:
        exc_type (type): The type of the exception that caused the context manager to exit.
        exc_value (Exception): The instance of the exception that caused the context manager to exit.
        traceback (traceback): A traceback object encapsulating the call stack at the point 
                               where the exception originally occurred.
    
        Side-effects:
        Modifies internal state to include variable name mappings and validates logical constraints.
        '''
        super().__exit__(exc_type, exc_value, traceback)
        
        # Get the current frame and then go one level back
        frame = inspect.currentframe().f_back
        
        # Populate varContext from the caller's frame
        self._populate_var_context(frame)

        from .lcUtils import checkLcCorrectness
        # Validate logical constraints
        checkLcCorrectness(self)
       
    @classmethod
    def clear(cls):
        super().clear()
        cls.varNameReversedMap.clear()
    
    def get_constraint_concept(self):
        if self.constraint is None:
            raise ValueError('Constraint has not been defined yet, initialize the Graph by calling with Graph(...) first.')

        return self.constraint
      
    def findRootConceptOrRelation(self, relationConcept):
        """Find the root concept or relation in the hierarchy.
        
        Returns the top-level concept/relation that has no parent (is_a) relations.
        Results are cached in self.cacheRootConcepts.
        
        Args:
            relationConcept: A concept or relation object to find the root for.
            
        Returns:
            The root concept/relation object. Should NOT return a string.
            
        Note:
            ISSUE TRACKING: This method should always return an object, never a string.
            If string is returned, it indicates:
            1. Input relationConcept was a string (caller error)
            2. Cache contains strings (data corruption)
            Both cases should be investigated.
        """
        # Type check: warn if input is a string (input should be concept/relation object)
        if isinstance(relationConcept, str):
            import warnings
            warnings.warn(
                f"findRootConceptOrRelation() received a string '{relationConcept}' instead of a concept/relation object. "
                f"This will cause .name attribute error in caller. Check caller context.",
                RuntimeWarning,
                stacklevel=2
            )
        
        # If the result is already in cache, return it
        if relationConcept in self.cacheRootConcepts:
            cached_result = self.cacheRootConcepts[relationConcept]
            # ISSUE TRACKING: Validate cache doesn't contain strings
            if isinstance(cached_result, str):
                import warnings
                warnings.warn(
                    f"Cache contains string '{cached_result}' for key {relationConcept}. "
                    f"Cache corruption detected. Clearing and recomputing.",
                    RuntimeWarning,
                    stacklevel=2
                )
                # Clear corrupted cache entry and recompute
                del self.cacheRootConcepts[relationConcept]
            else:
                return cached_result

        try:
            isAs = relationConcept.is_a()
        except (AttributeError, TypeError):
            isAs = []
        
        for _isA in isAs:
            parentRelationConcept = _isA.dst
            
            # Recursive call, but result is cached if previously computed
            root = self.findRootConceptOrRelation(parentRelationConcept)
            
            # Store result in cache (validate it's not a string)
            if isinstance(root, str):
                import warnings
                warnings.warn(
                    f"Recursive call returned string '{root}'. This should not happen.",
                    RuntimeWarning,
                    stacklevel=2
                )
            else:
                self.cacheRootConcepts[relationConcept] = root
            return root

        # If the provided concept or relation is root (has no parents)
        # Store result in cache
        self.cacheRootConcepts[relationConcept] = relationConcept
        
        return relationConcept
    
    def findConcept(self, conceptName):
        '''Finds a concept by name in the graph hierarchy.

        This method searches for a concept in the following order:
        1. Current graph's direct concepts
        2. All subgraphs (recursive)
        3. Supergraph (parent) if it exists
        4. Sibling graphs (other subgraphs of the parent)
        
        Args:
            conceptName (str): The name of the concept to find.

        Returns:
            Concept: The concept if found, None otherwise.
        '''
        # 1. Check current graph's direct concepts
        if conceptName in self._concepts:
            return self._concepts[conceptName]
        
        # 2. Check subgraphs recursively
        for subGraphKey in self.subgraphs:
            subGraph = self.subgraphs[subGraphKey]
            
            if conceptName in subGraph.concepts:
                return subGraph.concepts[conceptName]
            
            # Recursive search in nested subgraphs
            found = subGraph.findConcept(conceptName)
            if found is not None:
                return found
        
        # 3. Check supergraph (parent graph)
        if self.sup is not None and isinstance(self.sup, Graph):
            if conceptName in self.sup._concepts:
                return self.sup._concepts[conceptName]
        
        # 4. Check sibling graphs (other subgraphs of parent)
        if self.sup is not None and isinstance(self.sup, Graph):
            for siblingKey in self.sup.subgraphs:
                sibling = self.sup.subgraphs[siblingKey]
                # Skip self
                if sibling is self:
                    continue
                
                if conceptName in sibling.concepts:
                    return sibling.concepts[conceptName]
                
                # Check sibling's subgraphs
                found = sibling.findConcept(conceptName)
                if found is not None:
                    return found
        
        return None
    
    def collectAllConcepts(self, include_subgraphs=True, include_supergraph=True, include_siblings=True):
        '''Collects and returns all concepts from the graph hierarchy.
        
        This method collects concepts from:
        1. Current graph's direct concepts
        2. All subgraphs (if include_subgraphs=True)
        3. Supergraph/parent (if include_supergraph=True)
        4. Sibling graphs (if include_siblings=True)
        
        Args:
            include_subgraphs (bool): Whether to include concepts from subgraphs. Defaults to True.
            include_supergraph (bool): Whether to include concepts from parent graph. Defaults to True.
            include_siblings (bool): Whether to include concepts from sibling graphs. Defaults to True.
        
        Returns:
            OrderedDict: Dictionary mapping concept names to concept objects.
                        Note: If duplicate names exist, later entries overwrite earlier ones.
        '''
        collected = OrderedDict()
        
        # 1. Collect from current graph
        collected.update(self._concepts)
        
        # 2. Collect from subgraphs recursively
        if include_subgraphs:
            for subGraphKey in self.subgraphs:
                subGraph = self.subgraphs[subGraphKey]
                # Recursive collection from subgraphs
                sub_concepts = subGraph.collectAllConcepts(
                    include_subgraphs=True,
                    include_supergraph=False,  # Don't go back up
                    include_siblings=False     # Don't go sideways
                )
                collected.update(sub_concepts)
        
        # 3. Collect from supergraph (parent)
        if include_supergraph and self.sup is not None and isinstance(self.sup, Graph):
            collected.update(self.sup._concepts)
        
        # 4. Collect from sibling graphs
        if include_siblings and self.sup is not None and isinstance(self.sup, Graph):
            for siblingKey in self.sup.subgraphs:
                sibling = self.sup.subgraphs[siblingKey]
                # Skip self
                if sibling is self:
                    continue
                
                # Collect sibling's concepts
                collected.update(sibling.concepts)
                
                # Collect from sibling's subgraphs
                sibling_sub_concepts = sibling.collectAllConcepts(
                    include_subgraphs=True,
                    include_supergraph=False,  # Don't go back up
                    include_siblings=False     # Don't go sideways
                )
                collected.update(sibling_sub_concepts)
        
        return collected
    
    def getAllConceptNames(self, include_subgraphs=True, include_supergraph=True, include_siblings=True):
        '''Returns a list of all concept names in the graph hierarchy.
        
        This is a convenience method that returns just the names from collectAllConcepts().
        
        Args:
            include_subgraphs (bool): Whether to include concepts from subgraphs. Defaults to True.
            include_supergraph (bool): Whether to include concepts from parent graph. Defaults to True.
            include_siblings (bool): Whether to include concepts from sibling graphs. Defaults to True.
        
        Returns:
            list: List of concept names (strings).
        '''
        all_concepts = self.collectAllConcepts(
            include_subgraphs=include_subgraphs,
            include_supergraph=include_supergraph,
            include_siblings=include_siblings
        )
        return list(all_concepts.keys())

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
    def executableLCs(self):
        """Getter for the executable logical constraints.

        Returns:
            dict: Dictionary of executable logical constraints.
        """
        return self._executableLCs

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
    def executableLCsRecursive(self):
        """Generator function that yields executable logical constraints recursively.

        This method goes through all nodes and yields the executable logical constraints 
        if the node is an instance of Graph.

        Yields:
            tuple: A tuple containing key-value pairs of executable logical constraints.
        """
        def func(node):
            yield from node.executableLCs.items()
        yield from self.traversal_apply(func)
        
    @property
    def allLogicalConstrains(self):
        """Generator function that yields all logical constraints.

        This method yields both logical constraints and executable logical constraints.

        Yields:
            tuple: A tuple containing key-value pairs of all logical constraints.
        """
        yield from self.logicalConstrains.items()
        for key, lc in self.executableLCs.items():
            yield (key, lc.innerLC)
        
    @property
    def allLogicalConstrainsRecursive(self):
        """Generator function that yields all logical constraints recursively.

        This method goes through all nodes and yields both logical constraints 
        and executable logical constraints if the node is an instance of Graph.

        Yields:
            tuple: A tuple containing key-value pairs of all logical constraints.
        """
        def func(node):
            if isinstance(node, Graph):
                yield from node.allLogicalConstrains
        yield from self.traversal_apply(func)   
        
    @property
    def executableLCsLabels(self):
        """Getter for the executable logical constraints labels.

        Returns:
            dict: Dictionary of executable logical constraints labels.
        """
        return self._executableLCsLabels
    
    @executableLCsLabels.setter
    def executableLCsLabels(self, value):
        """Sets the executable logical constraints labels.
        
        Args:
            value: The value to set for executable logical constraints labels.
        """
        self._executableLCsLabels = value

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

    def compile_executable(
        self,
        data,
        logic_keyword='constraint',
        logic_label_keyword='label',
        extra_namespace_values={},
        verbose=False
    ):
        """
        Takes a dataset containing logical constraint expressions and compiles them
        as executable constraints. All constraints are wrapped with execute() and
        stored in graph.executableLCs. Labels are stored in graph.executableLCsLabels.
        
        Args:
            data: Iterable of dicts containing keys specified by logic_keyword and logic_label_keyword
            logic_keyword: Key in data items containing constraint string expression
            logic_label_keyword: Key in data items containing the label (True/False)
            extra_namespace_values: Additional variables to add to evaluation namespace
            verbose: If True, print debug information during compilation
            
        Returns:
            List of executable constraint names (e.g., ['ELC0', 'ELC1', ...])
        """
        from .executable import get_full_funcs, LogicDataset
        import importlib
        
        elc_name_list = []
        
        # Ensure we're in graph context for constraint creation
        cls = type(self)
        needs_context = not cls._context or cls._context[-1] is not self
                
        if needs_context:
            with self:
                self._process_executable(
                    data, logic_keyword, logic_label_keyword, 
                    extra_namespace_values, verbose, elc_name_list
                )
        else:
            self._process_executable(
                data, logic_keyword, logic_label_keyword, 
                extra_namespace_values, verbose, elc_name_list
            )
    
        return LogicDataset(
                data,
                elc_name_list,
                logic_keyword=logic_keyword,
                logic_label_keyword=logic_label_keyword
            )

    def _process_executable(
            self, data, logic_keyword, logic_label_keyword, 
            extra_namespace_values, verbose, elc_name_list
        ):
        """
        Internal method that processes each data item to create executable constraints.
        
        For each data item:
        1. Reads the constraint string expression
        2. Compiles and evaluates to create LogicalConstrain
        3. Wraps with execute() if not already wrapped
        4. Stores label in executableLCsLabels dictionary
        
        Args:
            data: Iterable of data items
            logic_keyword: Key for constraint expression in data items
            logic_label_keyword: Key for label in data items  
            extra_namespace_values: Additional namespace variables
            verbose: Debug output flag
            elc_name_list: List to accumulate created constraint names
        """
        from .executable import get_full_funcs, LogicDataset
        from .logicalConstrain import execute
        import importlib
        from ..sensor.pytorch.sensors import ReaderSensor
        
        # Ensure varContext is populated - go back 2 frames to reach the caller of compile_executable
        if self.varContext is None or not self.varContext:
            frame = inspect.currentframe().f_back.f_back
            self._populate_var_context(frame)
        
        for i, data_item in enumerate(data):
            # --- Step 1: Validate data item has required keys ---
            if logic_keyword not in data_item:
                raise ValueError(
                    f'Invalid data_item at index {i}: must contain key {logic_keyword} '
                    f'but instead just found: {data_item.keys()}'
                )
            
            # --- Step 2: Extract the constraint string ---
            lc_string = data_item[logic_keyword]
            
            # --- Step 3: Wrap with execute() if not already wrapped ---
            # All constraints processed here must be executable constraints
            if not lc_string.strip().startswith('execute('):
                lc_string = f'execute({lc_string})'
            
            # --- Step 4: Convert to fully qualified names ---
            # e.g., "execute(andL(...))" -> "domiknows.graph.logicalConstrain.execute(...)"
            lc_string_fmt = get_full_funcs(lc_string)
            
            # --- Step 5: Build namespace for evaluation ---
            # Includes domiknows module, graph variables, and any extra values
            target_namespace = {
                'domiknows': importlib.import_module('domiknows'),
                **(self.varContext or {}),
                **(extra_namespace_values or {}),
                'path': lambda *args: args,
            }
            
            if verbose:
                print(f'Compiling constraint {i}: {lc_string_fmt}')
            
            # --- Step 6: Compile and evaluate the constraint expression ---
            # This creates the execute wrapper object
            try:
                code = compile(lc_string_fmt, f'<executable_{i}>', 'eval')
                elc = eval(code, target_namespace)
            except NameError as e:
                var_name = str(e).split("'")[1]
                raise NameError(
                    f"Variable '{var_name}' used in constraint '{lc_string_fmt}' is not defined. "
                    f"Make sure all variables are defined in the graph context or passed via extra_namespace_values. "
                    f"Available variables: {sorted(target_namespace.keys())}"
                ) from None
            except Exception as e:
                raise Exception(
                    f"Failed to evaluate constraint '{lc_string_fmt}'. "
                    f"Error: {str(e)}"
                ) from None
            
            # --- Step 7: Store label in executableLCsLabels ---
            # Labels indicate expected satisfaction (True/False) for supervised training
            if logic_label_keyword in data_item:
                label = data_item[logic_label_keyword]
                self.executableLCsLabels[elc.lcName] = label
            
            elc_name_list.append(elc.lcName)
            
            new_lc_name = str(elc)
            
            constr_reader_key = LogicDataset.KEYWORD_FMT.format(lc_name=new_lc_name)
            elc.name = constr_reader_key

            self.constraint[elc] = ReaderSensor(
                keyword=constr_reader_key,
                is_constraint=True,
                label=True
            )