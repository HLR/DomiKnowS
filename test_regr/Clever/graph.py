try:
    from .dataset import g_attribute_concepts, g_relational_concepts
    from .execution import create_execution_for_question
except ImportError:
    from dataset import g_attribute_concepts, g_relational_concepts
    from execution import create_execution_for_question

# {'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
#  'material': ['rubber', 'metal'],
#  'shape': ['cube', 'sphere', 'cylinder'],
#  'size': ['small', 'large']}

def create_graph(dataset, return_graph_text=False, include_query_questions=False):
    """
    Create the DomiKnows graph for CLEVR dataset.
    
    Args:
        dataset: The CLEVR dataset
        return_graph_text: Whether to return the graph text
        include_query_questions: Whether to include query-type questions (uses queryL with iotaL)
    """
    graph_text = """from domiknows.graph import Graph, Concept
from domiknows.graph.logicalConstrain import ifL, andL, existsL, iotaL, queryL

with Graph('image_graph') as graph:

\timage = Concept(name='image')
\tobj = Concept(name='obj')
\timage_object_contains, = image.contains(obj)\n\n"""

    if include_query_questions:
        # Create parent attribute concepts and their subclasses for queryL
        for attr, values in g_attribute_concepts.items():
            # Create parent concept (e.g., material = obj(name='material'))
            graph_text += f"\t{attr} = obj(name='{attr}')\n"
            # Create subclasses (e.g., metal = material(name='metal'))
            for val in values:
                graph_text += f"\t{val} = {attr}(name='{val}')\n"
            graph_text += '\n'
    else:
        # Original: flat attribute concepts (without parent hierarchy)
        for attr, values in g_attribute_concepts.items():
            for val in values:
                graph_text += f"\t{val} = obj(name='{val}')\n"
            graph_text += '\n'

    # Add relational concepts
    for attr, values in g_relational_concepts.items():
        graph_text += "\trelaton_2_obj = Concept('relation_2_objects')\n"
        graph_text += "\t(obj1, obj2) = relaton_2_obj.has_a(arg1 = obj, arg2 = obj)\n"
        for val in values:
            graph_text += f"\t{val} = relaton_2_obj(name='{val}')\n"
        graph_text += '\n'
    
    # Add same_attribute relations if needed
    if include_query_questions:
        for attr in ['size', 'color', 'material', 'shape']:
            graph_text += f"\tsame_{attr} = relaton_2_obj(name='same_{attr}')\n"
        graph_text += '\n'
            
    executions = []
    query_types = []
    
    for i in range(len(dataset)):
        current_instance = dataset[i]
        program = current_instance.get('program', [])
        question_raw = current_instance.get('question_raw', '')
        
        execution, query_type = create_execution_for_question(program, i)
        
        if " or " in question_raw:
            print(f"Found 'or' in question {i}")
        
        executions.append(execution)
        query_types.append(query_type)

    local_vars = {}
    exec(graph_text, {}, local_vars)
    
    # Build attribute names dict
    attribute_names_dict = {}
    
    if include_query_questions:
        # Include parent concepts for queryL
        for attr, values in g_attribute_concepts.items():
            if attr in local_vars:
                attribute_names_dict[attr] = local_vars[attr]
            for val in values:
                if val in local_vars:
                    attribute_names_dict[val] = local_vars[val]
    else:
        for attr, values in g_attribute_concepts.items():
            for val in values:
                if val in local_vars:
                    attribute_names_dict[val] = local_vars[val]
    
    for attr, values in g_relational_concepts.items():
        for val in values:
            if val in local_vars:
                attribute_names_dict[val] = local_vars[val]
    
    if include_query_questions:
        for attr in ['size', 'color', 'material', 'shape']:
            key = f'same_{attr}'
            if key in local_vars:
                attribute_names_dict[key] = local_vars[key]

    if return_graph_text:
        return (
            executions, 
            local_vars["graph"], 
            local_vars["image"], 
            local_vars["obj"], 
            local_vars["image_object_contains"], 
            local_vars["obj1"], 
            local_vars["obj2"], 
            local_vars["relaton_2_obj"], 
            attribute_names_dict, 
            graph_text,
            query_types
        )
    
    return (
        executions, 
        local_vars["graph"], 
        local_vars["image"], 
        local_vars["obj"], 
        local_vars["image_object_contains"], 
        local_vars["obj1"], 
        local_vars["obj2"], 
        local_vars["relaton_2_obj"], 
        attribute_names_dict,
        query_types
    )


def create_graph_for_query_questions(dataset, return_graph_text=False):
    """
    Specialized graph creation for query-type questions only.
    Uses queryL with iotaL for selecting unique objects and querying attributes.
    """
    return create_graph(dataset, return_graph_text=return_graph_text, include_query_questions=True)