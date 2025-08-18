from dataset import g_attribute_concepts, g_relational_concepts
from pprint import pprint
from execution import create_execution_existL
import os

# {'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
#  'material': ['rubber', 'metal'],
#  'shape': ['cube', 'sphere', 'cylinder'],
#  'size': ['small', 'large']}


# {'spatial_relation': ['left', 'right', 'front', 'behind'] }

def create_graph(dataset,return_graph_text=False):
    graph_text = """from domiknows.graph import Graph, Concept\nfrom domiknows.graph.logicalConstrain import ifL, andL, existsL\n\nwith Graph('image_graph') as graph:\n
\timage = Concept(name='image')
\tobj = Concept(name='obj')
\timage_object_contains, = image.contains(obj)\n\n"""

    for attr, values in g_attribute_concepts.items():
        for val in values:
            prop_name = f"is_{val}"
            graph_text += f"\t{prop_name} = obj(name='{prop_name}')\n"
        graph_text+='\n'

    for attr, values in g_relational_concepts.items():
        graph_text += "\trelaton_2_obj = Concept('relation_2_objects')\n"
        graph_text += "\t(obj1, obj2) = relaton_2_obj.has_a(arg1 = obj, arg2 = obj)\n"
        for val in values:
            relation_name = f"is_{val}"
            graph_text += f"\t{relation_name} = relaton_2_obj(name='{relation_name}')\n"
        graph_text+='\n'
            
    executions = []
    for i in range(len(dataset)):
        current_instance = dataset[i]
        execution = create_execution_existL(current_instance['program'],i)
        #graph_text+="\n\t"+execution
        #print(execution)
        if " or " in current_instance["question_raw"]:
            print("Found or")
        executions.append(execution)

    # print(graph_text)
    local_vars = {}
    exec(graph_text, {}, local_vars)

    #print("variables:")
    #for name, value in local_vars.items():
    #    print(f"{name} = {value!r}")
    if return_graph_text:
        return executions, local_vars["graph"], local_vars["image"], local_vars["obj"], local_vars["image_object_contains"], local_vars["obj1"], local_vars["obj2"], local_vars["relaton_2_obj"], {attr_name: local_vars[attr_name] for attr_name in local_vars.keys() if "is_" in attr_name}, graph_text
    return executions, local_vars["graph"],local_vars["image"],local_vars["obj"],local_vars["image_object_contains"], local_vars["obj1"], local_vars["obj2"], local_vars["relaton_2_obj"], {attr_name:local_vars[attr_name] for attr_name in local_vars.keys() if "is_" in attr_name}