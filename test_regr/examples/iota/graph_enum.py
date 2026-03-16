from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.concept import EnumConcept
from domiknows.graph.logicalConstrain import iotaL, andL, existsL, queryL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('visual_qa_enum') as graph:
    image = Concept(name='image')
    
    object_node = Concept(name='object')
    (image_contains_object,) = image.contains(object_node)
    
    # Object properties
    big = object_node(name='big')
    large = object_node(name='large')
    brown = object_node(name='brown')
    cylinder = object_node(name='cylinder')
    sphere = object_node(name='sphere')
    
    # Spatial relations 
    pair = Concept(name='pair')
    (rel_arg1, rel_arg2) = pair.has_a(arg1=object_node, arg2=object_node)
    
    right_of = pair(name='right_of')
    left_of = pair(name='left_of')
    
    # =========================================================
    # Material as EnumConcept with explicit values
    # =========================================================
    material = EnumConcept(name='material', values=['metal', 'rubber'])

    # =========================================================
    # Query: "What material is the big object that is 
    #         right of the brown cylinder and 
    #         left of the large brown sphere?"
    # =========================================================
    
    
    the_material_answer = queryL(
        material,          # EnumConcept with values=['metal', 'rubber']
        iotaL(
            andL(
                big('z'), # z is the big object
                iotaL(andL(brown('x'), cylinder('x'))), # x is the brown cylinder
                iotaL(andL(large('y'), brown('y'), sphere('y'))), # y is the large brown sphere
                right_of('z', 'x'), # z is right of x
                left_of('z', 'y') # z is left of y
            )
        )
    )