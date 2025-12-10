from domiknows.graph import Graph, Concept
from domiknows.graph.logicalConstrain import iotaL, andL, existsL

with Graph('visual_qa') as graph:
    object_node = Concept(name='object')
    
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
    
    # Material property (what we want to query)
    material = object_node(name='material') 

    # =========================================================
    # Query: "What material is the big object that is 
    #         right of the brown cylinder and 
    #         left of the large brown sphere?"
    # =========================================================
    
    # Step 1: THE brown cylinder
    the_brown_cylinder = iotaL(
        andL(brown('x'), cylinder('x'))
    )
    
    # Step 2: THE large brown sphere  
    the_large_brown_sphere = iotaL(
        andL(large('y'), brown('y'), sphere('y'))
    )
    
    # Step 3: THE big object that is right of #1 and left of #2
    the_target_object = iotaL(
        andL(
            big('z'),
            right_of('r1', path=('z', rel_arg1.reversed)),
            the_brown_cylinder,  # nested iota for arg2 of right_of
            left_of('r2', path=('z', rel_arg1.reversed)),
            the_large_brown_sphere  # nested iota for arg2 of left_of
        )
    )