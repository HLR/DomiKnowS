from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import iotaL, andL, existsL, queryL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('visual_qa') as graph:
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
    # Material as multiclass concept with subclasses
    # =========================================================
    material = object_node(name='material')
    
    # Subclasses of material
    metal = material(name='metal')
    rubber = material(name='rubber')

    # =========================================================
    # Query: "What material is the big object that is 
    #         right of the brown cylinder and 
    #         left of the large brown sphere?"
    # =========================================================
    
    # Step 1: THE brown cylinder
    the_brown_cylinder = iotaL(
        andL(brown('x'), cylinder(path='x'))
    )
    
    # Step 2: THE large brown sphere  
    the_large_brown_sphere = iotaL(
        andL(large('y'), brown(path='y'), sphere(path='y'))
    )
    
    # Step 3: THE big object that is right of #1 and left of #2
    the_target_object = iotaL(
        andL(
            big('z'),
            right_of('r1', path=('z', rel_arg1.reversed)),
            the_brown_cylinder,
            left_of('r2', path=('z', rel_arg1.reversed)),
            the_large_brown_sphere
        )
    )
    
    # Step 4: Query material of the target object
    # queryL returns the most probable subclass (metal or rubber)
    the_material_answer = queryL(
        material,          # Parent multiclass concept
        the_target_object  # Entity selection from iotaL
    )
    
    # Alternatively, inline all steps:
    '''
    the_material_answer = queryL(
        material,          # Parent multiclass concept
        iotaL(
            andL(
                big('z'),
                iotaL(andL(brown('x'), cylinder(path='x')))
                andL(brown('x'), cylinder(path='x')),
                left_of('r2', path=('z', rel_arg1.reversed)),
                iotaL(andL(large('y'), brown(path='y'), sphere(path='y')))
            )
        )  # Entity selection from iotaL
    )
    '''