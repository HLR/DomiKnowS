from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import iotaL, andL, existsL, queryL, sumL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('visual_qa') as graph:
    image = Concept(name='image')
    
    object_node = Concept(name='object')
    (image_contains_object,) = image.contains(object_node)
    
    # Object properties
    
    # -- Size ['big', 'large', 'small', 'tiny']
    big = object_node(name='big')
    small = object_node(name='small')
    large = object_node(name='large')
    tiny = object_node(name='tiny')
    
    # -- Color  ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
    brown = object_node(name='brown')
    gray = object_node(name='gray')
    red = object_node(name='red')   
    blue = object_node(name='blue')
    green = object_node(name='green')
    purple = object_node(name='purple')
    cyan = object_node(name='cyan')
    yellow = object_node(name='yellow')
    
    # -- Shape ['cube', 'sphere', 'cylinder']
    cylinder = object_node(name='cylinder')
    sphere = object_node(name='sphere')
    cube = object_node(name='cube')
    
    # -- Texture  ['shiny', 'matte']
    shiny = object_node(name='shiny')
    matte = object_node(name='matte')
    
    # Spatial relations ['left_of', 'right_of', 'front_of', 'behind', 'above', 'below']
    pair = Concept(name='pair')
    (rel_arg1, rel_arg2) = pair.has_a(arg1=object_node, arg2=object_node)
    
   
    
    right_of = pair(name='right_of')
    left_of = pair(name='left_of')
    front_of = pair(name='front_of')
    behind = pair(name='behind')
    above = pair(name='above')
    below = pair(name='below')

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
    #         left of the big brown sphere?"
    # =========================================================
    
    the_material_answer = queryL(
        material,          # Parent multiclass concept
        iotaL(
            andL(
                big('z'), # z is the big object
                iotaL(andL(brown('x'), cylinder('x'))), # x is the brown cylinder
                iotaL(andL(big('y'), brown('y'), sphere('y'))), # y is the big brown sphere
                left_of('z', 'y'), # z is left of y
                right_of('z', 'x'), # z is right of x
            )
        )  # Entity selection from iotaL
    )
    
    # =========================================================
    # Query: "How many objects are there that are right of the big brown sphere?
    # =========================================================
    
    count_right_of_big_brown_sphere = queryL(
        sumL(
            andL(
                big('z'), # z is the big object
                brown('z'), # z is brown
                sphere('z'), # z is a sphere
                right_of('z', 'y') # z is right of y
            )
        )
    )