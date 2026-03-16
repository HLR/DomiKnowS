"""
Auto-generated DomiKnowS graph with executable constraints.
Source dataset: C:\Users\auszok\git\RelationalGraph\test_regr\Clever\20_examples_string_CLEVR.json
Total questions: 20  |  Encoded: 0  |  Failed: 20
"""

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


with graph:

    # Q1: What material is the big object that is right of the brown cylinder and left of the large brown sphere?
    # FAILED — could not encode this question

    # Q2: How big is the brown shiny sphere?
    # FAILED — could not encode this question

    # Q3: What is the shape of the brown thing to the right of the large brown metallic thing that is on the left side of the brown sphere?
    # FAILED — could not encode this question

    # Q4: The brown object that is the same shape as the green shiny thing is what size?
    # FAILED — could not encode this question

    # Q5: What color is the tiny matte block left of the blue block?
    # FAILED — could not encode this question

    # Q6: There is a rubber thing that is the same color as the cylinder; what shape is it?
    # FAILED — could not encode this question

    # Q7: The matte ball that is the same size as the gray rubber object is what color?
    # FAILED — could not encode this question

    # Q8: There is a small matte block that is on the left side of the large rubber thing that is left of the gray ball; what is its color?
    # FAILED — could not encode this question

    # Q9: There is a gray object that is in front of the rubber cube behind the metallic ball behind the small brown thing; what shape is it?
    # FAILED — could not encode this question

    # Q10: There is a thing that is both to the right of the big rubber cube and left of the small gray ball; what color is it?
    # FAILED — could not encode this question

    # Q11: What color is the metallic thing behind the big metal object?
    # FAILED — could not encode this question

    # Q12: What is the shape of the object that is both behind the large metallic thing and left of the gray metallic sphere?
    # FAILED — could not encode this question

    # Q13: The other object that is the same shape as the large metal thing is what color?
    # FAILED — could not encode this question

    # Q14: What material is the tiny cube that is to the right of the blue thing?
    # FAILED — could not encode this question

    # Q15: What is the material of the small thing to the left of the small purple block in front of the small cyan rubber cube?
    # FAILED — could not encode this question

    # Q16: What is the shape of the tiny cyan matte thing?
    # FAILED — could not encode this question

    # Q17: There is a small cube on the right side of the tiny blue thing that is behind the tiny cyan matte thing; what is its color?
    # FAILED — could not encode this question

    # Q18: There is a metallic block that is the same size as the cyan rubber cube; what color is it?
    # FAILED — could not encode this question

    # Q19: There is a cyan metal thing behind the cyan metal ball; what shape is it?
    # FAILED — could not encode this question

    # Q20: The purple ball that is the same material as the tiny cyan cylinder is what size?
    # FAILED — could not encode this question
