from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.concept import EnumConcept
from domiknows.graph import (
    ifL, andL, orL, nandL, norL, xorL, notL, equivalenceL,
    eqL, fixedL, forAllL,
    existsL, atLeastL, atMostL, exactL,
    existsAL, atLeastAL, atMostAL, exactAL,
    greaterL, greaterEqL, lessL, lessEqL, equalCountsL,
    sumL, iotaL, queryL,
    execute
)

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('visual_qa') as graph:
    image = Concept(name='image')
    
    object_node = Concept(name='object')
    (image_contains_object,) = image.contains(object_node)
    
    # Attribute concepts (EnumConcepts give us dot‑notation access)
    size = EnumConcept('size', values=['big', 'large', 'small', 'tiny'])
    color = EnumConcept('color', values=[
        'gray', 'red', 'blue', 'green', 'brown',
        'purple', 'cyan', 'yellow'
    ])
    shape = EnumConcept('shape', values=['cube', 'sphere', 'cylinder'])
    texture = EnumConcept('texture', values=['shiny', 'matte'])
    material = EnumConcept('material', values=['metal', 'rubber'])
    
    # Spatial relations ['left_of', 'right_of', 'front_of', 'behind', 'above', 'below']
    pair = Concept(name='pair')
    (rel_arg1, rel_arg2) = pair.has_a(arg1=object_node, arg2=object_node)
    
    # Define spatial relation concepts using the pair relation
    right_of = pair(name='right_of')
    left_of = pair(name='left_of')
    front_of = pair(name='front_of')
    behind = pair(name='behind')
    above = pair(name='above')
    below = pair(name='below')

    # =========================================================
    # Material as multiclass concept with subclasses
    # =========================================================
    material = EnumConcept('material', values=['metal', 'rubber'])
    
    # Subclasses of material
    metal = material(name='metal')
    rubber = material(name='rubber')
