from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.concept import EnumConcept
from domiknows.graph.logicalConstrain import sameL, differentL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('same_different_test') as graph:
    image = Concept(name='image')

    object_node = Concept(name='object')
    (image_contains_object,) = image.contains(object_node)

    # Color as EnumConcept connected to object_node via is_a
    color = object_node(name='color', ConceptClass=EnumConcept, values=['red', 'blue', 'green'])

    # =========================================================
    # Standalone sameL/differentL constraints.
    # Variables x and y both range over all objects and get aligned
    # row-by-row, so each entity is compared to itself.
    # sameL should be satisfied (each entity's color equals itself).
    # differentL should be violated.
    # =========================================================
    same_color = sameL(color, 'x', 'y', name='same_color')
    diff_color = differentL(color, 'x', 'y', name='diff_color')
