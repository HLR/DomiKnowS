"""
visual_reasoning_graph.py
"""

from domiknows.graph import Graph, Concept, Relation, equivalenceL, notL
from domiknows.graph.concept import EnumConcept

DEFAULT_COLORS    = ["red","green","blue","yellow","brown","gray","cyan","purple","orange","white","black"]
DEFAULT_SHAPES    = ["cube","sphere","cylinder","cone","torus"]
DEFAULT_MATERIALS = ["metal","rubber","glass","wood","plastic"]
DEFAULT_SIZES     = ["small","large"]


def build_visual_reasoning_graph(
    graph_name: str = "visual_reasoning",
    colors: list | None = None,
    shapes: list | None = None,
    materials: list | None = None,
    sizes: list | None = None,
):
    colors    = colors    or DEFAULT_COLORS
    shapes    = shapes    or DEFAULT_SHAPES
    materials = materials or DEFAULT_MATERIALS
    sizes     = sizes     or DEFAULT_SIZES

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph(graph_name) as graph:

        # ==============================================================
        # 1. Image / scene
        # ==============================================================
        image = Concept(name="image")

        # ==============================================================
        # 2. Object
        # ==============================================================
        object_node = Concept(name="object")
        (image_contains_object,) = image.contains(object_node)

        # ==============================================================
        # 3. Attributes
        # ==============================================================
        color_concepts = {c:  object_node(name=c)  for c in colors}
        shape_concepts = {s:  object_node(name=s)  for s in shapes}
        size_concepts  = {sz: object_node(name=sz) for sz in sizes}
        material = EnumConcept(name="material", values=materials)

        # ==============================================================
        # 4. Directed pair nodes  (reification + OWL inverse properties)
        #
        # Reification: instead of a bare binary predicate left_of(A,B),
        # the relation is promoted to a first-class Concept node (pair)
        # that has_a two object participants.  This lets the relation
        # carry its own sub-concepts, sensors, and ILP variables.
        #
        # OWL inverse properties: in OWL, leftOf and rightOf are two
        # *distinct* object properties related by owl:inverseOf.
        # A single undirected pair node cannot represent this — swapping
        # argument slot names ('a','b') → ('b','a') on the same node
        # resolves to the same ILP variable, making the inverse constraint
        # identical to the opposite constraint and causing infeasibility.
        #
        # The fix is two separate reified nodes:
        #   pair_forward — subject → referent: "A is left of B"
        #   pair_reverse — referent → subject: same objects, other direction
        #
        # Now left_of on pair_forward and right_of on pair_reverse are
        # distinct ILP variables, so both constraints are satisfiable:
        #   opposite: left_of_fwd(a,b) ↔ ¬right_of_fwd(a,b)
        #   inverse:  left_of_fwd(a,b) ↔  right_of_rev(a,b)
        # ==============================================================
        pair_forward = Concept(name="pair_forward")
        (rel_arg1_fwd, rel_arg2_fwd) = pair_forward.has_a(arg1=object_node, arg2=object_node)

        pair_reverse = Concept(name="pair_reverse")
        (rel_arg1_rev, rel_arg2_rev) = pair_reverse.has_a(arg1=object_node, arg2=object_node)

        # spatial relations — forward direction
        left_of_fwd   = pair_forward(name="left_of")
        right_of_fwd  = pair_forward(name="right_of")
        above_fwd     = pair_forward(name="above")
        below_fwd     = pair_forward(name="below")
        in_front_fwd  = pair_forward(name="in_front_of")
        behind_fwd    = pair_forward(name="behind")

        # spatial relations — reverse direction
        left_of_rev   = pair_reverse(name="left_of")
        right_of_rev  = pair_reverse(name="right_of")
        above_rev     = pair_reverse(name="above")
        below_rev     = pair_reverse(name="below")
        in_front_rev  = pair_reverse(name="in_front_of")
        behind_rev    = pair_reverse(name="behind")

    ctx = {
        "graph":  graph,
        "image":  image,
        "object": object_node,
        "image_contains_object": image_contains_object,
        # forward pair
        "pair_forward": pair_forward,
        "rel_arg1_fwd": rel_arg1_fwd,
        "rel_arg2_fwd": rel_arg2_fwd,
        # reverse pair
        "pair_reverse": pair_reverse,
        "rel_arg1_rev": rel_arg1_rev,
        "rel_arg2_rev": rel_arg2_rev,
        # attributes
        "colors":   color_concepts,
        "shapes":   shape_concepts,
        "sizes":    size_concepts,
        "material": material,
        # spatial — forward (reification of left_of(a,b) as pair_forward(arg1=a,arg2=b) with left_of property)
        "left_of":     left_of_fwd,
        "right_of":    right_of_fwd,
        "above":       above_fwd,
        "below":       below_fwd,
        "in_front_of": in_front_fwd,
        "behind":      behind_fwd,
        # spatial — reverse (OWL inverse properties)
        "left_of_rev":     left_of_rev,
        "right_of_rev":    right_of_rev,
        "above_rev":       above_rev,
        "below_rev":       below_rev,
        "in_front_of_rev": in_front_rev,
        "behind_rev":      behind_rev,
    }
    return graph, ctx