"""
visual_reasoning_graph.py

Generic visual-reasoning ontology for DomiKnowS.
Defines reusable concepts, relations, and enumeration types
for any image-based QA or scene-understanding task.

Usage:
    from visual_reasoning_graph import build_visual_reasoning_graph
    graph, ctx = build_visual_reasoning_graph()
"""

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.concept import EnumConcept
from domiknows.graph.relation import disjoint


# ---------------------------------------------------------------------------
# Default enum value lists – override via kwargs
# ---------------------------------------------------------------------------
DEFAULT_COLORS = [
    "red", "green", "blue", "yellow", "brown",
    "gray", "cyan", "purple", "orange", "white", "black",
]

DEFAULT_SHAPES = [
    "cube", "sphere", "cylinder", "cone", "torus",
]

DEFAULT_MATERIALS = [
    "metal", "rubber", "glass", "wood", "plastic",
]

DEFAULT_SIZES = ["small", "large"]


def build_visual_reasoning_graph(
    graph_name: str = "visual_reasoning",
    colors: list | None = None,
    shapes: list | None = None,
    materials: list | None = None,
    sizes: list | None = None,
):
    """
    Build and return a DomiKnowS Graph populated with generic
    visual-reasoning concepts, attributes, and spatial relations.

    Returns
    -------
    graph : Graph
        The fully-constructed DomiKnowS graph.
    ctx : dict
        Maps concept/relation names to their objects for sensor
        attachment and constraint definition.
    """
    colors    = colors    or DEFAULT_COLORS
    shapes    = shapes    or DEFAULT_SHAPES
    materials = materials or DEFAULT_MATERIALS
    sizes     = sizes     or DEFAULT_SIZES

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph(graph_name) as graph:

        # ==================================================================
        # 1. Image / scene — the batch root
        # ==================================================================
        image = Concept(name="image")

        # ==================================================================
        # 2. Object — individual detected entity
        # ==================================================================
        object_node = Concept(name="object")
        (image_contains_object,) = image.contains(object_node)

        # ==================================================================
        # 3. Attribute sub-concepts (is_a object)
        #    Each value becomes a boolean sub-concept of object.
        # ==================================================================
        # --- colors ---
        color_concepts = {}
        for c in colors:
            color_concepts[c] = object_node(name=c)

        # --- shapes ---
        shape_concepts = {}
        for s in shapes:
            shape_concepts[s] = object_node(name=s)

        # --- sizes ---
        size_concepts = {}
        for sz in sizes:
            size_concepts[sz] = object_node(name=sz)

        # --- material as EnumConcept (for queryL) ---
        material = EnumConcept(name="material", values=materials)

        # ==================================================================
        # 4. Pair — binary relation between two objects
        # ==================================================================
        pair = Concept(name="pair")
        (rel_arg1, rel_arg2) = pair.has_a(arg1=object_node, arg2=object_node)

        # --- spatial relation sub-types ---
        left_of   = pair(name="left_of")
        right_of  = pair(name="right_of")
        above     = pair(name="above")
        below     = pair(name="below")
        in_front  = pair(name="in_front_of")
        behind_of = pair(name="behind")

        # Mutual exclusion among opposite relations is handled by constraints

    # ------------------------------------------------------------------
    # Collect everything into a context dict
    # ------------------------------------------------------------------
    ctx = {
        "graph": graph,
        "image": image,
        "object": object_node,
        "image_contains_object": image_contains_object,
        "pair": pair,
        "rel_arg1": rel_arg1,
        "rel_arg2": rel_arg2,
        # attributes
        "colors": color_concepts,       # dict[str, Concept]
        "shapes": shape_concepts,       # dict[str, Concept]
        "sizes": size_concepts,         # dict[str, Concept]
        "material": material,           # EnumConcept
        # spatial relations
        "left_of": left_of,
        "right_of": right_of,
        "above": above,
        "below": below,
        "in_front_of": in_front,
        "behind": behind_of,
    }
    return graph, ctx