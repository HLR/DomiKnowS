"""
graph.py

Scene graph for the visual-constraint example.

Query: "What material is the large red sphere
        that is right of the small blue cube?"

Objects:
  1 = small blue cube       (THE small blue cube)
  2 = large red sphere      (THE target — material=metal)
  3 = small green cylinder   (distractor — material=rubber)

Spatial:
  (2,1) = right_of   →  object 2 is right of object 1

Answer: metal
"""

from domiknows.graph.visual.visual_reasoning_graph import build_visual_reasoning_graph
from domiknows.graph.logicalConstrain import iotaL, andL, queryL

# ── 1. Build generic graph ───────────────────────────────────────────
graph, ctx = build_visual_reasoning_graph(
    graph_name="visual_constraint_example",
    colors=["red", "green", "blue"],
    shapes=["cube", "sphere", "cylinder"],
    materials=["metal", "rubber"],
    sizes=["small", "large"],
)

# Unpack handles we need
image                = ctx["image"]
object_node          = ctx["object"]
image_contains_object = ctx["image_contains_object"]
pair                 = ctx["pair_forward"]
rel_arg1             = ctx["rel_arg1_fwd"]
rel_arg2             = ctx["rel_arg2_fwd"]

# Attribute sub-concepts
red      = ctx["colors"]["red"]
green    = ctx["colors"]["green"]
blue     = ctx["colors"]["blue"]
cube     = ctx["shapes"]["cube"]
sphere   = ctx["shapes"]["sphere"]
cylinder = ctx["shapes"]["cylinder"]
small    = ctx["sizes"]["small"]
large    = ctx["sizes"]["large"]

# Spatial relations
right_of = ctx["right_of"]
left_of  = ctx["left_of"]

# Material EnumConcept
material = ctx["material"]

with graph:

    # Step 1: THE small blue cube
    the_small_blue_cube = iotaL(
        andL(small('x'), blue(path='x'), cube(path='x')),
        name="the_small_blue_cube",
    )

    # Step 2: THE large red sphere that is right of the small blue cube
    the_target_object = iotaL(
        andL(
            large('z'),
            red(path='z'),
            sphere(path='z'),
            right_of('r1', path=('z', rel_arg1.reversed)),
            the_small_blue_cube,
        ),
        name="the_target_object",
    )

    # Step 3: Query material of the target
    the_material_answer = queryL(
        material,
        the_target_object,
        name="the_material_answer",
    )
