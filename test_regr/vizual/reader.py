"""
reader.py

Test data for the visual-constraint example.

Objects:
    1 = small blue cube       (THE small blue cube)
    2 = large red sphere      (THE target — material=metal)
    3 = small green cylinder   (distractor — material=rubber)

Spatial:
    (2, 1) = right_of   (object 2 is right of object 1)
"""


class SceneReader:
    def __init__(self):
        self.images = [[0]]
        self.objects = [[1, 2, 3]]
        self.pairs = [
            [(i, j) for i in [1, 2, 3] for j in [1, 2, 3] if i != j]
        ]

        # --- ground-truth object attributes ---
        self.small_objects    = [{1, 3}]      # objects 1,3 are small
        self.large_objects    = [{2}]         # object 2 is large
        self.red_objects      = [{2}]         # object 2 is red
        self.green_objects    = [{3}]         # object 3 is green
        self.blue_objects     = [{1}]         # object 1 is blue
        self.cube_objects     = [{1}]         # object 1 is cube
        self.sphere_objects   = [{2}]         # object 2 is sphere
        self.cylinder_objects = [{3}]         # object 3 is cylinder

        # --- ground-truth spatial relations ---
        self.right_of_pairs = [{(2, 1)}]      # obj 2 is right of obj 1
        self.left_of_pairs  = [set()]         # none explicitly

        # --- ground-truth materials ---
        # (used by EnumConcept learner: index 0=metal, 1=rubber)
        self.metal_objects  = [{2}]           # object 2 is metal
        self.rubber_objects = [{3}]           # object 3 is rubber

    def run(self):
        for i in range(len(self.objects)):
            yield {
                "image":    self.images[i],
                "objects":  self.objects[i],
                "pairs":    self.pairs[i],
                # attributes
                "small":    self.small_objects[i],
                "large":    self.large_objects[i],
                "red":      self.red_objects[i],
                "green":    self.green_objects[i],
                "blue":     self.blue_objects[i],
                "cube":     self.cube_objects[i],
                "sphere":   self.sphere_objects[i],
                "cylinder": self.cylinder_objects[i],
                # spatial
                "right_of": self.right_of_pairs[i],
                "left_of":  self.left_of_pairs[i],
                # material
                "metal":    self.metal_objects[i],
                "rubber":   self.rubber_objects[i],
            }