class VisualQAReader:
    """
    Test data for visual QA iotaL constraints.
    
    Objects:
        1 = brown cylinder (THE brown cylinder)
        2 = large brown sphere (THE large brown sphere)  
        3 = big target object, right of 1, left of 2 (THE target object)
        4 = other/distractor object
    
    Spatial Relations:
        (3, 1) = right_of  (object 3 is right of object 1)
        (3, 2) = left_of   (object 3 is left of object 2)
    """
    
    def __init__(self):
        # Image/scene container
        self.images = [[0]]
        
        # Object IDs
        self.objects = [[1, 2, 3, 4]]
        
        # Pairs for spatial relations (all possible object pairs)
        self.pairs = [[(i, j) for i in [1, 2, 3, 4] for j in [1, 2, 3, 4] if i != j]]
        
        # Ground truth object properties
        self.big_objects = [{3}]           # Object 3 is big
        self.large_objects = [{2}]         # Object 2 is large
        self.brown_objects = [{1, 2}]      # Objects 1, 2 are brown
        self.cylinder_objects = [{1}]      # Object 1 is cylinder
        self.sphere_objects = [{2}]        # Object 2 is sphere
        
        # Ground truth spatial relations
        # right_of[(a,b)] means a is right of b
        self.right_of_pairs = [{(3, 1)}]   # Object 3 is right of object 1 (brown cylinder)
        self.left_of_pairs = [{(3, 2)}]    # Object 3 is left of object 2 (large brown sphere)
        
        # Material for target object
        self.materials = [{3: 'metal'}]

    def run(self):
        for i in range(len(self.objects)):
            item = {
                'image': self.images[i],
                'objects': self.objects[i],
                'pairs': self.pairs[i],
                'big': self.big_objects[i],
                'large': self.large_objects[i],
                'brown': self.brown_objects[i],
                'cylinder': self.cylinder_objects[i],
                'sphere': self.sphere_objects[i],
                'right_of': self.right_of_pairs[i],
                'left_of': self.left_of_pairs[i],
                'materials': self.materials[i]
            }
            yield item