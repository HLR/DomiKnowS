class SameDifferentReader:
    """
    Test data for sameL and differentL constraint testing.

    Objects:
        1 = red
        2 = red   (same color as 1)
        3 = blue  (different color from 1 and 2)
    """

    def __init__(self):
        self.images = [[0]]
        self.objects = [[1, 2, 3]]

    def run(self):
        for i in range(len(self.objects)):
            item = {
                'image': self.images[i],
                'objects': self.objects[i],
            }
            yield item
