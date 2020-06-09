import json


class CityReader:
    def __init__(self,):
        self.data = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        self.links = [{
            1: [1, 2, 3, 4, 5],
            2: [1, 2, 6],
            3: [1, 3],
            4: [1, 4],
            5: [1, 5],
            6: [6, 7, 8, 9, 2],
            7: [6, 7],
            8: [6, 8],
            9: [6, 9]
        }]

    def run(self,):
        for i in range(len(self.data)):
            item = {'world':[0], 'city': self.data[i], 'links': self.links[i]}
            yield item
