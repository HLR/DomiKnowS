import json


class CityReader:
    def __init__(self,):
        self.data = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        self.links = [{
            1: [2, 3, 4, 5],
            2: [1, 6],
            3: [1],
            4: [1],
            5: [1],
            6: [7, 8, 9, 2],
            7: [6],
            8: [6],
            9: [6]
        }]

    def run(self,):
        for i in range(len(self.data)):
            item = {'world': self.data[i], 'links': self.links[i]}
            yield item
