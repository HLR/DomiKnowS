import json


class CityReader:

    def __init__(self, path):
        self.filename = path
        self.data = {}
        with open(self.filename) as json_file:
            self.data = json.load(json_file)

    def all_cities(self):

        my_map = []
        for c in self.data:
            my_map.append(c)
        return my_map

    def get_neighbors(self, city):
        neighbors = []
        for n in self.data[city]:
            neighbors.append(n)
        return neighbors


if __name__ == "__main__":

    myMap = CityReader("city.json")
    assert myMap.all_cities() == ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    assert myMap.get_neighbors('1') == ['2', '3', '4', '5']
    assert myMap.get_neighbors('2') == ['1']
    assert myMap.get_neighbors('3') == ['1']
    assert myMap.get_neighbors('4') == ['1']
    assert myMap.get_neighbors('6') == ['7','8','9']
