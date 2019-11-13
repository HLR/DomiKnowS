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
    print(myMap.all_cities())
    for a in myMap.all_cities():
       print(myMap.get_neighbors(a))
