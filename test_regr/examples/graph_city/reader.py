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
        
        # Pre-defined attributes for testing (can be made learnable)
        # City IDs that have specific services
        self.main_firestations = [{1}]  # City 1 is main firestation
        self.ancillary_firestations = [{6}]  # City 6 is ancillary firestation
        self.emergency_services = [{1, 6, 3}]  # Cities 1, 6, 3 have emergency services
        self.grocery_shops = [{2, 4, 5, 7}]  # Cities 2, 4, 5, 7 have grocery shops

    def run(self,):
        for i in range(len(self.data)):
            item = {
                'world': [0], 
                'city': self.data[i], 
                'links': self.links[i],
                'main_firestations': self.main_firestations[i],
                'ancillary_firestations': self.ancillary_firestations[i],
                'emergency_services': self.emergency_services[i],
                'grocery_shops': self.grocery_shops[i]
            }
            yield item