
import sys
sys.path.append('../../../../domiknows/')

import torch, argparse, time
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor
from domiknows.program import SolverPOIProgram

from domiknows.sensor.pytorch.learners import TorchLearner
from graph import get_graph

for test_number, test_cases in enumerate([
    [],
    ["--firestations", "4","--atleastaL", "3"],
    ["--firestations", "4","5","6","7","--atmostaL", "2"],
    ["--firestations", "1","2","6","--constraint","ifLnotLexistL"],
    ["--firestations", "1", "--constraint", "orLnotLexistL"]
]):
    from domiknows.sensor import Sensor
    Sensor.clear()

    parser = argparse.ArgumentParser(description='Check a graphical structure and existL constraint in domiknows')
    parser.add_argument('--firestations', dest='firestations', default=[], help='list of cities to be defauled as firestation from 1 to 9', type=int, nargs='+')
    parser.add_argument('--constraint', dest='constraint',default="None", choices=["None","orL", "existL","ifLnotLexistL","orLnotLexistL"], help="Choose a constraint")
    parser.add_argument('--atmostaL', dest='atmostaL',default=10,type=int)
    parser.add_argument('--atleastaL', dest='atleastaL',default=0,type=int)

    args = parser.parse_args(test_cases)
    
    graph, world, city, world_contains_city, neighbor, city1, city2, firestationCity = get_graph(args.constraint,args.atmostaL,args.atleastaL,test_number)
    

    class DummyCityLearner(TorchLearner):
        def __init__(self, *pre, fire_stations=[i+1 for i in range(9)]):
            TorchLearner.__init__(self,*pre)
            self.fire_stations=fire_stations

        def forward(self, x):
            result = torch.zeros(len(x), 2)
            result[:, 1] = -1000
            for i in self.fire_stations:
                result[i-1,1]=1000
            return result

    dataset = [{'world': [0], 'city': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'links': {1: [2, 3, 4, 5], 2: [1, 6], 3: [1], 4: [1], 5: [1], 6: [2, 7, 8, 9], 7: [6], 8: [6], 9: [6]}}]
    world['index'] = ReaderSensor(keyword='world')
    city['index'] = ReaderSensor(keyword='city')
    city[world_contains_city] = EdgeSensor(city['index'], world['index'], relation=world_contains_city, forward=lambda x, _: torch.ones_like(x).unsqueeze(-1))

    def readNeighbors(index, data, arg1, arg2):
        city1, city2 = arg1, arg2
        if city1.getAttribute('index') in data[int(city2.getAttribute('index'))]:
            return True
        else:
            return False

    neighbor[city1.reversed, city2.reversed] = CompositionCandidateReaderSensor(city['index'], keyword='links', relations=(city1.reversed, city2.reversed), forward=readNeighbors)


    city[firestationCity] = DummyCityLearner('index',fire_stations=list(args.firestations))
    program = SolverPOIProgram(graph, poi=[world, city, city[firestationCity], neighbor])

    for datanode in program.populate(dataset=dataset):    
        datanode.inferILPResults() 
        fire_stations=[int(child_node.getAttribute('<' + firestationCity.name + '>',"ILP").item()) for child_node in datanode.getChildDataNodes()]
        if test_number==0:
            assert sum(fire_stations)==0
        if test_number==1:
            assert sum(fire_stations)==3
            assert fire_stations[4-1]==1
        if test_number==2:
            assert sum(fire_stations)==2
            assert fire_stations[4-1]+fire_stations[5-1]+fire_stations[6-1]+fire_stations[7-1]==2
        if test_number==3:
            assert sum(fire_stations)==2
            assert fire_stations[1-1] and fire_stations[6-1]
        if test_number==4:
            assert sum(fire_stations)==1
            fire_stations[1-1]==0
