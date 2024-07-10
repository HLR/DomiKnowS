
import sys
sys.path.append('../../../../domiknows/')
from domiknows.graph import Graph
from domiknows.sensor import Sensor
import torch, argparse, time
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor
from domiknows.program import SolverPOIProgram

from domiknows.sensor.pytorch.learners import TorchLearner
from graph import get_graph


Sensor.clear()

parser = argparse.ArgumentParser(description='Check a graphical structure and existL constraint in domiknows')
parser.add_argument('--firestations', dest='firestations', default=[4], help='list of cities to be defauled as firestation from 1 to 9', type=int, nargs='+')
parser.add_argument('--constraint', dest='constraint',default="None", choices=["None","orL", "existL","ifLnotLexistL","orLnotLexistL"], help="Choose a constraint")
parser.add_argument('--atmostaL', dest='atmostaL',default=10,type=int)
parser.add_argument('--atleastaL', dest='atleastaL',default=3,type=int)

args = parser.parse_args()

graph, world, city, world_contains_city, neighbor, city1, city2, firestationCity = get_graph(args.constraint,args.atmostaL,args.atleastaL)


#Graph.clear()
#Sensor.clear()

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

    print("before inference")
    print("city index:     :",list(range(1,10,1)))
    print("is firestation? :", [int(child_node.getAttribute('<' + firestationCity.name + '>',"local/softmax")[1].item()) for child_node in datanode.getChildDataNodes()])

    datanode.inferILPResults() 
    
    print("\nafter inference")
    print("city index:     :",list(range(1,10,1)))
    print("is firestation? :", [int(child_node.getAttribute('<' + firestationCity.name + '>',"ILP").item()) for child_node in datanode.getChildDataNodes()])
