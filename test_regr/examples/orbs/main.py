import sys, torch, argparse, time
sys.path.append('../../domiknows/')


from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor
from domiknows.program import SolverPOIProgram

from domiknows.sensor.pytorch.learners import TorchLearner
from graph import get_graph
parser = argparse.ArgumentParser(description='Check a csp structure and atmostal/atleastal constraint in domiknows')
parser.add_argument('--colored', dest='colored', default=True,action='store_true',help="color every orb")
parser.add_argument('--constraint', dest='constraint',default="foreach_IfL_atleastL_atmostL",
                    choices=["None","simple_constraint","foreach_bag_existsL","foreach_bag_existsL_notL","foreach_bag_atLeastAL","foreach_bag_atMostAL","foreach_IfL_atleastL_bag_existsL_notL","foreach_IfL_atleastL_bag_existsL","foreach_IfL_atleastL_atmostL"], help="Choose a constraint")
parser.add_argument('--atmostaL', dest='atmostaL',default=1,type=int)
parser.add_argument('--atleastaL', dest='atleastaL',default=5,type=int)

args = parser.parse_args()

graph, csp,csp_range,orbs,csp_contains_csp_range,csp_range_contains_orbs,colored_orbs,enforce_csp_range = get_graph(args.constraint,args.atmostaL,args.atleastaL)
graph.detach()

class DummyCityLearner(TorchLearner):
    def __init__(self, *pre, colored=args.colored):
        TorchLearner.__init__(self,*pre)
        self.colored=colored

    def forward(self, x):
        result = torch.zeros(len(x), 2)
        result[:, 1] = 1000*self.colored-1000*(1-self.colored)
        return result
    
class DummyCityLearner2(TorchLearner):
    def forward(self, x):
        result = torch.zeros(len(x), 2)
        result[:, 1] = 1000
        return result

dataset = [{'csp': [0], 'csp_range': [0, 1, 2, 3, 4], 'orbs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]},]

csp['index'] = ReaderSensor(keyword='csp')
csp_range['index'] = ReaderSensor(keyword='csp_range')
orbs['index'] = ReaderSensor(keyword='orbs')

def connect(x,y):
    connection=torch.zeros((len(x),len(y)))
    for i in x.tolist():
        for j in x.tolist():
            connection[i-1,j+5*(i-1)]=1
    return connection.T

csp_range[csp_contains_csp_range] = EdgeSensor(csp['index'], csp_range['index'], relation=csp_contains_csp_range, forward=lambda _, x: torch.ones_like(x).unsqueeze(-1))
orbs[csp_range_contains_orbs] = EdgeSensor(csp_range['index'], orbs['index'], relation=csp_range_contains_orbs, forward=connect)
csp_range[enforce_csp_range] = DummyCityLearner2('index')
orbs[colored_orbs] = DummyCityLearner('index')

program = SolverPOIProgram(graph, poi=[orbs[colored_orbs],csp_range[enforce_csp_range],csp_range[csp_contains_csp_range],orbs[csp_range_contains_orbs]])

orbs_before_inference,orbs_after_inference=[],[]

for datanode in program.populate(dataset=dataset):
    print("before inference")
    print("orb color:",end="")
    for csp_range_datanode in datanode.getChildDataNodes():
        print(*[int(orb_node.getAttribute('<colored_orbs>',"local/softmax")[1].item()) for orb_node in csp_range_datanode.getChildDataNodes()],end=" | ")
        
    datanode.inferILPResults()
        
    print("\n\nafter inference")
    print("orb color:",end="")
    for csp_range_datanode in datanode.getChildDataNodes():
        print(*[int(orb_node.getAttribute('<colored_orbs>',"ILP").item()) for orb_node in csp_range_datanode.getChildDataNodes()],end=" | ")

print()