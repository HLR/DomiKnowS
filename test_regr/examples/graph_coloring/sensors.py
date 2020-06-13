from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor, ReaderSensor
from regr.sensor.pytorch.learners import TorchLearner
from typing import Any
import torch

#  --- City
class DummyCityEdgeSensor(TorchEdgeSensor): # Get world to city edge
    def forward(self,) -> Any:
        self.inputs.append(self.context_helper[self.edges[0].fullname])
        return self.inputs[0]
    
class DummyCityLearner(TorchLearner):  # Learn Fire station classification for City
    def forward(self,) -> Any:
        result = torch.zeros(len(self.inputs[0]), 2)
        
        for t in result: # Initially all cities are firestation cities
            t[1] = 1
            t[0] = 0

        return result

class DummyCityLabelSensor(TorchSensor): # Get Truth for Fire station classification
    def __init__(self, *pres, label=True):
        super().__init__(*pres, label=label)

    def forward(self,) -> Any:
        return []

# --- CityLink

class DummyCityLinkEdgeSensor(TorchEdgeSensor): # Get CityLink to city edge
    def forward(self,) -> Any:
        return self.inputs[0]

class DummyCityLinkCandidateGenerator(TorchSensor): # Generate candidates for CityLinks relation
    def forward(self,) -> Any:
        try:
            pairs = torch.ones(len(self.context_helper['city']), len(self.context_helper['city']), dtype=torch.int) # torch.int8 or torch.bool
            
            return pairs
        except:
            print("the key you requested from the reader doesn't exist")
            raise

class DummyNeighborLearner(TorchLearner):  # Learn Neighbor classification for CityLink
    def forward(self,) -> Any:
        neighbor = torch.zeros(*self.inputs[0].shape, 2)
                
        # Add neighbor connection from input data
        info = self.context_helper['links']
        for city, targets in info.items():
            for target in targets:
                neighbor[city - 1, target - 1][1] = 1 ## cities are connected
                
        # Fix negative probability for not connected cities
        for t in neighbor:
            for t1 in t:
                if t1[1] == 0: # Cities are not connected 
                    t1[0] = 1 # Negative probability is 1
                    
        return neighbor

class DummyCityLinkLabelSensor(TorchSensor): # Get Truth for Neighbor classification
    def __init__(self, *pres, label=True):
        super().__init__(*pres, label=label)

    def forward(self,) -> Any:
        return []