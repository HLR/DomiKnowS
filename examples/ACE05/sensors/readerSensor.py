from regr.sensor.pytorch.sensors import TorchSensor, FunctionalSensor, ReaderSensor, TorchEdgeSensor, TriggerPrefilledSensor
from typing import Any
from regr.sensor.pytorch.query_sensor import DataNodeSensor
from regr.graph import Concept
import torch

class MultiLevelReaderSensor(ReaderSensor):
    def fill_data(self, data_item):
        try:
            if isinstance(self.keyword, tuple):
                self.data = (self.fetch_key(data_item, keyword) for keyword in self.keyword)
            else:
                self.data = self.fetch_key(data_item, self.keyword)
        except KeyError as e:
            raise KeyError("The key you requested from the reader doesn't exist: %s" % str(e))

    def fetch_key(self, data_item, key):
        data = []
        if "." in key:
            keys = key.split(".")
            items = data_item
            loop = 0
            direct_loop = True
            for key in keys:
                if key == "*":
                    loop += 1
                    if loop == 1:
                        keys = items.keys()
                        items = [items[key] for key in keys]
                    if loop > 1:
                        keys = [item.keys() for item in items]
                        new_items = []
                        for index, item in enumerate(items):
                            for index1, key in enumerate(keys[index]):
                                new_items.append(item[key])
                        items = new_items
                else:
                    if loop == 0:
                        items = items[key]
                    if loop > 0:
                        items = [it[key] for it in items]
            data = items
        else:
            data = data_item[key]

        return data


class SpanLabelSensor(DataNodeSensor):
    def __init__(self, *pres, edges=None, forward=None, label=False, device='auto', concept=None):
        super().__init__(*pres, edges=edges, label=label, device=device)
        self.forward_ = forward
        self.concept_name = concept

    def forward(self, match, datanode):
        if len(datanode.getEqualTo(conceptName=self.concept_name)):
            return True
        else:
            return False
        
        
        
class CustomMultiLevelReaderSensor(ReaderSensor):
    def fill_data(self, data_item):
        try:
            if isinstance(self.keyword, tuple):
                self.data = (self.fetch_key(data_item, keyword) for keyword in self.keyword)
            else:
                self.data = self.fetch_key(data_item, self.keyword)
        except KeyError as e:
            raise KeyError("The key you requested from the reader doesn't exist: %s" % str(e))

    def fetch_key(self, data_item, key):
        data = []
        if "." in key:
            keys = key.split(".")
            items = data_item
            loop = 0
            direct_loop = True
            for key in keys:
                if key == "*":
                    loop += 1
                    if loop == 1:
                        keys = items.keys()
                        items = [items[key] for key in keys]
                    if loop > 1:
                        keys = [item.keys() for item in items]
                        new_items = []
                        for index, item in enumerate(items):
                            for index1, key in enumerate(keys[index]):
                                new_items.append(item[key])
                        items = new_items
                else:
                    if key == "subtype":
                        new_items = []
                        for item in items:
                            for i in range(len(item['mentions'])):
                                if  isinstance(item[key], Concept):
                                    new_items.append(item[key].name)
                                else:
                                    new_items.append(None)
                        items = new_items
                        
                    elif key == "type":
                        new_items = []
                        for item in items:
                            for i in range(len(item['mentions'])):
                                if  isinstance(item[key], Concept):
                                    new_items.append(item[key].name)
                                else:
                                    new_items.append(None)
                        items = new_items
                    elif loop == 0:
                        items = items[key]
                    elif loop > 0:
                        items = [it[key] for it in items]
                    
            data = items
        else:
            data = data_item[key]
        
        return data
    
    
class LabelConstantSensor(FunctionalSensor):
    def __init__(self, *pres, concept, edges=None, label=False, device='auto'):
        super().__init__(*pres, edges=edges, forward=None, label=label, device=device)
        self.concept_name = concept
        
    def forward(self, input) -> Any:
        output = []
        for data in input:
            if self.concept_name == "Entity":
                if data != "value" and data != "Timex2":
                    output.append(1)
                else:
                    output.append(0)
            else:
                if data == self.concept_name:
                    output.append(1)
                else:
                    output.append(0)
        if self.label == True:
            output = torch.tensor(output)
        return output
