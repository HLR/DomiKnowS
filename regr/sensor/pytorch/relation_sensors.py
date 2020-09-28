from regr.sensor.pytorch.sensors import TorchSensor, FunctionalSensor, TriggerPrefilledSensor, non_label_sensor, JointSensor
from regr.sensor.sensor import Sensor
from regr.graph.graph import Property
from regr.sensor.pytorch.query_sensor import QuerySensor
from typing import Any, Dict
import torch
from itertools import product


class EdgeSensor(FunctionalSensor):
    modes = ("forward", "backward")

    def __init__(self, *pres, relation, mode="forward", edges=None, forward=None, label=False, device='auto'):
        super().__init__(*pres, edges=edges, forward=forward, label=label, device=device)
        self.relation = relation
        self.mode = mode
        if self.mode not in self.modes:
            raise ValueError('The mode passed to the edge sensor must be one of %s' % self.modes)
        if self.mode == "forward":
            self.src = self.relation.src
            self.dst = self.relation.dst
        elif self.mode == "backward":
            self.src = self.relation.dst
            self.dst = self.relation.src

    def attached(self, sup):
        super().attached(sup)
        if self.dst != self.concept:
            raise ValueError('the assignment of Edge sensor is not correct!')
        if isinstance(self.prop, tuple):
            for to_ in self.prop:
                self.dst[to_] = TriggerPrefilledSensor(callback_sensor=self)
        else:
            self.dst[self.prop] = TriggerPrefilledSensor(callback_sensor=self)

    def update_context(
            self,
            data_item: Dict[str, Any],
            force=False
    ) -> Dict[str, Any]:

        if not force and self.fullname in data_item:
            val = data_item[self.fullname]
        else:
            self.update_pre_context(data_item)
            self.define_inputs()
            val = self.forward()

        if val is not None:
            data_item[self.fullname] = val
            if not self.label:
                if isinstance(self.prop, tuple):
                    index = 0
                    for to_ in self.prop:
                        data_item[to_.fullname] = val[index]
                        index += 1
                else:
                    data_item[self.prop.fullname] = val
        else:
            data_item[self.fullname] = None
            if not self.label:
                if isinstance(self.prop, tuple):
                    for to_ in self.prop:
                        data_item[to_.fullname] = None
                else:
                    data_item[self.prop.fullname] = None

        return data_item

    def update_pre_context(
            self,
            data_item: Dict[str, Any]
    ) -> Any:
        for edge in self.edges:
            for sensor in edge.find(non_label_sensor):
                sensor(data_item)
        for pre in self.pres:
            if isinstance(pre, str):
                for sensor in self.src[pre].find(non_label_sensor):
                    sensor(data_item)
            elif isinstance(pre, (Property, Sensor)):
                for sensor in pre.find(non_label_sensor):
                    sensor(data_item)
        # besides, make sure src exist
        self.src['index'](data_item=data_item)

    def fetch_value(self, pre, selector=None):
        if isinstance(pre, str):
            if selector:
                try:
                    return self.context_helper[next(self.src[pre].find(selector)).fullname]
                except:
                    print("The key you are trying to access to with a selector doesn't exist")
                    raise
            else:
                return self.context_helper[self.src[pre].fullname]
        elif isinstance(pre, (Property, Sensor)):
            return self.context_helper[pre.fullname]
        return pre


class CandidateSensor(QuerySensor):
    def __init__(self, *pres, edges=None, forward=None, label=False, device='auto'):
        super().__init__(*pres, edges=edges, forward=forward, label=label, device=device)

        # Add identification of candidate
        self.name += "_Candidate_"

    @property
    def args(self):
        return [rel.dst for rel in self.concept.has_a()]

    def update_pre_context(
            self,
            data_item: Dict[str, Any]
    ) -> Any:
        super().update_pre_context(data_item)
        for concept in self.args:
            if "index" in concept:
                concept['index'](data_item)  # call index property to make sure it is constructed

    def define_inputs(self):
        super().define_inputs()
        args = []
        for concept in self.args:
            datanodes = self.builder.findDataNodesInBuilder(select=concept)
            args.append(datanodes)
        self.inputs = self.inputs[:1] + args + self.inputs[1:]

    def forward_wrap(self):
        # current existing datanodes (if any)
        datanodes = self.inputs[0]
        # args
        args = self.inputs[1:len(self.args) + 1]
        # functional inputs
        inputs = self.inputs[len(self.args) + 1:]

        arg_lists = []
        dims = []
        for arg_list in args:
            arg_lists.append(enumerate(arg_list))
            dims.append(len(arg_list))
        output = torch.zeros(dims, dtype=torch.uint8).to(device=self.device)
        for arg_enum in product(*arg_lists):
            index, arg_list = zip(*arg_enum)
            output[(*index,)] = self.forward(datanodes, *arg_list, *inputs)
        return output


class CandidateRelationSensor(CandidateSensor):
    @property
    def args(self):
        concept = self.concept
        return [(rel.dst if concept is rel.src else rel.src) for rel in self.relations]

    def __init__(self, *pres, relations, edges=None, forward=None, label=False, device='auto'):
        super().__init__(*pres, edges=edges, forward=forward, label=label, device=device)
        self.relations = relations


class CandidateReaderSensor(CandidateSensor):
    def __init__(self, *pres, edges=None, forward=None, label=False, keyword=None, device='auto'):
        super().__init__(*pres, edges=edges, forward=forward, label=label, device=device)
        self.data = None
        self.keyword = keyword
        if keyword is None:
            raise ValueError('{} "keyword" must be assign.'.format(type(self)))

    def fill_data(self, data):
        self.data = data[self.keyword]

    def forward_wrap(self):
        # current existing datanodes (if any)
        datanodes = self.inputs[0]
        # args
        args = self.inputs[1:len(self.args)+1]
        # functional inputs
        inputs = self.inputs[len(self.args)+1:]

        arg_lists = []
        dims = []
        for arg_list in args:
            arg_lists.append(enumerate(arg_list))
            dims.append(len(arg_list))

        if self.data is None and self.keyword in self.context_helper:
            self.data = self.context_helper[self.keyword]
        output = torch.zeros(tuple(dims), dtype=torch.uint8).to(device=self.device)
        for arg_enum in product(*arg_lists):
            index, arg_list = zip(*arg_enum)
            output[(*index,)] = self.forward(self.data, datanodes, *arg_list, *inputs)
        return output


class CandidateEqualSensor(CandidateSensor):
    def __init__(self, *pres, edges=None, forward=None, label=False, device='auto', relations=None):
        super().__init__(*pres, edges=edges, forward=forward, label=label, device=device)

        if relations:
            self.relations = relations
            # Add identification of equality and the  name of the equal concept type
            self.name += "_Equality_" +  self.relations[0].dst.name
        else:
            self.relations = []

    @property
    def args(self):
        if len(self.relations):
            return [self.concept, self.relations[0].dst]
        else:
            return [self.concept, self.concept.equal()[0].dst]

    def forward_wrap(self):
        # current existing datNnodes (if any) for first element of equality
        conceptDns = self.inputs[1]
        equalDns = self.inputs[2]

        dims = (len(conceptDns), len(equalDns))
        output = torch.zeros(dims, dtype=torch.uint8).to(device=self.device)

        for dns_product in product(conceptDns,equalDns):
            index = (dns_product[0].getInstanceID(), dns_product[1].getInstanceID() )
            output[(*index,)] = self.forward("", dns_product[0], dns_product[1])

        return output