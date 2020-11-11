from regr.sensor.pytorch.sensors import TorchSensor, FunctionalSensor, TriggerPrefilledSensor, JointSensor
from regr.sensor.sensor import Sensor
from regr.graph.graph import Property
from regr.sensor.pytorch.query_sensor import QuerySensor
from typing import Any, Dict
import torch
from itertools import product


class EdgeSensor(FunctionalSensor):
    modes = ("forward", "backward")

    def __init__(self, *pres, relation, mode="forward", **kwargs):
        super().__init__(*pres, **kwargs)
        self.relation = relation
        self.mode = mode

    @property
    def relation(self):
        return self._relation

    @relation.setter
    def relation(self, relation):
        self._relation = relation
        # try to update
        try:
            self.mode = self.mode
        except:
            pass

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in self.modes:
            raise ValueError('The mode passed to the edge sensor must be one of %s' % self.modes)
        self._mode = mode
        if self.mode == "forward":
            self.src = self.relation.src
            self.dst = self.relation.dst
        elif self.mode == "backward":
            self.src = self.relation.dst
            self.dst = self.relation.src

    def update_pre_context(
        self,
        data_item: Dict[str, Any],
        concept=None
    ) -> Any:
        concept = concept or self.src
        super().update_pre_context(data_item, concept)

    def fetch_value(self, pre, selector=None, concept=None):
        concept = concept or self.src
        return super().fetch_value(pre, selector, concept)


class CandidateSensor(EdgeSensor, QuerySensor):
    @property
    def args(self):
        return [self.dst, self.src]

    def update_pre_context(
            self,
            data_item: Dict[str, Any]
    ) -> Any:
        super().update_pre_context(data_item)
        # for concept in self.args:
        #     if "index" in concept:
        #         concept['index'](data_item)  # call index property to make sure it is constructed

    def define_inputs(self):
        super(QuerySensor, self).define_inputs()  # skip QuerySensor.define_inputs
        if self.inputs is None:
            self.inputs = []
        args = []
        for concept in self.args:
            datanodes = self.builder.findDataNodesInBuilder(select=concept)
            args.append(datanodes)
        self.inputs = args + self.inputs

    def forward_wrap(self):
        # args
        args = self.inputs[:len(self.args)]
        # functional inputs
        inputs = self.inputs[len(self.args):]

        arg_lists = []
        dims = []
        for arg_list in args:
            arg_lists.append(enumerate(arg_list))
            dims.append(len(arg_list))
        output = torch.zeros(dims, dtype=torch.long, device=self.device)
        for arg_enum in product(*arg_lists):
            index, arg_list = zip(*arg_enum)
            output[(*index,)] = self.forward(*arg_list, *inputs)
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