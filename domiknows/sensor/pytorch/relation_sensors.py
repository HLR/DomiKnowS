from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor, JointReaderSensor
from domiknows.sensor.sensor import Sensor
from domiknows.graph.graph import Property
from domiknows.sensor.pytorch.query_sensor import QuerySensor
from typing import Any, Dict
from collections import OrderedDict
import torch
from itertools import product


class EdgeSensor(FunctionalSensor):
    def __init__(self, *pres, relation, **kwargs):
        super().__init__(*pres, **kwargs)
        self.relation = relation

    @property
    def relation(self):
        return self._relation

    @relation.setter
    def relation(self, relation):
        self._relation = relation
        self.src = self.relation.src
        self.dst = self.relation.dst

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


class BaseCandidateSensor(QuerySensor):
    @property
    def args(self):
        raise NotImplementedError

    def define_inputs(self):
        super(QuerySensor, self).define_inputs()  # skip QuerySensor.define_inputs
        args = {}
        for name, concept in self.args.items():
            datanodes = self.builder.getDataNode(device=self.device).findDatanodes(select=concept)
            args[name] = datanodes
        self.kwinputs['datanodes'] = args

class CandidateSensor(EdgeSensor, BaseCandidateSensor):
    @property
    def args(self):
        return OrderedDict((('dst', self.dst), ('src', self.src)))

    def forward_wrap(self):
        # args
        args = self.kwinputs['datanodes']
        # functional inputs
        inputs = self.inputs

        arg_lists = []
        dims = []
        for arg_list in args.values():
            arg_lists.append(enumerate(arg_list))
            dims.append(len(arg_list))
        output = torch.zeros(dims, dtype=torch.long, device=self.device)
        for arg_enum in product(*arg_lists):
            index, arg_list = zip(*arg_enum)
            candidates = dict(zip(self.args.keys(), arg_list))
            output[(*index,)] = self.forward(*inputs, **candidates)
        return output


class CompositionCandidateSensor(JointSensor, BaseCandidateSensor):
    @property
    def args(self):
        return OrderedDict((relation.reversed.name, relation.src) for relation in self.relations)

    def __init__(self, *args, relations, **kwargs):
        super().__init__(*args, **kwargs)
        self.relations = relations

    def forward_wrap(self):
        # args
        args = self.kwinputs['datanodes']
        # functional inputs
        inputs = self.inputs

        arg_lists = []
        indexes = []
        dims = []
        for arg_list in args.values():
            arg_lists.append(enumerate(arg_list))
            dims.append(len(arg_list))
            indexes.append([])

        for arg_enum in product(*arg_lists):
            index, arg_list = zip(*arg_enum)
            candidates = dict(zip(self.args.keys(), arg_list))
            if self.forward(*inputs, **candidates):
                for i, index_ in enumerate(index):
                    indexes[i].append(index_)

        mappings = []
        for index, dim in zip(indexes, dims):
            mapping = torch.zeros(len(index), dim)
            if len(index):
                index = torch.tensor(index, dtype=torch.long).view(-1, 1)
                mapping.scatter_(1, index, 1)
            mappings.append(mapping)

        return mappings


class CandidateRelationSensor(CandidateSensor):
    @property
    def args(self):
        # guess which side of the relation?
        concept = self.concept
        return OrderedDict((rel.name, rel.dst) if concept is rel.src else (rel.reversed.name, rel.src) for rel in self.relations)

    def __init__(self, *pres, relations, edges=None, forward=None, label=False, device='auto'):
        super().__init__(*pres, edges=edges, forward=forward, label=label, device=device)
        self.relations = relations


class CompositionCandidateReaderSensor(CompositionCandidateSensor, FunctionalReaderSensor):
    pass

class EdgeReaderSensor(ReaderSensor, EdgeSensor):
    pass

class JointEdgeReaderSensor(JointReaderSensor, EdgeSensor):
    pass


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