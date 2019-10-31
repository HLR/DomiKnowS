from regr.sensor.sensor import Sensor
from typing import Dict, Any
import torch
from flair.data import Sentence


class CallingSensor(Sensor):
    def __init__(self, *pres, output=None):
        super().__init__()
        self.pres = pres
        self.output = output
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        for pre in self.pres:
            for _, sensor in pre.find(Sensor):
                sensor(context=context)
        if self.output:
            return context[self.output.fullname]


class ReaderSensor(CallingSensor):
    def __init__(self, *pres, reader):
        super().__init__(*pres)
        self.reader = reader
        self.data = None

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        super(ReaderSensor, self).forward(context=context)
        # return Sentence("I am highly motivated to capture the relationships of washington with berlin"), ["FAC",
        #                                                                                                       'LOC',
        #                                                                                                       'NONE',
        #                                                                                                       'NONE',
        #                                                                                                       "FAC",
        #                                                                                                       'LOC',
        #                                                                                                       'NONE',
        #                                                                                                       'NONE',
        #                                                                                                       "FAC",
        #                                                                                                       'LOC',
        #                                                                                                       'NONE',
        #                                                                                                       'NONE']
        return next(self.data)


class SequenceConcatSensor(CallingSensor):
    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        super(SequenceConcatSensor, self).forward(context=context)
        _list = []
        for item in self.pres:
            _list.append(context[item.fullname])
        it = 0
        _data = []
        for item in _list[0]:
            _data.append(torch.cat((item, _list[1][it]), 1))
            it += 1
        return torch.stack(_data)


class FlairEmbeddingSensor(CallingSensor):
    def __init__(self, *pres, embedding_dim):
        super(FlairEmbeddingSensor, self).__init__(*pres)
        self.embedding_dim = embedding_dim

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        super(FlairEmbeddingSensor, self).forward(context=context)
        _list = []
        for token in context[self.pres[0].fullname]:
            _list.append(token.embedding.view(1, self.embedding_dim))
        _tensor = torch.stack(_list)
        return _tensor

