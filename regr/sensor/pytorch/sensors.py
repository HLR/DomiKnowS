from typing import Dict, Any
import torch

from .. import Sensor
from ...graph import Property


class TorchSensor(Sensor):
    def __init__(self, *pres, edges=None, label=False):
        super().__init__()
        if not edges:
            edges = []
        self.pres = pres
        self.context_helper = None
        self.inputs = []
        self.edges = edges
        self.label = label
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def __call__(
        self,
        data_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.context_helper = data_item
        try:
            data_item = self.update_context(data_item)
        except:
            print('Error during updating data item with sensor {}'.format(self.fullname))
            raise
        return data_item[self.fullname]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.update_pre_context(data_item)
            self.define_inputs()
            val = self.forward()

        if val is not None:
            data_item[self.fullname] = val
            if not self.label:
                data_item[self.prop.fullname] = val  # override state under property name
        else:
            data_item[self.fullname] = None
            if not self.label:
                data_item[self.prop.fullname] = None

        return data_item

    def update_pre_context(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        for edge in self.edges:
            for sensor in edge.find(Sensor):
                sensor(data_item=data_item)
        for pre in self.pres:
            for sensor in self.concept[pre].find(Sensor):
                sensor(data_item=data_item)

    def fetch_value(self, pre, selector=None):
        if selector:
            try:
                return self.context_helper[next(self.concept[pre].find(selector)).fullname]
            except:
                print("The key you are trying to access to with a selector doesn't exist")
                raise
        else:
            return self.context_helper[self.concept[pre].fullname]

    def define_inputs(self):
        self.inputs = []
        for pre in self.pres:
            self.inputs.append(self.fetch_value(pre))

    def forward(self,) -> Any:
        raise NotImplementedError

    @property
    def prop(self):
        if self.sup is None:
            raise ValueError('{} must be used with with property assignment.'.format(type(self)))
        return self.sup

    @property
    def concept(self):
        if self.prop.sup is None:
            raise ValueError('{} must be used with with concept[property] assignment.'.format(type(self)))
        return self.prop.sup


class FunctionalSensor(TorchSensor):
    def __init__(self, *pres, edges=None, forward=None, label=False):
        super().__init__(*pres, edges=edges, label=label)
        self.forward_ = forward

    def update_pre_context(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        for edge in self.edges:
            for sensor in edge.find(Sensor):
                sensor(data_item)
        for pre in self.pres:
            if isinstance(pre, str):
                for sensor in self.concept[pre].find(Sensor):
                    sensor(data_item)
            elif isinstance(pre, (Property, Sensor)):
                pre(data_item)

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False,
        override=True
    ) -> Dict[str, Any]:
        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.update_pre_context(data_item)
            self.define_inputs()
            val = self.forward_wrap()

        data_item[self.fullname] = val
        if override and not self.label:
            data_item[self.prop.fullname] = val  # override state under property name

        return data_item

    def fetch_value(self, pre, selector=None):
        if isinstance(pre, str):
            return super().fetch_value(pre, selector)
        elif isinstance(pre, (Property, Sensor)):
            return self.context_helper[pre.fullname]
        return pre

    def forward_wrap(self):
        return self.forward(*self.inputs)

    def forward(self, *inputs):
        if self.forward_ is not None:
            return self.forward_(*inputs)
        return super().forward()


class ConstantSensor(TorchSensor):
    def __init__(self, *pres, data, edges=None, label=False):
        super().__init__(*pres, edges=edges, label=label)
        self.data = data

    def forward(
        self,
    ) -> Any:
        try:
            return torch.tensor(self.data, device=self.device)
        except (TypeError, RuntimeError):
            return self.data


class PrefilledSensor(TorchSensor):
    def forward(self,) -> Any:
        return self.context_helper[self.prop.fullname]


class TriggerPrefilledSensor(PrefilledSensor):
    def __init__(self, *pres, callback_sensor=None, edges=None, label=False):
        super().__init__(*pres, edges=edges, label=label)
        self.callback_sensor = callback_sensor

    def forward(self,) -> Any:
        self.callback_sensor(self.context_helper)
        return super().forward()


class ReaderSensor(ConstantSensor):
    def __init__(self, *pres, keyword=None, edges=None, label=False):
        super().__init__(*pres, data=None, edges=edges, label=label)
        self.keyword = keyword

    def fill_data(self, data_item):
        try:
            self.data = data_item[self.keyword]
        except KeyError as e:
            raise KeyError("The key you requested from the reader doesn't exist: %s" % str(e))


class NominalSensor(TorchSensor):
    def __init__(self, *pres, vocab=None, edges=None):
        super().__init__(*pres, edges=edges)
        self.vocab = vocab

    def complete_vocab(self):
        if not self.vocab:
            self.vocab = []
        value = self.forward()
        if value not in self.vocab:
            self.vocab.append(value)

    def one_hot_encoder(self, value):
        if not isinstance(value, list):
            output = torch.zeros([1, len(self.vocab)], device=self.device)
            output[0][self.vocab.index(value)] = 1
        else:
            if len(value):
                output = torch.zeros([len(value), 1, len(self.vocab)], device=self.device)
                for _it in range(len(value)):
                    output[_it][0][self.vocab.index(value[_it])] = 1
            else:
                output = torch.zeros([1, 1, len(self.vocab)], device=self.device)
        return output

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.update_pre_context(data_item)
            self.define_inputs()
            val = self.forward()
            val = self.one_hot_encoder(val)

        if val is not None:
            data_item[self.fullname] = val
            if not self.label:
                data_item[self.prop.fullname] = val  # override state under property name
        else:
            data_item[self.fullname] = None
            if not self.label:
                data_item[self.prop.fullname] = None

        return data_item


class ModuleSensor(FunctionalSensor):
    def __init__(self, *pres, module, edges=None, label=False):
        super().__init__(*pres, edges=edges, label=label)
        self.module = module

    def forward(self, *inputs):
        return self.module(*inputs)


class TorchEdgeSensor(FunctionalSensor):
    modes = ("forward", "backward", "selection")

    def __init__(self, *pres, to, mode="forward", edges=None, label=False):
        super().__init__(*pres, edges=edges, label=label)
        self.to = to
        self.mode = mode
        if self.mode not in self.modes:
            raise ValueError('The mode passed to the edge sensor must be one of %s' % self.modes)
        self.src = None
        self.dst = None

    def attached(self, sup):
        super().attached(sup)
        self.relation = sup.sup
        if self.mode == "forward":
            self.src = self.relation.src
            self.dst = self.relation.dst
        elif self.mode == "backward" or self.mode == "selection":
            self.src = self.relation.dst
            self.dst = self.relation.src
        else:
            raise ValueError('The mode passed to the edge is invalid!')
        self.dst[self.to] = TriggerPrefilledSensor(callback_sensor=self)

    def __call__(
        self,
        data_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        super().__call__(data_item)
        return data_item[self.dst[self.to].fullname]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        super().update_context(data_item, force=force)
        data_item[self.dst[self.to].fullname] = data_item[self.fullname]
        return data_item

    def update_pre_context(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        for edge in self.edges:
            for sensor in edge.find(Sensor):
                sensor(data_item)
        for pre in self.pres:
            if isinstance(pre, str):
                for sensor in self.src[pre].find(Sensor):
                    sensor(data_item)
            elif isinstance(pre, (Property, Sensor)):
                pre(data_item)
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


    @property
    def concept(self):
        raise TypeError('{} is not to be associated with a concept.'.format(type(self)))


class TorchEdgeReaderSensor(TorchEdgeSensor, ReaderSensor):
    def __init__(self, *pres, to, keyword, mode="forward", edges=None, label=False):
        super().__init__(*pres, to=to, mode=mode, edges=edges, label=label)
        self.keyword = keyword
        self.data = None


class ForwardEdgeSensor(TorchEdgeSensor):
    def forward(self, input) -> Any:
        return input


class AggregationSensor(TorchSensor):
    def __init__(self, *pres, edges, map_key, deafault_dim = 480):
        super().__init__(*pres, edges=edges)
        self.edge_node = self.edges[0].sup
        self.map_key = map_key
        self.map_value = None
        self.data = None
        self.default_dim = deafault_dim
        if self.edges[0].name == "backward":
            self.src = self.edges[0].sup.dst
            self.dst = self.edges[0].sup.src
        else:
            print("the mode should always be passed as backward to the edge used in aggregator sensor")
            raise Exception('not valid')

    def get_map_value(self, ):
        self.map_value = self.context_helper[self.src[self.map_key].fullname]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:

        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.define_inputs()
            self.get_map_value()
            self.get_data()
            val = self.forward()
        if val is not None:
            data_item[self.fullname] = val
            data_item[self.prop.fullname] = val # override state under property name
        return data_item

    def get_data(self):
        result = []
        for item in self.inputs[0]:
            result.append(self.map_value[item[0]:item[1]+1])
        self.data = result


class MaxAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        for item in self.data:
            results.append(torch.max(item, dim=0)[0])
        return torch.stack(results)


class MinAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        for item in self.data:
            results.append(torch.min(item, dim=0)[0])
        return torch.stack(results)


class MeanAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        if len(self.data):
            for item in self.data:
                results.append(torch.mean(item, dim=0))
            return torch.stack(results)
        else:
            return torch.zeros(1, 1, self.default_dim, device=self.device)


class ConcatAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        for item in self.data:
            results.append(torch.cat([x for x in item], dim=-1))
        return torch.stack(results)


class LastAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        if len(self.data):
            for item in self.data:
                results.append(item[-1])
            return torch.stack(results)
        else:
            return torch.zeros(1, 1, self.default_dim, device=self.device)


class FirstAggregationSensor(AggregationSensor):
    def forward(self,) -> Any:
        results = []
        if len(self.data):
            for item in self.data:
                results.append(item[0])
            return torch.stack(results)
        else:
            return torch.zeros(1, 1, self.default_dim, device=self.device)


class SelectionEdgeSensor(TorchEdgeSensor):
    def __init__(self, *pres, mode="selection"):
        super().__init__(*pres, mode=mode)
        self.selection_helper = None

    def __call__(
        self,
        data_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.get_initialized()
        try:
            self.update_pre_context(data_item)
        except:
            print('Error during updating pre data item with sensor {}'.format(self.fullname))
            raise
        self.context_helper = data_item
        try:
            data_item = self.update_context(data_item)
        except:
            print('Error during updating data item with sensor {}'.format(self.fullname))
            raise
        return data_item[self.src[self.dst].fullname]

    def get_selection_helper(self):
        self.selection_helper = self.context_helper[self.src[self.dst].fullname]

    def update_pre_context(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        for sensor in self.src[self.dst].find(Sensor):
            sensor(data_item)

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:

        if not force and self.fullname in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            self.define_inputs()
            self.get_selection_helper()
            val = self.forward()
        if val is not None:
            data_item[self.fullname] = val
            data_item[self.prop.fullname] = val  # override state under property name
        return data_item


class ProbabilitySelectionEdgeSensor(SelectionEdgeSensor):
    def forward(self,) -> Any:
        return self.selection_helper


class ThresholdSelectionEdgeSensor(SelectionEdgeSensor):
    def __init__(self, *pres, threshold=0.5):
        # FIXME: @hfaghihi, do you mean to call super class of `SelectionEdgeSensor`, so here we skip the constructor of `SelectionEdgeSensor`?
        super(SelectionEdgeSensor).__init__(*pres)
        self.threshold = threshold

    def forward(self,) -> Any:
        return torch.tensor([x for x in self.selection_helper if x >= self.threshold], device=self.device)


class ConcatSensor(TorchSensor):
    def forward(self,) -> Any:
        return torch.cat(self.inputs, dim=-1)


class ListConcator(TorchSensor):
    def forward(self,) -> Any:
        for it in range(len(self.inputs)):
            if isinstance(self.inputs[it], list):
                self.inputs[it] = torch.stack(self.inputs[it])
        return torch.cat(self.inputs, dim=-1)


class SpacyTokenizorSensor(FunctionalSensor):
    from spacy.lang.en import English
    nlp = English()

    def forward(self, sentences):
        tokens = self.nlp.tokenizer.pipe(sentences)
        return list(tokens)


class BertTokenizorSensor(FunctionalSensor):
    from transformers import BertTokenizer
    TRANSFORMER_MODEL = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)

    def forward(self, sentences):
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            return_attention_mask=True,
            #return_offsets_mapping=True,
        )
        tokens['tokens'] = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'], skip_special_tokens=True)
        return tokens
