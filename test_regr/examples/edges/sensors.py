from regr.sensor.pytorch.sensors import TorchSensor, FunctionalSensor, TriggerPrefilledSensor, non_label_sensor
from regr.sensor.sensor import Sensor
from regr.graph.graph import Property
from typing import Any, Dict


class TorchEdgeSensor(FunctionalSensor):
    modes = ("forward", "backward")

    def __init__(self, *pres, to, relation, mode="forward", edges=None, forward=None, label=False, device='auto'):
        super().__init__(*pres, edges=edges, forward=forward, label=label, device=device)
        self.to = to
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
        if self.dst != self.concept:
            raise ValueError('the assignment of Edge sensor is not correct!')

    def attached(self, sup):
        super().attached(sup)  # skip TorchEdgeSensor
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

        if not force and  self.fullname in data_item:
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


    @property
    def concept(self):
        raise TypeError('{} is not to be associated with a concept.'.format(type(self)))


class TokenizerEdgeSensor(TorchEdgeSensor):
    def __init__(self, *pres, to, mode="forward", edges=None, label=False, device='auto', tokenizer=None):
        super().__init__(*pres, to=to, mode=mode, edges=edges, label=label, device=device)
        if not tokenizer:
            raise ValueError('You should select a default Tokenizer')
        self.tokenizer = tokenizer

    def forward(self, text) -> Any:
        tokenized = self.tokenizer.encode_plus(text, return_tensors="pt", return_offsets_mapping=True)
        tokens = tokenized['input_ids'].view(-1).to(device=self.device)
        offset = tokenized['offset_mapping'].to(device=self.device)
        index = list(range(len(tokens)))
        return index, tokens, offset


    def update_context(self, data_item, force=False):
        super(TorchEdgeSensor, self).update_context(data_item, force=force)  # skip TorchEdgeSensor
        if isinstance(self.to, tuple):
            for to_, value in zip(self.to, data_item[self.fullname]):
                data_item[self.dst[to_].fullname] = value
        else:
            data_item[self.dst[self.to].fullname] = data_item[self.fullname]
        return data_item


class TokenizerSpan(TorchSensor):
    def __init__(self, *pres, edges=None, label=False, device='auto', tokenizer=None):
        super().__init__(*pres, edges=edges, label=label, device=device)
        if not tokenizer:
            raise ValueError('You should select a default Tokenizer')
        self.tokenizer = tokenizer

    def forward(self, ) -> Any:
        ids_index = []
        start = 0
        for item in self.tokenizer.convert_ids_to_tokens(self.inputs[0], skip_special_tokens=True):
            if item[0] == "Ġ":
                ids_index.append((start + 1, start + len(item) - 1))
            else:
                ids_index.append((start, start + len(item) - 1))
            start += len(item)
        return [(None, None)] + ids_index + [(None, None)]
