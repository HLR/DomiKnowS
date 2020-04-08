import abc
from typing import Any, Dict
from regr.sensor.sensor import Sensor
from regr.graph.property import Property

import torch


class SkipSensor(Exception):
    pass


class TorchSensor(Sensor):
    def __init__(self, target=False):
        super().__init__()
        self.target = target

    def propagate_context(self, context, node, force=False):
        if not self.target:  # avoid target to be mixed in forward calculation
            super().propagate_context(context, node, force)

    def mask(self, context: Dict[str, Any]) -> Any:
        # allow safely skip mask
        raise SkipSensor


class DataSensor(TorchSensor):
    def __init__(self, key, target=False):
        super().__init__(target=target)
        self.key = key

    def forward(self, context):
        return context[self.key]


class LabelSensor(DataSensor):
    def __init__(self, key):
        super().__init__(key, target=True)


class FunctionalSensor(TorchSensor):
    def __init__(self, *pres, target=False):
        super().__init__(target)
        self.pres = pres

    def get_args(self, context, skip_none_prop=False, sensor_fn=None, sensor_filter=None):
        for pre in self.pres:
            if isinstance(pre, Property):
                for _, sensor in pre.items():
                    if sensor_filter and not sensor_filter(sensor):
                        continue
                    try:
                        if sensor_fn:
                            yield sensor_fn(sensor, context)
                        else:
                            yield sensor(context)
                        break
                    except SkipSensor:
                        pass
                else:  # no break
                    raise RuntimeError('Not able to find a sensor for {} as prereqiured by {}'.format(
                        pre.fullname, self.fullname))
            else:
                if not skip_none_prop:
                    yield pre

    @abc.abstractmethod
    def forward_func(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        args = self.get_args(context, sensor_filter=lambda s: not s.target)
        return self.forward_func(*args)

    def mask(self, context):
        masks = list(self.get_args(context, sensor_fn=lambda s, c: s.mask(c)))
        masks_num = len(masks)
        mask = masks[0]
        mask = mask.float()
        for i in range(1, masks_num):
            for j in range(i, masks_num):
                masks[j].unsqueeze_(-2)
            masks[i] = masks[i].float()
            mask = mask.unsqueeze_(-1).matmul(masks[i])
        return mask


class CartesianSensor(FunctionalSensor):
    def forward_func(self, *inputs):
        # torch cat is not broadcasting, do repeat manually
        input_iter = iter(inputs)
        output = next(input_iter)
        for input in input_iter:
            dob, *dol, dof = output.shape
            dib, *dil, dif = input.shape
            assert dob == dib
            output = output.view(dob, *dol, *(1,)*len(dil), dof).repeat(1, *(1,)*len(dol), *dil, 1)
            input = input.view(dib, *(1,)*len(dol), *dil, dif).repeat(1, *dol, *(1,)*len(dil), 1)
            output = torch.cat((output, input), dim=-1)
        return output
