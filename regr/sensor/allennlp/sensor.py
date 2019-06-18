from typing import List, Dict, NoReturn, Any, Optional
import torch
from torch.nn import Module
from ...graph import Property
from .. import Sensor


class AllenNlpSensor(Sensor):
    def __init__(
        self,
        *pres: List[Property],
        output_only: Optional[bool]=False
    ) -> NoReturn:
        Sensor.__init__(self)
        self.pres = pres
        self.output_only = output_only

    def update_context(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        # update prerequired first
        for pre in self.pres:
            if not isinstance(pre, dict):
                raise RuntimeError('{}'.format(self.fullname))
            for name, sensor in pre.items():
                # choose one result to update finally
                # TODO: consider priority or confidence or merge somehow
                if isinstance(sensor, AllenNlpSensor) and not sensor.output_only:
                    context = sensor(context)
                    break
            else:  # no break
                raise RuntimeError('Not able to find a sensor for {} as prereqiured by {}'.format(
                    pre.fullname, self.fullname))

        # then call forward
        return Sensor.update_context(self, context)


class AllenNlpReaderSensor(AllenNlpSensor):
    def __init__(
        self,
        fieldname: str,
        output_only: Optional[bool]=False
    ) -> NoReturn:
        AllenNlpSensor.__init__(self, output_only=output_only) # *pres=[]
        self.fieldname = fieldname

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return context[self.fieldname]


class AllenNlpModuleSensor(AllenNlpSensor):
    class WrapperModule(Module):
        # use a wrapper to keep pre-requireds and avoid side-effect of sequencial or other modules
        def __init__(self, module):
            super(AllenNlpModuleSensor.WrapperModule, self).__init__()
            self.main_module = module

        def forward(self, *args, **kwargs):
            return self.main_module(*args, **kwargs)

    def __init__(
        self,
        module: Module,
        *pres: List[Sensor],
        output_only: Optional[bool]=False
    ) -> NoReturn:
        AllenNlpSensor.__init__(self, *pres, output_only=output_only)
        self.module = AllenNlpModuleSensor.WrapperModule(module)
        for pre in pres:
            for name, sensor in pre.items():
                if isinstance(sensor, AllenNlpModuleSensor) and not sensor.output_only:
                    self.module.add_module(sensor.fullname, sensor.module)


class SentenceSensor(AllenNlpReaderSensor):
    pass


class PhraseSensor(AllenNlpReaderSensor):
    pass


class PhraseSequenceSensor(AllenNlpReaderSensor):
    def __init__(
        self,
        vocab,
        fieldname: str,
        tokenname: str='tokens',
        output_only: Optional[bool]=False
    ) -> NoReturn:
        AllenNlpReaderSensor.__init__(self, fieldname, output_only=output_only) # *pres=[]
        self.vocab = vocab
        self.tokenname = tokenname

class LabelSensor(AllenNlpSensor):
    pass


class LabelSequenceSensor(AllenNlpReaderSensor):
    pass


class CartesianProductSensor(AllenNlpModuleSensor):
    class CP(Module):
        def forward(self, x, y):  # (b,l1,f1) x (b,l2,f2) -> (b, l1, l2, f1+f2)
            xs = x.size()
            ys = y.size()
            assert xs[0] == ys[0]
            # torch cat is not broadcasting, do repeat manually
            xx = x.view(xs[0], xs[1], 1, xs[2]).repeat(1, 1, ys[1], 1)
            yy = y.view(ys[0], 1, ys[1], ys[2]).repeat(1, xs[1], 1, 1)
            return torch.cat([xx, yy], dim=3)

    class SelfCP(CP):
        def forward(self, x):
            return CartesianProductSensor.CP.forward(self, x, x)

    def __init__(
        self,
        *pres: List[Property]
    ) -> NoReturn:
        if len(pres) != 1:
            raise ValueError(
                '{} take one pre-required sensor, {} given.'.format(type(self), len(pres)))
        module = CartesianProductSensor.SelfCP()
        AllenNlpModuleSensor.__init__(self, module, *pres)
        self.pre = self.pres[0]

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return self.module(context[self.pre.fullname])
