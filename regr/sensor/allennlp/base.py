from typing import List, Dict, NoReturn, Any, Optional
import torch
from torch.nn import Module
from ...graph import Property
from .. import Sensor, Learner


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


class AllenNlpLearner(AllenNlpModuleSensor, Learner):
    def parameters(self):
        return self.module.parameters()


class PreArgsSensor(AllenNlpModuleSensor):
    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        try:
            return self.module(*(context[pre.fullname] for pre in self.pres))
        except:
            print('Error during forward with sensor {}'.format(self.fullname))
            print('module:', self.module)
            raise
        return None


class SinglePreSensor(PreArgsSensor):
    def __init__(
        self,
        module: Module,
        *pres: List[Property]
    ) -> NoReturn:
        if len(pres) != 1:
            raise ValueError(
                '{} take one pre-required sensor, {} given.'.format(type(self), len(pres)))
        PreArgsSensor.__init__(self, module, *pres)
        self.pre = self.pres[0]


class SinglePreLearner(SinglePreSensor, AllenNlpLearner):
    pass


class MaskedSensor(AllenNlpSensor):
    def get_mask(self, context: Dict[str, Any]):
        pass


class SinglePreMaskedSensor(MaskedSensor):
    def get_mask(self, context: Dict[str, Any]):
        for sensor in self.pre.values():
            if isinstance(sensor, MaskedSensor):
                # TODO: just retieve the first? better approach?
                break
        else:
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))
        return sensor.get_mask(context)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return self.module(context[self.pre.fullname], self.get_mask(context))


class SinglePreMaskedLearner(SinglePreMaskedSensor, SinglePreLearner):
    pass
