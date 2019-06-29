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
        context: Dict[str, Any],
        force=False
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
        return Sensor.update_context(self, context, force)


class ReaderSensor(AllenNlpSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: Optional[bool]=False
    ) -> NoReturn:
        AllenNlpSensor.__init__(self, output_only=output_only) # *pres=[]
        self.key = key
        self.reader = reader
        reader.claim(key, self)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return context[self.fullname]


class ModuleSensor(AllenNlpSensor):
    class WrapperModule(Module):
        # use a wrapper to keep pre-requireds and avoid side-effect of sequencial or other modules
        def __init__(self, module):
            super(ModuleSensor.WrapperModule, self).__init__()
            self.main_module = module

        def forward(self, *args, **kwargs):
            return self.main_module(*args, **kwargs)

    def __init__(
        self,
        module: Module,
        *pres: List[Property],
        output_only: Optional[bool]=False
    ) -> NoReturn:
        AllenNlpSensor.__init__(self, *pres, output_only=output_only)
        self.module = ModuleSensor.WrapperModule(module)
        for pre in pres:
            for name, sensor in pre.items():
                if isinstance(sensor, ModuleSensor) and not sensor.output_only:
                    self.module.add_module(sensor.fullname, sensor.module)


class AllenNlpLearner(ModuleSensor, Learner):
    def parameters(self):
        return self.module.parameters()


class PreArgsModuleSensor(ModuleSensor):
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


class PreArgsModuleLearner(PreArgsModuleSensor, AllenNlpLearner):
    pass


class SinglePreSensor(AllenNlpSensor):
    def __init__(
        self,
        pre: Property,
        output_only: Optional[bool]=False
    ) -> NoReturn:
        AllenNlpSensor.__init__(self, pre, output_only=output_only)
        self.pre = pre


class SinglePreLearner(SinglePreSensor, PreArgsModuleLearner):
    def __init__(
        self,
        module: Module,
        pre: Property,
        output_only: Optional[bool]=False
    ) -> NoReturn:
        SinglePreSensor.__init__(self, pre, output_only=output_only)
        PreArgsModuleLearner.__init__(self, module, pre, output_only=output_only)


class MaskedSensor(AllenNlpSensor):
    def get_mask(self, context: Dict[str, Any]):
        pass


class SinglePreMaskedSensor(SinglePreSensor, MaskedSensor):
    def get_mask(self, context: Dict[str, Any]):
        for name, sensor in self.pre.find(MaskedSensor):
            break
        else:
            print(self.pre)
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))
        return sensor.get_mask(context)


class SinglePreMaskedLearner(SinglePreMaskedSensor, SinglePreLearner):
    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return self.module(context[self.pre.fullname], self.get_mask(context))
