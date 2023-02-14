from typing import List, Dict, Tuple, NoReturn, Any
import torch
from ...graph import Property
from .. import Sensor, Learner
from .module import WrapperModule


class AllenNlpSensor(Sensor):
    def __init__(
        self,
        *pres: List[Property],
        output_only: bool=False
    ) -> NoReturn:
        Sensor.__init__(self)
        self.pres = pres
        self.output_only = output_only
        self.pre_dims = []
        for pre in pres:
            for sensor in pre.find(AllenNlpSensor):
                dim = sensor.output_dim
                self.pre_dims.append(dim)
                break
            else:
                raise RuntimeError('Could not determin input dim for {} from pre-requirement {}'.format(self.fullname, pre.fullname))

    def update_context(
        self,
        data_item: Dict[str, Any],
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
                    sensor(data_item)
                    break
            else:  # no break
                raise RuntimeError('Not able to find a sensor for {} as prereqiured by {}'.format(
                    pre.fullname, self.fullname))

        # then call forward
        Sensor.update_context(self, data_item, force)

    @property
    def output_dim(self):
        return None

class ReaderSensor(AllenNlpSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: bool=False
    ) -> NoReturn:
        AllenNlpSensor.__init__(self, output_only=output_only) # *pres=[]
        self.key = key
        self.reader = reader
        reader.claim(key, self)

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        return data_item[self.fullname]



class ModuleSensor(AllenNlpSensor):
    def __init__(
        self,
        *pres: List[Property],
        output_only: bool=False
    ) -> NoReturn:
        AllenNlpSensor.__init__(self, *pres, output_only=output_only)
        self._module = None

    def create_module(self):
        raise NotImplementedError('Implemented of create_module is required in subclass of ModuleSensor.')

    @property
    def module(self):
        if self._module is None:
            module = self.create_module()
            self._module = WrapperModule(module, self.pres)
        return self._module

    @property
    def output_dim(self):
        return (self.module.get_output_dim(),)


class AllenNlpLearner(ModuleSensor, Learner):
    def parameters(self):
        return self.module.parameters()


class PreArgsModuleSensor(ModuleSensor):
    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        try:
            return self.module(*(data_item[pre.fullname] for pre in self.pres))
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
        output_only: bool=False
    ) -> NoReturn:
        AllenNlpSensor.__init__(self, pre, output_only=output_only)

    @property
    def pre(self):
        return self.pres[0]

    @property
    def pre_dim(self):
        return self.pre_dims[0]


class SinglePreLearner(SinglePreSensor, PreArgsModuleLearner):
    def __init__(
        self,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        PreArgsModuleLearner.__init__(self, pre, output_only=output_only)


class MaskedSensor(AllenNlpSensor):
    def get_mask(self, data_item: Dict[str, Any]):
        raise NotImplementedError('Implement get_mask in subclass of MaskedSensor.')


class SinglePreMaskedSensor(SinglePreSensor, MaskedSensor):
    def get_mask(self, data_item: Dict[str, Any]):
        for sensor in self.pre.find(MaskedSensor):
            break
        else:
            print(self.pre)
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))
        return sensor.get_mask(data_item)


class SinglePreMaskedLearner(SinglePreMaskedSensor, SinglePreLearner):
    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        return self.module(data_item[self.pre.fullname], self.get_mask(data_item))


class SinglePreArgMaskedPairSensor(PreArgsModuleSensor, SinglePreMaskedSensor):
    pass
