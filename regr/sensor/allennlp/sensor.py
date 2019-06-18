from typing import List, Dict, NoReturn, Any, Optional
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
    def __init__(
        self,
        module: Module,
        *pres: List[Sensor],
        output_only: Optional[bool]=False
    ) -> NoReturn:
        AllenNlpSensor.__init__(self, *pres, output_only=output_only)
        self.module = module
        for pre in pres:
            for name, sensor in pre.items():
                if isinstance(sensor, AllenNlpModuleSensor) and not sensor.output_only:
                    module.add_module(sensor.fullname, sensor.module)


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
