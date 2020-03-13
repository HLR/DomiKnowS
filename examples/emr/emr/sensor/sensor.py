import abc
from typing import Any, Dict
from regr.sensor.sensor import Sensor


class TorchSensor(Sensor):
    def __init__(self, output_only=None):
        super().__init__()
        self.output_only = output_only

    def __call__(self, context: Dict[str, Any], hard=False) -> Dict[str, Any]:
        if hard or self.fullname not in context:
            try:
                self.update_context(context)
            except:
                print('Error during updating context with sensor %s', self.fullname)
                raise
        return context[self.fullname]

    def update_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            val = self.forward(context)
        except SkipSensor:
            pass
        else:
            context[self.fullname] = val
            context[self.sup.fullname] = val # override state under property name

    @abc.abstractmethod
    def forward(self, *args, **kwargs,) -> Any:
        return None

class DataSensor(TorchSensor):
    def __init__(self, key, output_only=False):
        super().__init__(output_only=output_only)
        self.key = key

    def forward(self, context):
        return context[self.key]


class LabelSensor(DataSensor):
    def __init__(self, key):
        super().__init__(key, output_only=True)


class SkipSensor(Exception):
    pass

class FunctionalSensor(TorchSensor):
    def __init__(self, *pres, output_only=None):
        super().__init__(output_only)
        self.pres = pres

    def update_context(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.pres:
            super().update_context(context)

        parameters = []
        for pre in self.pres:
            for name, item in pre.items():
                # choose one result to update finally
                if isinstance(item, Sensor):
                    if item.output_only:
                        continue
                    parameters.append(item(context))
                    break
                else:
                    parameters.append(item)
                    break
            else:  # no break
                raise RuntimeError('Not able to find a sensor for {} as prereqiured by {}'.format(
                    pre.fullname, self.fullname))
        try:
            val = self.forward(*parameters)
        except SkipSensor:
            pass
        else:
            context[self.fullname] = val
            context[self.sup.fullname] = val # override state under property name


class CartesianSensor(FunctionalSensor):
    pass
