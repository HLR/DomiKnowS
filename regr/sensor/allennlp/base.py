from typing import List, Dict, NoReturn, Any
from allennlp.data.dataset_readers import DatasetReader
from .. import Sensor, Learner


class BaseAllenNlpSensor(Sensor):
    def __init__(
        self,
        reader: DatasetReader,
        fieldname: str
    ) -> NoReturn:
        Sensor.__init__(self)
        self.reader = reader # not sure how to use this reader
        self.fieldname = fieldname

    def update_context(
        self,
        context: Dict[str, Any]
    ) -> Dict:
        context[self.fullname] = context[self.fieldname]
        return context

class BaseAllenNlpLearner(BaseAllenNlpSensor, Learner):
    def __init__(
        self,
        *pres: List[Sensor]
    ) -> NoReturn: 
        Learner.__init__(self)

    def update_context(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        for pre in pres:
            context = pre(context)
        context[self.fullname] = self.forward(context)
        return context

    def forward(
        context: Dict[str, Any]
    ) -> Any:
        return None