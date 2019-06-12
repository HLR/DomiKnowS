from typing import Dict, NoReturn, Any, Optional
from allennlp.data.dataset_readers import DatasetReader
from .. import Sensor


class AllenNlpSensor(Sensor):
    def __init__(
        self,
        reader: DatasetReader,
        fieldname: str,
        output_only: Optional[bool]=False
    ) -> NoReturn:
        Sensor.__init__(self)
        self.reader = reader # not sure how to use this reader
        self.fieldname = fieldname
        self.output_only = output_only

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return context[self.fieldname]


class TokenSensor(AllenNlpSensor):
    pass


class TokenSequenceSensor(AllenNlpSensor):
    pass


class LabelSensor(AllenNlpSensor):
    pass


class LabelSequenceSensor(AllenNlpSensor):
    pass
