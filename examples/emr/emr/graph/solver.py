import torch

from regr.solver.nologInferenceSolver import NoLogInferenceSolver

from ..sensor.learner import ModuleLearner
from ..sensor.sensor import DataSensor


class Solver(NoLogInferenceSolver):
    def __init__(self, graph, ontologiesTuple, _ilpConfig,):
        def input_sensor_type(sensor):
            return isinstance(sensor, DataSensor) and not sensor.target
        super().__init__(graph, ontologiesTuple, _ilpConfig, input_sensor_type, ModuleLearner)

    def get_prop_result(self, prop, data):
        output_sensor, target_sensor = self.prop_dict[prop]

        logit = output_sensor(data)
        logit = torch.stack((1-logit, logit), dim=-1)
        mask = output_sensor.mask(data)
        labels = target_sensor(data)
        labels = labels.float()
        return labels, logit, mask

    def inferSelection(self, graph, data, prop_list, prop_dict):
        self.prop_dict = prop_dict
        return super().inferSelection(graph, data, prop_list)