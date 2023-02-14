import warnings

import torch

from .pytorch import SolverModel
from ...graph.concept import EnumConcept


class IMLModel(SolverModel):
    def poi_loss(self, data_item, prop, sensors):
        output_sensor, target_sensor = sensors
        logit = output_sensor(data_item)
        labels = target_sensor(data_item)
        if len(logit) == 0:
            return None

        builder = data_item
        if (builder.needsBatchRootDN()):
            builder.addBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        concept = prop.sup
        values = []
        try:
            for cdn in datanode.findDatanodes(select=concept):
                value = cdn.getAttribute(f'<{prop.name}>/ILP')
                if isinstance(prop.name, EnumConcept):
                    # if multi-class
                    values.append(value)
                else:
                    values.append(torch.cat((1-value, value), dim=-1))
            if values:
                inference = torch.stack(values)
            else:
                assert logit.shape == (0, 2)
                inference = torch.zeros_like(logit)
        except TypeError:
            message = (f'Failed to get inference result for {prop}. '
                       'Is it included in the inference (with `inference_with` attribute)? '
                       'Continue with predicted value.')
            warnings.warn(message)
            inference = logit.softmax(dim=-1).detach()

        if self.loss:
            local_loss = self.loss[output_sensor, target_sensor](logit, inference, labels)
            return local_loss
