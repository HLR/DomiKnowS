from domiknows.sensor.pytorch.learners import TorchLearner
import torch


class BigLearner(TorchLearner):
    """Returns high probability for object 3 being big"""
    def forward(self, x):
        batch_size = len(x)
        device = x.device if hasattr(x, 'device') else 'cpu'
        result = torch.zeros(batch_size, 2, device=device)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 3:
                result[i] = torch.tensor([0.1, 0.9], device=device)
            else:
                result[i] = torch.tensor([0.9, 0.1], device=device)
        return result


class LargeLearner(TorchLearner):
    """Returns high probability for object 2 being large"""
    def forward(self, x):
        batch_size = len(x)
        device = x.device if hasattr(x, 'device') else 'cpu'
        result = torch.zeros(batch_size, 2, device=device)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 2:
                result[i] = torch.tensor([0.1, 0.9], device=device)
            else:
                result[i] = torch.tensor([0.9, 0.1], device=device)
        return result


class BrownLearner(TorchLearner):
    """Returns high probability for objects 1 and 2 being brown"""
    def forward(self, x):
        batch_size = len(x)
        device = x.device if hasattr(x, 'device') else 'cpu'
        result = torch.zeros(batch_size, 2, device=device)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id in [1, 2]:
                result[i] = torch.tensor([0.1, 0.9], device=device)
            else:
                result[i] = torch.tensor([0.9, 0.1], device=device)
        return result


class CylinderLearner(TorchLearner):
    """Returns high probability for object 1 being cylinder"""
    def forward(self, x):
        batch_size = len(x)
        device = x.device if hasattr(x, 'device') else 'cpu'
        result = torch.zeros(batch_size, 2, device=device)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 1:
                result[i] = torch.tensor([0.1, 0.9], device=device)
            else:
                result[i] = torch.tensor([0.9, 0.1], device=device)
        return result


class SphereLearner(TorchLearner):
    """Returns high probability for object 2 being sphere"""
    def forward(self, x):
        batch_size = len(x)
        device = x.device if hasattr(x, 'device') else 'cpu'
        result = torch.zeros(batch_size, 2, device=device)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 2:
                result[i] = torch.tensor([0.1, 0.9], device=device)
            else:
                result[i] = torch.tensor([0.9, 0.1], device=device)
        return result


class MaterialEnumLearner(TorchLearner):
    """
    Returns probabilities for EnumConcept material with values ['metal', 'rubber'].
    Output shape: (batch_size, 2) where index 0=metal, index 1=rubber
    
    Object 3 is metal (high prob at index 0)
    Object 4 is rubber (high prob at index 1)
    """
    def forward(self, x):
        batch_size = len(x)
        device = x.device if hasattr(x, 'device') else 'cpu'
        # Output: [metal_prob, rubber_prob] for each object
        result = torch.zeros(batch_size, 2, device=device)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 3:
                # Object 3 is metal
                result[i] = torch.tensor([0.9, 0.1], device=device)
            elif obj_id == 4:
                # Object 4 is rubber
                result[i] = torch.tensor([0.1, 0.9], device=device)
            else:
                # Other objects - uniform distribution
                result[i] = torch.tensor([0.5, 0.5], device=device)
        return result