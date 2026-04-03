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


class RightOfLearner(TorchLearner):
    """Returns high probability for pair (3,1) being right_of."""
    def forward(self, arg1, arg2):
        num_pairs = arg1.shape[0]
        device = arg1.device if hasattr(arg1, 'device') else 'cpu'
        result = torch.zeros(num_pairs, 2, device=device)
        
        for i in range(num_pairs):
            if arg1.dim() > 1 and arg1.shape[1] > 1:
                a1_idx = arg1[i].argmax().item() + 1
                a2_idx = arg2[i].argmax().item() + 1
            else:
                a1_idx = arg1[i].item() if hasattr(arg1[i], 'item') else int(arg1[i])
                a2_idx = arg2[i].item() if hasattr(arg2[i], 'item') else int(arg2[i])
            
            if a1_idx == 3 and a2_idx == 1:
                result[i] = torch.tensor([0.1, 0.9], device=device)
            else:
                result[i] = torch.tensor([0.9, 0.1], device=device)
        
        return result


class LeftOfLearner(TorchLearner):
    """Returns high probability for pair (3,2) being left_of."""
    def forward(self, arg1, arg2):
        num_pairs = arg1.shape[0]
        device = arg1.device if hasattr(arg1, 'device') else 'cpu'
        result = torch.zeros(num_pairs, 2, device=device)
        
        for i in range(num_pairs):
            if arg1.dim() > 1 and arg1.shape[1] > 1:
                a1_idx = arg1[i].argmax().item() + 1
                a2_idx = arg2[i].argmax().item() + 1
            else:
                a1_idx = arg1[i].item() if hasattr(arg1[i], 'item') else int(arg1[i])
                a2_idx = arg2[i].item() if hasattr(arg2[i], 'item') else int(arg2[i])
            
            if a1_idx == 3 and a2_idx == 2:
                result[i] = torch.tensor([0.1, 0.9], device=device)
            else:
                result[i] = torch.tensor([0.9, 0.1], device=device)
        
        return result


class MaterialLearner(TorchLearner):
    """Returns high probability for object 3 having material property"""
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
    

class MetalLearner(TorchLearner):
    """Returns high probability for object 3 being metal."""
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


class RubberLearner(TorchLearner):
    """Returns high probability for object 4 being rubber."""
    def forward(self, x):
        batch_size = len(x)
        device = x.device if hasattr(x, 'device') else 'cpu'
        result = torch.zeros(batch_size, 2, device=device)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 4:
                result[i] = torch.tensor([0.1, 0.9], device=device)
            else:
                result[i] = torch.tensor([0.9, 0.1], device=device)
        return result