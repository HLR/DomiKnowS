from domiknows.sensor.pytorch.learners import TorchLearner
import torch


class BigLearner(TorchLearner):
    """Returns high probability for object 3 being big"""
    def forward(self, x):
        batch_size = len(x)
        result = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 3:  # Object 3 is big
                result[i] = torch.tensor([0.1, 0.9])
            else:
                result[i] = torch.tensor([0.9, 0.1])
        return result


class LargeLearner(TorchLearner):
    """Returns high probability for object 2 being large"""
    def forward(self, x):
        batch_size = len(x)
        result = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 2:  # Object 2 is large
                result[i] = torch.tensor([0.1, 0.9])
            else:
                result[i] = torch.tensor([0.9, 0.1])
        return result


class BrownLearner(TorchLearner):
    """Returns high probability for objects 1 and 2 being brown"""
    def forward(self, x):
        batch_size = len(x)
        result = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id in [1, 2]:  # Objects 1, 2 are brown
                result[i] = torch.tensor([0.1, 0.9])
            else:
                result[i] = torch.tensor([0.9, 0.1])
        return result


class CylinderLearner(TorchLearner):
    """Returns high probability for object 1 being cylinder"""
    def forward(self, x):
        batch_size = len(x)
        result = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 1:  # Object 1 is cylinder
                result[i] = torch.tensor([0.1, 0.9])
            else:
                result[i] = torch.tensor([0.9, 0.1])
        return result


class SphereLearner(TorchLearner):
    """Returns high probability for object 2 being sphere"""
    def forward(self, x):
        batch_size = len(x)
        result = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 2:  # Object 2 is sphere
                result[i] = torch.tensor([0.1, 0.9])
            else:
                result[i] = torch.tensor([0.9, 0.1])
        return result


class RightOfLearner(TorchLearner):
    """
    Returns high probability for pair (3,1) being right_of.
    Inputs from CompositionCandidateSensor are 2D tensors [num_pairs, num_objects].
    """
    def forward(self, arg1, arg2):
        # arg1, arg2 are [num_pairs, num_objects] shaped tensors from CompositionCandidateSensor
        # Each row is a one-hot or index representation
        num_pairs = arg1.shape[0]
        result = torch.zeros(num_pairs, 2)
        
        for i in range(num_pairs):
            # Get the object indices - argmax gives the object index for this pair element
            if arg1.dim() > 1 and arg1.shape[1] > 1:
                a1_idx = arg1[i].argmax().item() + 1  # +1 because object IDs are 1-indexed
                a2_idx = arg2[i].argmax().item() + 1
            else:
                a1_idx = arg1[i].item() if hasattr(arg1[i], 'item') else int(arg1[i])
                a2_idx = arg2[i].item() if hasattr(arg2[i], 'item') else int(arg2[i])
            
            # Object 3 is right of object 1 (brown cylinder)
            if a1_idx == 3 and a2_idx == 1:
                result[i] = torch.tensor([0.1, 0.9])
            else:
                result[i] = torch.tensor([0.9, 0.1])
        
        return result


class LeftOfLearner(TorchLearner):
    """
    Returns high probability for pair (3,2) being left_of.
    Inputs from CompositionCandidateSensor are 2D tensors [num_pairs, num_objects].
    """
    def forward(self, arg1, arg2):
        # arg1, arg2 are [num_pairs, num_objects] shaped tensors from CompositionCandidateSensor
        num_pairs = arg1.shape[0]
        result = torch.zeros(num_pairs, 2)
        
        for i in range(num_pairs):
            # Get the object indices
            if arg1.dim() > 1 and arg1.shape[1] > 1:
                a1_idx = arg1[i].argmax().item() + 1  # +1 because object IDs are 1-indexed
                a2_idx = arg2[i].argmax().item() + 1
            else:
                a1_idx = arg1[i].item() if hasattr(arg1[i], 'item') else int(arg1[i])
                a2_idx = arg2[i].item() if hasattr(arg2[i], 'item') else int(arg2[i])
            
            # Object 3 is left of object 2 (large brown sphere)
            if a1_idx == 3 and a2_idx == 2:
                result[i] = torch.tensor([0.1, 0.9])
            else:
                result[i] = torch.tensor([0.9, 0.1])
        
        return result


class MaterialLearner(TorchLearner):
    """Returns high probability for object 3 having material property"""
    def forward(self, x):
        batch_size = len(x)
        result = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id == 3:  # Object 3 has material
                result[i] = torch.tensor([0.1, 0.9])
            else:
                result[i] = torch.tensor([0.9, 0.1])
        return result