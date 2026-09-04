import torch
from domiknows.sensor.pytorch.learners import TorchLearner


class ColorEnumLearner(TorchLearner):
    """
    Returns probabilities for EnumConcept color with values ['red', 'blue', 'green'].
    Output shape: (batch_size, 3) where index 0=red, 1=blue, 2=green

    Object 1 -> red, Object 2 -> red, Object 3 -> blue
    """
    def forward(self, x):
        batch_size = len(x)
        device = x.device if hasattr(x, 'device') else 'cpu'
        result = torch.zeros(batch_size, 3, device=device)
        for i in range(batch_size):
            obj_id = x[i].item() if hasattr(x[i], 'item') else int(x[i])
            if obj_id in [1, 2]:
                # red
                result[i] = torch.tensor([0.9, 0.05, 0.05], device=device)
            elif obj_id == 3:
                # blue
                result[i] = torch.tensor([0.05, 0.9, 0.05], device=device)
            else:
                result[i] = torch.tensor([0.33, 0.33, 0.34], device=device)
        return result


class CompareLearner(TorchLearner):
    """
    Returns high probability for all test pairs being 'compare' pairs.
    All pairs in the test data should be compared.
    Output shape: (batch_size, 2) where [not_compare, compare]
    """
    def forward(self, *args):
        # All pairs are compare pairs
        x = args[0] if args else args
        batch_size = len(x) if hasattr(x, '__len__') else 1
        device = x.device if hasattr(x, 'device') else 'cpu'
        result = torch.zeros(batch_size, 2, device=device)
        for i in range(batch_size):
            result[i] = torch.tensor([0.05, 0.95], device=device)
        return result
