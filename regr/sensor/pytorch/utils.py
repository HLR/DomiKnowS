import torch

class UnBatchWrap(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def batch(self, value):
        if isinstance(value, torch.Tensor):
            return value.unsqueeze(0)
        return value

    def unbatch(self, value):
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == 1
            return value.squeeze(0)
        return value

    def forward(self, *args, **kwargs):
        args = list(map(self.batch, args))
        kwargs = dict(map(lambda item: (item[0], self.batch(item[1])), kwargs.items()))
        out = self.module(*args, **kwargs)
        if isinstance(out, tuple):
            return tuple(map(self.unbatch, out))
        return self.unbatch(out)
