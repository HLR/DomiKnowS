import torch
from torch.nn import functional as F


class BWithLogitsIMLoss(torch.nn.Module):
    def __init__(self, lmbd, reduction='mean'):
        super().__init__()
        self.lmbd = lmbd
        self.reduction = reduction

    def forward(self, input, inference, target, weight=None):
        if weight is None:
            weight = 1

        logp = F.logsigmoid(input)
        lognp = logp - input  # log(1-1/(1+exp(-x))) = log(exp(-x)/(1+exp(-x))) = log(exp(-x)) + log(1/(1+exp(-x)))
        # make sure target is float
        target = target.to(dtype=logp.dtype)
        inference = inference.to(dtype=logp.dtype)
        # FL(p_t) = - alpha_t * (1 - p_t) ** gamma  * log(p_t)
        loss = - (1 - (1 - self.lmbd) * inference) * target * logp
        loss += - (self.lmbd + (1 - self.lmbd) * inference) * (1 - target) * lognp
        loss *= weight

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum

        raise ValueError('Unknown reduction method "{}"'.format(self.reduction))
