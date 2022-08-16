import torch
from torch.nn import functional as F


class NBCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, input, target, *args, **kwargs):
        epsilon = 1e-5
        input = input.view(-1, input.shape[-1])
#         input = input.clamp(min=epsilon, max=1-epsilon)
        target = target.view(-1).to(dtype=torch.long, device=input.device)
        return super().forward(input, target, *args, **kwargs)
    

class NBCrossEntropyDictLoss(torch.nn.CrossEntropyLoss):
    def forward(self, builder, prop, input, target, *args, **kwargs):
        epsilon = 1e-5
        input = input.view(-1, input.shape[-1])
#       input = input.clamp(min=epsilon, max=1-epsilon)
        target = target.view(-1).to(dtype=torch.long, device=input.device)
        return super().forward(input, target, *args, **kwargs)


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight
        # make sure target is float
        target = target.float()
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)


class BCEFocalLoss(BCEWithLogitsLoss):
    def __init__(self, weight=None, pos_weight=None, reduction='mean', alpha=1, gamma=2, with_logits=True):
        super().__init__(weight=weight, pos_weight=pos_weight, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits

    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight or 1
        if self.with_logits:
            bce = F.binary_cross_entropy_with_logits(input, target, weight,
                                                     pos_weight=self.pos_weight,
                                                     reduction='none')
        else:
            # TODO: update weight based on pos_weight
            bce = F.binary_cross_entropy(input, target, weight, reduction='none')

        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt)**self.gamma * bce

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum

        raise ValueError('Unknown reduction method "{}"'.format(self.reduction))


class BCEWithLogitsFocalLoss(torch.nn.Module):
    def __init__(self, weight=None, reduction='mean', alpha=0.5, gamma=2):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight or 1
        logp = F.logsigmoid(input)
        lognp = logp - input  # log(1-1/(1+exp(-x))) = log(exp(-x)/(1+exp(-x))) = log(exp(-x)) + log(1/(1+exp(-x)))
        p = torch.exp(logp)
        # make sure target is float
        target = target.float()
        # FL(p_t) = - alpha_t * (1 - p_t) ** gamma  * log(p_t)
        loss = - self.alpha * (1 - p)**self.gamma * target * logp
        loss += - (1 - self.alpha) * p**self.gamma * (1 - target) * lognp
        loss *= weight

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum

        raise ValueError('Unknown reduction method "{}"'.format(self.reduction))


class BCEWithLogitsIMLoss(torch.nn.Module):
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
        inference = inference.to(device=logp.device)
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


class NBCrossEntropyIMLoss(BCEWithLogitsIMLoss):
    def forward(self, input, inference, target, weight=None):
        num_classes = input.shape[-1]
        target = target.to(dtype=torch.long)
        target = F.one_hot(target, num_classes=num_classes)
        return super().forward(input, inference, target, weight=weight)
