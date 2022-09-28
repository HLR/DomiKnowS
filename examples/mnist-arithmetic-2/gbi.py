import copy
import torch
from torch.optim import SGD

def get_lambda(model, lr=1e-3):
    model_l = copy.deepcopy(model)

    # freeze weights
    #for param in model.parameters():
    #    param.requires_grad = False

    c_opt = SGD(model_l.parameters(), lr=lr)

    return model_l, c_opt


def reg_loss(model_lambda, model, exclude_names=set()):
    orig_params = []
    lambda_params = []

    for (name, w_orig), (_, w_curr) in zip(model.named_parameters(), model_lambda.named_parameters()):
        if name not in exclude_names:
            orig_params.append(w_orig.flatten())
            lambda_params.append(w_curr.flatten())
        else:
            print('skipping %s' % name)

    orig_params = torch.cat(orig_params, dim=0)
    lambda_params = torch.cat(lambda_params, dim=0)

    return torch.linalg.norm(orig_params - lambda_params, dim=0, ord=2)
