import copy
import torch
from torch.optim import SGD
import pickle

def get_lambda(model, lr=1e-3):
    model_l = copy.deepcopy(model)

    # freeze weights
    #for param in model.parameters():
    #    param.requires_grad = False

    c_opt = SGD(model_l.parameters(), lr=lr)

    #for param in model.parameters():
    #    param.requires_grad = False

    return model_l, c_opt


def reg_loss(model, model_lambda, exclude_names=set()):
    orig_params = []
    lambda_params = []

    for (orig_name, w_orig), (lmbd_name, w_curr) in zip(model.named_parameters(), model_lambda.named_parameters()):
        if orig_name not in exclude_names:
            assert lmbd_name in orig_name

            orig_params.append(w_orig.flatten())
            lambda_params.append(w_curr.flatten())

    orig_params = torch.cat(orig_params, dim=0)
    lambda_params = torch.cat(lambda_params, dim=0)

    return torch.linalg.norm(orig_params - lambda_params, dim=0, ord=2)
