import copy
from torch.optim import SGD

def get_lambda(model, lr=1e-3):
    model_l = copy.deepcopy(model)

    # freeze weights
    #for param in model.parameters():
    #    param.requires_grad = False

    c_opt = SGD(model_l.parameters(), lr=lr)

    return model_l, c_opt
