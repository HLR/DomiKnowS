import functools
import operator
import torch


# Selecting the maximum values from a batchified set of inputs
def max_value_select(_input):
    temp = torch.norm(_input.float(), dim=-1, p=2)
    return torch.max(temp, dim=1)[0]


# selecting the maximum index from a batchified set of inputs
def max_index_select(_input):
    temp = torch.norm(_input.float(), dim=-1, p=2)
    return torch.max(temp, dim=1)[1]


# selecting the minimum index from a batchified set of inputs
def min_index_select(_input):
    temp = torch.norm(_input.float(), dim=-1, p=2)
    return torch.min(temp, dim=1)[1]


# selecting the minimum value from a batchified set of inputs
def min_value_select(_input):
    temp = torch.norm(_input.float(), dim=-1, p=2)
    return torch.min(temp, dim=1)[0]


# selecting the first index from a batchified set of inputs
def first_value_select(_input):
    return _input.gather[:, 0]


# selecting the last index from a batchified set of inputs
def last_value_select(_input):
    return _input.gather[:, -1]
