import torch

p = torch.tensor([1/10] * 10)
joint = torch.zeros((10, 10, 19))
for i in range(10):
    for j in range(10):
        for s in range(19):
            if i + j == s:
                joint[i, j, s] = p[i] * p[j]


def get_digit_label(summation):
    # summation: label summation
    # digit, digit_idx: digit to get label for

    # P(d_0 = i and sum = s)
    margin = torch.sum(joint, dim=1)[:, summation]

    # P(sum = s)
    psum = torch.sum(torch.sum(joint, dim=0), dim=0)[summation]

    # P(d_0 = i | sum = 2)
    cond = margin/psum

    return torch.unsqueeze(cond, dim=0)


digit_labels = torch.empty((19, 10))
for i in range(19):
    digit_labels[i] = get_digit_label(i)
