from itertools import product
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = torch.rand(10)
    x[1] += 2
    x[4] += 2
    x = torch.nn.functional.softmax(x)

    y = torch.rand(10)
    y[1] += 2
    y[5] += 2
    y = torch.nn.functional.softmax(y)

    # if we assume we have two valid probability distributes from
    # a neural net x and y, we can combine them as follows
    # This assumes that the index of the vector is the same as the class
    # which allows us to use index additon to model the probabilistic addition
    z = torch.zeros(19)
    for i, j in product(range(10),repeat=2):
        z[i + j] = z[i + j] + x[i]*y[j]

    print(x.sum(), y.sum(), z.sum())
    plt.bar(range(19), z, tick_label=range(19))
    plt.savefig('test.png')