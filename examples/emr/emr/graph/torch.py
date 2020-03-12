import torch


class TorchModel(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def forward(self, data):
        pass


def train(model, dataset, opt):
    model.train()
    for data in dataset:
        opt.zero_grad()
        loss, metric, output = model(data)
        loss.backward()
        opt.step()
        yield loss, metric, output


def test(model, dataset):
    model.eval()
    with torch.no_grad():
        for data in dataset:
            loss, metric, output = model(data)
            yield loss, metric, output


def eval_many(model, dataset):
    model.eval()
    with torch.no_grad():
        for data in dataset:
            _, _, output = model(data)
            yield output


def eval_one(model, data):
    model.eval()
    with torch.no_grad():
        _, _, output = model(data)
        return output
