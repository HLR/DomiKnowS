def model_declaration():
    import torch
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import PoiModel
    from regr.program.loss import BCEWithLogitsLoss
    from regr.program.metric import MacroAverageTracker
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.sensor.pytorch.learners import FullyConnectedLearner
    from graph import graph

    graph.detach()

    x = graph['input/x']
    y0 = graph['output/y0']
    y1 = graph['output/y1']

    x['val'] = ReaderSensor(keyword='x')
    x[y0] = ReaderSensor(keyword='y0', label=True)
    x[y1] = ReaderSensor(keyword='y1', label=True)
    x[y0] = FullyConnectedLearner('val', input_dim=1, output_dim=2)
    x[y1] = FullyConnectedLearner('val', input_dim=1, output_dim=2)

    program = LearningBasedProgram(graph, lambda graph: PoiModel(graph, loss=MacroAverageTracker(BCEWithLogitsLoss())))
    return program


def main():
    import torch

    program = model_declaration()
    data = [{
        'x': torch.tensor([[[1.]]]),
        'y0': torch.tensor([[1.,0.]]),
        'y1': torch.tensor([[0.,1.]])
        }]
    program.train(data, train_epoch_num=10, Optim=lambda param: torch.optim.SGD(param, lr=1))
    datanode = next(program.populate(data))
    print(datanode.getAttribute('<y0>'))
    print(datanode.getAttribute('<y1>'))


if __name__ == '__main__':
    main()