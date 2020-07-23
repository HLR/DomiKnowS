def model_declaration():
    import torch
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import PoiModel, IMLModel
    from regr.program.loss import BCEWithLogitsLoss, BCEWithLogitsIMLoss
    from regr.program.metric import MacroAverageTracker, ValueTracker
    from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeReaderSensor
    from regr.sensor.pytorch.learners import FullyConnected2Learner
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    from regr.solver.contextsolver.pytorch import Solver
    from regr.graph import Property

    from graph import graph, world_contains_x

    graph.detach()

    world = graph['world']
    x = graph['x']
    y0 = graph['y0']
    y1 = graph['y1']

    world['x'] = ReaderSensor(keyword='x')
    world_contains_x['forward'] = TorchEdgeReaderSensor('x', mode='forward', keyword='x')
    with x:
        Property('x')
    # x['x'] = ReaderSensor(keyword='x')
    x[y0] = ReaderSensor(keyword='y0', label=True)
    x[y1] = ReaderSensor(keyword='y1', label=True)
    x[y0] = FullyConnected2Learner('x', edges=[world_contains_x['forward']], input_dim=1, output_dim=2)
    x[y1] = FullyConnected2Learner('x', edges=[world_contains_x['forward']], input_dim=1, output_dim=2)

    program = LearningBasedProgram(
        graph, 
        lambda graph: IMLModel(
            graph, 
            loss=MacroAverageTracker(BCEWithLogitsIMLoss(0.5)),
            metric=ValueTracker(lambda pr, gt: pr.data),
            Solver=ilpOntSolverFactory.getOntSolverInstance))
    return program


def main():
    import logging
    import torch

    logging.basicConfig(level=logging.INFO)

    program = model_declaration()
    data = [{
        'x': torch.tensor([[[1.]]]),
        'y0': torch.tensor([[[1.,0.]]]),
        'y1': torch.tensor([[[0.,1.]]])
        }]
    program.train(data, train_epoch_num=10, Optim=lambda param: torch.optim.SGD(param, lr=1))
    for metric, x_node in program.test(data):
        print(metric)
        print(x_node.getAttribute('<y0>'))
        print(x_node.getAttribute('<y1>'))


if __name__ == '__main__':
    main()
