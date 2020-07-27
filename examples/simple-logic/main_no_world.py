def model_declaration():
    import torch
    from regr.program import LearningBasedProgram
    from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeReaderSensor
    from regr.sensor.pytorch.learners import FullyConnected2Learner
    from regr.graph import Property

    from graph_no_world import graph
    from model import MyModel, MyIMLModel

    graph.detach()

    x = graph['x']
    y0 = graph['y0']
    y1 = graph['y1']

    x['x'] = ReaderSensor(keyword='x')
    x[y0] = ReaderSensor(keyword='y0', label=True)
    x[y1] = ReaderSensor(keyword='y1', label=True)
    x[y0] = FullyConnected2Learner('x', input_dim=1, output_dim=2)
    x[y1] = FullyConnected2Learner('x', input_dim=1, output_dim=2)

    program = LearningBasedProgram(graph, MyIMLModel)
    return program


def main():
    import logging
    import torch

    from graph import x

    logging.basicConfig(level=logging.INFO)

    program = model_declaration()
    data = [{
        'x': [1.],
        'y0': [[1.,0.]],
        'y1': [[0.,1.]]
        }]
    program.train(data, train_epoch_num=10, Optim=lambda param: torch.optim.SGD(param, lr=1))
    for loss, metric, x_node in program.test(data):
        print('loss:', loss)
        # print(metric)
        print('y0:', torch.softmax(x_node.getAttribute('<y0>'), dim=-1))
        print('y1:', torch.softmax(x_node.getAttribute('<y1>'), dim=-1))

if __name__ == '__main__':
    main()
