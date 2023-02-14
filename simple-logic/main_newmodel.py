def model_declaration():
    import torch
    from domiknows.program import POILossProgram
    from domiknows.program.model.pytorch import PoiModelToWorkWithLearnerWithLoss
    from domiknows.program.loss import BCEWithLogitsLoss
    from domiknows.sensor.pytorch.sensors import ConstantSensor, ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.graph import Property

    from graph import graph, world_contains_x
    from model import Net

    graph.detach()

    world = graph['world']
    x = graph['x']
    y0 = graph['y0']
    y1 = graph['y1']

    world['x'] = ReaderSensor(keyword='x')
    x[world_contains_x] = EdgeSensor(world['x'], relation=world_contains_x, forward=lambda x: x)

    x[y0] = ReaderSensor(keyword='y0', label=True)
    x[y1] = ReaderSensor(keyword='y1', label=True)
    # x[y0] = ModuleLearner(world_contains_x('x'), module=Net(), loss=BCEWithLogitsLoss())
    x[y0] = ModuleLearner(world_contains_x('x'), module=Net())
    x[y1] = ModuleLearner(world_contains_x('x'), module=Net(), loss=BCEWithLogitsLoss())

    program = POILossProgram(graph)
    return program


def main():
    import logging
    import torch

    from graph import x

    logging.basicConfig(level=logging.INFO)

    program = model_declaration()
    data = [{
        'x': [[1.]],
        'y0': [[1.,0.]],
        'y1': [[0.,1.]]
        }]
    program.train(data, train_epoch_num=10, Optim=lambda param: torch.optim.SGD(param, lr=1))
    print('Train loss:', program.model.loss)

    program.test(data)
    print('Test loss:', program.model.loss)

    for world_node in program.populate(data):
        world_node.inferILPResults(fun=lambda val: torch.tensor(val).softmax(dim=-1).detach().cpu().numpy().tolist(), epsilon=None)
        x_node = world_node.getChildDataNodes(x)[0]
        print('y0:', torch.softmax(x_node.getAttribute('<y0>'), dim=-1))
        print('y1:', torch.softmax(x_node.getAttribute('<y1>'), dim=-1))
        print('y0:', x_node.getAttribute('<y0>/ILP'))
        print('y1:', x_node.getAttribute('<y1>/ILP'))

if __name__ == '__main__':
    main()
