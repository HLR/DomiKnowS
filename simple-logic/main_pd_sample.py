import sys
sys.path.append("../..")

def model_declaration():
    import torch
    from domiknows.program import LearningBasedProgram, IMLProgram
    from domiknows.program.lossprogram import SampleLossProgram
    from domiknows.program.model.pytorch import SolverModel
    from domiknows.sensor.pytorch.sensors import ConstantSensor, ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.graph import Property
    from domiknows.program.loss import BCEWithLogitsLoss, BCEWithLogitsIMLoss, NBCrossEntropyLoss
    
    from domiknows.program.metric import MacroAverageTracker, ValueTracker, PRF1Tracker, DatanodeCMMetric

    from graph import graph, world_contains_x
    from model import MyModel, MyIMLModel, Net, prediction_softmax

    graph.detach()

    world = graph['world']
    x = graph['x']
    y0 = graph['y0']
    y1 = graph['y1']

    world['x'] = ReaderSensor(keyword='x')
    x[world_contains_x] = EdgeSensor(world['x'], relation=world_contains_x, forward=lambda x: x)

    x[y0] = ReaderSensor(keyword='y0', label=True)
    x[y1] = ReaderSensor(keyword='y1', label=True)
    x[y0] = ModuleLearner(world_contains_x('x'), module=Net())
    x[y1] = ModuleLearner(world_contains_x('x'), module=Net())

    # program = LearningBasedProgram(graph, MyIMLModel)
    program = SampleLossProgram(
        graph, SolverModel,
        poi=(world, x,),
        # inferTypes=['ILP', 'local/argmax'],
        
        metric=ValueTracker(prediction_softmax),

        #metric={ 'softmax' : ValueTracker(prediction_softmax),
        #       'ILP': PRF1Tracker(DatanodeCMMetric()),
        #        'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))
        #       },
        loss=MacroAverageTracker(BCEWithLogitsLoss()),
        
        sample = True,
        sampleSize=100, 
        sampleGlobalLoss = True
        )
   
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
    program.train(data, train_epoch_num=2000, c_warmup_iters=0, Optim=lambda param: torch.optim.SGD(param, lr=1), device='auto')
    print('Train loss:', program.model.loss)

    program.test(data)
    print('Test loss:', program.model.loss)

    for world_node in program.populate(data):
        world_node.inferILPResults()
        x_node = world_node.getChildDataNodes(x)[0]
        
        print('\n')
        print('y0 Label  :', x_node.getAttribute('<y0>/label')[1].item())
        print('y0 ILP    :', x_node.getAttribute('<y0>/ILP').item())
        print('y0 Softmax:', x_node.getAttribute('<y0>').softmax(dim=-1))
        print('\n')
        print('y1 Label :', x_node.getAttribute('<y1>/label')[1].item())
        print('y1 ILP    :', x_node.getAttribute('<y1>/ILP').item())
        print('y1 Softmax:', x_node.getAttribute('<y1>').softmax(dim=-1))

if __name__ == '__main__':
    main()
