import torch
from regr.program import POIProgram
from regr.sensor.pytorch.sensors import ReaderSensor

def model(graph, ):
    graph.detach()

    ling_graph = graph['linguistic']
    sentence = ling_graph['sentence']

    sentence['raw'] = ReaderSensor(keyword='text')

    program = POIProgram(graph, poi=(sentence['raw'],))

    return program
