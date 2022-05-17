import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from models import *
from utils import *
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor


def program_declaration(cur_device):
    from graph import graph, paragraph, paragraph_contain, event, sub_relation, temp_relation,\
        symmetric, s_event1, s_event2

    graph.detach()
    # Reading directly from data table
    paragraph["context"] = ReaderSensor(keyword="context", device=cur_device)
    paragraph["eiids"] = ReaderSensor(keyword="eiids", device=cur_device)
    paragraph["x_events"] = ReaderSensor(keyword="x_events", device=cur_device)
    paragraph["y_events"] = ReaderSensor(keyword="y_events", device=cur_device)
    paragraph["x_token_list"] = ReaderSensor(keyword="x_token_list", device=cur_device)
    paragraph["y_token_list"] = ReaderSensor(keyword="y_token_list", device=cur_device)
    paragraph["x_position_list"] = ReaderSensor(keyword="x_position_list", device=cur_device)
    paragraph["y_position_list"] = ReaderSensor(keyword="y_position_list", device=cur_device)
    paragraph["relation_list"] = ReaderSensor(keyword="relation_list", device=cur_device)

    #TODO: Follow the paper code for these steps

    program = None
    return program
