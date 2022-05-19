import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from models import *
from utils import *
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor


def program_declaration(cur_device):
    from graph import graph, paragraph, paragraph_contain, event_relation, relation

    graph.detach()
    # Reading directly from data table
    paragraph["context"] = ReaderSensor(keyword="context", device=cur_device)
    paragraph["eiids1"] = ReaderSensor(keyword="eiids1", device=cur_device)
    paragraph["eiids2"] = ReaderSensor(keyword="eiids2", device=cur_device)
    paragraph["x_tokens_list"] = ReaderSensor(keyword="x_tokens_list", device=cur_device)
    paragraph["y_tokens_list"] = ReaderSensor(keyword="y_tokens_list", device=cur_device)
    paragraph["x_position_list"] = ReaderSensor(keyword="x_position_list", device=cur_device)
    paragraph["y_position_list"] = ReaderSensor(keyword="y_position_list", device=cur_device)
    paragraph["relation_list"] = ReaderSensor(keyword="relation_list", device=cur_device)

    # TODO: Follow the paper code for these steps
    def str_to_int_list(x):
        return torch.LongTensor([int(i) for i in x]).to(cur_device)

    def str_to_token_list(x):
        tokens_list = x.split("@@")
        return torch.IntTensor([[int(i) for i in eval(tokens)] for tokens in tokens_list]).to(cur_device)

    def relation_str_to_list(relations):
        rel = []
        flags = []  # 0 for temporal, otherwise 1
        rel_index = {"BEFORE": '4', "AFTER": '5', "EQUAL": '6', "VAGUE": '7',
                     "SuperSub": '0', "SubSuper": '1', "Coref": '2', "NoRel": '3'}
        for relation in relations:
            rel += [rel_index[relation]]
            flags.append(0 if int(rel_index[relation]) < 4 else 0)
        return str_to_int_list(rel), str_to_int_list(flags)

    def make_event(context, eiids1, eiids2, x_token_list, y_token_list,
                   x_position_list, y_position_list, relation_list):
        # Seperate them from batch to seperate dataset
        # Note that x_tokens_list need to use split -> eval -> torch.tensor
        eiid1_list = str_to_int_list(eiids1.split("@@"))
        eiid2_list = str_to_int_list(eiids2.split("@@"))
        x_token_list = str_to_token_list(x_token_list)
        y_token_list = str_to_token_list(y_token_list)
        x_pos_list = str_to_int_list(x_position_list.split("@@"))
        y_pos_list = str_to_int_list(y_position_list.split("@@"))
        rel, flags = relation_str_to_list(relation_list.split("@@"))
        return torch.ones(len(context.split("@@")), 1), context.split("@@"), \
               eiid1_list, eiid2_list, x_token_list, y_token_list, x_pos_list, y_pos_list, rel, flags

    event_relation[paragraph_contain,
                   "context", "eiid1", "eiid2", "x_sent", "y_sent", "x_pos", "y_pos", "rel_", "flags"] = \
        JointSensor(paragraph["context"], paragraph["eiids1"], paragraph["eiids2"],
                    paragraph["x_tokens_list"], paragraph["y_tokens_list"],
                    paragraph["x_position_list"], paragraph["y_position_list"],
                    paragraph["relation_list"], forward=make_event, device=cur_device)

    def label_reader(_, label):
        return label

    #BiLSTM_model = BiLSTM(512, 128, 1)
    out_model = Robert_Model()

    event_relation["x_output"] = ModuleLearner("x_sent", "x_pos", module=out_model, device=cur_device)
    event_relation["y_output"] = ModuleLearner("y_sent", "y_pos", module=out_model, device=cur_device)

    def make_MLP_input(_, x, y):
        subXY = torch.sub(x, y)
        mulXY = torch.mul(x, y)
        return_input = torch.cat((x, y, subXY, mulXY), 1)
        return return_input

    event_relation["MLP_input"] = FunctionalSensor(paragraph_contain, "x_output", "y_output",
                                                    forward=make_MLP_input, device=cur_device)

    event_relation[relation] = ModuleLearner("MLP_input", module=BiLSTM_MLP(out_model.last_layer_size, 256, 8), device=cur_device)

    event_relation[relation] = FunctionalSensor(paragraph_contain, "rel_",
                                                forward=label_reader, label=True, device=cur_device)

    from regr.program.primaldualprogram import PrimalDualProgram
    from regr.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
    from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
    from regr.program import LearningBasedProgram, IMLProgram, SolverPOIProgram
    from regr.program.model.pytorch import model_helper, PoiModel, SolverModel

    # Define the same weight as original paper
    HierPC = 1802.0
    HierCP = 1846.0
    HierCo = 758.0
    HierNo = 63755.0
    HierTo = HierPC + HierCP + HierCo + HierNo # total number of event pairs
    weights = torch.FloatTensor([0.25*818.0/412.0, 0.25*818.0/263.0, 0.25*818.0/30.0, 0.25*818.0/113.0,
               0.25*HierTo/HierPC, 0.25*HierTo/HierCP, 0.25*HierTo/HierCo, 0.25*HierTo/HierNo]).to(cur_device)

    poi_list = [event_relation, relation]
    # Initial program using only ILP
    program = SolverPOIProgram(graph,
                               inferTypes=['local/argmax'],
                               poi=poi_list,
                               loss=MacroAverageTracker(NBCrossEntropyLoss(weight=weights)),
                               metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                       'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    # print("Using IML Program")
    # program = IMLProgram(graph, poi=poi_list,
    #                      inferTypes=['ILP', 'local/argmax'],
    #                      loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=beta)),
    #                      metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
    #                              'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program
