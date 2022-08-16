import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner, LSTMLearner
from models import *
from utils import *
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from sklearn import preprocessing
from transformers import RobertaModel
import numpy as np


def program_declaration(cur_device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False, constrains=False):
    from graph import graph, story, question, answer, story_contain, \
        symmetric, s_quest1, s_quest2, reverse, r_quest1, r_quest2,\
        transitive, t_quest1, t_quest2, t_quest3, transitive_topo, tt_quest1, tt_quest2, tt_quest3, tt_quest4

    story["questions"] = ReaderSensor(keyword="questions", device=cur_device)
    story["stories"] = ReaderSensor(keyword="stories", device=cur_device)
    story["relations"] = ReaderSensor(keyword="relation", device=cur_device)
    story["question_ids"] = ReaderSensor(keyword="question_ids", device=cur_device)
    story["labels"] = ReaderSensor(keyword="labels", device=cur_device)

    def str_to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        label_nums = [0 if label == "Yes" else 1 for label in labels]
        return str_to_int_list(label_nums)

    def make_question(questions, stories, relations, q_ids, labels):
        num_labels = make_labels(labels)
        ids = str_to_int_list(q_ids.split("@@"))
        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), \
        relations.split("@@"), ids, num_labels

    question[story_contain, "question", "story", "relation", "id", "label"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=cur_device)

    def read_label(_, label):
        return label

    question[answer] = FunctionalSensor(story_contain, "label", forward=read_label, label=True, device=cur_device)

    question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                        forward=BERTTokenizer(), device=cur_device)

    clf = BertForBooleanQuestion3ClassYN.from_pretrained('bert-base-uncased', device=cur_device, drp=dropout)

    question[answer] = ModuleLearner("input_ids", module=clf, device=cur_device)

    poi_list = [question, answer]
    if constrains:
        symmetric[s_quest1.reversed, s_quest2.reversed] = \
            CompositionCandidateSensor(
            relations=(s_quest1.reversed, s_quest2.reversed),
            forward=check_symmetric, device=cur_device)

        reverse[r_quest1.reversed, r_quest2.reversed] = \
            CompositionCandidateSensor(
            relations=(r_quest1.reversed, r_quest2.reversed),
            forward=check_reverse, device=cur_device)

        transitive[t_quest1.reversed, t_quest2.reversed, t_quest3.reversed] = \
            CompositionCandidateSensor(
            relations=(t_quest1.reversed, t_quest2.reversed, t_quest3.reversed),
            forward=check_transitive, device=cur_device)

        transitive_topo[tt_quest1.reversed, tt_quest2.reversed, tt_quest3.reversed, tt_quest4.reversed] =\
            CompositionCandidateSensor(
            relations=(tt_quest1.reversed, tt_quest2.reversed, tt_quest3.reversed, tt_quest4.reversed),
            forward=check_transitive_topo, device=cur_device)

        poi_list.extend([symmetric, reverse, transitive, transitive_topo])

    from regr.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker
    from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
    from regr.program import LearningBasedProgram, SolverPOIProgram
    from regr.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from regr.program.model.pytorch import model_helper, PoiModel, SolverModel

    # Initial program using only ILP
    # symmetric[s_event1.reversed, s_event2.reversed] = CompositionCandidateSensor(
    #     relations=(s_event1.reversed, s_event2.reversed),
    #     forward=check_symmetric, device=cur_device)
    #
    # transitive[t_event1.reversed, t_event2.reversed, t_event3.reversed] = CompositionCandidateSensor(
    #     relations=(t_event1.reversed, t_event2.reversed, t_event3.reversed),
    #     forward=check_transitive, device=cur_device)

    infer_list = ['ILP', 'local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    beta=beta,
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    device=cur_device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=True,
                                    device=cur_device)
    else:
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                           'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                   device=cur_device)

    return program