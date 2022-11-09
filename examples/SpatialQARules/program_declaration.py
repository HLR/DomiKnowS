import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner, LSTMLearner
from models import *
from utils import *
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from sklearn import preprocessing
from transformers import RobertaModel
import numpy as np


def program_declaration(cur_device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                        constrains=False):
    from graph import graph, story, question, answer_class, story_contain, \
        symmetric, s_quest1, s_quest2, reverse, r_quest1, r_quest2, \
        transitive, t_quest1, t_quest2, t_quest3, transitive_topo, tt_quest1, tt_quest2, tt_quest3, tt_quest4

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

    def str_to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
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

    question[answer_class] = FunctionalSensor(story_contain, "label", forward=read_label, label=True, device=cur_device)

    question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                        forward=BERTTokenizer(), device=cur_device)

    clf = MultipleClassYN.from_pretrained('bert-base-uncased', device=cur_device, drp=dropout)

    question[answer_class] = ModuleLearner("input_ids", module=clf, device=cur_device)

    poi_list = [question, answer_class]

    # Including the constrains relation check
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

        transitive_topo[tt_quest1.reversed, tt_quest2.reversed, tt_quest3.reversed, tt_quest4.reversed] = \
            CompositionCandidateSensor(
                relations=(tt_quest1.reversed, tt_quest2.reversed, tt_quest3.reversed, tt_quest4.reversed),
                forward=check_transitive_topo, device=cur_device)

        poi_list.extend([symmetric, reverse, transitive, transitive_topo])

    from regr.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker, ValueTracker
    from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss, BCEFocalLoss
    from regr.program import LearningBasedProgram, SolverPOIProgram
    from regr.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from regr.program.model.pytorch import model_helper, PoiModel, SolverModel

    infer_list = ['local/argmax']  # ['ILP', 'local/argmax']
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
                                    sampleGlobalLoss=False,
                                    beta=1,
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


def program_declaration_spartun_fr(device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                                   constrains=False, spartun=True):
    program = None
    from graph_spartun_rel import graph, story, story_contain, question, \
        left, right, above, below, behind, front, near, far, disconnected, touch, \
        overlap, coveredby, inside, cover, contain

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")
    all_labels = ["left", "right", "above", "below", "behind", "front", "near", "far", "dc", "ec", "po", "tpp", "ntpp",
                  "tppi", "ntppi"]

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        all_labels_list = [[] for _ in range(15)]
        for bits_label in labels:
            bits_label = int(bits_label)
            cur_label = 1
            for ind, label in enumerate(all_labels):
                all_labels_list[ind].append(1 if bits_label & cur_label else 0)
                cur_label *= 2

        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return [to_int_list(labels_list) for labels_list in all_labels_list]

    def make_question(questions, stories, relations, q_ids, labels):
        all_labels = make_labels(labels)
        ids = to_int_list(q_ids.split("@@"))
        left_list, right_list, above_list, below_list, behind_list, \
        front_list, near_list, far_list, dc_list, ec_list, po_list, \
        tpp_list, ntpp_list, tppi_list, ntppi_list = all_labels
        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), \
               relations.split("@@"), ids, left_list, right_list, above_list, below_list, behind_list, \
               front_list, near_list, far_list, dc_list, ec_list, po_list, \
               tpp_list, ntpp_list, tppi_list, ntppi_list

    question[story_contain, "question", "story", "relation", "id", "left_label", "right_label",
             "above_label", "below_label", "behind_label", "front_label", "near_label", "far_label", "dc_label", "ec_label", "po_label",
             "tpp_label", "ntpp_label", "tppi_label", "ntppi_label"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=device)

    def read_label(_, label):
        return label

    # question[answer_class] =
    # FunctionalSensor(story_contain, "label", forward=read_label, label=True, device=cur_device)
    # Replace with all classes

    question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                        forward=BERTTokenizer(), device=device)

    clf1 = MultipleClassYN_Hidden.from_pretrained('bert-base-uncased', device=device, drp=dropout)

    question["hidden_layer"] = ModuleLearner("input_ids", module=clf1, device=device)

    question[left] = ModuleLearner("hidden_layer",
                                   module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                   device=device)
    question[left] = FunctionalSensor(story_contain, "left_label", forward=read_label, label=True, device=device)

    question[right] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[right] = FunctionalSensor(story_contain, "right_label", forward=read_label, label=True, device=device)

    question[above] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[above] = FunctionalSensor(story_contain, "above_label", forward=read_label, label=True, device=device)

    question[below] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[below] = FunctionalSensor(story_contain, "below_label", forward=read_label, label=True, device=device)

    question[behind] = ModuleLearner("hidden_layer",
                                     module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                     device=device)
    question[behind] = FunctionalSensor(story_contain, "behind_label", forward=read_label, label=True, device=device)

    question[front] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[front] = FunctionalSensor(story_contain, "front_label", forward=read_label, label=True, device=device)

    question[near] = ModuleLearner("hidden_layer",
                                   module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                   device=device)
    question[near] = FunctionalSensor(story_contain, "near_label", forward=read_label, label=True, device=device)

    question[far] = ModuleLearner("hidden_layer",
                                  module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                  device=device)
    question[far] = FunctionalSensor(story_contain, "far_label", forward=read_label, label=True, device=device)

    question[disconnected] = ModuleLearner("hidden_layer",
                                           module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                           device=device)
    question[disconnected] = FunctionalSensor(story_contain, "dc_label", forward=read_label, label=True, device=device)

    question[touch] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device)
    question[touch] = FunctionalSensor(story_contain, "ec_label", forward=read_label, label=True, device=device)

    question[overlap] = ModuleLearner("hidden_layer",
                                      module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device)
    question[overlap] = FunctionalSensor(story_contain, "po_label", forward=read_label, label=True, device=device)

    question[coveredby] = ModuleLearner("hidden_layer",
                                        module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                        device=device)
    question[coveredby] = FunctionalSensor(story_contain, "tpp_label", forward=read_label, label=True, device=device)

    question[inside] = ModuleLearner("hidden_layer",
                                     module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                     device=device)
    question[inside] = FunctionalSensor(story_contain, "ntpp_label", forward=read_label, label=True, device=device)

    question[cover] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[cover] = FunctionalSensor(story_contain, "tppi_label", forward=read_label, label=True, device=device)

    question[contain] = ModuleLearner("hidden_layer",
                                      module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                      device=device)
    question[contain] = FunctionalSensor(story_contain, "ntppi_label", forward=read_label, label=True, device=device)

    poi_list = [question, left, right, above, below, behind, front, near, far,
                disconnected, touch, overlap, coveredby, inside, cover, contain]

    from regr.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker, ValueTracker
    from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss, BCEFocalLoss
    from regr.program import LearningBasedProgram, SolverPOIProgram
    from regr.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from regr.program.model.pytorch import model_helper, PoiModel, SolverModel

    infer_list = ['local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    beta=beta,
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    device=device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=False,
                                    beta=1,
                                    device=device)
    else:
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                   device=device)

    return program


def program_declaration_StepGame(device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False,
                                 constrains=False, spartun=True):
    program = None
    from graph_stepgame import graph, story, story_contain, question, \
        left, right, above, below, lower_left, lower_right, upper_left, upper_right, overlap
    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")
    all_labels = ["left", "right", "above", "below", "lower-left",
                  "lower-right", "upper-left", "upper-right", "overlap"]

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        all_labels_list = [[] for _ in range(15)]
        for cur_label in labels:
            for ind, label in enumerate(all_labels):
                all_labels_list[ind].append(1 if label == cur_label else 0)
        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return [to_int_list(labels_list) for labels_list in all_labels_list]

    def make_question(questions, stories, relations, q_ids, labels):
        all_labels = make_labels(labels)
        ids = to_int_list(q_ids.split("@@"))
        left_list, right_list, above_list, below_list, lower_left_list, \
        lower_right_list, upper_left_list, upper_right_list, over_lap_list = all_labels
        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), \
               relations.split("@@"), ids, left_list, right_list, above_list, below_list, lower_left_list, \
               lower_right_list, upper_left_list, upper_right_list, over_lap_list

    question[story_contain, "question", "story", "relation", "id", "left_label", "right_label",
             "above_label", "below_label", "behind_label", "front_label", "near_label", "far_label", "dc_label", "ec_label", "po_label",
             "tpp_label", "ntpp_label", "tppi_label", "ntppi_label"] = \
        JointSensor(story["questions"], story["stories"], story["relations"],
                    story["question_ids"], story["labels"], forward=make_question, device=device)

    def read_label(_, label):
        return label

    # question[answer_class] =
    # FunctionalSensor(story_contain, "label", forward=read_label, label=True, device=cur_device)
    # Replace with all classes

    question["input_ids"] = JointSensor(story_contain, 'question', "story",
                                        forward=BERTTokenizer(), device=device)

    clf1 = MultipleClassYN_Hidden.from_pretrained('bert-base-uncased', device=device, drp=dropout)

    question["hidden_layer"] = ModuleLearner("input_ids", module=clf1, device=device)

    question[left] = ModuleLearner("hidden_layer",
                                   module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                   device=device)
    question[left] = FunctionalSensor(story_contain, "left_label", forward=read_label, label=True, device=device)

    question[right] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[right] = FunctionalSensor(story_contain, "right_label", forward=read_label, label=True, device=device)

    question[above] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[above] = FunctionalSensor(story_contain, "above_label", forward=read_label, label=True, device=device)

    question[below] = ModuleLearner("hidden_layer",
                                    module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                    device=device)
    question[below] = FunctionalSensor(story_contain, "below_label", forward=read_label, label=True, device=device)

    question[lower_left] = ModuleLearner("hidden_layer",
                                         module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                         device=device)
    question[lower_left] = FunctionalSensor(story_contain, "lower_left_label", forward=read_label, label=True,
                                            device=device)

    question[lower_right] = ModuleLearner("hidden_layer",
                                          module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                          device=device)
    question[lower_right] = FunctionalSensor(story_contain, "lower_right_label", forward=read_label, label=True,
                                             device=device)

    question[upper_left] = ModuleLearner("hidden_layer",
                                         module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                         device=device)
    question[upper_left] = FunctionalSensor(story_contain, "upper_left_label", forward=read_label, label=True,
                                            device=device)

    question[upper_right] = ModuleLearner("hidden_layer",
                                          module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                          device=device)
    question[upper_right] = FunctionalSensor(story_contain, "upper_right_label", forward=read_label, label=True,
                                             device=device)

    question[overlap] = ModuleLearner("hidden_layer",
                                      module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout),
                                      device=device)
    question[overlap] = FunctionalSensor(story_contain, "overlap_label", forward=read_label, label=True, device=device)

    poi_list = [question, left, right, above, below, lower_left, lower_right, upper_left, upper_right,
                overlap]

    from regr.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker, ValueTracker
    from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss, BCEFocalLoss
    from regr.program import LearningBasedProgram, SolverPOIProgram
    from regr.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from regr.program.model.pytorch import model_helper, PoiModel, SolverModel

    infer_list = ['local/argmax']  # ['ILP', 'local/argmax']
    if pmd:
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    beta=beta,
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    device=device)
    elif sampling:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=False,
                                    beta=1,
                                    device=device)
    else:
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                   device=device)

    return program
