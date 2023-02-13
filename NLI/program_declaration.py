import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from model import RobertClassification, NLI_Robert, RobertaTokenizerMulti
from utils import check_symmetric, check_transitive
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor


def program_declaration(cur_device, *,
                        sym_relation: bool = False, tran_relation: bool = False,
                        primaldual: bool = False, sample: bool = False, beta=0.5, sampling_size=100):
    from graph import graph, group_pairs, pairs, symmetric, s_sent1, s_sent2, \
        transitive, t_sent1, t_sent2, t_sent3, answer_class, group_pair_contains

    graph.detach()
    # Reading directly from data table
    group_pairs["premises_raw"] = ReaderSensor(keyword="premises")
    group_pairs["hypothesises_raw"] = ReaderSensor(keyword="hypothesises")
    group_pairs["labels_raw"] = ReaderSensor(keyword="label_list")

    def str_to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    # Making individual data from batch
    def make_pair(premises, hypothesises, label_list):
        label_list = str_to_int_list(label_list.split("@@"))
        return torch.ones(len(premises.split("@@")), 1), premises.split("@@"), hypothesises.split("@@"), label_list

    def read_label(_, label):
        return label

    pairs[group_pair_contains, "premise", "hypothesis", "label"] = JointSensor(
        group_pairs["premises_raw"],
        group_pairs["hypothesises_raw"],
        group_pairs["labels_raw"], forward=make_pair, device=cur_device)

    # Create token_ids and mask
    pairs["token_ids", "Mask"] = JointSensor(group_pair_contains, 'premise',
                                             "hypothesis",
                                             forward=RobertaTokenizerMulti(), device=cur_device)
    roberta_model = NLI_Robert()
    pairs["robert_emb"] = ModuleLearner("token_ids", "Mask", module=roberta_model, device=cur_device)

    # number of hidden layer excluding the first layer and the last layer
    hidden_layer_size = 2
    # Predict the result from classify layer
    pairs[answer_class] = FunctionalSensor(group_pair_contains, "label",
                                           forward=read_label, label=True, device=cur_device)

    pairs[answer_class] = ModuleLearner("robert_emb",
                                        module=RobertClassification(roberta_model.last_layer_size,
                                                                    hidden_layer_size=hidden_layer_size),
                                        device=cur_device)

    poi_list = [group_pairs, pairs, answer_class]
    # Using symmetric relation
    if sym_relation:
        symmetric[s_sent1.reversed, s_sent2.reversed] = CompositionCandidateSensor(
            relations=(s_sent1.reversed, s_sent2.reversed),
            forward=check_symmetric, device=cur_device)
        poi_list.append(symmetric)
    # Using transitive relation
    if tran_relation:
        transitive[t_sent1.reversed, t_sent2.reversed, t_sent3.reversed] = CompositionCandidateSensor(
            relations=(t_sent1.reversed, t_sent2.reversed, t_sent3.reversed),
            forward=check_transitive, device=cur_device)
        poi_list.append(transitive)

    from regr.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
    from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
    from regr.program import LearningBasedProgram, SolverPOIProgram
    from regr.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from regr.program.model.pytorch import model_helper, PoiModel, SolverModel

    if primaldual:
        print("Using Primal Dual Program")
        program1 = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                     inferTypes=['local/argmax'],
                                     loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                     beta=beta,
                                     device=cur_device,
                                     metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

        program2 = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                     inferTypes=['ILP', 'local/argmax'],
                                     loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                     beta=beta,
                                     device=cur_device,
                                     metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                             'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    elif sample:
        print("Using Sampling Program")
        program1 = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                     inferTypes=['local/argmax'],
                                     loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                     beta=beta,
                                     metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                     sample=True,
                                     sampleSize=sampling_size,
                                     sampleGlobalLoss=True
                                     )

        program2 = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                     inferTypes=['ILP', 'local/argmax'],
                                     loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                     beta=beta,
                                     metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                             'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                     sample=True,
                                     sampleSize=sampling_size,
                                     sampleGlobalLoss=True
                                     )

    else:
        print("Using simple Program")
        program1 = SolverPOIProgram(graph, poi=poi_list,
                                    inferTypes=['local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
        program2 = SolverPOIProgram(graph, poi=poi_list,
                                    inferTypes=['ILP', 'local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program1, program2
