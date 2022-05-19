import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from model import RobertClassification, NLI_Robert, RobertaTokenizerMulti
from utils import check_symmetric, check_transitive
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor


def program_declaration(cur_device, *,
                        sym_relation: bool = False, tran_relation: bool = False,
                        primaldual: bool = False, sample:bool = False, iml: bool = False, beta= 0.5):
    from graph_senetences import graph, sentence, entailment, neutral, \
        contradiction, sentence_group, sentence_group_contains, symmetric, s_sent1, s_sent2, \
        transitive, t_sent1, t_sent2, t_sent3

    graph.detach()
    # Reading directly from data table
    sentence_group["premises"] = ReaderSensor(keyword="premises", device=cur_device)
    sentence_group["hypothesises"] = ReaderSensor(keyword="hypothesises", device=cur_device)
    sentence_group["entailment_list"] = ReaderSensor(keyword="entailment_list", device=cur_device)
    sentence_group["contradiction_list"] = ReaderSensor(keyword="contradiction_list", device=cur_device)
    sentence_group["neutral_list"] = ReaderSensor(keyword="neutral_list", device=cur_device)

    def str_to_int_list(x):
        return torch.LongTensor([int(i) for i in x])
    # Making individual data from batch
    def make_sentence(premises, hypothesises, ent_list, cont_list, neu_list):
        ent_int_list = str_to_int_list(ent_list.split("@@"))
        cont_int_list = str_to_int_list(cont_list.split("@@"))
        neu_int_list = str_to_int_list(neu_list.split("@@"))
        return torch.ones(len(premises.split("@@")), 1), premises.split("@@"), hypothesises.split("@@"), \
               ent_int_list, cont_int_list, neu_int_list

    def read_label(_, label):
        return label

    sentence[sentence_group_contains, "premise", "hypothesis", "entail_list", "cont_list", "neutral_list"] = \
        JointSensor(sentence_group["premises"], sentence_group["hypothesises"], sentence_group["entailment_list"],
                    sentence_group["contradiction_list"], sentence_group["neutral_list"], forward=make_sentence,
                    device=cur_device)

    sentence["token_ids", "Mask"] = JointSensor(sentence_group_contains, 'premise', "hypothesis",
                                                forward=RobertaTokenizerMulti(), device=cur_device)
    roberta_model = NLI_Robert()
    sentence["robert_emb"] = ModuleLearner("token_ids", "Mask", module=roberta_model, device=cur_device)

    # number of hidden layer excluding the first layer and the last layer
    hidden_layer_size = 2
    sentence[entailment] = ModuleLearner("robert_emb",
                                         module=RobertClassification(roberta_model.last_layer_size,
                                                                     hidden_layer_size=hidden_layer_size),
                                         device=cur_device)

    sentence[neutral] = ModuleLearner("robert_emb", module=RobertClassification(roberta_model.last_layer_size,
                                                                                hidden_layer_size=hidden_layer_size),
                                      device=cur_device)

    sentence[contradiction] = ModuleLearner("robert_emb", module=RobertClassification(roberta_model.last_layer_size,
                                                                                      hidden_layer_size=
                                                                                      hidden_layer_size),
                                            device=cur_device)
    sentence[entailment] = FunctionalSensor(sentence_group_contains, "entail_list", forward=read_label, label=True,
                                            device=cur_device)
    sentence[neutral] = FunctionalSensor(sentence_group_contains, "neutral_list", forward=read_label, label=True,
                                         device=cur_device)
    sentence[contradiction] = FunctionalSensor(sentence_group_contains, "cont_list", forward=read_label,
                                               label=True, device=cur_device)
    poi_list = [sentence[entailment], sentence[contradiction], sentence[neutral]]
    # Using symmetric relation
    if sym_relation:
        symmetric[s_sent1.reversed, s_sent2.reversed] = CompositionCandidateSensor(
            relations=(s_sent1.reversed, s_sent2.reversed),
            forward=check_symmetric, device=cur_device)
        poi_list.append(symmetric)

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

    program = None
    if primaldual:
        print("Using Primal Dual Program")
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=['ILP', 'local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    beta=beta,
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    elif sample:
        print("Using Sampling Program")
        program = SampleLossProgram(
            graph, SolverModel,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            beta=beta,
            metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                    'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
            sample= True,
            sampleGlobalLoss= True
        )
    else:
        print("Using simple Program")
        program = SolverPOIProgram(graph, poi=poi_list,
                                   inferTypes=['ILP', 'local/argmax'],
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                           'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program
