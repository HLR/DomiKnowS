import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from model import RobertClassification, NLI_Robert, RobertaTokenizerMulti
from utils import check_symmetric
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor


def program_declaration(cur_device, *, sym_relation:bool =True):
    from graph_senetences import graph, sentence, entailment, neutral, \
        contradiction, sentence_group, sentence_group_contains, symmetric, s_sent1, s_sent2

    graph.detach()
    sentence_group["premises"] = ReaderSensor(keyword="premises", device=cur_device)
    sentence_group["hypothesises"] = ReaderSensor(keyword="hypothesises", device=cur_device)
    sentence_group["entailment_list"] = ReaderSensor(keyword="entailment_list", device=cur_device)
    sentence_group["contradiction_list"] = ReaderSensor(keyword="contradiction_list", device=cur_device)
    sentence_group["neutral_list"] = ReaderSensor(keyword="neutral_list", device=cur_device)

    def str_to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

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

    sentence["token_ids", "Mask"] = JointSensor(sentence_group_contains, 'hypothesis', "premise",
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
    if sym_relation:
        symmetric[s_sent1.reversed, s_sent2.reversed] = CompositionCandidateSensor(
            relations=(s_sent1.reversed, s_sent2.reversed),
            forward=check_symmetric, device=cur_device)

    from regr.program import POIProgram, IMLProgram, SolverPOIProgram
    from regr.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
    from regr.program.loss import NBCrossEntropyLoss

    # Creating the program to create model
    # Pdual with lr = 5 1 3
    # IMP with lr = [0, 1] 0.5
    poi_list = [sentence, entailment, contradiction, neutral]
    if sym_relation:
        poi_list.append(symmetric)
    program = SolverPOIProgram(graph, poi=poi_list,
                               inferTypes=['ILP', 'local/argmax'],
                               loss=MacroAverageTracker(NBCrossEntropyLoss()),
                               metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                       'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program
