from domiknows.program.metric import MetricTracker
from transformers import AutoTokenizer

from models import Tokenizer, T5WithLoraGenerativeCLF
import torch
from utils import check_symmetric, check_transitive, check_transitive_topo


class ValueTracker(MetricTracker):
    def forward(self, values):
        return values


# class T5LossFunction(torch.nn.CrossEntropyLoss):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, target, *args, **kwargs):
#         """
#         :param input: (batch size, seq length, num class)
#         :param target: (batch size, seq length)
#         """
#         # Changing the input from (batch size, seq length, num class) -> (batch size, num class, seq length)
#         # ori_loss = super().forward(input, target)
#         input = input.transpose(1, 2)
#         # loss = super().forward(input, target)
#         # print(loss, ori_loss)
#         return super().forward(input, target)


def program_declaration(device='cpu', pmd=False, beta=1, constraints=False):
    from graph import (graph, context, question, rel_context_contain_question,
                       rel_question_contain_answer, answer_relations,
                       answer, inverse, inv_question1, inv_question2,
                       transitive, tran_quest1, tran_quest2, tran_quest3,
                       tran_topo, tran_topo_quest1, tran_topo_quest2, tran_topo_quest3, tran_topo_quest4)
    from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor
    from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor, EdgeSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    context["questions"] = ReaderSensor(keyword="questions")
    context["stories"] = ReaderSensor(keyword="stories")
    context["relations"] = ReaderSensor(keyword="relation")
    context["question_ids"] = ReaderSensor(keyword="question_ids")
    context["labels"] = ReaderSensor(keyword="labels")
    all_labels = ["left", "right", "above", "below", "behind", "front",
                  "near", "far", "disconnected", "touch", "overlap", "coveredby",
                  "inside", "cover", "contain"]

    map_label_index = {text: i for i, text in enumerate(all_labels)}
    max_label = 0
    max_size = 5

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        text_label = [[] for _ in range(len(labels))]
        for ind, bits_label in enumerate(labels):
            bits_label = int(bits_label)
            cur_bit = 1
            for i, label in enumerate(all_labels):
                if bits_label & cur_bit:
                    text_label[ind].append(i)
                cur_bit *= 2

        for i, label in enumerate(text_label):
            label.append(len(all_labels))
            while len(label) < max_size:
                label.append(len(all_labels) + 1)
        text_label = torch.LongTensor(text_label)
        return text_label

    def make_question(questions, stories, relations, q_ids, labels):
        text_label = make_labels(labels)
        ids = to_int_list(q_ids.split("@@"))

        return torch.ones(len(questions.split("@@")), 1), questions.split("@@"), stories.split("@@"), relations.split(
            "@@"), ids, text_label

    question[rel_context_contain_question, "question", "story", "relation", "id", "labels"] = \
        JointSensor(context["questions"], context["stories"], context["relations"],
                    context["question_ids"], context["labels"], forward=make_question, device=device)

    tokenizer = Tokenizer("t5-small", label=all_labels)

    def make_answer(labels):
        question_size = labels.size()[0]
        all_answer = question_size * labels.size()[1]

        contain_rel = []
        for i in range(question_size):
            contain = []
            for j in range(all_answer):
                contain.append(i * labels.size()[1] <= j < (i + 1) * labels.size()[1])
            contain_rel.append(contain)

        return torch.Tensor(contain_rel).T

    answer[rel_question_contain_answer] = EdgeSensor(question["labels"], forward=make_answer,
                                                     relation=rel_question_contain_answer, device=device)

    def forward_answer(label):
        # print(torch.unsqueeze(label, -1))
        labels = torch.flatten(label)
        return labels
    answer[answer_relations] = JointSensor(question["labels"], forward=forward_answer, device=device, label=True)

    # question[rel_question_contain_answer.reversed] = EdgeSensor(question[rel_context_contain_question],
    # max_size,
    # relation=rel_question_contain_answer.reversed,
    # forward=match_question)

    print("USING T5")

    T5Model = T5WithLoraGenerativeCLF("google/flan-t5-small", tokenizer=tokenizer, device=device, max_length=5)

    question["input_ids"] = JointSensor(rel_context_contain_question, "question", "story",
                                        forward=tokenizer, device=device)

    # question[answer_relations] = JointSensor(question["labels"], forward=lambda x: x, device=device, label=True)

    answer[answer_relations] = ModuleLearner(question[rel_context_contain_question], question["input_ids"], question["labels"],
                                               module=T5Model, device=device)

    poi_list = [answer_relations, question, answer]

    if constraints:
        inverse[inv_question1.reversed, inv_question2.reversed] = \
            CompositionCandidateSensor(
                relations=(inv_question1.reversed, inv_question2.reversed),
                forward=check_symmetric, device=device)

        poi_list.append(inverse)

    from domiknows.program.metric import PRF1Tracker, DatanodeCMMetric, MacroAverageTracker
    from domiknows.program.loss import NBCrossEntropyLoss
    from domiknows.program import SolverPOIProgram
    from domiknows.program.lossprogram import PrimalDualProgram
    from domiknows.program.model.pytorch import SolverModel

    infer_list = ['local/softmax']  #
    if pmd:
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=infer_list,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    beta=beta,
                                    device=device)
    else:
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=infer_list,
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   device=device)

    return program
