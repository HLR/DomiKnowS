from transformers import AutoTokenizer

from models import Tokenizer, T5WithLoraGenerativeCLF
import torch


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
#         input = input.transpose(1, 2)
#         return super().forward(input, target)


def program_declaration(device='cpu', pmd=False, beta=0.5):
    from graph import graph, context, question, rel_context_contain_question, \
        rel_question_contain_answer, relation, answer, raw_answer
    from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor
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

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list, max_size=5):
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

    tokenizer = Tokenizer("t5-base", label=all_labels)

    print("USING T5")

    T5Model = T5WithLoraGenerativeCLF("google/flan-t5-base", tokenizer=tokenizer, device=device, max_length=5)

    question["input_ids"] = JointSensor(rel_context_contain_question, "question", "story",
                                        forward=tokenizer, device=device)

    question["raw_answer"] = JointSensor("labels", forward=lambda x: x, device=device, label=True)

    question["raw_answer"] = ModuleLearner(rel_context_contain_question, "input_ids", "labels",
                                           module=T5Model, device=device)

    def make_answer(_, raw_answer, label):
        answer = raw_answer.argmax(dim=-1)
        # print(answer, label)
        return torch.ones(len(raw_answer), 1), answer, label

    answer[rel_question_contain_answer, "_answer", "_label"] \
        = JointSensor(question[rel_context_contain_question], question["raw_answer"], question["labels"],
                      forward=make_answer, device=device)

    answer[relation] = JointSensor("_answer", forward = lambda y: y, device=device)

    # def make_relation_labels(_, labels):
    #     # print("Label:", labels)
    #     # labels = [label[0] for label in labels]
    #     # labels = torch.LongTensor(labels)
    #     return labels
    #
    # answer[relations] = JointSensor(rel_question_contain_answer, question["labels"], label=True,
    #                                 forward=make_relation_labels,
    #                                 device=device)

    # answer[relations] = ModuleLearner(rel_question_contain_answer, "input_ids", question["labels"],
    #                                   module=T5Model)

    # def make_relation(question, stories):
    #     return torch.ones(len(question.split("@@")), 1)
    #
    # relations[rel_context_contain_question] = JointSensor(question["question"], question["story"],
    #                                                       forward=make_relation)

    poi_list = [relation, question, raw_answer, answer]

    from domiknows.program.metric import PRF1Tracker, DatanodeCMMetric, MacroAverageTracker
    from domiknows.program.loss import NBCrossEntropyLoss
    from domiknows.program import SolverPOIProgram
    from domiknows.program.lossprogram import PrimalDualProgram
    from domiknows.program.model.pytorch import SolverModel

    infer_list = ['local/argmax']
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
