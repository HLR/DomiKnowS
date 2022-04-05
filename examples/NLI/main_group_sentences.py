import pandas as pd
from data.reader import DataReaderMulti
from transformers import AdamW
import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from model import RobertClassification, NLI_Robert, RobertaTokenizerMulti
import argparse


def program_declaration():
    from graph_senetences import graph, sentence, entailment, neutral, \
        contradiction, sentence_group, sentence_group_contains

    graph.detach()
    sentence_group["premises"] = ReaderSensor(keyword="premises")
    sentence_group["hypothesises"] = ReaderSensor(keyword="hypothesises")
    sentence_group["entailment_list"] = ReaderSensor(keyword="entailment_list")
    sentence_group["contradiction_list"] = ReaderSensor(keyword="contradiction_list")
    sentence_group["neutral_list"] = ReaderSensor(keyword="neutral_list")

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
                    sentence_group["contradiction_list"], sentence_group["neutral_list"], forward=make_sentence)

    sentence["token_ids", "Mask"] = JointSensor(sentence_group_contains, 'hypothesis', "premise",
                                                forward=RobertaTokenizerMulti())
    roberta_model = NLI_Robert()
    sentence["robert_emb"] = ModuleLearner("token_ids", "Mask", module=roberta_model)

    # number of hidden layer excluding the first layer and the last layer
    hidden_layer_size = 2
    sentence[entailment] = ModuleLearner("robert_emb", module=RobertClassification(roberta_model.last_layer_size,
                                                                                   hidden_layer_size=hidden_layer_size))
    sentence[neutral] = ModuleLearner("robert_emb", module=RobertClassification(roberta_model.last_layer_size,
                                                                                hidden_layer_size=hidden_layer_size))
    sentence[contradiction] = ModuleLearner("robert_emb", module=RobertClassification(roberta_model.last_layer_size,
                                                                                      hidden_layer_size=
                                                                                      hidden_layer_size))
    sentence[entailment] = FunctionalSensor(sentence_group_contains, "entail_list", forward=read_label, label=True)
    sentence[neutral] = FunctionalSensor(sentence_group_contains, "neutral_list", forward=read_label, label=True)
    sentence[contradiction] = FunctionalSensor(sentence_group_contains, "cont_list", forward=read_label,
                                               label=True)

    from regr.program import POIProgram, IMLProgram, SolverPOIProgram
    from regr.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
    from regr.program.loss import NBCrossEntropyLoss

    # Creating the program to create model
    program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'],
                               loss=MacroAverageTracker(NBCrossEntropyLoss()),
                               metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                       'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program


def main(args):
    from graph_senetences import entailment, neutral, contradiction

    # Set the cuda number we want to use
    cuda_number = args.cuda_number
    cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'
    training_file = "train.csv" if not args.adver_data else "adver_nli_train.jsonl"
    testing_file = "test.csv" if not args.adver_data else "adver_nli_test.jsonl"
    test_dataset = DataReaderMulti(file="data/" + testing_file, size=args.testing_samples,
                                   batch_size=args.batch_size, adver_data_set=args.adver_data)
    train_dataset = DataReaderMulti(file="data/" + training_file, size=args.training_samples,
                                    batch_size=args.batch_size, adver_data_set=args.adver_data)
    model = program_declaration()
    model.train(train_dataset, test_set=test_dataset, train_epoch_num=args.cur_epoch,
                Optim=lambda params: torch.optim.AdamW(params, lr=args.learning_rate), device=cur_device)
    model.test(test_dataset, device=cur_device)

    correct = 0
    index = 0
    result = {"premise": [],
              "hypothesis": [],
              "actual": [],
              "predict": []}
    for datanode in model.populate(test_dataset, device=cur_device):
        for sentence in datanode.getChildDataNodes():
            # print(sentence.getAttribute)
            # print("Actual")
            # print(sentence.getAttribute(entailment, 'label'))
            # print(sentence.getAttribute(neutral, 'label'))
            # print(sentence.getAttribute(contradiction, 'label'))
            #
            # print("ILP")
            # print(sentence.getAttribute(entailment, 'ILP'))
            # print(sentence.getAttribute(neutral, 'ILP'))
            # print(sentence.getAttribute(contradiction, 'ILP'))

            result["premise"].append(sentence.getAttribute("premise"))
            result["hypothesis"].append(sentence.getAttribute("hypothesis"))
            result["actual"].append('entailment' if sentence.getAttribute(entailment, 'label')
                                     else 'neutral' if sentence.getAttribute(neutral, 'label') else 'contrast')
            result["predict"].append('entailment' if sentence.getAttribute(entailment, 'ILP')
                                     else 'neutral' if sentence.getAttribute(neutral, 'ILP') else 'contrast')

            # Should ask it early about argmax value
            # in neutral be [2.2, 0.0. 2.0]
            # same as others
            correct += sentence.getAttribute(entailment, 'ILP') if sentence.getAttribute(entailment, 'label') else \
                sentence.getAttribute(neutral, 'ILP') if sentence.getAttribute(neutral, 'label') else \
                    sentence.getAttribute(contradiction, 'ILP')
    print("Accuracy = %.2f%%" % (correct / len(result["predict"]) * 100))
    result = pd.DataFrame(result)
    training_size = 10000 if args.training_samples > 10000 and args.adver_data else args.training_samples
    result.to_csv("report-{:}-{:}-{:}--adver:{:}.csv".format(training_size, args.testing_samples,
                                                             args.cur_epoch, args.adver_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLI Learning Code")
    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
    parser.add_argument('--epoch', dest='cur_epoch', default=10, help='number of epochs to train model', type=int)
    parser.add_argument('--lr', dest='learning_rate', default=1e-6, help='learning rate of the adamW optimiser',
                        type=float)
    parser.add_argument('--training_sample', dest='training_samples', default=550146,
                        help="number of data to train model", type=int)
    parser.add_argument('--testing_sample', dest='testing_samples', default=10000, help="number of data to test model",
                        type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=4, help="batch size of sample", type=int)
    parser.add_argument('--adver_data', dest='adver_data', default=0, help="Using adversarial data set ot not",
                        type=bool)
    args = parser.parse_args()
    main(args)
