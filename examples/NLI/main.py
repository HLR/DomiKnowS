import pandas as pd
from data.reader import DataReader
from transformers import AdamW
import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from model import RobertaTokenizer, UFT_Robert, RobertClassification
import argparse


def program_declaration():
    from graph import graph, sentence, entailment, neutral, contradiction

    graph.detach()
    # Reading from sentence
    sentence['premise'] = ReaderSensor(keyword="premise")
    sentence['hypothesis'] = ReaderSensor(keyword="hypothesis")

    # Creating the ROBERTA representation of premise and hypothesis
    sentence["token_ids", "Mask"] = JointSensor('hypothesis', 'premise', forward=RobertaTokenizer())
    roberta_model = UFT_Robert()
    sentence["robert_emb"] = ModuleLearner("token_ids", "Mask", module=roberta_model)

    # Define label
    sentence[entailment] = ModuleLearner("robert_emb", module=RobertClassification(roberta_model.last_layer_size))
    sentence[neutral] = ModuleLearner("robert_emb", module=RobertClassification(roberta_model.last_layer_size))
    sentence[contradiction] = ModuleLearner("robert_emb", module=RobertClassification(roberta_model.last_layer_size))
    sentence[entailment] = ReaderSensor(keyword="entailment", label=True)
    sentence[neutral] = ReaderSensor(keyword="neutral", label=True)
    sentence[contradiction] = ReaderSensor(keyword="contradiction", label=True)

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
    from graph import sentence, entailment, neutral, contradiction

    # Set the cuda number we want to use
    cuda_number = args.cuda_number
    cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'

    test_dataset = DataReader(file="data/test.csv", size=args.testing_samples)
    train_dataset = DataReader(file="data/train.csv", size=args.training_samples)
    model = program_declaration()
    model.train(train_dataset, test_set=test_dataset, train_epoch_num=args.cur_epoch,
                Optim=lambda params: torch.optim.AdamW(params, lr=args.learning_rate), device=cur_device)
    model.test(test_dataset, device=cur_device)

    correct = 0
    index = 0
    result = {"premise": [data['premise'][0] for data in test_dataset],
              "hypothesis": [data['hypothesis'][0] for data in test_dataset],
              "actual": ['entailment' if data['entailment'] else 'neutral' if data['neutral'] else 'contrast' for data
                         in test_dataset],
              "predict": []}
    for datanode in model.populate(test_dataset):
        #print(datanode)
        #print("Entailment: ", datanode.getAttribute(entailment, 'ILP'))
        #print("Neutral: ", datanode.getAttribute(neutral, 'ILP'))
        #print("Contrast: ", datanode.getAttribute(contradiction, 'ILP'))
        result["predict"].append('entailment' if datanode.getAttribute(entailment, 'ILP')
                                 else 'neutral' if datanode.getAttribute(neutral, 'ILP') else 'contrast')
        correct += datanode.getAttribute(entailment, 'ILP') if result["actual"][index] == 'entailment' else \
            datanode.getAttribute(neutral, 'ILP') if result["actual"][index] == 'neutral' else \
            datanode.getAttribute(contradiction, 'ILP')
        index += 1
    #print("Accuracy = %.2f%%" % (correct / index * 100))
    result = pd.DataFrame(result)
    result.to_csv("report-{:}-{:}-{:}.csv".format(args.training_samples, args.testing_samples, args.cur_epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLI Learning Code")
    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
    parser.add_argument('--epoch', dest='cur_epoch', default=10, help='number of epochs to train model', type=int)
    parser.add_argument('--lr', dest='learning_rate', default=1e-5, help='learning rate of the adamW optimiser',
                        type=float)
    parser.add_argument('--training_sample', dest='training_samples', default=550152,
                        help="number of data to train model", type=int)
    parser.add_argument('--testing_sample', dest='testing_samples', default=10000, help="number of data to test model",
                        type=int)
    args = parser.parse_args()
    main(args)
