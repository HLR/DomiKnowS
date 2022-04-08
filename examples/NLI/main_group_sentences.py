import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import pandas as pd
from data.reader import DataReaderMulti, DataReaderMultiRelation
from program_declaration import program_declaration
import torch
import argparse


def main(args):
    from graph_senetences import entailment, neutral, contradiction

    # Set the cuda number we want to use
    cuda_number = args.cuda_number
    cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'
    training_file = "train.csv"
    testing_file = "test.csv"

    test_dataset = DataReaderMultiRelation(file="data/" + testing_file, size=args.testing_samples,
                                   batch_size=args.batch_size, augment_file="data/snli_genadv_1000_dev.jsonl")

    train_dataset = DataReaderMultiRelation(file="data/" + training_file, size=args.training_samples,
                                    batch_size=args.batch_size)

    model = program_declaration(cur_device, sym_relation=args.sym_relation)
    model.train(train_dataset, test_set=test_dataset, train_epoch_num=args.cur_epoch,
                Optim=lambda params: torch.optim.AdamW(params, lr=args.learning_rate), device=cur_device)
    model.test(test_dataset, device=cur_device)

    correct = 0
    result = {"premise": [],
              "hypothesis": [],
              "actual": [],
              "predict": []}
    for datanode in model.populate(test_dataset, device=cur_device):
        for sentence in datanode.getChildDataNodes():
            result["premise"].append(sentence.getAttribute("premise"))
            result["hypothesis"].append(sentence.getAttribute("hypothesis"))
            result["actual"].append('entailment' if sentence.getAttribute(entailment, 'label')
                                    else 'neutral' if sentence.getAttribute(neutral, 'label') else 'contrast')
            result["predict"].append('entailment' if sentence.getAttribute(entailment, 'ILP')
                                     else 'neutral' if sentence.getAttribute(neutral, 'ILP') else 'contrast')

            correct += sentence.getAttribute(entailment, 'ILP') if sentence.getAttribute(entailment, 'label') else \
                sentence.getAttribute(neutral, 'ILP') if sentence.getAttribute(neutral, 'label') else \
                    sentence.getAttribute(contradiction, 'ILP')
    print("Accuracy = %.2f%%" % (correct / len(result["predict"]) * 100))
    result = pd.DataFrame(result)
    training_size = 10000 if args.training_samples > 10000 and args.adver_data else args.training_samples
    result.to_csv("report-{:}-{:}-{:}--sym:{:}.csv".format(training_size, args.testing_samples,
                                                             args.cur_epoch, args.sym_relation))


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

    parser.add_argument('--sym_relation', dest='sym_relation', default=0, help="Using symmetric relation",
                        type=int)
    args = parser.parse_args()
    main(args)
