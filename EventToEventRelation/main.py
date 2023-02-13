import sys

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import pandas as pd
import torch
import argparse
import numpy as np
from regr.graph import Graph, Concept, Relation
from program_declaration import program_declaration
from doc_reader import load_dataset


def main(args):
    from graph import graph, paragraph, paragraph_contain, event_relation, relation_classes
    # Set the cuda number we want to use
    cuda_number = args.cuda_number
    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'

    print('Using: %s' % (cur_device))

    train_dataset, valid_dataset, test_dataset = load_dataset(args.training_size, args.validation_size,
                                                              args.testing_size, batch_size=args.batch_size)

    # Declare Program
    program = program_declaration(cur_device, PMD=args.PMD, sampleloss=args.sampleloss)

    base_round = 10
    round = args.epoch // base_round + (1 if args.epoch % base_round else 0)
    remain_epoch = args.epoch
    count_round = 0
    for _ in range(round):
        program.train(train_dataset, train_epoch_num=base_round if remain_epoch >= base_round else remain_epoch,
                      Optim=lambda params: torch.optim.Adam(params, lr=args.learning_rate, amsgrad=True)
                      , device=cur_device)
        count_round += base_round if base_round < remain_epoch else remain_epoch
        remain_epoch -= base_round
        classes = ["before", "after",
                   "EQUAL", "VAGUE"]

        output_file = {"file": [], "eiid1": [], "eiid2": [], "actual": [], "argmax": [], "ILP": []}
        accuracy_argmax = 0
        accuracy_ILP = 0
        total = 0
        for datanode in program.populate(test_dataset, device=cur_device):
            for event_relation in datanode.getChildDataNodes():
                actual_class = classes[int(event_relation.getAttribute(relation_classes, "label"))]
                pred_argmax = classes[int(torch.argmax(event_relation.getAttribute(relation_classes, "local/argmax")))]
                pred_ILP = classes[int(torch.argmax(event_relation.getAttribute(relation_classes, "ILP")))]
                output_file["file"].append(event_relation.getAttribute("file"))
                output_file["eiid1"].append(int(event_relation.getAttribute("eiid1")))
                output_file["eiid2"].append(int(event_relation.getAttribute("eiid2")))
                output_file["actual"].append(actual_class)
                output_file["argmax"].append(pred_argmax)
                output_file["ILP"].append(pred_ILP)
                accuracy_argmax += 1 if actual_class == pred_argmax else 0
                accuracy_ILP += 1 if actual_class == pred_ILP else 0
                total += 1
        output_file = pd.DataFrame(output_file)
        output_file.to_csv("result_epoch" + str(count_round) + ".csv")
        file = open("result_run.txt", "a")
        file.write("Result:\n")
        explain = "EPOCH: " + str(count_round) + "\nlearning rate: " + str(args.learning_rate) \
        + "\nPMD: " + str(args.PMD) + "\nSample Loss: " + str(args.sampleloss) + "\n"
        file.write(explain)
        file.write("Argmax accuracy = " + str(accuracy_argmax * 100 / total) + "%" + "\n")
        file.write("ILP accuracy = " + str(accuracy_ILP * 100 / total) + "%" + "\n")
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Event to Event Relation Learning Code")

    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
    parser.add_argument('--epoch', dest='epoch', default=1, help='number of epoch to train the model', type=int)
    parser.add_argument('--lr', dest='learning_rate', default=1e-7, help='learning rate of the model', type=float)
    parser.add_argument('--batch_size', dest='batch_size', default=3, help="batch size of sample", type=int)
    parser.add_argument('--PMD', dest='PMD', default=False, help="using primal dual program or not", type=bool)
    parser.add_argument('--beta', dest='beta', default=0.5, help="beta value for primal dual program", type=float)
    parser.add_argument('--sampleloss', dest='sampleloss', default=False, help="using sample loss program or not",
                        type=bool)
    parser.add_argument('--sampleSize', dest='sampleSize', default=1, help="Sample Size for sample loss program",
                        type=int)
    parser.add_argument('--training_size', dest='training_size', default=100000, help="", type=int)
    parser.add_argument('--testing_size', dest='testing_size', default=100000, help="", type=int)
    parser.add_argument('--validation_size', dest='validation_size', default=100000, help="", type=int)
    args = parser.parse_args()
    main(args)
