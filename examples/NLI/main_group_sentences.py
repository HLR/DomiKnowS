import sys

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import pandas as pd
from data.reader import DataReaderMultiRelation
from program_declaration import program_declaration
import torch
import argparse
import numpy as np
import random
from tqdm import tqdm
from regr.graph import Graph, Concept, Relation


def eval(program, testing_set, cur_device, args, filename=""):
    """
    Evaluate the model by given testing dataset
    Parameters
    ----------
    program: program to be assessed
    testing_set: training dataset to evaluate the model
    cur_device: cuda device to be used
    args: namespace containing the condition parameters from users
    filename: filename to save result

    Returns
    -------
    None
    """
    from graph import answer_class
    labels = ["Yes", "No"]
    accuracy_ILP = 0
    accuracy = 0
    count = 0
    count_datanode = 0
    satisfy_constrain_rate = 0
    for group_pair in tqdm(program.populate(testing_set, device=cur_device), "Manually Testing"):
        count_datanode += 1
        for pair in group_pair.getChildDataNodes():
            count += 1
            label = int(pair.getAttribute(answer_class, "label"))
            pred = int(torch.argmax(pair.getAttribute(answer_class, "local/argmax")))
            pred_ILP = int(torch.argmax(pair.getAttribute(answer_class, "ILP")))
            accuracy_ILP += 1 if pred_ILP == label else 0
            accuracy += 1 if pred == label else 0
        verify_constrains = group_pair.verifyResultsLC()
        count_verify = 0
        if verify_constrains:
            for lc in verify_constrains:
                count_verify += verify_constrains[lc]["satisfied"]
        satisfy_constrain_rate += count_verify / len(verify_constrains)
    satisfy_constrain_rate /= count_datanode
    accuracy /= count
    accuracy_ILP /= count

    result_file = open("result.txt", 'a')
    print("Program:", "Primal Dual" if args.primaldual else "Sampling Loss" if args.sampleloss else "DomiKnowS",
          file=result_file)

    print("Training info", file=result_file)
    print("Batch Size:", args.batch_size, file=result_file)
    print("Epoch:", args.cur_epoch, file=result_file)
    print("Learning Rate:", args.learning_rate, file=result_file)
    print("Beta:", args.beta, file=result_file)
    # print("Sampling Size:", args.sampling_size, file=result_file)
    print("Evaluation File: ", filename, file=result_file)
    print("Accuracy:", accuracy, file=result_file)
    print("ILP Accuracy:", accuracy_ILP, file=result_file)
    print("Constrains Satisfied rate:", satisfy_constrain_rate, "%", file=result_file)
    result_file.close()


def main(args):
    from graph import answer_class

    SEED = 2022
    np.random.seed(SEED)
    random.seed(SEED)
    # pl.seed_everything(SEED)
    torch.manual_seed(SEED)

    # Set the cuda number we want to use
    cuda_number = args.cuda_number
    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'

    print('Using: %s' % (cur_device))

    training_file = "train.csv"
    testing_file = "test.csv"

    augment_file_test = "data/snli_genadv_1000_test.jsonl"
    augment_file_dev = "data/snli_genadv_1000_dev.jsonl"
    # Loading Test and Train data
    test_dataset = DataReaderMultiRelation(file="data/" + testing_file, size=args.testing_sample,
                                           batch_size=args.batch_size, augment_file=augment_file_test)

    train_dataset = DataReaderMultiRelation(file="data/" + training_file, size=args.training_sample,
                                            batch_size=args.batch_size)
    # Load Augmentation data
    augment_dataset_dev = DataReaderMultiRelation(file=None, size=None, batch_size=args.batch_size,
                                                  augment_file=augment_file_dev)

    augment_dataset_test = DataReaderMultiRelation(file=None, size=None, batch_size=args.batch_size,
                                                   augment_file=augment_file_test)
    # Declare Program
    train_program, eval_program = program_declaration(cur_device,
                                                      sym_relation=args.sym_relation,
                                                      tran_relation=args.tran_relation,
                                                      primaldual=args.primaldual,
                                                      sample=args.sampleloss,
                                                      beta=args.beta,
                                                      sampling_size=args.sampling_size)

    train_program.train(train_dataset, train_epoch_num=args.cur_epoch,
                        Optim=lambda params: torch.optim.AdamW(params, lr=args.learning_rate), device=cur_device)

    # Loading train parameter to evaluation program
    train_program.save(args.model_name + ".pth")  # Save model
    eval_program.load(args.model_name + ".pth")  # Load model to train
    eval(eval_program, test_dataset, cur_device, args, "ALL")
    eval(eval_program, augment_dataset_dev, cur_device, args, "Augmented_dev")
    eval(eval_program, augment_dataset_test, cur_device, args, "Augmented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLI Learning Code")

    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)

    parser.add_argument('--epoch', dest='cur_epoch', default=5, help='number of epochs to train model', type=int)

    parser.add_argument('--lr', dest='learning_rate', default=1e-5, help='learning rate of the adamW optimiser',
                        type=float)

    parser.add_argument('--training_sample', dest='training_sample', default=600000,
                        help="number of data to train model", type=int)

    parser.add_argument('--testing_sample', dest='testing_sample', default=600000, help="number of data to test model",
                        type=int)

    parser.add_argument('--batch_size', dest='batch_size', default=2, help="batch size of sample", type=int)

    parser.add_argument('--sym_relation', dest='sym_relation', default=False, help="Using symmetric relation",
                        type=bool)
    parser.add_argument('--tran_relation', dest='tran_relation', default=False, help="Using transitive relation",
                        type=bool)
    parser.add_argument('--pmd', dest='primaldual', default=False, help="Using primaldual model or not",
                        type=bool)
    parser.add_argument('--sampleloss', dest='sampleloss', default=False, help="Using sampling loss model or not",
                        type=bool)
    parser.add_argument('--beta', dest='beta', default=0.5, help="Beta value to use in PMD",
                        type=float)
    parser.add_argument('--sampling_size', dest='sampling_size', default=100,
                        help="Sampling size to use in sampling loss",
                        type=int)
    parser.add_argument('--model_name', dest='model_name', default="Models",
                        help="Model name to save model after training",
                        type=str)
    args = parser.parse_args()
    main(args)
