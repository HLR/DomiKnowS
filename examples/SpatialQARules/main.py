import os
import sys
import random
import pytorch_lightning as pl

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import pandas as pd
import torch
import argparse
import numpy as np
from regr.graph import Graph, Concept, Relation
from program_declaration import program_declaration
from reader import DomiKnowS_reader
import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


def eval(program, testing_set, cur_device, args):
    from graph import answer_class
    labels = ["Yes", "No", "DK"]
    accuracy_ILP = 0
    accuracy = 0
    count = 0
    count_datanode = 0
    satisfy_constrain_rate = 0
    pred = []
    actual = []
    result_csv = {"story": [], "question": [], "label": [], "argmax": [], "ILP": []}
    for datanode in tqdm.tqdm(program.populate(testing_set, device=cur_device), "Manually Testing"):
        count_datanode += 1
        for question in datanode.getChildDataNodes():
            count += 1
            label = labels[int(question.getAttribute(answer_class, "label"))]
            pred_label = int(torch.argmax(question.getAttribute(answer_class, "local/argmax")))
            actual_label = int(torch.argmax(question.getAttribute(answer_class, "ILP")))
            pred_argmax = labels[pred_label]
            pred_ILP = labels[actual_label]
            pred.append(pred_label)
            actual.append(actual_label)
            accuracy_ILP += 1 if pred_ILP == label else 0
            accuracy += 1 if pred_argmax == label else 0
            result_csv["story"].append(question.getAttribute("story"))
            result_csv["question"].append(question.getAttribute("question"))
            result_csv["label"].append(label)
            result_csv["argmax"].append(pred_argmax)
            result_csv["ILP"].append(pred_ILP)
        verify_constrains = datanode.verifyResultsLC()
        count_verify = 0
        if verify_constrains:
            for lc in verify_constrains:
                count_verify += verify_constrains[lc]["satisfied"]
        satisfy_constrain_rate += count_verify / len(verify_constrains)
    satisfy_constrain_rate /= count_datanode
    accuracy /= count
    accuracy_ILP /= count

    result_file = open("result.txt", 'a')
    print("Program:", "Primal Dual" if args.pmd else "Sampling Loss" if args.sampling else "DomiKnowS",
          file=result_file)
    if not args.loaded:
        print("Training info", file=result_file)
        print("Batch Size:", args.batch_size, file=result_file)
        print("Epoch:", args.epoch, file=result_file)
        print("Learning Rate:", args.lr, file=result_file)
        print("Beta:", args.beta, file=result_file)
        print("Sampling Size:", args.sampling_size, file=result_file)
    else:
        print("Loaded Model Name:", args.loaded_file, file=result_file)
    print("Evaluation File:", args.test_file, file=result_file)
    print("Accuracy:", accuracy, file=result_file)
    print("ILP Accuracy:", accuracy_ILP, file=result_file)
    print("Constrains Satisfied rate:", satisfy_constrain_rate, "%", file=result_file)
    print("Precious:", precision_score(actual, pred), file=result_file)
    print("Recall:", precision_score(actual, pred), file=result_file)
    print("F1:", precision_score(actual, pred), file=result_file)
    result_file.close()

    #df = pd.DataFrame(result_csv)
    #df.to_csv("result.csv")


def train(program, train_set, eval_set, cur_device, limit, lr, check_epoch=4, program_name="DomiKnow", args=None):
    from graph import answer_class

    def evaluate():
        labels = ["Yes", "No", "DK"]
        count = 0
        accuracy = 0.0
        for datanode in tqdm.tqdm(program.populate(eval_set, device=cur_device), "Manually Evaluation"):
            for question in datanode.getChildDataNodes():
                count += 1
                label = labels[int(question.getAttribute(answer_class, "label"))]
                pred_argmax = labels[int(torch.argmax(question.getAttribute(answer_class, "local/argmax")))]
                accuracy += 1 if pred_argmax == label else 0
        return accuracy / count

    best_accuracy = 0
    best_epoch = 0
    old_file = None
    training_file = open("training.txt", 'a')
    print("-" * 10, file=training_file)
    print("Training by ", program_name, file=training_file)
    print("Learning Rate:", args.lr, file=training_file)
    training_file.close()
    for epoch in range(check_epoch, limit, check_epoch):
        training_file = open("training.txt", 'a')
        program.train(train_set, train_epoch_num=check_epoch,
                      Optim=lambda param: torch.optim.Adam(param, lr=lr, amsgrad=True),
                      device=cur_device)
        accuracy = evaluate()
        print("Epoch:", epoch, file=training_file)
        print("Dev Accuracy:", accuracy * 100, "%", file=training_file)
        if accuracy > best_accuracy:
            best_epoch = epoch + check_epoch
            best_accuracy = accuracy
            # if old_file:
            #     os.remove(old_file)
            program_addition = ""
            if program_name == "PMD":
                program_addition = "_beta_" + str(args.beta)
            else:
                program_addition = "_size_" + str(args.sampling_size)
            new_file = program_name + "_" + str(epoch) + "epoch" + "_lr_" + str(args.lr) + program_addition
            old_file = new_file
            program.save("Models/" + new_file)
        training_file.close()

    training_file = open("training.txt", 'a')
    if epoch < limit:
        program.train(train_set, train_epoch_num=limit - epoch,
                      Optim=lambda param: torch.optim.Adam(param, lr=lr, amsgrad=True),
                      device=cur_device)
        accuracy = evaluate()
        print("Epoch:", limit, file=training_file)
        print("Dev Accuracy:", accuracy * 100, "%", file=training_file)
        if accuracy > best_accuracy:
            best_epoch = epoch + check_epoch
            # if old_file:
            #     os.remove(old_file)
            new_file = program_name + "_" + str(limit) + "epoch" + "_lr_" + str(args.lr)
            old_file = new_file
            program.save("Models/" + new_file)
    print("Best epoch ", best_epoch, file=training_file)
    training_file.close()
    return best_epoch


def main(args):
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    pl.seed_everything(SEED)
    torch.manual_seed(SEED)

    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'

    program = program_declaration(cur_device,
                                  pmd=args.pmd, beta=args.beta,
                                  sampling=args.sampling, sampleSize=args.sampling_size,
                                  dropout=args.dropout, constrains=args.constrains)

    train_file = "DataSet/train_with_rules.json" if args.train_file.upper() == "SPARTUN" \
        else "DataSet/boolQ/train.json" if args.train_file.upper() == "BOOLQ" else "DataSet/human_train.json"
    training_set = DomiKnowS_reader(train_file, "YN",
                                    size=args.train_size, upward_level=8,
                                    augmented=args.train_file.upper() == "SPARTUN", batch_size=args.batch_size)

    test_file = "DataSet/human_test.json" if args.test_file.upper() == "HUMAN" \
         else "DataSet/test.json"
    testing_set = DomiKnowS_reader(test_file, "YN", size=args.test_size,
                                   augmented=False, batch_size=args.batch_size)

    eval_file = "DataSet/human_dev.json" if args.test_file.upper() == "HUMAN" \
        else "DataSet/boolQ/train.json" if args.train_file.upper() == "BOOLQ" else "DataSet/dev_Spartun.json"

    eval_set = DomiKnowS_reader(eval_file, "YN", size=args.test_size,
                                augmented=False, batch_size=args.batch_size)
    program_name = "PMD" if args.pmd else "Sampling" if args.sampling else "Base"
    if args.loaded:
        program.load("Models" + args.loaded_file, map_location={'cuda:0': cur_device, 'cuda:1': cur_device})
        eval(program, testing_set, cur_device, args)
    elif args.loaded_train:
        program.load("Models/" + args.loaded_file, map_location={'cuda:0': cur_device, 'cuda:1': cur_device})
        train(program, training_set, eval_set, cur_device, args.epoch, args.lr, program_name=program_name, args=args)
    else:
        train(program, training_set, eval_set, cur_device, args.epoch, args.lr, program_name=program_name, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpaRTUN Rules Base")
    parser.add_argument("--epoch", dest="epoch", type=int, default=1)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-5)
    parser.add_argument("--cuda", dest="cuda", type=int, default=0)
    parser.add_argument("--test_size", dest="test_size", type=int, default=100000)
    parser.add_argument("--train_size", dest="train_size", type=int, default=100000)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=100000)
    parser.add_argument("--train_file", type=str, default="SpaRTUN", help="Option: SpaRTUN or Human")
    parser.add_argument("--test_file", type=str, default="SpaRTUN", help="Option: SpaRTUN or Human")
    parser.add_argument("--dropout", dest="dropout", type=bool, default=False)
    parser.add_argument("--pmd", dest="pmd", type=bool, default=False)
    parser.add_argument("--beta", dest="beta", type=float, default=0.5)
    parser.add_argument("--sampling", dest="sampling", type=bool, default=False)
    parser.add_argument("--sampling_size", dest="sampling_size", type=int, default=1)
    parser.add_argument("--constrains", dest="constrains", type=bool, default=False)
    parser.add_argument("--loaded", dest="loaded", type=bool, default=False)
    parser.add_argument("--loaded_file", dest="loaded_file", type=str, default="train_model")
    parser.add_argument("--loaded_train", type=bool, default=False, help="Option to load and then further train")
    parser.add_argument("--save", dest="save", type=bool, default=False)
    parser.add_argument("--save_file", dest="save_file", type=str, default="train_model")

    args = parser.parse_args()
    main(args)
