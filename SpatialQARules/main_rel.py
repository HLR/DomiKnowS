import os
import sys
import random

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import pandas as pd
import torch
import argparse
import numpy as np
import transformers
from domiknows.graph import Graph, Concept, Relation
from program_declaration import program_declaration_spartun_fr, program_declaration_StepGame, \
    program_declaration_spartun_fr_T5, program_declaration_spartun_fr_T5_v2, \
    program_declaration_spartun_fr_T5_v3, program_declaration_spartun_fr_T5_v4
from reader import DomiKnowS_reader
import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def eval(program, testing_set, cur_device, args, print_result=False, StepGame_number=None, multilabel=False):
    if args.test_file.upper() != "STEPGAME":
        from graph_spartun_rel import left, right, above, below, behind, front, near, far, disconnected, touch, \
            overlap, coveredby, inside, cover, contain, output_for_loss
        all_labels = [left, right, above, below, behind, front, near, far, disconnected,
                      touch, overlap, coveredby, inside, cover, contain]

        all_labels_text = ["left", "right", "above", "below", "behind", "front",
                           "near", "far", "disconnect", "touch", "overlap", "covered by",
                           "inside", "cover", "contain"]
    else:
        from graph_stepgame import left, right, above, below, lower_left, lower_right, upper_left, upper_right, overlap
        all_labels = [left, right, above, below, lower_left, lower_right, upper_left, upper_right, overlap]
        all_labels_text = ["left", "right", "above", "below", "lower-left",
                           "lower-right", "upper-left", "upper-right", "overlap"]

    def remove_opposite(ind1, ind2, result_set, result_list):
        if ind1 in pred_set and ind2 in pred_set:
            if result_list[ind1] > result_list[ind2]:
                result_set.remove(ind2)
            else:
                result_set.remove(ind1)

    pred_list = []
    correct = 0
    total = 0
    pred_set = set()
    for datanode in tqdm.tqdm(program.populate(testing_set, device=cur_device), "Checking accuracy"):
        for question in datanode.getChildDataNodes():
            pred_set.clear()
            pred_list.clear()
            total += 1
            # Getting predict label
            for ind, label in enumerate(all_labels):
                pred = question.getAttribute(label, 'local/softmax')
                if pred.argmax().item() == 1:
                    pred_set.add(ind)
                pred_list.append(pred[1].item())
            if args.train_file.upper() == "STEPGAME":
                pred = np.array(pred_list).argmax()
                pred_set = {pred}
            else:
                remove_opposite(0, 1, pred_set, pred_list)
                remove_opposite(2, 3, pred_set, pred_list)
                remove_opposite(4, 5, pred_set, pred_list)
                remove_opposite(6, 7, pred_set, pred_list)
                remove_opposite(8, 9, pred_set, pred_list)
            accuracy_check = True
            # Getting acutal label
            # if args.model == "t5-adapter":
            #     expected_text = question.getAttribute("text_labels")
            #     pred_text = ""
            #     for i, label in enumerate(all_labels_text):
            #         if multilabel:
            #             pred_text += label + ":" + ("yes" if i in pred_set else "no") + " "
            #         else:
            #             if i in pred_set:
            #                 pred_text += label if not pred_text else (", " + label)
            #     correct += int(expected_text.strip() == pred_text.strip())
            # else:
            for ind, label_ind in enumerate(all_labels):
                label = question.getAttribute(label_ind, 'label').item()
                pred = 1 if ind in pred_set else 0
                accuracy_check = accuracy_check and label == pred
            if accuracy_check:
                correct += 1
    accuracy = correct / total

    if print_result:
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
        if StepGame_number:
            print("Testing on StepGame {:} steps".format(StepGame_number), file=result_file)
        print("Accuracy:", accuracy, file=result_file)

    return accuracy


def train(program, train_set, eval_set, cur_device, limit, lr, check_epoch=1, program_name="DomiKnow", args=None):
    def get_avg_loss():
        from domiknows.program.model.base import Mode
        if cur_device is not None:
            program.model.to(cur_device)
        program.model.mode(Mode.TEST)
        program.model.reset()
        train_loss = 0
        total_loss = 0
        with torch.no_grad():
            for data_item in tqdm.tqdm(train_set, "Calculating Loss of training"):
                loss, _, *output = program.model(data_item)
                total_loss += 1
                train_loss += loss
        return train_loss / total_loss

    best_accuracy = 0
    best_epoch = 0
    old_file = None
    check_epoch = args.check_epoch
    training_file = open("training.txt", 'a')
    print("-" * 10, file=training_file)
    print("Training by {:s} of ({:s} {:s})".format(program_name, args.train_file, "FR"), file=training_file)
    print("Learning Rate:", args.lr, file=training_file)
    training_file.close()
    cur_epoch = 0
    if args.model == "t5-adapter":
        optimizer = lambda param: transformers.optimization.Adafactor(param, lr=lr, scale_parameter=False,
                                                                      relative_step=False)
    else:
        optimizer = lambda param: torch.optim.AdamW(param, lr=lr)
    for epoch in range(check_epoch, limit, check_epoch):
        print("Training")
        if args.pmd:
            program.train(train_set, c_warmup_iters=0, train_epoch_num=check_epoch,
                          Optim=optimizer,
                          device=cur_device)
        else:
            program.train(train_set, train_epoch_num=check_epoch,
                          Optim=optimizer,
                          device=cur_device)
        cur_epoch += check_epoch
        # loss = get_avg_loss()
        training_file = open("training.txt", 'a')
        accuracy = eval(program, eval_set, cur_device, args)
        print("Epoch:", epoch, file=training_file)
        # print("Loss:", loss, file=training_file)
        print("Dev Accuracy:", accuracy * 100, "%", file=training_file)
        if accuracy >= best_accuracy:
            best_epoch = epoch
            best_accuracy = accuracy
            # if old_file:
            #     os.remove(old_file)
            program_addition = ""
            if program_name == "PMD":
                program_addition = "_beta_" + str(args.beta)
            else:
                program_addition = "_size_" + str(args.sampling_size)
            new_file = program_name + "_" + str(epoch) + "epoch" + "_lr_" + str(
                args.lr) + program_addition + "_model_" + args.model
            program.save("Models/" + new_file)
        training_file.close()

    training_file = open("training.txt", 'a')
    if cur_epoch < limit:
        if args.pmd:
            program.train(train_set, c_warmup_iters=0, train_epoch_num=check_epoch,
                          Optim=optimizer,
                          device=cur_device)
        else:
            program.train(train_set, train_epoch_num=check_epoch,
                          Optim=optimizer,
                          device=cur_device)
        accuracy = eval(program, eval_set, cur_device, args)
        print("Epoch:", limit, file=training_file)
        print("Dev Accuracy:", accuracy * 100, "%", file=training_file)
        if accuracy >= best_accuracy:
            best_epoch = limit
            # if old_file:
            #     os.remove(old_file)
            if program_name == "PMD":
                program_addition = "_beta_" + str(args.beta)
            else:
                program_addition = "_size_" + str(args.sampling_size)
            new_file = program_name + "_" + str(limit) + "epoch" + "_lr_" + str(
                args.lr) + program_addition + "_model_" + args.model
            old_file = new_file
            program.save("Models/" + new_file)
    print("Best epoch ", best_epoch, file=training_file)
    training_file.close()
    return best_epoch


def main(args):
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'

    if args.train_file.upper() == "STEPGAME":
        program = program_declaration_StepGame(cur_device,
                                                   pmd=args.pmd, beta=args.beta,
                                                   sampling=args.sampling, sampleSize=args.sampling_size,
                                                   dropout=args.dropout, constraints=args.constrains)
    else:
        if args.model == "t5-adapter":
            print("call T5")
            program_declaration_function = None
            if args.version == 2:
                program_declaration_function = program_declaration_spartun_fr_T5_v2
            elif args.version == 3:
                program_declaration_function = program_declaration_spartun_fr_T5_v3
            elif args.version == 4:
                program_declaration_function = program_declaration_spartun_fr_T5_v4
            else:
                program_declaration_function = program_declaration_spartun_fr_T5

            program = program_declaration_function(cur_device,
                                                   pmd=args.pmd, beta=args.beta,
                                                   sampling=args.sampling, sampleSize=args.sampling_size,
                                                   dropout=args.dropout, constraints=args.constrains)
        else:
            program = program_declaration_spartun_fr(cur_device,
                                                     pmd=args.pmd, beta=args.beta,
                                                     sampling=args.sampling, sampleSize=args.sampling_size,
                                                     dropout=args.dropout, constraints=args.constrains,
                                                     model=args.model)

    boolQ = args.train_file.upper() == "BOOLQ"
    train_file = "train.json" if args.train_file.upper() == "ORIGIN" \
        else "train_FR_v3.json" if args.train_file.upper() == "SPARTUN" \
        else "boolQ/train.json" if args.train_file.upper() == "BOOLQ" \
        else "StepGame" if args.train_file.upper() == "STEPGAME" \
        else "human_train.json"

    training_set = DomiKnowS_reader("DataSet/" + train_file, "FR",
                                    type_dataset=args.train_file.upper(),
                                    size=args.train_size,
                                    upward_level=12,
                                    augmented=args.train_file.upper() == "SPARTUN",
                                    batch_size=args.batch_size,
                                    rule_text=args.text_rules,
                                    STEPGAME_status="train" if args.train_file.upper() == "STEPGAME" else None)

    test_file = "human_test.json" if args.test_file.upper() == "HUMAN" \
        else "StepGame" if args.train_file.upper() == "STEPGAME" \
        else "test.json"

    testing_set = DomiKnowS_reader("DataSet/" + test_file, "FR",
                                   type_dataset=args.train_file.upper(),
                                   size=args.test_size,
                                   augmented=False,
                                   batch_size=args.batch_size,
                                   rule_text=args.text_rules,
                                   STEPGAME_status="test" if args.train_file.upper() == "STEPGAME" else None,
                                   )

    eval_file = "human_dev.json" if args.test_file.upper() == "HUMAN" \
        else "StepGame" if args.train_file.upper() == "STEPGAME" \
        else "boolQ/train.json" if args.train_file.upper() == "BOOLQ" else "dev_Spartun.json"

    eval_set = DomiKnowS_reader("DataSet/" + eval_file, "FR",
                                type_dataset=args.train_file.upper(),
                                size=args.test_size,
                                augmented=False,
                                batch_size=args.batch_size,
                                rule_text=args.text_rules,
                                STEPGAME_status="dev" if args.train_file.upper() == "STEPGAME" else None)

    program_name = "PMD" if args.pmd else "Sampling" if args.sampling else "Base"

    # eval(program, testing_set, cur_device, args)
    if args.loaded:
        if args.model_change:
            pretrain_model = torch.load("Models/" + args.loaded_file,
                                        map_location={'cuda:0': cur_device, 'cuda:1': cur_device, 'cuda:2': cur_device,
                                                      'cuda:3': cur_device, 'cuda:4': cur_device, 'cuda:5': cur_device})
            pretrain_dict = pretrain_model.state_dict()
            current_dict = program.model.state_dict()
            # Filter out unnecessary keys
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in current_dict}
            program.model.load_state_dict(pretrain_dict)
        else:
            program.load("Models/" + args.loaded_file,
                         map_location={'cuda:0': cur_device, 'cuda:1': cur_device, 'cuda:2': cur_device,
                                       'cuda:3': cur_device, 'cuda:4': cur_device, 'cuda:5': cur_device})
        if args.test_each:
            for i in range(10):
                print("Testing {:} steps".format(i))
                testing_set = DomiKnowS_reader("DataSet/" + test_file, "FR",
                                               type_dataset=args.train_file.upper(),
                                               size=args.test_size,
                                               augmented=False,
                                               batch_size=args.batch_size,
                                               rule_text=args.text_rules,
                                               STEPGAME_status="test" if args.train_file.upper() == "STEPGAME" else None,
                                               reasoning_steps=i)
                eval(program, testing_set, cur_device, args, print_result=True)
        else:
            eval(program, testing_set, cur_device, args, print_result=True)
    elif args.loaded_train:
        if args.model_change:
            pretrain_model = torch.load("Models/" + args.loaded_file,
                                        map_location={'cuda:0': cur_device, 'cuda:1': cur_device, 'cuda:2': cur_device,
                                                      'cuda:3': cur_device, 'cuda:4': cur_device, 'cuda:5': cur_device})
            pretrain_dict = pretrain_model
            current_dict = program.model.state_dict()
            # Filter out unnecessary keys
            # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in current_dict}
            # Loaded same parameters
            new_state_dict = {k: v if k not in pretrain_dict else pretrain_dict[k]
                              for k, v in current_dict.items()}
            program.model.load_state_dict(new_state_dict)
        else:
            program.load("Models/" + args.loaded_file,
                         map_location={'cuda:0': cur_device, 'cuda:1': cur_device, 'cuda:2': cur_device,
                                       'cuda:3': cur_device, 'cuda:4': cur_device, 'cuda:5': cur_device})
        train(program, training_set, eval_set, cur_device, args.epoch, args.lr, program_name=program_name, args=args)
    else:
        train(program, training_set, eval_set, cur_device, args.epoch, args.lr, program_name=program_name, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpaRTUN Rules Base")
    parser.add_argument("--epoch", dest="epoch", type=int, default=1)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-5)
    parser.add_argument("--cuda", dest="cuda", type=int, default=0)
    parser.add_argument("--test_size", dest="test_size", type=int, default=12)
    parser.add_argument("--train_size", dest="train_size", type=int, default=16)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--train_file", type=str, default="SPARTUN", help="Option: SpaRTUN or Human")
    parser.add_argument("--test_file", type=str, default="SPARTUN", help="Option: SpaRTUN or Human")
    parser.add_argument("--text_rules", type=bool, default=False, help="Including rules as text or not")
    parser.add_argument("--dropout", dest="dropout", type=bool, default=False)
    parser.add_argument("--pmd", dest="pmd", type=bool, default=False)
    parser.add_argument("--beta", dest="beta", type=float, default=0.5)
    parser.add_argument("--sampling", dest="sampling", type=bool, default=False)
    parser.add_argument("--sampling_size", dest="sampling_size", type=int, default=1)
    parser.add_argument("--constrains", dest="constrains", type=bool, default=False)
    parser.add_argument("--loaded", dest="loaded", type=bool, default=False)
    parser.add_argument("--loaded_file", dest="loaded_file", type=str, default="train_model")
    parser.add_argument("--loaded_train", type=bool, default=False, help="Option to load and then further train")
    parser.add_argument("--model_change", type=bool, default=False, help="Option to load and then further train")
    parser.add_argument("--save", dest="save", type=bool, default=False)
    parser.add_argument("--save_file", dest="save_file", type=str, default="train_model")
    parser.add_argument("--step_game_test_each", dest="test_each", type=bool, default=False)
    parser.add_argument("--model", dest="model", type=str, default="bert")
    parser.add_argument("--check_epoch", dest="check_epoch", type=int, default=1)
    parser.add_argument("--version", dest="version", type=int, default=0)

    args = parser.parse_args()
    main(args)
