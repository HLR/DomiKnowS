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
from tqdm import tqdm
from regr.graph import Graph, Concept, Relation


def eval(program, testing_set, cur_device, args, filename=""):
    from graph_senetences import answer_class
    labels = ["Yes", "No"]
    accuracy_ILP = 0
    accuracy = 0
    count = 0
    count_datanode = 0
    satisfy_constrain_rate = 0
    for datanode in tqdm(program.populate(testing_set, device=cur_device), "Manually Testing"):
        count_datanode += 1
        for question in datanode.getChildDataNodes():
            count += 1
            label = int(question.getAttribute(answer_class, "label"))
            pred = int(torch.argmax(question.getAttribute(answer_class, "local/argmax")))
            pred_ILP = int(torch.argmax(question.getAttribute(answer_class, "ILP")))
            accuracy_ILP += 1 if pred_ILP == label else 0
            accuracy += 1 if pred == label else 0
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

    # df = pd.DataFrame(result_csv)
    # df.to_csv("result.csv")


# def train(program, train_set, eval_set, cur_device, limit, lr, check_epoch=4, program_name="DomiKnow", args=None):
#     from graph import answer_class
#
#     def evaluate():
#         labels = ["Yes", "No"]
#         count = 0
#         actual = []
#         pred = []
#         for datanode in tqdm.tqdm(program.populate(eval_set, device=cur_device), "Manually Evaluation"):
#             for question in datanode.getChildDataNodes():
#                 count += 1
#                 actual.append(int(question.getAttribute(answer_class, "label")))
#                 pred.append(int(torch.argmax(question.getAttribute(answer_class, "local/argmax"))))
#         return f1_score(actual, pred, average="macro")
#
#     best_accuracy = 0
#     best_epoch = 0
#     old_file = None
#     training_file = open("training.txt", 'a')
#     print("-" * 10, file=training_file)
#     print("Training by ", program_name, file=training_file)
#     print("Learning Rate:", args.lr, file=training_file)
#     training_file.close()
#     for epoch in range(check_epoch, limit, check_epoch):
#         training_file = open("training.txt", 'a')
#         program.train(train_set, train_epoch_num=check_epoch,
#                       Optim=lambda param: torch.optim.Adam(param, lr=lr, amsgrad=True),
#                       device=cur_device)
#         accuracy = evaluate()
#         print("Epoch:", epoch, file=training_file)
#         print("Dev Accuracy:", accuracy * 100, "%", file=training_file)
#         if accuracy > best_accuracy:
#             best_epoch = epoch
#             best_accuracy = accuracy
#             # if old_file:
#             #     os.remove(old_file)
#             if program_name == "PMD":
#                 program_addition = "_beta_" + str(args.beta)
#             else:
#                 program_addition = "_size_" + str(args.sampling_size)
#             new_file = program_name + "_" + str(epoch) + "epoch" + "_lr_" + str(args.lr) + program_addition
#             old_file = new_file
#             program.save("Models/" + new_file)
#         training_file.close()
#
#     training_file = open("training.txt", 'a')
#     if epoch < limit:
#         program.train(train_set, train_epoch_num=limit - epoch,
#                       Optim=lambda param: torch.optim.Adam(param, lr=lr, amsgrad=True),
#                       device=cur_device)
#         accuracy = evaluate()
#         print("Epoch:", limit, file=training_file)
#         print("Dev Accuracy:", accuracy * 100, "%", file=training_file)
#         if accuracy > best_accuracy:
#             best_epoch = epoch + check_epoch
#             # if old_file:
#             #     os.remove(old_file)
#             new_file = program_name + "_" + str(limit) + "epoch" + "_lr_" + str(args.lr)
#             old_file = new_file
#             program.save("Models/" + new_file)
#     print("Best epoch ", best_epoch, file=training_file)
#     training_file.close()
#     return best_epoch

def main(args):
    from graph_senetences import answer_class

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

    eval_program.train(train_dataset, train_epoch_num=0,
                        Optim=lambda params: torch.optim.AdamW(params, lr=args.learning_rate), device=cur_device)

    # Loading train parameter to evaluation program
    #train_program.save("Models.pth")
    #eval_program.load("Models.pth")
    eval(eval_program, test_dataset, cur_device, args, "ALL")
    eval(eval_program, augment_dataset_dev, cur_device, args, "Augmented_dev")
    eval(eval_program, augment_dataset_test, cur_device, args, "Augmented")

    correct_ILP = 0
    correct_softmax = 0
    result = {"premise": [],
              "hypothesis": [],
              "actual": [],
              "predict_softmax": [],
              "predict_ILP": []}

    # for datanode in program.populate(test_dataset, device=cur_device):
    #     for sentence in datanode.getChildDataNodes():
    #         result["premise"].append(sentence.getAttribute("premise"))
    #         result["hypothesis"].append(sentence.getAttribute("hypothesis"))
    #         result["actual"].append('entailment' if sentence.getAttribute(entailment, 'label')
    #                                 else 'neutral' if sentence.getAttribute(neutral, 'label') else 'contrast')
    #         result["predict_ILP"].append('entailment' if sentence.getAttribute(entailment, 'ILP')
    #                                  else 'neutral' if sentence.getAttribute(neutral, 'ILP') else 'contrast')
    #
    #         correct_ILP += sentence.getAttribute(entailment, 'ILP').item() if sentence.getAttribute(entailment, 'label') else \
    #             sentence.getAttribute(neutral, 'ILP').item() if sentence.getAttribute(neutral, 'label') else \
    #                 sentence.getAttribute(contradiction, 'ILP').item()
    #
    #         predict_ent = sentence.getAttribute(entailment, 'local/softmax')[1].item()
    #         predict_neu = sentence.getAttribute(neutral, 'local/softmax')[1].item()
    #         predict_con = sentence.getAttribute(contradiction, 'local/softmax')[1].item()
    #         label = ["entailment", "neutral", "contrast"]
    #         predict = label[np.array([predict_ent, predict_neu, predict_con]).argmax()]
    #         result["predict_softmax"].append(predict)
    #         actual_check = entailment if predict == "entailment" else \
    #             neutral if predict == "neutral" else contradiction
    #         correct_softmax += 1 if sentence.getAttribute(actual_check, 'label') else 0
    #
    # correct_augment_ILP = 0
    # correct_augment_softmax = 0
    # count_augment = 0
    # for datanode in program.populate(augment_dataset, device=cur_device):
    #     for sentence in datanode.getChildDataNodes():
    #         correct_augment_ILP += sentence.getAttribute(entailment, 'ILP').item() if sentence.getAttribute(entailment, 'label') else \
    #             sentence.getAttribute(neutral, 'ILP').item() if sentence.getAttribute(neutral, 'label') else \
    #                 sentence.getAttribute(contradiction, 'ILP').item()
    #
    #         predict_ent = sentence.getAttribute(entailment, 'local/softmax')[1].item()
    #         predict_neu = sentence.getAttribute(neutral, 'local/softmax')[1].item()
    #         predict_con = sentence.getAttribute(contradiction, 'local/softmax')[1].item()
    #         label = ["entailment", "neutral", "contrast"]
    #         predict = label[np.array([predict_ent, predict_neu, predict_con]).argmax()]
    #         actual_check = entailment if predict == "entailment" else \
    #             neutral if predict == "neutral" else contradiction
    #         correct_augment_softmax += 1 if sentence.getAttribute(actual_check, 'label') else 0
    #         count_augment += 1
    # print("Using Symmetric:", args.sym_relation)
    # print("Using PML:", args.primaldual)
    # print("Accuracy Softmax = %.3f%%" % (correct_softmax / len(result["predict_softmax"]) * 100))
    # print("Accuracy ILP = %.3f%%" % (correct_ILP / len(result["predict_ILP"]) * 100))
    # print("Accuracy Softmax on augment data = %.3f%%" % (correct_augment_softmax * 100 / count_augment))
    # print("Accuracy ILP on augment data = %.3f%%" % (correct_augment_ILP * 100 / count_augment))
    result = pd.DataFrame(result)
    training_size = args.training_sample
    import os
    output_file = "report-{:}-{:}-{:}--sym:{:}.csv".format(args.training_sample, args.testing_sample,
                                                           args.cur_epoch, args.sym_relation)
    result.to_csv(os.path.join(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLI Learning Code")

    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)

    parser.add_argument('--epoch', dest='cur_epoch', default=5, help='number of epochs to train model', type=int)

    parser.add_argument('--lr', dest='learning_rate', default=1e-5, help='learning rate of the adamW optimiser',
                        type=float)

    parser.add_argument('--training_sample', dest='training_sample', default=600,
                        help="number of data to train model", type=int)

    parser.add_argument('--testing_sample', dest='testing_sample', default=600, help="number of data to test model",
                        type=int)

    parser.add_argument('--batch_size', dest='batch_size', default=2, help="batch size of sample", type=int)

    parser.add_argument('--sym_relation', dest='sym_relation', default=False, help="Using symmetric relation",
                        type=bool)
    parser.add_argument('--tran_relation', dest='tran_relation', default=False, help="Using transitive relation",
                        type=bool)
    parser.add_argument('--pmd', dest='primaldual', default=False, help="Using primaldual model or not",
                        type=bool)
    parser.add_argument('--sampleloss', dest='sampleloss', default=False, help="Using IML model or not",
                        type=bool)
    parser.add_argument('--beta', dest='beta', default=0.5, help="Using IML model or not",
                        type=float)
    parser.add_argument('--sampling_size', dest='sampling_size', default=100, help="Using IML model or not",
                        type=int)
    args = parser.parse_args()
    main(args)
