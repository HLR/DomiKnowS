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
from regr.graph import Graph, Concept, Relation


def main(args):
    from graph_senetences import entailment, neutral, contradiction

    # Set the cuda number we want to use
    cuda_number = args.cuda_number
    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'
        
    print('Using: %s'%(cur_device))
    
    training_file = "train.csv"
    testing_file = "test.csv"

    augment_file = "data/snli_genadv_1000_test.jsonl"
    # Loading Test and Train data
    test_dataset = DataReaderMultiRelation(file="data/" + testing_file, size=args.testing_sample,
                                           batch_size=args.batch_size, augment_file=augment_file)

    train_dataset = DataReaderMultiRelation(file="data/" + training_file, size=args.training_sample,
                                            batch_size=args.batch_size)
    # Load Augmentation data
    augment_dataset = DataReaderMultiRelation(file=None, size=None, batch_size=args.batch_size,
                                              augment_file=augment_file)
    # Declare Program
    program = program_declaration(cur_device, sym_relation=args.sym_relation, tran_relation=args.tran_relation,
                                  primaldual=args.primaldual, sample=args.sampleloss, beta=args.beta)

    program.train(train_dataset, test_set=test_dataset, train_epoch_num=args.cur_epoch,
                  Optim=lambda params: torch.optim.AdamW(params, lr=args.learning_rate), device=cur_device)

    correct_ILP = 0
    correct_softmax = 0
    result = {"premise": [],
              "hypothesis": [],
              "actual": [],
              "predict_softmax": [],
              "predict_ILP": []}
    for datanode in program.populate(test_dataset, device=cur_device):
        for sentence in datanode.getChildDataNodes():
            result["premise"].append(sentence.getAttribute("premise"))
            result["hypothesis"].append(sentence.getAttribute("hypothesis"))
            result["actual"].append('entailment' if sentence.getAttribute(entailment, 'label')
                                    else 'neutral' if sentence.getAttribute(neutral, 'label') else 'contrast')
            result["predict_ILP"].append('entailment' if sentence.getAttribute(entailment, 'ILP')
                                     else 'neutral' if sentence.getAttribute(neutral, 'ILP') else 'contrast')

            correct_ILP += sentence.getAttribute(entailment, 'ILP').item() if sentence.getAttribute(entailment, 'label') else \
                sentence.getAttribute(neutral, 'ILP').item() if sentence.getAttribute(neutral, 'label') else \
                    sentence.getAttribute(contradiction, 'ILP').item()

            predict_ent = sentence.getAttribute(entailment, 'local/softmax')[1].item()
            predict_neu = sentence.getAttribute(neutral, 'local/softmax')[1].item()
            predict_con = sentence.getAttribute(contradiction, 'local/softmax')[1].item()
            label = ["entailment", "neutral", "contrast"]
            predict = label[np.array([predict_ent, predict_neu, predict_con]).argmax()]
            result["predict_softmax"].append(predict)
            actual_check = entailment if predict == "entailment" else \
                neutral if predict == "neutral" else contradiction
            correct_softmax += 1 if sentence.getAttribute(actual_check, 'label') else 0

    correct_augment_ILP = 0
    correct_augment_softmax = 0
    count_augment = 0
    for datanode in program.populate(augment_dataset, device=cur_device):
        for sentence in datanode.getChildDataNodes():
            correct_augment_ILP += sentence.getAttribute(entailment, 'ILP').item() if sentence.getAttribute(entailment, 'label') else \
                sentence.getAttribute(neutral, 'ILP').item() if sentence.getAttribute(neutral, 'label') else \
                    sentence.getAttribute(contradiction, 'ILP').item()

            predict_ent = sentence.getAttribute(entailment, 'local/softmax')[1].item()
            predict_neu = sentence.getAttribute(neutral, 'local/softmax')[1].item()
            predict_con = sentence.getAttribute(contradiction, 'local/softmax')[1].item()
            label = ["entailment", "neutral", "contrast"]
            predict = label[np.array([predict_ent, predict_neu, predict_con]).argmax()]
            actual_check = entailment if predict == "entailment" else \
                neutral if predict == "neutral" else contradiction
            correct_augment_softmax += 1 if sentence.getAttribute(actual_check, 'label') else 0
            count_augment += 1
    print("Using Symmetric:", args.sym_relation)
    print("Using PML:", args.primaldual)
    print("Accuracy Softmax = %.3f%%" % (correct_softmax / len(result["predict_softmax"]) * 100))
    print("Accuracy ILP = %.3f%%" % (correct_ILP / len(result["predict_ILP"]) * 100))
    print("Accuracy Softmax on augment data = %.3f%%" % (correct_augment_softmax * 100 / count_augment))
    print("Accuracy ILP on augment data = %.3f%%" % (correct_augment_ILP * 100 / count_augment))
    result = pd.DataFrame(result)
    training_size = args.training_sample
    import os
    output_file = "report-{:}-{:}-{:}--sym:{:}.csv".format(args.training_sample, args.testing_sample,
                                                           args.cur_epoch, args.sym_relation)
    result.to_csv(os.path.join(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLI Learning Code")

    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)

    parser.add_argument('--epoch', dest='cur_epoch', default=3, help='number of epochs to train model', type=int)

    parser.add_argument('--lr', dest='learning_rate', default=1e-5, help='learning rate of the adamW optimiser',
                        type=float)

    parser.add_argument('--training_sample', dest='training_sample', default=550146,
                        help="number of data to train model", type=int)

    parser.add_argument('--testing_sample', dest='testing_sample', default=10000, help="number of data to test model",
                        type=int)

    parser.add_argument('--batch_size', dest='batch_size', default=4, help="batch size of sample", type=int)

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
    args = parser.parse_args()
    main(args)
