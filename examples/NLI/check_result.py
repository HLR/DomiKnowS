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


    augment_file = "data/snli_genadv_1000_test.jsonl"
    # Loading Test and Train data
    # Load Augmentation data
    augment_dataset = DataReaderMultiRelation(file=None, size=None, batch_size=args.batch_size,
                                              augment_file=augment_file)
    # Declare Program
    program = program_declaration(cur_device, sym_relation=args.sym_relation, tran_relation=args.tran_relation,
                                  primaldual=args.primaldual, sample=args.sampleloss, beta=args.beta)

    program.load(args.loaded_file, map_location={'cuda:0': cur_device, 'cuda:1': cur_device})
    total_data_node = 0
    satisfy_ILP = 0
    satisfy_argmax = 0
    for datanode in program.populate(augment_dataset, device=cur_device):
        total_data_node += 1

        verify_constrainsILP = datanode.verifyResultsLC(key="/ILP")
        count_verify = 0
        if verify_constrainsILP:
            for lc in verify_constrainsILP:
                count_verify += verify_constrainsILP[lc]['satisfied']
            satisfy_ILP += count_verify / len(verify_constrainsILP)

        verify_constrains = datanode.verifyResultsLC(key="")
        count_verify = 0
        if verify_constrains:
            for lc in verify_constrains:
                count_verify += verify_constrains[lc]['satisfied']
            satisfy_argmax += count_verify / len(verify_constrains)
    satisfy_argmax = satisfy_argmax / total_data_node
    satisfy_ILP = satisfy_ILP / total_data_node
    result_file = open("constrain_result", 'a')
    print("PMD" if args.primaldual else "Sampling Loss" if args.sampleloss else "DomiknowS", file=result_file)
    print("Without ILP, constrains is {:.2f}% satisfy".format(satisfy_argmax), file=result_file)
    print("With ILP, constrains is {:.2f}% satisfy".format(satisfy_ILP), file=result_file)
    result_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLI Learning Code")

    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)

    parser.add_argument('--epoch', dest='cur_epoch', default=3, help='number of epochs to train model', type=int)

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
    parser.add_argument('--loaded_file', dest='loaded_file', default="", help="Load parameter file",
                        type=float)
    args = parser.parse_args()
    main(args)
