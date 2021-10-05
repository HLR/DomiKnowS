import sys

sys.path.append('/home/admiraldarius/DomiKnowS/')
sys.path.append('/home/admiraldarius/DomiKnowS/regr')
sys.path.append('/home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/regr')
sys.path.append('/home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS')

import argparse
from dataset_multiclass import loadDataset
from model_multiclass import model_declaration,graph
from graph_multiclass import family,subFamily
import logging, random
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
parser.add_argument('--solver', help='the model solver', default='primal_dual')
parser.add_argument('--samplenum', dest='samplenum', default=4521, help='number of samples to choose from the dataset',
                    type=int)
parser.add_argument('--verbose', dest='verbose', default=0, help='print the errors', type=int)
parser.add_argument('--debugmode', dest='debugmode', default=0, help='different debugging setting to test the code',
                    type=int)
parser.add_argument('--epochs', dest='epochs', default=7, help='number of training epoch', type=int)
parser.add_argument('--lambdaValue', dest='lambdaValue', default=0.1, help='value of learning rate', type=float)
parser.add_argument('--model_size', dest='model_size', default=10, help='size of the model', type=int)
parser.add_argument('--lr', dest='learning_rate', default=0.0001, help='learning rate of the adam optimiser', type=float)

# TODO add  model size and other things here
args = parser.parse_args()

import math


def main():
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = "cuda:" + str(args.cuda_number)
    else:
        device = 'cpu'
    print("selected device is:", device)

    torch.manual_seed(777)
    random.seed(777)
    logging.basicConfig(level=logging.INFO)
    # trainset = loadDataset(mode="train")
    train_ds = loadDataset(mode="test")
    import gc
    gc.collect()
    print("Data Loaded")
    solver = args.solver
    program = model_declaration(device=device, solver=solver, lambdaValue=args.lambdaValue, model_size=args.model_size)
    program.train(train_ds, train_epoch_num=args.epochs,
                  Optim=lambda param: torch.optim.Adam(param, lr=args.learning_rate), device=device)
    
    guessed_subFamily = {
        "local/softmax": [],
        "ILP": []
    }
    real_subFamily = []
    guessed_family = {
        "local/softmax": [],
        "ILP": []
    }
    subFamily = graph['subFamily']
    family = graph['family']
    real_family = []
    for pic_num, picture_group in enumerate(program.populate(train_ds, device=device)):
        for image_ in picture_group.getChildDataNodes():
            for key in ["local/softmax", "ILP"]:
                if key == "ILP":
                    guessed_subFamily[key].append(int(torch.argmax(image_.getAttribute(subFamily, key))))
                else:
                    guessed_subFamily[key].append(int(torch.argmax(image_.getAttribute(subFamily, key))))

                if key == "ILP":
                    guessed_family[key].append(int(torch.argmax(image_.getAttribute(family, key))))
                else:
                    guessed_family[key].append(int(torch.argmax(image_.getAttribute(family, key))))
            real_subFamily.append(int(image_.getAttribute(subFamily, "label")[0]))
            real_family.append(int(image_.getAttribute(family, "label")[0]))

    for key in ["local/softmax", "ILP"]:
        print(f"##############################{key}#########################")
        guessed_labels = guessed_subFamily[key]
        real_labels = real_subFamily
        print(guessed_labels)
        print(real_labels)
        correct = sum(1 if x == y else 0 for x, y in zip(real_labels, guessed_labels))
        total = len(real_labels)
        print("subFamilys accuracy", correct / total)
        guessed_labels = guessed_family[key]
        real_labels = real_family
        print(guessed_labels)
        print(real_labels)
        correct = sum(1 if x == y else 0 for x, y in zip(real_labels, guessed_labels))
        total = len(real_labels)
        print("family accuracy", correct / total)



if __name__ == '__main__':
    main()
