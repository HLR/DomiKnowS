import sys

sys.path.append('/home/admiraldarius/DomiKnowS/')
sys.path.append('/home/admiraldarius/DomiKnowS/regr')
sys.path.append('/home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/regr')
sys.path.append('/home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS')

import torch
from torch.utils.data import random_split, DataLoader
import argparse
from dataset_multiclass import load_animals_and_flowers
from model_shared_resnet_multiclass import model_declaration
import logging, random
import torch
from graph_multiclass import graph

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
parser.add_argument('--solver', help='the model solver', default='primal_dual')
parser.add_argument('--samplenum', dest='samplenum', default=4521, help='number of samples to choose from the dataset',
                    type=int)
parser.add_argument('--verbose', dest='verbose', default=0, help='print the errors', type=int)
parser.add_argument('--debugmode', dest='debugmode', default=0, help='different debugging setting to test the code',
                    type=int)
parser.add_argument('--epochs', dest='epochs', default=5, help='number of training epoch', type=int)
parser.add_argument('--lambdaValue', dest='lambdaValue', default=0.1, help='value of learning rate', type=float)
parser.add_argument('--model_size', dest='model_size', default=10, help='size of the model', type=int)
parser.add_argument('--lr', dest='learning_rate', default=2e-5, help='learning rate of the adam optimiser', type=float)

# TODO add  model size and other things here
args = parser.parse_args()

import math


def main():
    if torch.cuda.is_available():
        device = "cuda:" + str(args.cuda_number)
    else:
        device = 'cpu'
    print("selected device is:", device)

    torch.manual_seed(777)
    random.seed(777)
    logging.basicConfig(level=logging.INFO)
    trainset = load_animals_and_flowers(args, root='./data/', size=100, )
    train_ds, val_ds = trainset[:math.floor(len(trainset) * 0.8)], trainset[math.floor(len(trainset) * 0.8):]
    solver = args.solver
    program = model_declaration(device=device, solver=solver, lambdaValue=args.lambdaValue, model_size=args.model_size)
    # program.load("./checkpoint")
    program.train(train_ds, valid_set=val_ds, train_epoch_num=args.epochs,
                  Optim=lambda param: torch.optim.Adam(param, lr=args.learning_rate), device=device)
    # program.save("./checkpoint")
    guessed_tag = {
        "local/softmax": [],
        "ILP": []
    }
    real_tag = []
    guessed_category = {
        "local/softmax": [],
        "ILP": []
    }
    tag = graph['tag']
    category = graph['category']
    real_category = []
    for pic_num, picture_group in enumerate(program.populate(train_ds, device=device)):
        for image_ in picture_group.getChildDataNodes():
            for key in ["local/softmax", "ILP"]:
                if key == "ILP":
                    guessed_tag[key].append(int(torch.argmax(image_.getAttribute(tag, key))))
                else:
                    guessed_tag[key].append(int(torch.argmax(image_.getAttribute(tag, key)[:-1])))

                if key == "ILP":
                    guessed_category[key].append(int(torch.argmax(image_.getAttribute(category, key))))
                else:
                    guessed_category[key].append(int(torch.argmax(image_.getAttribute(category, key)[1:])))
            real_tag.append(int(image_.getAttribute(tag, "label")[0]))
            real_category.append(int(image_.getAttribute(category, "label")[0]))

    for key in ["local/softmax", "ILP"]:
        print(f"##############################{key}#########################")
        guessed_labels = guessed_tag[key]
        real_labels = real_tag
        print(guessed_labels)
        print(real_labels)
        correct = sum(1 if x == y else 0 for x, y in zip(real_labels, guessed_labels))
        total = len(real_labels)
        print("tags accuracy", correct / total)
        guessed_labels = guessed_category[key]
        real_labels = real_category
        print(guessed_labels)
        print(real_labels)
        correct = sum(1 if x == y else 0 for x, y in zip(real_labels, guessed_labels))
        total = len(real_labels)
        print("category accuracy", correct / total)


if __name__ == '__main__':
    main()
