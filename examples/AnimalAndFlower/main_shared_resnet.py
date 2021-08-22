import sys

sys.path.append('/home/admiraldarius/DomiKnowS/')
sys.path.append('/home/admiraldarius/DomiKnowS/regr')
sys.path.append('/home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/regr')
sys.path.append('/home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS')

import torch
from torch.utils.data import random_split, DataLoader
import argparse
from dataset import load_animals_and_flowers
from model_shared_resnet import model_declaration
from graph import graph, daisy, dandelion, rose, sunflower, tulip, cat, dog, monkey, squirrel, animal, flower
import logging, random
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on',type=int)
parser.add_argument('--solver', help='the model solver', default='primal_dual')
parser.add_argument('--samplenum', dest='samplenum', default=4521, help='number of samples to choose from the dataset',
                    type=int)
parser.add_argument('--verbose', dest='verbose', default=0, help='print the errors', type=int)
parser.add_argument('--debugmode', dest='debugmode', default=0, help='different debugging setting to test the code',
                    type=int)
parser.add_argument('--epochs', dest='epochs', default=30, help='number of training epoch', type=int)
parser.add_argument('--lambdaValue', dest='lambdaValue', default=0.1, help='value of learning rate', type=float)
parser.add_argument('--model_size', dest='model_size', default=10, help='size of the model',type=int)
parser.add_argument('--lr', dest='learning_rate', default=2e-5, help='learning rate of the adam optimiser',type=float)

# TODO add  model size and other things here
args = parser.parse_args()

import math
def main():
    if torch.cuda.is_available():
        device="cuda:"+str(args.cuda_number)
    else:
        device='cpu'
    print("selected device is:",device)

    torch.manual_seed(777)
    random.seed(777)
    logging.basicConfig(level=logging.INFO)
    trainset = load_animals_and_flowers(args, root='./data/', size=100, )

    train_ds, val_ds = trainset[:math.floor(len(trainset)*0.8)], trainset[math.floor(len(trainset)*0.8):]
    solver = args.solver
    program = model_declaration(device=device,solver=solver, lambdaValue=args.lambdaValue,model_size=args.model_size)
    # program.load("./checkpoint")
    program.train(train_ds, valid_set=val_ds, train_epoch_num=args.epochs,
                  Optim=lambda param: torch.optim.Adam(param, lr=args.learning_rate), device=device)
    #program.save("./checkpoint")

    pic_list = [ daisy, dandelion, rose, sunflower, tulip, cat, dog, monkey, squirrel]
    parent_list=[animal, flower]
    for key in ["local/argmax", "ILP"]:
         print(f"#################################{key}######################")
         accuracy=0
         inaccuracy=0

         accuracy_parent=0
         inaccuracy_parent=0

         for pic_num, picture_group in enumerate(program.populate(val_ds, device=device)):
             for image_ in picture_group.getChildDataNodes():
                guessed_labels=[]
                real_labels=[]

                for class_number, image_class in enumerate(pic_list):
                     guessed_labels.append(int(image_.getAttribute(image_class, key).item()))
                     real_labels.append(int(image_.getAttribute(image_class, "label")[0]))
                f = lambda i: guessed_labels[i]
                chosen_class=max(range(len(l)), key=f)
                if real_labels[chosen_class]:
                    accuracy+=1
                else:
                    inaccuracy+=1

                guessed_labels=[]
                real_labels=[]

                for class_number, image_class in enumerate(parent_list):
                     guessed_labels.append(int(image_.getAttribute(image_class, key).item()))
                     real_labels.append(int(image_.getAttribute(image_class, "label")[0]))
                f = lambda i: guessed_labels[i]
                chosen_class=max(range(len(l)), key=f)
                if real_labels[chosen_class]:
                    accuracy_parent+=1
                else:
                    inaccuracy_parent+=1

    print("accuracy:",accuracy/(accuracy+inaccuracy))
    print("parent accuracy:",accuracy_parent/(accuracy_parent+inaccuracy_parent))

if __name__ == '__main__':
    main()
