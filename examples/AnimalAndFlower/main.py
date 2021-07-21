import sys
import torch
from torch.utils.data import random_split, DataLoader
import argparse
from dataset import load_animals_and_flowers
from model import model_declaration
from graph import graph,daisy, dandelion, rose, sunflower, tulip, cat, dog, monkey, squirrel,animal,flower

sys.path.append('.')
sys.path.append('../..')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--solver', help='the model solver', default='iml') #TODO add solverPOImodel
parser.add_argument('--samplenum', dest='samplenum', default=100, help='number of samples to choose from the dataset',type=int)
parser.add_argument('--verbose', dest='verbose', default=1, help='print the errors',type=int)
parser.add_argument('--debugmode', dest='debugmode', default=0, help='different debugging setting to test the code',type=int)
#TODO add epoch, lambda , model size and other things here
args = parser.parse_args()


def main():
    #TODO set seeds here to get similar results after each run
    import logging,random
    logging.basicConfig(level=logging.INFO)
    trainset=load_animals_and_flowers(args,root='./data/', size=100,)

    trainset=list(trainset)

    random.shuffle(trainset)
    train_ds, val_ds = random_split(trainset[:args.samplenum], [args.samplenum//5*4, args.samplenum-args.samplenum//5*4])

    solver = args.solver
    program = model_declaration(solver=solver)
    program.train(train_ds, valid_set=val_ds, train_epoch_num=1, Optim=lambda param: torch.optim.Adam(param, lr=.0001),device=device)

    pic_list=[daisy, dandelion, rose, sunflower, tulip, cat, dog, monkey, squirrel,animal,flower]
    accuracy_ILP=[0 for i in range(11)]
    accuracy_SM=[0 for i in range(11)]
    precision_ILP=[0 for i in range(11)]
    precision_SM=[0 for i in range(11)]
    total=[0 for i in range(11)]
    precision_total=[0 for i in range(11)]
    for pic_num,picture in enumerate(program.populate(val_ds, device=device)):
        for j,i in enumerate(pic_list):
            if picture.getAttribute(i, "ILP").item() == picture.getAttribute(i,"label")[0]:
                accuracy_ILP[j]+=1
            if picture.getAttribute(i)[picture.getAttribute(i,"label")[0]]>picture.getAttribute(i)[1-picture.getAttribute(i,"label")[0]]:
                accuracy_SM[j]+=1
            if picture.getAttribute(i,"label")[0]:

                if picture.getAttribute(i, "ILP").item() == picture.getAttribute(i,"label")[0]:
                    precision_ILP[j]+=1
                if picture.getAttribute(i)[picture.getAttribute(i,"label")[0]]>picture.getAttribute(i)[1-picture.getAttribute(i,"label")[0]]:
                    precision_SM[j]+=1
                precision_total[j]+=1
            total[j]+=1

    print("ILP accuracy",[i/j for i,j in zip(accuracy_ILP,total)])
    print("softmax accuracy",[i/j for i,j in zip(accuracy_SM,total)])
    print("ILP precision",[i/j for i,j in zip(precision_ILP,total)])
    print("softmax precision",[i/j for i,j in zip(precision_SM,total)])

if __name__ == '__main__':
    main()
