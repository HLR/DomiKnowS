import sys
import torch
from torch.utils.data import random_split, DataLoader
import argparse
from dataset import load_animals_and_flowers
from model import model_declaration
from graph import graph, daisy, dandelion, rose, sunflower, tulip, cat, dog, monkey, squirrel, animal, flower
import logging, random

sys.path.append('.')
sys.path.append('../..')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--solver', help='the model solver', default='iml')  # TODO add solverPOImodel
parser.add_argument('--samplenum', dest='samplenum', default=4521, help='number of samples to choose from the dataset',
                    type=int)
parser.add_argument('--verbose', dest='verbose', default=0, help='print the errors', type=int)
parser.add_argument('--debugmode', dest='debugmode', default=0, help='different debugging setting to test the code',
                    type=int)
parser.add_argument('--epochs', dest='epochs', default=10, help='number of training epoch', type=int)
parser.add_argument('--lambdaValue', dest='lambdaValue', default=0.5, help='number of training epoch', type=float)
# TODO add  model size and other things here
args = parser.parse_args()


def main():
    torch.manual_seed(777)
    random.seed(777)
    logging.basicConfig(level=logging.INFO)
    trainset = load_animals_and_flowers(args, root='./data/', size=100, )

    trainset = list(trainset)

    random.shuffle(trainset)
    train_ds, val_ds = random_split(trainset[:args.samplenum],
                                    [args.samplenum // 5 * 4, args.samplenum - args.samplenum // 5 * 4])

    solver = args.solver
    program = model_declaration(solver=solver, lambdaValue=args.lambdaValue)
    program.load("./checkpoint")
    program.train(train_ds, valid_set=val_ds, train_epoch_num=args.epochs,
                  Optim=lambda param: torch.optim.Adam(param, lr=.0001), device=device)
    # program.save("./checkpoint")
    program.test(val_ds)
    pic_list = [animal, flower, daisy, dandelion, rose, sunflower, tulip, cat, dog, monkey, squirrel]
    for key in ["local/argmax", "ILP"]:
        print(f"#################################{key}######################")
        tps = [0 for i in range(11)]
        tns = [0 for i in range(11)]
        fps = [0 for i in range(11)]
        fns = [0 for i in range(11)]
        for pic_num, picture in enumerate(program.populate(val_ds, device=device)):
            for j, i in enumerate(pic_list):
                # predictionValue = int(picture.getAttribute(i, key).item())
                # predictionValue = int(torch.argmax(picture.getAttribute(i)))
                if key == "ILP":
                    predictionValue = int(picture.getAttribute(i, key).item())
                else:
                    predictionValue = int(torch.argmax(picture.getAttribute(i, key)))
                labelValue = int(picture.getAttribute(i, "label")[0])
                if predictionValue == 1:
                    if labelValue == 1:
                        tps[j] += 1
                    elif labelValue == 0:
                        fps[j] += 1
                elif predictionValue == 0:
                    if labelValue == 1:
                        fns[j] += 1
                    elif labelValue == 0:
                        tns[j] += 1
        for index, image in enumerate(pic_list):
            print(image.__str__() + " " * (10 - len(image.__str__())), end="\t")
            precision = tps[index] * 100 / (tps[index] + fps[index])
            print("Precision:", round(precision, 2), end="\t")
            if tps[index] + fns[index]:
                recall = tps[index] * 100 / (tps[index] + fns[index])
                print("Recall:", round(recall, 2), end="\t")
                f1 = 2 * precision * recall / (precision + recall)
                print("F1:", round(f1, 2), end="\t")
            acc = (tps[index] + tns[index]) * 100 / (tps[index] + tns[index] + fps[index] + fns[index])
            print("Accuracy:", round(acc, 2))


if __name__ == '__main__':
    main()
