import argparse
import sys



sys.path.append('.')
sys.path.append('../..')
from regr.program.lossprogram import SampleLossProgram
from regr.program.model.pytorch import SolverModel
from regr.program.primaldualprogram import PrimalDualProgram

import torch
from torchvision.models import resnet18

from regr.program import SolverPOIProgram, IMLProgram
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from regr.sensor.pytorch.learners import ModuleLearner
from reader import create_readers
import torch.nn as nn
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from graph import graph, image_group_contains,image,category,Label,image_group


class ImageNetwork(torch.nn.Module):
    def __init__(self):
        super(ImageNetwork, self).__init__()
        self.conv = resnet18(pretrained=True)

    def forward(self, x):
        x = self.conv(x)
        return x

def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
    parser.add_argument('--solver', help='the model solver', default='poi')
    parser.add_argument('--samplenum', dest='samplenum', default=5000,help='number of samples to choose from the dataset',type=int)
    parser.add_argument('--epochs', dest='epochs', default=10, help='number of training epoch', type=int)
    parser.add_argument('--lambdaValue', dest='lambdaValue', default=0.5, help='value of learning rate', type=float)
    parser.add_argument('--lr', dest='learning_rate', default=2e-4, help='learning rate of the adam optimiser',type=float)
    parser.add_argument('--beta', dest='beta', default=0.1, help='primal dual or IML multiplier', type=float)
    parser.add_argument('--test', dest='test', default=False, help='test the final resultss', type=bool)

    args = parser.parse_args()

    device = "cuda:"+str(args.cuda_number)

    image_group['pixels_group'] = ReaderSensor(keyword='pixels', device=device)
    image_group['category_group'] = ReaderSensor(keyword='corase_label', device=device)
    image_group['tag_group'] = ReaderSensor(keyword='fine_label', device=device)

    def str_to_int_list(x):
        return torch.LongTensor([[int(i)] for i in x])

    def make_images(pixels_group, category_group, tag_group):
        return torch.ones((len(category_group.split("@@")), 1)), torch.squeeze(pixels_group, 0), str_to_int_list(
            category_group.split("@@")), str_to_int_list(tag_group.split("@@"))

    image[image_group_contains, "pixels", 'category_', "tag_"] = JointSensor(image_group['pixels_group'],
                                                                             image_group["category_group"],
                                                                             image_group["tag_group"],
                                                                             forward=make_images)
    def label_reader(_, label):
        return label

    image[category] = FunctionalSensor(image_group_contains, "category_", forward=label_reader, label=True)
    image[Label] = FunctionalSensor(image_group_contains, "tag_", forward=label_reader, label=True)


    image['emb'] = ModuleLearner('pixels', module=resnet18(pretrained=True))
    image[category] = ModuleLearner('emb', module=nn.Linear(1000, 20))
    image[Label] = ModuleLearner('emb', module=nn.Linear(1000, 100))
    f = open(str(args.solver)+"_"+str(args.samplenum)+"_"+"_"+str(args.beta)+".txt", "w")
    if args.solver=="poi":
        print("POI")
        program = SolverPOIProgram(graph,inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss())\
                               ,metric={'ILP': PRF1Tracker(DatanodeCMMetric()),'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},f=f)
    elif args.solver=="primal_dual":

        print("PrimalDualProgram")
        program = PrimalDualProgram(graph, SolverModel, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},beta=args.lambdaValue,f=f)
    elif args.solver=="sam":

        print("sam")
        program = SampleLossProgram(graph, SolverModel,inferTypes=['ILP','local/argmax'],
        metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},loss=MacroAverageTracker(NBCrossEntropyLoss()),sample=True,sampleSize=50,sampleGlobalLoss=True,beta=args.beta,device=device)

    elif args.solver=="iml":

        print("iml")
        program = IMLProgram(graph, inferTypes=['ILP', 'local/argmax'], \
                             loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=args.beta)),metric={'ILP': PRF1Tracker(DatanodeCMMetric()), \
                                     'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    train_reader,test_reader=create_readers(train_num=args.samplenum)
    for i in range(args.epochs):
        program.train(train_reader,valid_set=test_reader, train_epoch_num=1, Optim=lambda param: torch.optim.Adam(param, lr=args.learning_rate),device=device)
        program.save(str(args.solver)+"_"+str(args.samplenum)+"_"+str(i)+"_"+str(args.beta))
    f.close()
    guessed_tag = {
        "local/softmax": [],
        "ILP": []
    }
    real_tag = []
    guessed_category = {
        "local/softmax": [],
        "ILP": []
    }

    real_category = []
    for pic_num, picture_group in enumerate(program.populate(test_reader, device=device)):
        for image_ in picture_group.getChildDataNodes():
            for key in ["local/softmax", "ILP"]:
                if key == "ILP":
                    guessed_tag[key].append(int(torch.argmax(image_.getAttribute(Label, key))))
                else:
                    guessed_tag[key].append(int(torch.argmax(image_.getAttribute(Label, key))))

                if key == "ILP":
                    guessed_category[key].append(int(torch.argmax(image_.getAttribute(category, key))))
                else:
                    guessed_category[key].append(int(torch.argmax(image_.getAttribute(category, key))))
            real_tag.append(int(image_.getAttribute(Label, "label")[0]))
            real_category.append(int(image_.getAttribute(category, "label")))

    for key in ["local/softmax", "ILP"]:
        print(f"##############################{key}#########################")
        guessed_labels = guessed_tag[key]
        real_labels = real_tag
        #print(guessed_labels)
        #print(real_labels)
        correct = sum(1 if x == y else 0 for x, y in zip(real_labels, guessed_labels))
        total = len(real_labels)+1
        print("tags accuracy", correct / total)
        guessed_labels = guessed_category[key]
        real_labels = real_category
        #print(guessed_labels)
        #print(real_labels)
        correct = sum(1 if x == y else 0 for x, y in zip(real_labels, guessed_labels))
        total = len(real_labels)+1
        print("category accuracy", correct / total)


if __name__ == '__main__':
    main()