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
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torchvision.models.resnet import resnet50
import math

import torch
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
parser.add_argument('--solver', help='the model solver', default='poi')
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


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


output_size = 1000
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import torch


class AnimalAndFlowers(Dataset):

    def __init__(self, args, root, transform=None):

        self.root = root
        self.data = []
        self.targets = []
        self.transform = transform
        index = 0
        if args.verbose:
            print("current directory", os.getcwd())
        for folder_name in os.listdir(root):
            if "." in folder_name or "food" in folder_name:
                continue
            for category_name in os.listdir(f"{root}/{folder_name}"):
                print(category_name)
                for file_name in os.listdir(f"{root}/{folder_name}/{category_name}/"):
                    file_path = os.path.join(self.root, folder_name, category_name, file_name)
                    with open(file_path, 'rb') as f:
                        self.data.append(np.array(Image.open(file_path).resize((100, 100))))
                        self.targets.append(
                            ["cat", "dog", "monkey", "squirrel", "daisy", "dandelion", "rose", "sunflower",
                             "tulip"].index(category_name))
                        index += 1
        self.data = np.vstack(self.data).reshape(-1, 3, 100, 100)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict:
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # img = img.unsqueeze(0)

        return img, target


def load_animals_and_flowers(args, root='./data/', size=100, ):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=round(size // 8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    )
    return AnimalAndFlowers(args, root=root, transform=transform)


class ImageNetwork(torch.nn.Module):
    def __init__(self, n_outputs=9):
        super(ImageNetwork, self).__init__()
        self.num_outputs = n_outputs
        self.conv = resnet50(pretrained=True)
        set_parameter_requires_grad(self.conv, True)

        self.dense1 = nn.Linear(output_size, output_size)
        self.dense2 = nn.Linear(output_size, n_outputs)
        # self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = self.drop(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def loadData():
    trainset = load_animals_and_flowers(args, root='./data/', size=100)
    train_ds, val_ds = random_split(trainset, [len(trainset) // 5 * 4, len(trainset) - len(trainset) // 5 * 4])
    trainloader = DataLoader(train_ds, batch_size=32)
    testloader = DataLoader(val_ds, batch_size=32)

    return trainloader, testloader


def calSubAccuracy(dataset, name, network):
    classes = ["cat", "dog", "monkey", "squirrel", "daisy", "dandelion", "rose", "sunflower", "tulip"]
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            outputs = network(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} in {} %".format(classname,
                                                                   accuracy, name))


def calTotalAccuracy(dataset, name, network):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} {name} images: {100 * correct / total} %')


def main():
    if torch.cuda.is_available():
        device = "cuda:" + str(args.cuda_number)
    else:
        device = 'cpu'
    print("selected device is:", device)
    torch.manual_seed(777)
    random.seed(777)
    logging.basicConfig(level=logging.INFO)

    trainloader, testloader = loadData()

    net = ImageNetwork()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=[0.9, 0.999], eps=1e-07, amsgrad=False)

    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % 50 == 49:  # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 10))
        running_loss = 0.0

        calTotalAccuracy(trainloader, "Training", net)
        calTotalAccuracy(testloader, "Validation", net)

    print('Finished Training')


if __name__ == '__main__':
    main()
