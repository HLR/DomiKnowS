import torch
import torchvision
from torchvision import datasets


def create_readers(dataset,sample_size,batch_size):
    Numbers = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four",\
               5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}
    reader=[]
    instance = {}
    dist=[i[1] for i in dataset]
    max_size=min([sum([i == j for i in dist]) for j in range(10)])
    max_size-=max_size%batch_size
    image_groups=[[] for i in range(10)]
    for i in dataset:
        image_groups[ i[1] ].append(i)
    dataset=[]
    for i in range(max_size):
        for j in range(10):
            dataset.append(image_groups[j][i])

    for number,i in enumerate(dataset):

        if not "pixels" in instance:
            instance["pixels"]=[i[0]]
        else:
            instance["pixels"].append(i[0])
        for j in range(10):
            if not Numbers[j] in instance:
                instance[Numbers[j]]=""

        for j in range(10):
            if not j==i[1]:
                instance[Numbers[j]] += "0"
            else:
                instance[Numbers[j]] += "1"

        if number%batch_size==batch_size-1:
            instance['pixels']=torch.stack(instance['pixels'], dim=0).unsqueeze(dim=0)
            reader.append(instance)
            instance={}
    if instance:
        instance['pixels'] = torch.stack(instance['pixels'], dim=0).unsqueeze(dim=0)
        reader.append(instance)
    return reader[:sample_size]

from torch import nn

class MNISTCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MNISTCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Linear(4 * 4 * 64, num_classes)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

transform_mnist=transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)


mnist_trainset_reader=create_readers(mnist_trainset,80,64)
mnist_testset_reader=create_readers(mnist_testset,130,64)

models = [MNISTCNN((1, 28, 28), 2) for i in range(10)]

device = "cuda:"+str(0) if torch.cuda.is_available() else 'cpu'

print("device is : ",device)
for model in models:
    model.to(device)
import torch.optim as optim

import itertools

criterion = torch.nn.CrossEntropyLoss()#weight =torch.FloatTensor([1,10]).to(device))
optimizer = optim.Adam(list(itertools.chain.from_iterable([list(model.parameters()) for model in models])), lr=2e-3,)
epoch_number=10

Numbers = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", \
           5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

for epoch_i in range(epoch_number):
    model.train()
    ac_=0
    t_=0
    train_loss=0
    for batch in mnist_trainset_reader:
        # get the inputs; data is a list of [inputs, labels]
        input_token=batch["pixels"].squeeze(dim=0).to(device)
        optimizer.zero_grad()
        for j in range(10):
            t_+=1
            labels=torch.LongTensor([int(k) for k in batch[Numbers[j]]]).to(device)
        # zero the parameter gradients


            # forward + backward + optimize
            outputs = models[j](input_token)

            loss = criterion(outputs, labels)
            train_loss+=loss.item()
            loss.backward()
        optimizer.step()

    print(epoch_i,"train:",train_loss/t_)
    model.eval()
    TP,TN,FP,FN=[0 for i in range(10)],[0 for i in range(10)],[0 for i in range(10)],[0 for i in range(10)]
    for batch in mnist_testset_reader:
        # get the inputs; data is a list of [inputs, labels]
        input_token=batch["pixels"].squeeze(dim=0).to(device)
        for j in range(10):

            labels=torch.LongTensor([int(k) for k in batch[Numbers[j]]]).to(device)
        # zero the parameter gradients
            outputs = models[j](input_token)
            for i_,j_ in zip(outputs.argmax(dim=1),labels):
                if i_==1 and j_==1:
                    TP[j]+=1
                if i_==0 and j_==0:
                    TN[j]+=1
                if i_==0 and j_==1:
                    FN[j]+=1
                if i_==1 and j_==0:
                    FP[j]+=1

            # forward + backward + optimize
    for j in range(10):

        print(j,epoch_i, "val:", 2*TP[j]/(1+2*TP[j]+FP[j]+FN[j]))
