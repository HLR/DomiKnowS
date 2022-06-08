import torch
import torchvision
from torchvision import datasets


def create_readers(dataset,sample_size,batch_size):
    Numbers = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four",\
               5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}
    reader=[]
    instance = {}
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


mnist_trainset_reader=create_readers(mnist_trainset,99999999,16)
mnist_testset_reader=create_readers(mnist_testset,99999999,16)

model = MNISTCNN((1, 28, 28), 2)

device = "cuda:"+str(0) if torch.cuda.is_available() else 'cpu'

print("device is : ",device)

model.to(device)
import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()#weight =torch.FloatTensor([1,10]).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4,)
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
        for j in range(1,2):
            t_+=1
            labels=torch.LongTensor([int(k) for k in batch[Numbers[j]]]).to(device)
        # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_token)

            loss = criterion(outputs, labels)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
        #print(epoch_i, "train:", loss)
    print(epoch_i,"train:",train_loss/t_)
    model.eval()
    TP,TN,FP,FN=0,0,0,0
    for batch in mnist_testset_reader:
        # get the inputs; data is a list of [inputs, labels]
        input_token=batch["pixels"].squeeze(dim=0).to(device)
        for j in range(1,2):

            labels=torch.LongTensor([int(k) for k in batch[Numbers[j]]]).to(device)
        # zero the parameter gradients
            outputs = model(input_token)
            for i_,j_ in zip(outputs.argmax(dim=1),labels):
                if i_==1 and j_==1:
                    TP+=1
                if i_==0 and j_==0:
                    TN+=1
                if i_==0 and j_==1:
                    FN+=1
                if i_==1 and j_==0:
                    FP+=1

            # forward + backward + optimize

    print(epoch_i, "val:", 2*TP/(1+2*TP+FP+FN))
