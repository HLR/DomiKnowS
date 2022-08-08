from torch import nn

class MNISTCNN(nn.Module):
    def __init__(self, input_size, num_classes, number):
        super(MNISTCNN, self).__init__()
        self.number=number

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Linear(4 * 4 * 64, num_classes)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

class MNISTLinear(nn.Module):
    def __init__(self,):
        super(MNISTLinear, self).__init__()


        self.fc1 = nn.Linear(28*28, 28*28)
        self.fc2 = nn.Linear(28*28, 2)
        self.tanh= nn.Tanh()

    def forward(self, x):

        x = x.reshape(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        return self.fc2(x)