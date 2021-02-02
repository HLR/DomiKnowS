# CIFAR10 Tutorial
This tutorial is created to show the ability of the framework to do inference on problems. 
## Problem
We address image classification problem on CIFAR10 data using our framework. CIFAR10 is a dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
Class labels are `airplane`, `automobile`, `bird`,`cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.
 
[Here](https://www.cs.toronto.edu/~kriz/cifar.html) is more information about CIFAR10 dataset.

## Framework Modeling
### Define the Graph
Each program in the Domiknows framework starts with a concept graph which defines the concepts interacting inside the problem world. 
Here we have an `image` as a concept, each class label `Is_A` `image` concept. Basically, we define the parent concept `image`,
 and all of the class labels as the children of `image`. For example, `ariplane` is an `image`.
 
 The graph declration for CIFAR10 problem is:
 ```python
Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph:
    image = Concept(name='image')
    airplane = image(name='airplane')
    dog = image(name='dog')
    truck = image(name='truck')
    automobile = image(name='automobile')
    bird = image(name='bird')
    cat = image(name='cat')
    deer = image(name='deer')
    frog = image(name='frog')
    horse = image(name='horse')
    ship = image(name='ship')
    nandL(truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship)
```

Notice the last line in the graph declaration; we defined a constraint that we want to apply on the inference. `nandL` constraint enforces that only one
concept among all its arguments should be selected. In particular, it means each image should be only one object (`airplane`, `dog`, `truck`, etc.).
 Please refer to [here](/docs/KNOWLEDGE.md) for more information about how you can define rules and constraints.

In [User Pipeline](/docs/PIPELINE.md#1-knowledge-declaration) and [Knowledge Declaration](/docs/KNOWLEDGE.md) in the documentation you could find more specification.



### Model Declaration
The next step toward solving a problem in our framework is to define a model flow or declaration for each example of the data.

Before jumping to the model declaration, we could first define the network. For example, for image classification task,
we defined a simple convolutional neural network as follows:

```python
import torch.nn.functional as F
import torch
import torch.nn as nn
class ImageNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2):
        super(ImageNetwork, self).__init__()
        self.num_outputs = n_outputs
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        return x
```
In above python code, we have a simple CNN consisting of two convolutional layers follwoing reLu activation function.

Now we start to connect the reader output data with our formatted domain knowledge defined in the graph.

```python
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.program import LearningBasedProgram
from torch import nn
from graph import graph
from regr.program.metric import MacroAverageTracker, PRF1Tracker

def model_declaration():
    graph.detach()
    image = graph['image']
    airplane = graph['airplane']
    dog = graph['dog']
    truck = graph['truck']
    automobile = graph['automobile']
    bird = graph['bird']
    cat = graph['cat']
    deer = graph['deer']
    frog = graph['frog']
    horse = graph['horse']
    ship = graph['ship']

    image['pixels'] = ReaderSensor(keyword='pixels')
    image[airplane] = ReaderSensor(keyword='airplane',label=True)
    image[dog] = ReaderSensor(keyword='dog',label=True)
    image[truck] = ReaderSensor(keyword='truck',label=True)
    image[automobile] = ReaderSensor(keyword='automobile',label=True)
    image[bird] = ReaderSensor(keyword='bird',label=True)
    image[cat] = ReaderSensor(keyword='cat',label=True)
    image[deer] = ReaderSensor(keyword='deer',label=True)
    image[frog] = ReaderSensor(keyword='frog',label=True)
    image[horse] = ReaderSensor(keyword='horse',label=True)
    image[ship] = ReaderSensor(keyword='ship',label=True)

    image['emb'] = ModuleLearner('pixels', module=ImageNetwork())
    image[airplane] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[dog] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[truck] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[automobile] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[bird] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[cat] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[deer] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[frog] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[horse] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[ship] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))

    program = LearningBasedProgram(graph, PoiModel, loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
    return program
```
First we link `ReaderSensor` to the concepts and properties of the graph. Basically, we are connecting the readers to
the the graph we declared for this problem.
 
 
Next, we define learners and connect them to the concepts. Specifically, the pixel of each image passes through the convolutional
neural network we defined as `ImageNetwork`; this CNN networked shared among all of the concepts. Each class labeld concept has its own
linear layer. The dimension of these linear layers are desinged such that they are consistent with the lasy CNN layer in our `ImageNetwork`.

By defining `LearningBasedProgram`, we have an executable instance from the declaration of the graph attached to the sensors and learners.
Basically, we make an executable version of our declared graph that is able to trace the dependencies of the sensors and fill the data from
 the reader to run examples on the declared model.

#### Load CIFAR10 dataset
In order to loaded the CIFAR10 dataset be consistent with this framework, we inherit from `datasets.CIFAR10`
class and change the `__getitem__` function:
 
```python
class CIFAR10_DomiKnows(datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10_1, self).__init__(root, transform=transform,
                                      target_transform=target_transform, download=download)

        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = img.unsqueeze(0)
        target_dict = {0:'airplane',1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7:'horse',8: 'ship', 9: 'truck'}
        dict = {}
        dict['pixels'] = img
        for i in range(10):
            dict[target_dict[i]] = [0]
        dict[target_dict[target]] = [1]
        return dict
```
Notice that in `__getitem__` function, we return a dictionary. The dictionary has a key as `pixel` for the image pixels,
and ten keys (class labels) with the value to be zero or one. It is really important the value for each class label be a
`list` not a scalar value; e.g., [0] or [1] not 0, or 1. 

After constructing the class, we could load CIFAR10 data:


```python
def load_cifar10(train=True, root='./data/', size=32):
    CIFAR100_TRAIN_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_TRAIN_STD = (0.2675, 0.2565, 0.2761)

    if train:
        transform = transforms.Compose(
            [transforms.Resize(size),
             transforms.RandomCrop(size, padding=round(size/8)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    else:
        transform = transforms.Compose(
            [transforms.Resize(size),
             transforms.ToTensor(),
             transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

    return CIFAR10_DomiKnows(root=root, train=train, transform=transform,download=True)
```

In the above code, we apply some transformers to the images, and then load the data based on our new class loader.

### Model Execution
To run the model, you have to call the reader and the model_declaration.

```python
program = model_declaration()
```

Then, you will load the data:
```python
trainset = load_cifar10(train=True)
```
    
And then run the model for training:
```python
program.train(trainset, train_epoch_num=10, Optim=lambda param: torch.optim.SGD(param, lr=.001))
```

Then, we populate datanode and call inference on each image.

```python
label_list = ['airplane', 'automobile','bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
testset = load_cifar10(train=False)
for datanode in program.populate(dataset=testset):
    print('----------before ILP---------')
    for label in label_list:
        print(label, datanode.getAttribute(eval(label)).softmax(-1))

    datanode.inferILPConstrains('dog', 'truck', 'airplane',
                                'automobile', 'bird', 'cat',
                                'deer', 'frog', 'horse', 'ship',fun=None)
    print('----------after ILP---------')
    prediction = ' '
    for label in label_list:
        predt_label = datanode.getAttribute(eval(label), 'ILP').item()
        if predt_label == 1.0:
            prediction = label
        print('inference ',label, predt_label )
```
After `datanode.inferILPConstrains`, we enforces the constraint we defined in our graph, which was
`nandL(truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship)`. With this inference constraint, we will have
only one class label as prediction. 



