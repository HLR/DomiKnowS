import pickle,torch
from torchvision import transforms
import torch.optim as optim
import itertools, argparse
from torchvision.models import resnet18,resnet50,resnet101,resnet152
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

parser = argparse.ArgumentParser()

parser.add_argument('--namesave', dest='namesave', default="modelname", help='model name to save', type=str)
parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)

parser.add_argument('--epochs', dest='epochs', default=100, help='number of training epoch', type=int)
parser.add_argument('--resnet', dest='resnet', default=18, help='value of learning rate', type=int)

parser.add_argument('--graph_type', dest='graph_type', default="exactL_nandL", help='type of constraints to be defined', type=str)
args = parser.parse_args()


def create_readers(train_num=50000,test_num=10000//4,batch_size=64):
    structure={}
    def unpickle(file_name):
        with open(file_name , 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    train_data = unpickle("cifar-100-python/train")
    test_data = unpickle("cifar-100-python/test")
    meta = unpickle("cifar-100-python/meta")

    train_reader,test_reader=[],[]
    batch_train_reader, batch_test_reader = [], []
    coarse_names=meta[b'coarse_label_names']
    fine_names = meta[b'fine_label_names']
    for i in range(train_num):
        instance={}
        instance['pixels']=preprocess(train_data[b'data'][i].reshape((32,32,3)))
        instance['corase_label'] = train_data[b'coarse_labels'][i]
        instance['fine_label'] = train_data[b'fine_labels'][i]
        instance['corase_label_name'] = coarse_names[(train_data[b'coarse_labels'][i])].decode('ascii')
        instance['fine_label_name'] = fine_names[(train_data[b'fine_labels'][i])].decode('ascii')
        if instance['corase_label_name'] in structure:
            structure[instance['corase_label_name']].add(instance['fine_label_name'])
        else:
            structure[instance['corase_label_name']]=set()
            structure[instance['corase_label_name']].add(instance['fine_label_name'])
        train_reader.append(instance)

    for i in range(0,train_num,batch_size):
        pixels_group,coarse_names_group,fine_names_group=[],[],[]
        for i_ in range(i,min(i+batch_size,train_num)):
            pixels_group.append(train_reader[i_]['pixels'])
            coarse_names_group.append(train_reader[i_]['corase_label'])
            fine_names_group.append(train_reader[i_]['fine_label'])
        intance={}
        intance['pixels']=torch.stack(pixels_group, dim=0).unsqueeze(dim=0)
        intance['corase_label']="@@".join([str(j) for j in coarse_names_group])
        intance['fine_label']="@@".join([str(j) for j in fine_names_group])
        batch_train_reader.append(intance)

    for i in range(test_num):
        instance={}
        instance['pixels']=preprocess(test_data[b'data'][i].reshape((32,32,3)))
        instance['corase_label'] = test_data[b'coarse_labels'][i]
        instance['fine_label'] = test_data[b'fine_labels'][i]
        instance['corase_label_name'] = coarse_names[(test_data[b'coarse_labels'][i])].decode('ascii')
        instance['fine_label_name'] = fine_names[(test_data[b'fine_labels'][i])].decode('ascii')
        test_reader.append(instance)

    for i in range(0,test_num,batch_size):
        pixels_group,coarse_names_group,fine_names_group=[],[],[]
        for i_ in range(i,min(i+batch_size,test_num)):
            pixels_group.append(test_reader[i_]['pixels'])
            coarse_names_group.append(test_reader[i_]['corase_label'])
            fine_names_group.append(test_reader[i_]['fine_label'])
        intance={}
        intance['pixels']=torch.stack(pixels_group, dim=0).unsqueeze(dim=0)
        intance['corase_label']="@@".join([str(j) for j in coarse_names_group])
        intance['fine_label']="@@".join([str(j) for j in fine_names_group])
        batch_test_reader.append(intance)

    return batch_train_reader,batch_test_reader

class CIFAR100Model(torch.nn.Module):

    def __init__(self):
        super(CIFAR100Model, self).__init__()
        #resnet18, resnet50, resnet101, resnet152
        if args.resnet==18:
            self.res_p = resnet18(pretrained=True)
            self.res_c = resnet18(pretrained=True)
        elif args.resnet==50:
            self.res_p = resnet50(pretrained=True)
            self.res_c = resnet50(pretrained=True)
        elif args.resnet==101:
            self.res_p = resnet101(pretrained=True)
            self.res_c = resnet101(pretrained=True)
        elif args.resnet==152:
            self.res_p = resnet152(pretrained=True)
            self.res_c = resnet152(pretrained=True)

        self.l1=torch.nn.Linear(1000, 20)
        self.l2=torch.nn.Linear(1000, 100)
    def forward(self,input,mode="parent"):
        if mode=="parent":
            return self.l1(self.res_p(input))
        return self.l2(self.res_c(input))

model = CIFAR100Model()
device = "cuda:"+str(args.cuda_number) if torch.cuda.is_available() else 'cpu'
print("device is : ",device)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4,)
epoch_number=args.epochs
train_reader,test_reader=create_readers()#train_num=2500,test_num=2500

for epoch_i in range(epoch_number):
    model.train()
    ac_c,ac_p=0,0
    t_=0
    for batch in train_reader:

        input_token=batch["pixels"].squeeze(dim=0).to(device)
        corase_label=torch.LongTensor([int(i) for i in batch["corase_label"].split("@@")]).to(device)
        fine_label=torch.LongTensor([int(i) for i in batch["fine_label"].split("@@")]).to(device)

        optimizer.zero_grad()
        outputs = model(input_token,mode="parent")
        ac_p += torch.sum(outputs.argmax(dim=1) == corase_label).item()
        loss = criterion(outputs, corase_label)
        #loss.backward()
        #optimizer.step()

        optimizer.zero_grad()
        outputs = model(input_token, mode="child")
        ac_c+=torch.sum(outputs.argmax(dim=1)==fine_label).item()
        loss+= criterion(outputs, fine_label)
        loss.backward()
        optimizer.step()

        t_+=outputs.shape[0]

    print(epoch_i,"train ac:",ac_p/t_*100,ac_c/t_*100)
    model.eval()
    ac_c, ac_p = 0, 0
    t_ = 0
    for batch in test_reader:
        input_token = batch["pixels"].squeeze(dim=0).to(device)
        corase_label = torch.LongTensor([int(i) for i in batch["corase_label"].split("@@")]).to(device)
        fine_label = torch.LongTensor([int(i) for i in batch["fine_label"].split("@@")]).to(device)

        outputs = model(input_token, mode="parent")
        ac_p += torch.sum(outputs.argmax(dim=1) == corase_label).item()

        outputs = model(input_token, mode="child")
        ac_c += torch.sum(outputs.argmax(dim=1) == fine_label).item()

        t_ += outputs.shape[0]
    if epoch_i%10==9:
        torch.save(model.state_dict(), args.namesave+str(epoch_i))
    print(epoch_i, "test ac:", ac_p / t_ * 100, ac_c / t_ * 100)
