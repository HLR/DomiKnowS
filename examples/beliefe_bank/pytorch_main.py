import json
import torch
from transformers import RobertaTokenizerFast, RobertaModel


def read_data(batch_size=128,sample_size=15):
    f = open("data/calibration_facts.json")
    f = json.load(f)

    sample_counter=0
    calibration_data = []
    for i in f.keys():
        facts=[]
        labels=[]
        for j in f[i]:
            facts.append(j)
            labels.append(f[i][j])
            if len(facts)==batch_size:
                calibration_data.append({"name":i,"facts":[facts],"labels":[labels]})
                facts = []
                labels = []
                sample_counter+=1
                if sample_counter>=sample_size:
                    break
        if sample_counter >= sample_size:
            break
        if not len(facts)==0:
            calibration_data.append({"name": i, "facts": [facts], "labels": [labels]})
            sample_counter += 1


    f = open("data/silver_facts.json")
    f = json.load(f)

    silver_data = []
    for i in f.keys():
        facts = []
        labels = []
        for j in f[i]:
            facts.append(j)
            labels.append(f[i][j])
            if len(facts)==batch_size:
                silver_data.append({"name":i,"facts":[facts],"labels":[labels]})
                facts = []
                labels = []

        if not len(facts) == 0:
            silver_data.append({"name": i, "facts": [facts], "labels": [labels]})

    f = open("data/constraints_v2.json")
    f = json.load(f)


    constraints_yes = dict()
    constraints_no = dict()
    for i in f["nodes"]:
        constraints_yes[i["id"]] = set()
        constraints_no[i["id"]] = set()

    print("number of links:",len(f["links"]))

    for i in f["links"]:
        if i["weight"]=="yes_yes":
            if i["direction"]=="forward":
                constraints_yes[i["source"]].add(i["target"])
            else:
                constraints_yes[i["target"]].add(i["source"])
        else:
            if (i["direction"]=="forward" and i["weight"]=="yes_no") or (i["direction"]=="back" and i["weight"]=="no_yes"):
                constraints_no[i["source"]].add(i["target"])
            else:
                constraints_no[i["target"]].add(i["source"])

    print("data sizes:",len(calibration_data),len(silver_data),len(constraints_yes),len(constraints_no))
    return calibration_data,silver_data,constraints_yes,constraints_no

class RobertaClassificationHead(torch.nn.Module):

    def __init__(self,last_layer_size):
        super(RobertaClassificationHead, self).__init__()
        self.dense = torch.nn.Linear(last_layer_size, last_layer_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.out_proj = torch.nn.Linear(last_layer_size, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class BBRobert(torch.nn.Module):

    def __init__(self):
        super(BBRobert, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        for name, param in list(self.bert.named_parameters())[:-32]:
            param.requires_grad = False
        self.last_layer_size = self.bert.config.hidden_size
        self.head=RobertaClassificationHead(self.last_layer_size)

    def forward(self, input_ids,attention_mask):
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)
        return self.head(last_hidden_state[:,0])

class RobertaTokenizer:
    def __init__(self,max_length=64):
        self.max_length=max_length
        self.tokenizer= RobertaTokenizerFast.from_pretrained("roberta-base")

    def __call__(self,name, sentence):
        preprocessed=[i+" "+j.replace("IsA" ,"is a").replace("CapableOf" ,"is capable of").replace("HasPart" ,"has the part").replace("HasA" ,"has").replace("," ," ").replace("MadeOf" ,"is made of ").replace("HasProperty" ,"Has the Property of") for i,j in zip(name, sentence)]
        encoded_input = self.tokenizer(preprocessed ,padding="max_length",max_length =self.max_length)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        return torch.LongTensor(input_ids),torch.LongTensor(attention_mask)

batch_size=128
calibration_data,silver_data,_,_=read_data(batch_size=batch_size,sample_size=4)
train_size=len(calibration_data)*3//4
calibration_data_dev=calibration_data[train_size:]
calibration_data=calibration_data[:train_size]
dataset_train,dataset_val,label_train,label_val=[],[],[],[]
tokenizer=RobertaTokenizer()

for i in calibration_data:
    dataset_train.append(tokenizer([i["name"] for j in range(batch_size)],i["facts"][0]))
    label_train.append(torch.LongTensor([0 if j=="no" else 1 for j in i['labels'][0]]))

for i in silver_data:
    dataset_val.append(tokenizer([i["name"] for j in range(batch_size)],i["facts"][0]))
    label_val.append(torch.LongTensor([0 if j=="no" else 1 for j in i['labels'][0]]))

device="cuda"
model=BBRobert()
model.to(device)
import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4,)
epoch_number=30
for epoch_i in range(epoch_number):
    model.train()
    ac_=0
    t_=0
    for batch,label in zip(dataset_train,label_train):
        # get the inputs; data is a list of [inputs, labels]
        input_token=batch[0].to(device)
        input_mask = batch[1].to(device)
        labels=label.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_token,input_mask)
        ac_+=sum(outputs.argmax(dim=1)==labels)
        t_+=outputs.shape[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(epoch_i,"train:",ac_ / t_ * 100)
    model.eval()
    ac_=0
    t_=0
    TP, TN, FP, FN =0,0,0,0
    if epoch_i==epoch_number-1:
        for batch, label in zip(dataset_val, label_val):
            # get the inputs; data is a list of [inputs, labels]
            input_token = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = label.to(device)
            # zero the parameter gradients

            # forward + backward + optimize

            outputs = model(input_token, input_mask)
            for i_, j_ in zip(outputs.argmax(dim=1), labels):
                if i_ == 1 and j_ == 1:
                    TP += 1
                if i_ == 0 and j_ == 0:
                    TN += 1
                if i_ == 0 and j_ == 1:
                    FN += 1
                if i_ == 1 and j_ == 0:
                    FP += 1

            # forward + backward + optimize
        print("test:", 2 * TP / (1 + 2 * TP + FP + FN))