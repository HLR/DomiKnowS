from datasets import load_dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy
from transformers import RobertaTokenizer, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class NewsRepModule(nn.Module):
    def __init__(self, roberta):
        super().__init__()
        self.roberta = roberta
        self.drop_layer = nn.Dropout(p=0.6)

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.pooler_output
        logits = self.drop_layer(logits)
        return logits

class Level2Calssification(nn.Module):
    def __init__(self,):
        super().__init__()
        self.classification = nn.Linear(768, 8)

    def forward(self, logits):
        _out = self.classification(logits)
        return _out
    
class Level1Calssification(nn.Module):
    def __init__(self,):
        super().__init__()
        self.classification = nn.Linear(768, 50)
        self.relu = nn.LeakyReLU()
        self.classification2 = nn.Linear(50, 16)
        

    def forward(self, logits):
        _out = self.classification(logits)
        _out = self.relu(_out)
        _out = self.classification2(_out)
        return _out


def trim_dataset(dataset, key):
    labels = {}
    for item in dataset[key]:
        if item['label'] not in labels:
            labels[item['label']] = 0
        labels[item['label']] += 1
    print(labels)
    before_len = len(dataset[key])
    print(before_len)
    dataset[key] = dataset[key].filter(lambda example: example['label'] and example['label'] != "None" and example['text'] and len(example["text"]) >= 10)
    after_len = len(dataset[key])
    print(after_len)
    print("the number of removed items : ", before_len - after_len)
    labels = {}
    for item in dataset[key]:
        if item['label'] not in labels:
            labels[item['label']] = 0
        labels[item['label']] += 1
    print(labels)
    return dataset


def collate_label_set(data):
    level1 = [
        "comp.os", "comp.sys", "comp.windows", "comp.graphics", "rec.motorcycles", "rec.sport", "rec.autos", "talk.religion",
        "sci.electronics", "sci.med", "sci.space", "misc.forsale", "talk.politics", "sci.crypt", "alt.atheism", "soc.religion"
    ]
    level2_pass = {
        "comp.windows", "comp.os", "talk.religion", "soc.religion"
    }
    level2 = [
        "misc", "guns", "ibm", "mac", "baseball", "hockey", "mideast", "None"
    ]
    for item in data:
        if item['text'] is None:
            item['text'] = " "
        try:
            label = item['label']
            if not label:
                label = "None"
                item['level1'] = level1.index(label)
                item['level2'] = level2.index("None")
            else:
                label = label.split(".")
                label_two = ".".join(label[:2])
                if label_two not in level1:
                    print(label[0])
                else:
                    item['level1'] = level1.index(label_two)
                if len(label) > 2:
                    if label_two in level2_pass:
                        item['level2'] = level2.index("None")
                    else:
                        if label[2] not in level2:
                            print(label[2])
                            item['level2'] = level2.index("None")
                        else:
                            item['level2'] = level2.index(label[2])
                else:
                    item['level2'] = level2.index("None")
        except:
            print(label, item['text'])
            raise
            
    final_data = {"id": None, "text": None, "level1": None, "level2": None}
    final_data['id'] = torch.tensor([item['id'] for item in data])
    final_data['text'] = [item['text'] for item in data]
    final_data['level1'] = torch.tensor([item['level1'] for item in data])
    final_data['level2'] = torch.tensor([item['level2'] for item in data])
    try:
        x = tokenizer.batch_encode_plus(final_data['text'], return_tensors="pt", padding="longest", max_length=512, truncation=True)
    except:
        print(final_data['text'])
        raise
    for key in x:
        final_data[key] = x[key]
        
    return final_data


dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
dataset = trim_dataset(dataset, "train")
dataset = trim_dataset(dataset, "test")

# First make the kfold object
folds = StratifiedKFold(n_splits=5)

# Now make our splits based off of the labels. 
# We can use `np.zeros()` here since it only works off of indices, we really care about the labels
splits = folds.split(np.zeros(dataset["train"].num_rows), dataset["train"]["label"])

# dataset2 = copy.deepcopy(dataset1)
new_datasets = [copy.deepcopy(dataset) for i in range(5)]
i = 0
for train_ids, val_ids in splits:
    new_datasets[i]["validation"] = new_datasets[i]["train"].select(val_ids)
    new_datasets[i]["train"] = new_datasets[i]["train"].select(train_ids)
    i += 1

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from transformers import AutoTokenizer, RobertaModel, BertModel
import json

level1 = [
    "comp.os", "comp.sys", "comp.windows", "comp.graphics", "rec.motorcycles", "rec.sport", "rec.autos", "talk.religion",
    "sci.electronics", "sci.med", "sci.space", "misc.forsale", "talk.politics", "sci.crypt", "alt.atheism", "soc.religion"
]
level2 = [
    "misc", "guns", "ibm", "mac", "baseball", "hockey", "mideast", "None"
]

best_performances = []
for split in range(5):
    loader = DataLoader(new_datasets[split]['train'], batch_size=36, collate_fn=collate_label_set)
    val_loader = DataLoader(new_datasets[split]['validation'], batch_size=36, collate_fn=collate_label_set)
    device = "cuda:6"
    rep_model = NewsRepModule(BertModel.from_pretrained("bert-base-uncased"))
    rep_model = rep_model.to(device)

    level1_classification = Level1Calssification().to(device)
    level2_classification = Level2Calssification().to(device)
    
    params = list(rep_model.parameters()) + list(level1_classification.parameters())
    params = list(params) + list(level2_classification.parameters())
    optim = torch.optim.AdamW(params, lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_performance = 0
    best_performance_details = None
    
    for epoch in range(5):
        rep_model.train()
        level1_classification.train()
        level2_classification.train()
        total_loss = 0
        for train_batch in tqdm(loader):
            try:
                rep = rep_model(train_batch['input_ids'].to(device),
                                               train_batch['attention_mask'].to(device))
                level1 = level1_classification(rep)
                level2 = level2_classification(rep)

            except:
                print(train_batch['text'])
                raise
            try:
                loss = 0 
                loss += criterion(level1, train_batch['level1'].long().to(device))
                loss += criterion(level2, train_batch['level2'].long().to(device))
                total_loss += loss.item()
                loss.backward()
            except:
                print(level1, train_batch['level1'])
                print(level2, train_batch['level2'])
                raise
            optim.step()
            optim.zero_grad()
        print(f"The loss of epoch {epoch} is {total_loss} \n")

        rep_model.eval()
        level1_classification.eval()
        level2_classification.eval()
        all_preds = {'level1': [], 'level2': []}
        all_gt = {'level1': [], 'level2': []}
        
        for train_batch in tqdm(val_loader):
            rep = rep_model(train_batch['input_ids'].to(device),
                                                   train_batch['attention_mask'].to(device))
            level1 = level1_classification(rep)
            level2 = level2_classification(rep)
            level1 = level1.argmax(-1)
            level2 = level2.argmax(-1)
            all_gt['level1'].extend(train_batch['level1'].cpu().detach().tolist())
            all_gt['level2'].extend(train_batch['level2'].cpu().detach().tolist())
            all_preds['level1'].extend(level1.cpu().detach().tolist())
            all_preds['level2'].extend(level2.cpu().detach().tolist())

            
        l1_pr, l1_rec, l1_f1, l1_sup = precision_recall_fscore_support(np.array(all_gt['level1']), np.array(all_preds['level1']))
        l2_pr, l2_rec, l2_f1, l2_sup = precision_recall_fscore_support(np.array(all_gt['level2']), np.array(all_preds['level2']))
        level1_aggregate = (torch.from_numpy(l1_sup) * torch.from_numpy(l1_f1)).sum().item() / torch.from_numpy(l1_sup).sum().item()
        level2_aggregate = (torch.from_numpy(l2_sup) * torch.from_numpy(l2_f1)).sum().item() / torch.from_numpy(l2_sup).sum().item()
        score = (level1_aggregate + level2_aggregate) / 2
        print(score, level1_aggregate, level2_aggregate)
        if score > best_performance:
            best_performance = score
            best_performance_details = [torch.from_numpy(l1_pr), torch.from_numpy(l1_rec), 
                                        torch.from_numpy(l1_f1), torch.from_numpy(l1_sup),
                                        torch.from_numpy(l2_pr), torch.from_numpy(l2_rec),
                                        torch.from_numpy(l2_f1), torch.from_numpy(l2_sup),
                                       level1_aggregate, level2_aggregate]
    if len(best_performances):
        for _ind in range(len(best_performances)):
            best_performances[_ind] = best_performances[_ind] + best_performance_details[_ind]
    else:
        best_performances = best_performance_details

for _ind in range(len(best_performances)):
    best_performances[_ind] = best_performances[_ind] / 5

torch.save(best_performances, "average_performance.pt")