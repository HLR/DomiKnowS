import pandas as pd
import networkx as nx
from collections import defaultdict

from examples.WIQA.WIQA_aug import s_arg1, s_arg2


def make_reader(file_address="data/WIQA_AUG/train.jsonl",sample_num=1000000000):

    counter=0
    para_quest_dict=defaultdict(list)
    quest_id_to_datum=defaultdict(list)
    reader=[]
    data = pd.read_json(file_address, orient="records", lines=True)

    G=nx.Graph()

    for i, row in data.iterrows():
        datum={}

        if row["question"]["answer_label"].strip()=='more':
            datum["more"]=[1]
            datum["less"]=[0]
            datum["no_effect"]=[0]
        elif row["question"]["answer_label"].strip()=='less':
            datum["more"]=[0]
            datum["less"]=[1]
            datum["no_effect"]=[0]
        else:
            datum["more"]=[0]
            datum["less"]=[0]
            datum["no_effect"]=[1]

        para = " ".join([p.strip() for p in row["question"]["para_steps"] if len(p) > 0])
        paragraph_list=[p.strip() for p in row["question"]["para_steps"] if len(p) > 0]

        datum["paragraph"]=para
        datum["paragraph_list"]=paragraph_list

        question = row["question"]["stem"].strip()
        datum["question"]=question
        datum["ques_id"]=row["metadata"]["ques_id"]

        quest_id_to_datum[row["metadata"]["ques_id"]]=datum
        para_quest_dict[para].append(question)

        G.add_node(row["metadata"]["ques_id"])
        if "_symmetric" in row["metadata"]["ques_id"]:
            G.add_edge(row["metadata"]["ques_id"].split("_symmetric")[0],row["metadata"]["ques_id"])
        elif "@" in row["metadata"]["ques_id"]:
            G.add_edge(row["metadata"]["ques_id"].split("@")[0],row["metadata"]["ques_id"])
            G.add_edge(row["metadata"]["ques_id"].split("@")[1].split("_transit")[0],row["metadata"]["ques_id"])

        counter+=1
        if counter>sample_num:
            break

    print()

    import seaborn as sns
    from matplotlib import pyplot as plt

    #sns.displot(x=[len(i) for i in nx.connected_components(G)],binwidth=1)
    #plt.show()
    #print([i  for i in nx.connected_components(G) if len(i)>15][0])

    comp_size=[len(i) for i in nx.connected_components(G)]
    comps=[i for i in nx.connected_components(G)]

    zipped_lists = zip(comp_size, comps)
    sorted_pairs = sorted(zipped_lists,reverse=True)
    tuples = zip(*sorted_pairs)

    comp_size, comps = [ list(tuple) for tuple in  tuples]

    #print(comp_size[:10])
    #print(comps[:10])
    #print(comp_size[:2],comp_size[-20:-1])
    #print(sorted([len(j) for i,j in para_quest_dict.items()])[-20:-1])
    #print(comps[:2],comps[-2:-1])

    print(len(list(nx.connected_components(G))))
    batch_size=30
    for i in range(len(comp_size)):
        if comp_size[i]>=batch_size or comp_size[i]==0:
            continue
        for j in range(i+1,len(comp_size)):
            if comp_size[j]==0 or comp_size[j]+comp_size[i]>batch_size+2 or \
                    not quest_id_to_datum[list(comps[j])[0]]["paragraph"]==quest_id_to_datum[list(comps[i])[0]]["paragraph"]:
                continue
            G.add_edge(list(comps[j])[0],list(comps[i])[0])
            comp_size[i]+=comp_size[j]
            comp_size[j]=0
            if comp_size[i]>=batch_size:
                break

    sns.displot(x=[i for i in comp_size if not i==0],binwidth=1)
    #plt.show()
    print(len(list(nx.connected_components(G))))
    final_list=[]
    for i in list(nx.connected_components(G)):
        if len(i)<=batch_size+2:
            final_list.append(list(i))
            continue
        cur_list=list(i)
        for j in range(0,len(i),batch_size):
            final_list.append(cur_list[j:j+batch_size])

    #print(len(final_list)*32)
    #print(len(reader))

    for i in final_list:
        more_list=[]
        less_list=[]
        no_effect_list=[]
        question_list=[]
        para=""
        for j in i:
            datum=quest_id_to_datum[j]
            para=datum["paragraph"]
            more_list.append(str(datum["more"][0]))
            less_list.append(str(datum["less"][0]))
            no_effect_list.append(str(datum["no_effect"][0]))
            question_list.append(datum["question"])
        reader.append({"paragraph":para,"more_list":"@@".join(more_list),"less_list":"@@".join(less_list),"no_effect_list":"@@".join(no_effect_list),
                       "question_list":"@@".join(question_list),"quest_ids":"@@".join(i)})
    return reader

from transformers import RobertaTokenizerFast
import torch

class RobertaTokenizer:
    def __init__(self,answer_option,max_length=256):
        self.answer_option=answer_option
        self.max_length=max_length
        self.tokenizer= RobertaTokenizerFast.from_pretrained("roberta-base")

    def __call__(self,_, question_paragraph, text):
        encoded_input = self.tokenizer(question_paragraph,[i+" </s> "+self.answer_option for i in text] ,padding="max_length",max_length =self.max_length)

        input_ids = encoded_input["input_ids"]
        #print(input_ids[0])
        #print(self.tokenizer.convert_ids_to_tokens(input_ids[0]))

        attention_mask = encoded_input["attention_mask"]
        return torch.LongTensor(input_ids),torch.LongTensor(attention_mask)


from transformers import RobertaModel
import torch
from torch import nn

class DariusWIQA_Robert(nn.Module):

  def __init__(self):
    super(DariusWIQA_Robert, self).__init__()
    self.bert = RobertaModel.from_pretrained('roberta-base')
    self.last_layer_size = self.bert.config.hidden_size

  def forward(self, input_ids,attention_mask,use_soft_max=False):
    last_hidden_state, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)
    return last_hidden_state[:,0]

from itertools import product

def make_pair(question_ids):
    print("im here 2")
    n=len(question_ids)
    p1=[]
    p2=[]
    for arg1, arg2 in product(range(n), repeat=2):
        if arg1 >= arg2:
            continue
        if question_ids[arg1] in question_ids[arg2] and "_symmetric" in question_ids[arg2]:
            p1.append([0 if i==arg1 else 1 for i in range(n)])
            p2.append([0 if i==arg2 else 1 for i in range(n)])

    return torch.LongTensor(p1),torch.LongTensor(p2),torch.ones((n,1))

def make_pair_with_labels(question_ids):

    n=len(question_ids)
    p1=[]
    p2=[]
    label_output=[]
    for arg1, arg2 in product(range(n), repeat=2):
        if arg1 >= arg2:
            continue
        p1.append([0 if i==arg1 else 1 for i in range(n)])
        p2.append([0 if i==arg2 else 1 for i in range(n)])
        if question_ids[arg1] in question_ids[arg2] and "_symmetric" in question_ids[arg2]:
            label_output.append([1])
        else:
            label_output.append([0])

    return torch.LongTensor(p1),torch.LongTensor(p2),torch.LongTensor(label_output)

def make_triple(question_ids):
    print("im here 3")
    n=len(question_ids)
    p1=[]
    p2=[]
    p3=[]
    for arg1, arg2, arg3 in product(range(n), repeat=3):
        if arg1 >= arg2 or arg2 >= arg3:
            continue
        if question_ids[arg1] in question_ids[arg3] and \
           question_ids[arg2] in question_ids[arg3] and \
                "_transit" in question_ids[arg3]:
            p1.append([0 if i==arg1 else 1 for i in range(n)])
            p2.append([0 if i==arg2 else 1 for i in range(n)])
            p3.append([0 if i==arg3 else 1 for i in range(n)])

    return torch.LongTensor(p1),torch.LongTensor(p2),torch.LongTensor(p3),torch.ones((n,1))

def make_triple_with_labels(question_ids):
    n=len(question_ids)
    p1=[]
    p2=[]
    p3=[]
    label_output=[]
    for arg1, arg2, arg3 in product(range(n), repeat=3):
        if arg1 >= arg2 or arg2 >= arg3:
            continue
        p1.append([0 if i==arg1 else 1 for i in range(n)])
        p2.append([0 if i==arg2 else 1 for i in range(n)])
        p3.append([0 if i==arg3 else 1 for i in range(n)])

        if question_ids[arg1] in question_ids[arg3] and \
           question_ids[arg2] in question_ids[arg3] and \
                "_transit" in question_ids[arg3]:
            label_output.append([1])
        else:
            label_output.append([0])

    return torch.LongTensor(p1),torch.LongTensor(p2),torch.LongTensor(p3),torch.LongTensor(label_output)

def guess_pair(quest_id, data, arg1, arg2):
    quest1, quest2 = arg1.getAttribute('quest_id'), arg2.getAttribute('quest_id')
    if quest1 in quest2 and "_symmetric" in quest2: #directed?
        return True
    else:
        return False

def guess_pair_datanode_2(*_, data, datanode):
    quest1_node = datanode.relationLinks[s_arg1.name][0]
    quest2_node = datanode.relationLinks[s_arg2.name][0]
    quest1=quest1_node.getAttribute('quest_id')
    quest2=quest2_node.getAttribute('quest_id')
    if quest1 in quest2 and "_symmetric" in quest2: #directed?
        return True
    else:
        return False
