import torch
import pandas as pd
from collections import Counter

class LabelReader():
    def __init__(self,device,tag_list):
        self.device=device
        self.tag_list=tag_list

    def __call__(self,_, label):

        label_list=[]
        for i in label:
            try:
                label_list.append(self.tag_list.index("tag__" + i, ))
            except:
                label_list.append(len(self.tag_list)-1)
        return torch.LongTensor(label_list).to(self.device)

def make_words( tokenized_text, tokenized_pos):
    return torch.ones((len(tokenized_text[0].split(" ")), 1)), tokenized_text[0].split(" "),tokenized_pos[0].split(" ")

def create_reader_and_vocuabulary(samplenum,top_pos):
    vocabulary = {"<UNK>": 0, "<begin>": 1, "<end>": 2}
    reader = []

    try:
        data = pd.read_csv("brown.csv")
    except:
        data = pd.read_csv("examples/POS_tagging/brown.csv")

    vocab_counter = 3
    sample_counter = 1

    tag_set = set()
    tag_counter = Counter()

    for num, i in data.iterrows():
        if not len(i["tokenized_pos"].split(" ")) ==len(i["tokenized_pos"].split(" ")):
            continue
        reader.append({"sentecne": [i["tokenized_text"]], "tags": [i["tokenized_pos"]]})
        for j in i["tokenized_pos"].split(" "):
            tag_counter[j] += 1
        tag_set.update(i["tokenized_pos"].split(" "))
        for j in i["tokenized_text"].split(" "):
            if not j in vocabulary:
                vocabulary[j] = vocab_counter
                vocab_counter += 1
        sample_counter += 1
        if sample_counter > samplenum:
            break

    print("10 most frequent tags: ", sorted(list(tag_counter.items()), key=lambda x: x[1], reverse=True)[:10])
    chosen_tags = [i for i, j in sorted(list(tag_counter.items()), key=lambda x: x[1], reverse=True)[:top_pos]]
    print("Size of the vocabulary", len(vocabulary))

    tag_list = ["tag__" + i for i in list(chosen_tags)]
    if len(tag_list) < len(tag_set):
        tag_list.append("tag__other")

    return vocabulary,tag_list,reader