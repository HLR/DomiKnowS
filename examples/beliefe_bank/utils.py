from transformers import RobertaTokenizerFast,RobertaModel
import torch
import numpy as np

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

class Generator(torch.nn.Module):

    def __init__(self,tokenizer,model):
        super(Generator, self).__init__()
        self.tokenizer=tokenizer
        self.model=model

    def forward(self,name,sentence):


        input_string = "$answer$ ; $mcoptions$  = (A) no (B) yes ; $question$ = "+name+" "+ sentence.replace(","," ").replace("IsA","is a").replace("CapableOf","is capable of")
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt")

        output = self.model.generate(input_ids, max_length=200,output_scores =True,return_dict_in_generate=True)
        return torch.Tensor((output.scores[6][0][150],output.scores[6][0][4273]))

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

def make_facts(name, facts, labels):
    return torch.ones((len(facts[0]), 1)), [name for i in range(len(facts[0]))],facts[0],labels[0]

def label_reader(_, label):
    return torch.LongTensor([{"no":0,"yes":1}.get(i) for i in label])

class SimpleTokenizer:

    def __init__(self,device):

        import spacy
        self.nlp= spacy.load("en_core_web_sm")
        self.device=device

    def __call__(self,name, sentence):
        preprocessed=[i+" "+j.replace("IsA" ,"is a").replace("CapableOf" ,"is capable of").replace("HasPart" ,"has the part").replace("HasA" ,"has").replace("," ," ").replace("MadeOf" ,"is made of ").replace("HasProperty" ,"Has the Property of") for i,j in zip(name, sentence)]
        return torch.FloatTensor([self.nlp(i).vector.tolist() for i in preprocessed]).to(self.device)
