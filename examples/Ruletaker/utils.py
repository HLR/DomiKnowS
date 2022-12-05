from transformers import RobertaTokenizerFast,RobertaModel
import torch
import numpy as np

class RobertaTokenizer:
    def __init__(self,max_length=384):
        self.max_length=max_length
        self.tokenizer= RobertaTokenizerFast.from_pretrained("roberta-base")

    def __call__(self,context, questionlist):
        preprocessed=[context[0] for i in questionlist]
        encoded_input = self.tokenizer(preprocessed,questionlist,padding="max_length",max_length =self.max_length)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        return torch.LongTensor(input_ids),torch.LongTensor(attention_mask)


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

def make_questions(context, questions, labels,proofs,strategies):
    return torch.ones((len(questions[0]), 1)), [context for i in range(len(questions[0]))],questions[0],labels[0],proofs[0],strategies[0]

def label_reader(_, label):
    #print(label)
    return label

class SimpleTokenizer:

    def __init__(self,device):

        import spacy
        self.nlp= spacy.load("en_core_web_sm")
        self.device=device

    def __call__(self,context, questionlist):
        preprocessed = [context[0] for i in questionlist]
        return torch.concat((torch.FloatTensor([self.nlp(i).vector.tolist() for i in preprocessed]),torch.FloatTensor([self.nlp(i).vector.tolist() for i in questionlist])),dim=1)
