from transformers import RobertaTokenizerFast
import torch
from itertools import product
import numpy as np

class RobertaTokenizer:
    def __init__(self,max_length=256):
        self.max_length=max_length
        self.tokenizer= RobertaTokenizerFast.from_pretrained("roberta-base")

    def __call__(self,_, question_paragraph, text):
        encoded_input = self.tokenizer(question_paragraph,text ,padding="max_length",max_length =self.max_length)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        return torch.LongTensor(input_ids),torch.LongTensor(attention_mask)

def make_pair(question_ids):
    n=len(question_ids)
    p1=[]
    p2=[]
    for arg1, arg2 in product(range(n), repeat=2):
        if arg1 == arg2:
            continue
        if question_ids[arg1] in question_ids[arg2] and "_symmetric" in question_ids[arg2]:
            p1.append([1 if i==arg1 else 0 for i in range(n)])
            p2.append([1 if i==arg2 else 0 for i in range(n)])
    return torch.LongTensor(p1),torch.LongTensor(p2)

def make_pair_with_labels(question_ids):
    n=len(question_ids)
    p1=[]
    p2=[]
    label_output=[]
    for arg1, arg2 in product(range(n), repeat=2):
        p1.append([1 if i==arg1 else 0 for i in range(n)])
        p2.append([1 if i==arg2 else 0 for i in range(n)])
        if arg1 == arg2:
            label_output.append([0])
            continue
        if question_ids[arg1] in question_ids[arg2] and "_symmetric" in question_ids[arg2]:
            label_output.append([1])
        else:
            label_output.append([0])

    return torch.LongTensor(p1),torch.LongTensor(p2),torch.LongTensor(label_output)

def make_triple(question_ids):
    n=len(question_ids)
    p1=[]
    p2=[]
    p3=[]
    for arg1, arg2, arg3 in product(range(n), repeat=3):
        if arg1 == arg2 or arg2 == arg3:
            continue
        if question_ids[arg1] in question_ids[arg3] and \
           question_ids[arg2] in question_ids[arg3] and \
                "_transit" in question_ids[arg3]:
            p1.append([1 if i==arg1 else 0 for i in range(n)])
            p2.append([1 if i==arg2 else 0 for i in range(n)])
            p3.append([1 if i==arg3 else 0 for i in range(n)])

    return torch.LongTensor(p1),torch.LongTensor(p2),torch.LongTensor(p3)

def make_triple_with_labels(question_ids):
    n=len(question_ids)
    p1=[]
    p2=[]
    p3=[]
    label_output=[]
    for arg1, arg2, arg3 in product(range(n), repeat=3):
        p1.append([1 if i==arg1 else 0 for i in range(n)])
        p2.append([1 if i==arg2 else 0 for i in range(n)])
        p3.append([1 if i==arg3 else 0 for i in range(n)])
        if arg1 == arg2 or arg2 == arg3:
            label_output.append([0])
            continue
        if question_ids[arg1] in question_ids[arg3] and \
           question_ids[arg2] in question_ids[arg3] and \
                "_transit" in question_ids[arg3]:
            label_output.append([1])
        else:
            label_output.append([0])
    return torch.LongTensor(p1),torch.LongTensor(p2),torch.LongTensor(p3),torch.LongTensor(label_output)

def guess_pair(quest_id, arg1, arg2):

    if len(quest_id)<2 or arg1==arg2:
        return False
    quest1, quest2 = arg1.getAttribute('quest_id'), arg2.getAttribute('quest_id')
    if quest1 in quest2 and "_symmetric" in quest2:
        return True
    else:
        return False

def guess_triple(quest_id, arg11, arg22,arg33):

    if len(quest_id)<3 or arg11==arg22 or arg22==arg33:
        return False
    quest1, quest2,quest3 = arg11.getAttribute('quest_id'), arg22.getAttribute('quest_id'), arg33.getAttribute('quest_id')
    if quest1 in quest3 and quest2 in quest3 and "_transit" in quest3:
        return True
    return False

def is_ILP_consistant(questions_id,results):
    for i in results:
        if i[0]==None or i[1]==None or i[2]==None:
            return False
        if not i[0]+i[1]+i[2]==1:
            return False
    n=len(questions_id)
    for arg1, arg2 in product(range(n), repeat=2):
        if arg1 == arg2:
            continue
        if questions_id[arg1] in questions_id[arg2] and "_symmetric" in questions_id[arg2]:
            if (results[arg1][0] and not results[arg2][1]) or (results[arg1][1] and not results[arg2][0]):
                return False

    for arg1, arg2, arg3 in product(range(n), repeat=3):
        if arg1 == arg2 or arg2 == arg3 or arg1 == arg3:
            continue
        if questions_id[arg1] in questions_id[arg3] and \
           questions_id[arg2] in questions_id[arg3] and \
                "_transit" in questions_id[arg3]:
            if (results[arg1][0] and results[arg2][0] and not results[arg3][0]) or\
                    (results[arg1][0] and results[arg2][1] and not results[arg3][1]):
                return False
    return True


def test_inference_results(program, reader,cur_device,is_more,is_less,no_effect):
    counter = 0
    ac_ = 0
    ILPac_ = 0
    for paragraph_ in program.populate(reader, device=cur_device):
        #print("paragraph:", paragraph_.getAttribute('paragraph_intext'))
        paragraph_.inferILPResults(is_more,is_less,no_effect,fun=None)
        questions_id, results = [], []
        sresult=[]
        for question_ in paragraph_.getChildDataNodes():
            #print(question_.getAttribute('text'))
            #print(question_.getRelationLinks())
            questions_id.append(question_.getAttribute('quest_id'))
            
            is_moreILP = question_.getAttribute(is_more, "ILP")
            is_lessILP = question_.getAttribute(is_less, "ILP")
            no_effectILP =  question_.getAttribute(no_effect, "ILP") 
            
            results.append((question_.getAttribute(is_more, "ILP").item(), question_.getAttribute(is_less, "ILP").item(),question_.getAttribute(no_effect, "ILP").item()))

            predict_is_more_value=question_.getAttribute(is_more).softmax(-1)[1].item()
            predict_is_less_value=question_.getAttribute(is_less).softmax(-1)[1].item()
            predict_no_effect_value=question_.getAttribute(no_effect).softmax(-1)[1].item()

            sresult.append([predict_is_more_value,predict_is_less_value,predict_no_effect_value])
            if not "_symmetric" in question_.getAttribute('quest_id') and not "_transit" in question_.getAttribute('quest_id'):
                counter += 1
                ac_+=np.array([predict_is_more_value,predict_is_less_value,predict_no_effect_value]).argmax()==np.array([question_.getAttribute("is_more_"),question_.getAttribute("is_less_"),question_.getAttribute("no_effect_")]).argmax()
                ILPac_+=np.array(list(results[-1])).argmax()==np.array([question_.getAttribute("is_more_"),question_.getAttribute("is_less_"),question_.getAttribute("no_effect_")]).argmax()
        #print(results)
        #print(sresult)
        if not is_ILP_consistant(questions_id, results):
            print("ILP inconsistency")

    print("accuracy:", ac_ / counter)
    print("ILP accuracy:", ILPac_ / counter)

