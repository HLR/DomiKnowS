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
    if (quest1 in quest2 and "_symmetric" in quest2) or (quest2 in quest1 and "_symmetric" in quest1):
        return True
    else:
        return False

def guess_triple(quest_id, arg11, arg22,arg33):

    if len(quest_id)<3 or arg11==arg22 or arg22==arg33 or arg11==arg33:
        return False
    quest1, quest2, quest3 = arg11.getAttribute('quest_id'), arg22.getAttribute('quest_id'), arg33.getAttribute('quest_id')
    if quest1 +"@" + quest2 in quest3 and "_transit" in quest3:
        return True

    return False

import gurobipy as gp
from gurobipy import GRB

def is_ILP_consistant(questions_id,results,verbose,probabilities,para_num):

    n=len(questions_id)
    tran_violated=False
    m = gp.Model("whatever")
    m.setParam(GRB.Param.OutputFlag, 0)
    obj = gp.LinExpr()
    g_vars=[]
    for i_var in range(n):
        g_vars.append( (m.addVar(vtype=GRB.BINARY, name=str(i_var)+" more"),m.addVar(vtype=GRB.BINARY, name=str(i_var)+" less"),m.addVar(vtype=GRB.BINARY, name=str(i_var)+" no effect") ) )
        obj-=1-g_vars[-1][0]*probabilities[i_var][0]+1-g_vars[-1][1]*probabilities[i_var][1]+1-g_vars[-1][2]*probabilities[i_var][2]
        m.addConstr(g_vars[-1][0]+g_vars[-1][1]+g_vars[-1][2]== 1, str(i_var)+" labels")

    if verbose:
        for i in results:
            if i[0]==None or i[1]==None or i[2]==None:
                print("Error: There is a None in paragraph : ",para_num)
            elif i[0]+i[1]+i[2] > 1:
                print("Error: There more than one correct label in paragraph : ",para_num)
            elif i[0]+i[1]+i[2] == 0:
                print("Error: There are no label in paragraph : ",para_num)
        
    for arg1, arg2 in product(range(n), repeat=2):
        if arg1 == arg2:
            continue
        if questions_id[arg1] in questions_id[arg2] and "_symmetric" in questions_id[arg2]:
            if (results[arg1][0] and not results[arg2][1]) or (results[arg1][1] and not results[arg2][0]):
                if verbose:
                    print("Symmetry is violated in paragraph : ",para_num)
            m.addConstr(g_vars[arg1][0]+g_vars[arg2][0]<= 1, str(arg1)+" "+str(arg2)+" s1")
            m.addConstr(g_vars[arg1][0]+g_vars[arg2][2]<= 1, str(arg1)+" "+str(arg2)+" s2")
            m.addConstr(g_vars[arg1][1]+g_vars[arg2][1]<= 1, str(arg1)+" "+str(arg2)+" s3")
            m.addConstr(g_vars[arg1][1]+g_vars[arg2][2]<= 1, str(arg1)+" "+str(arg2)+" s4")
            m.addConstr(g_vars[arg1][2]+g_vars[arg2][0]<= 1, str(arg1)+" "+str(arg2)+" s5")
            m.addConstr(g_vars[arg1][2]+g_vars[arg2][1]<= 1, str(arg1)+" "+str(arg2)+" s6")

    for arg1, arg2, arg3 in product(range(n), repeat=3):
        if arg1 == arg2 or arg2 == arg3 or arg1 == arg3:
            continue
        if questions_id[arg1] +"@" + questions_id[arg2] in questions_id[arg3] and \
                "_transit" in questions_id[arg3]:
            m.addConstr(g_vars[arg3][0]+1 >= g_vars[arg1][0]+ g_vars[arg2][0],str(arg1)+" "+str(arg2)+" "+ str(arg3)+" tran 1")
            m.addConstr(g_vars[arg3][1]+1 >= g_vars[arg1][0]+ g_vars[arg2][1],str(arg1)+" "+str(arg2)+" "+ str(arg3)+" tran 2")
            if (results[arg1][0] and results[arg2][0] and not results[arg3][0]) or\
                    (results[arg1][0] and results[arg2][1] and not results[arg3][1]):
                if verbose:
                    print("Transivity is violated in paragraph : ",para_num)
                    tran_violated=True
                    print(para_num)
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    vars_=list(m.getVars())
    return [i.x for i in vars_],tran_violated


def test_inference_results(program, reader, cur_device, is_more, is_less, no_effect, transitive, symmetric, verbose):
    counter = 0
    ac_ = 0
    ILPac_ = 0
    ac_test=0
    for para_num,paragraph_ in enumerate(program.populate(reader, device=cur_device)):

        conceptsRelations = [is_more, is_less, no_effect, transitive, symmetric]
        paragraph_.inferILPResults(*conceptsRelations, fun=None)
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
                ac_+=np.array([predict_is_more_value,predict_is_less_value,predict_no_effect_value]).argmax()==np.array([question_.getAttribute("is_more_").cpu().numpy()[0],question_.getAttribute("is_less_").cpu().numpy()[0],question_.getAttribute("no_effect_").cpu().numpy()[0]]).argmax()
                ILPac_+=np.array(list(results[-1])).argmax()==np.array([question_.getAttribute("is_more_").cpu().numpy()[0],question_.getAttribute("is_less_").cpu().numpy()[0],question_.getAttribute("no_effect_").cpu().numpy()[0]]).argmax()

        _vars,tran_violated=is_ILP_consistant(questions_id, results , verbose,sresult,para_num)

        for num,question_ in enumerate(paragraph_.getChildDataNodes()):
            if not "_symmetric" in question_.getAttribute('quest_id') and not "_transit" in question_.getAttribute('quest_id'):
                import logging
                if not np.array(list(results[num])).argmax()==np.array([_vars[num*3],_vars[num*3+1],_vars[num*3+2]]).argmax() and verbose:
                    logging.info(" This question in paragraph number {} is Incorrect".format(str(para_num)))
                    #print(para_num," This question in paragraph number {} is Incorrect".format(para_num),len(list(results)),question_.getAttribute('quest_id'),np.array(list(results[num])),np.array([_vars[num*3],_vars[num*3+1],_vars[num*3+2]]),questions_id, results , verbose,sresult,para_num)
                elif verbose:
                    logging.info(" This question in paragraph number {} is Correct".format(str(para_num)))
                    #print(para_num," This question in paragraph number {} is Correct".format(para_num))
                fb=np.array([_vars[num*3],_vars[num*3+1],_vars[num*3+2]]).argmax()
                sb=np.array([question_.getAttribute("is_more_").cpu().numpy()[0],question_.getAttribute("is_less_").cpu().numpy()[0],question_.getAttribute("no_effect_").cpu().numpy()[0]]).argmax()
                ac_test+=fb==sb
    print("accuracy:", ac_ / counter)
    print("ILP accuracy:", ILPac_ / counter)
    #print("ILP test accuracy:", ac_test / counter)

import os

def join_model(fromdir, tofile):
    output = open(tofile, 'wb')
    parts  = os.listdir(fromdir)
    parts.sort(  )
    for filename in parts:
        filepath = os.path.join(fromdir, filename)
        fileobj  = open(filepath, 'rb')
        while 1:
            filebytes = fileobj.read(int(90*1000*1024))
            if not filebytes: break
            output.write(filebytes)
        fileobj.close(  )
    output.close(  )


def split(fromfile, todir, chunksize=int(90*1000*1024)):
    if not os.path.exists(todir):                  # caller handles errors
        os.mkdir(todir)                            # make dir, read/write parts
    else:
        for fname in os.listdir(todir):            # delete any existing files
            os.remove(os.path.join(todir, fname))
    partnum = 0
    input = open(fromfile, 'rb')                   # use binary mode on Windows
    while 1:                                       # eof=empty string from read
        chunk = input.read(chunksize)              # get next part <= chunksize
        if not chunk: break
        partnum  = partnum+1
        filename = os.path.join(todir, ('part%04d' % partnum))
        fileobj  = open(filename, 'wb')
        fileobj.write(chunk)
        fileobj.close()                            # or simply open(  ).write(  )
    input.close(  )
    assert partnum <= 9999                         # join sort fails if 5 digits
    return partnum
