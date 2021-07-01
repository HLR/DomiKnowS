import pandas as pd
import networkx as nx
from collections import defaultdict

def make_reader(file_address="data/WIQA_AUG/train.jsonl",sample_num=1000000000,batch_size=14):

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

    comp_size=[len(i) for i in nx.connected_components(G)]
    #print(comp_size[:100])
    comps=[i for i in nx.connected_components(G)]
    zipped_lists = zip(comp_size, comps)
    sorted_pairs = sorted(zipped_lists,reverse=True)
    tuples = zip(*sorted_pairs)
    comp_size, comps = [ list(tuple) for tuple in  tuples]
    print(comp_size[-10:],comps[-10:])
    for i in range(len(comp_size)):
        if comp_size[i]>=batch_size or comp_size[i]==0:
            continue
        for j in range(i+1,len(comp_size)):
            if comp_size[j]==0 or comp_size[j]+comp_size[i]>batch_size or \
                    not quest_id_to_datum[sorted(list(comps[j]))[0]]["paragraph"]==quest_id_to_datum[sorted(list(comps[i]))[0]]["paragraph"]:
                continue
            G.add_edge(sorted(list(comps[j]))[0],sorted(list(comps[i]))[0])
            comp_size[i]+=comp_size[j]
            comp_size[j]=0
            if comp_size[i]>=batch_size:
                break

    final_list=[]
    print(len(list(nx.connected_components(G))))
    for i in sorted(list(nx.connected_components(G))):
        if len(i)<=batch_size:
            final_list.append(sorted(list(i)))
            continue
        cur_list=sorted(list(i))
        for j in range(0,len(i),batch_size):
            final_list.append(cur_list[j:j+batch_size])

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
        reader.append({"paragraph_intext":para,"more_list":"@@".join(more_list),"less_list":"@@".join(less_list),"no_effect_list":"@@".join(no_effect_list),
                       "question_list":"@@".join(question_list),"quest_ids":"@@".join(i)})
    return reader
