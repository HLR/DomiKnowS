from ast import dump
import json

name_path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/rescources/attribute.json"
is_a_path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/data/knowledge_base/extra_is_a.facts"
is_a_dic = {}
with open(name_path) as f_name, open(is_a_path) as f_kg:
    name = json.load(f_name)
    is_a_kg = f_kg.read().split("\n")
    for i in is_a_kg:
        if i:
            i = i.split('\t')
            if i[1] not in is_a_dic:
                is_a_dic[i[1]] = [i[0]]
            else:
                is_a_dic[i[1]].append(i[0])
filt_dict = {}
for key, value in is_a_dic.items():
    if key not in name.values():
        continue
    for item in value:
        if item in name.values():
            if key not in filt_dict:
                filt_dict[key] = [item]
            else:
                filt_dict[key].append(item)

res = {}
for key, value in filt_dict.items():
    if [key] == value:
        continue
    res[key] = list(set(value))

with open("/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning/joslin/output_attribute.json", 'w') as f_out:
    json.dump(res, f_out, indent=4)
    
    
