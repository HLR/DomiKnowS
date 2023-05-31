"""
To obtain the relationed labels for the names, attribute is not ready yet
"""


import json

name_path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/rescources/name.json"
attr_path ="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/rescources/attribute.json"
is_a_path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/data/knowledge_base/extra_is_a.facts"
is_a_dic = {}
with open(name_path) as f_name, open(is_a_path) as f_kg, open(attr_path) as f_att:
    name = json.load(f_name)
    attr = json.load(f_att)
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
    if key not in attr.values():
        continue
    for item in value:
        if item in attr.values():
            if key not in filt_dict:
                filt_dict[key] = [item]
            else:
                filt_dict[key].append(item)

# out_path = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning/joslin/new_is_a.facts"
# with open(out_path, 'w') as f_out:
#     for key, value in filt_dict.items():
#         for item in value:
#             if item != value:
#                 f_out.write(key+"\t"+item)
#                 f_out.write("\n")

print('yue')