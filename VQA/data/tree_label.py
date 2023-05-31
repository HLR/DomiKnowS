import json
from os import name
import pickle
from treelib import Node, Tree
from tqdm import tqdm

#input_data = pickle.load(open("/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning/joslin/rescources/new_label_data/train.pickle", 'rb'))

label_file = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/output_names.json"
input_file ="/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/data/dataset/scene_graph/gqa_train_formatted_scene_graph.pkl"
name_file = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/name.json"
input_data = pickle.load(open(input_file, 'rb'))

with open(label_file) as f_l, open(name_file) as f_n:
    labels = json.load(f_l)
    names = json.load(f_n)

### create label
outputs = {}
all_labels = []
for key_i, value_i in labels.items():
    outputs[key_i] = {}
    if key_i not in all_labels:
        all_labels.append(key_i)
    for each_v in value_i:
        if each_v not in all_labels:
            all_labels.append(each_v)
        outputs[each_v] = {}
        outputs[each_v]['parent'] = key_i
    for key_j, value_j in labels.items():
        if key_i == key_j:
            continue
        if key_i in value_j:
            outputs[key_i]['parent'] = key_j
            flag = 1
            break
    else:
        outputs[key_i]['parent'] = "Thing"
    
        
outputs["Thing"]={} 
outputs['Thing']["parent"]= None

### create a tree
dict_ = outputs
added = set()
tree = Tree()
while dict_:
    for key, value in dict_.items():
        if value['parent'] in added:
            tree.create_node(key, key, parent=value['parent'])
            added.add(key)
            dict_.pop(key)
            break
        elif value['parent'] is None:
            tree.create_node(key, key)
            added.add(key)
            dict_.pop(key)
            break

# levels of labels
tree_label = {}
for each_label in all_labels:
    node = tree.get_node(each_label)
    depth = tree.depth(node)
    if depth not in tree_label:
        tree_label[depth] = [each_label]
    else:
        tree_label[depth].append(each_label)

'''
### create labels for each level
tmp_level = []
for value_n in names.values():
    if value_n not in tree_label[1] and value_n not in tree_label[2] and \
        value_n not in tree_label[3] and value_n not in tree_label[4]:
        tmp_level.append(value_n)

level0_label, level1_label, level2_label, level3_label = {}, {}, {}, {}
tree_label[1] += tmp_level
for id0, item0 in enumerate(tree_label[1]):
    level0_label[id0] = item0
for id1, item1 in enumerate(tree_label[2]):
    level1_label[id1] = item1
for id2, item2 in enumerate(tree_label[3]):
    level2_label[id2] = item2
for id3, item3 in enumerate(tree_label[4]):
    level3_label[id3] = item3
        
file1 = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name1.json"
file2 = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name2.json"
file3 = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name3.json"
file4 = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name4.json"
with open(file1, 'w') as f1, open(file2, 'w') as f2, open(file3, 'w') as f3, open(file4, 'w') as f4:
    json.dump(level0_label, f1, indent=4)  
    json.dump(level1_label, f2, indent=4)  
    json.dump(level2_label, f3, indent=4)  
    json.dump(level3_label, f4, indent=4)  
'''

def name_to_id(input_dict):
    name_id_dict = {}
    for key, value in input_dict.items():
        name_id_dict[value] = key
    return name_id_dict

file1 = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name1.json"
file2 = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name2.json"
file3 = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name3.json"
file4 = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/labels/name4.json"
with open(file1) as f1, open(file2) as f2, open(file3) as f3, open(file4) as f4:
    label1 = json.load(f1)  
    label2 = json.load(f2)  
    label3 = json.load(f3)  
    label4 = json.load(f4)

name_id_dict1 = name_to_id(label1)
name_id_dict2 = name_to_id(label2)
name_id_dict3 = name_to_id(label3)
name_id_dict4 = name_to_id(label4)

### process the label in each data
for each_example in tqdm(input_data):
    each_example['hierachy_names'] = {}
    for name_key, name_value in each_example['names'].items():
        new_label = [None]*4
        text_name = names[str(name_value)]
        if text_name in label1.values():
            ### search the first level
            new_label[0] = int(name_id_dict1[text_name])
            new_label[1] = -1
            new_label[2] = -1
            new_label[3] = -1
        elif text_name in label2.values():
            ### search the second level
            path = list(tree.rsearch(text_name))
            new_label[0] = int(name_id_dict1[path[1]])
            new_label[1] = int(name_id_dict2[text_name])
            new_label[2] = -1
            new_label[3] = -1
        elif text_name in label3.values():
            ### search the third level
            path = list(tree.rsearch(text_name))
            new_label[0] = int(name_id_dict1[path[-2]])
            new_label[1] = int(name_id_dict2[path[-3]])
            new_label[2] = int(name_id_dict3[text_name])
            new_label[3] = -1
        elif text_name in label4.values():
            ### search the forth level
            path = list(tree.rsearch(text_name))
            new_label[0] = int(name_id_dict1[path[-2]])
            new_label[1] = int(name_id_dict2[path[-3]])
            new_label[2] = int(name_id_dict3[path[-4]])
            new_label[3] = int(name_id_dict4[text_name])
        else:
            ### put the out of level on the first level
            print('wrong case')
        each_example['hierachy_names'][name_key] = new_label
        

with open('/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/src/supervised_learning_joslin/joslin/rescources/new_label_data/train.pickle', 'wb') as handle:
    pickle.dump(input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('yue')

