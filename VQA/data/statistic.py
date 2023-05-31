import pickle
import numpy as np

train_path = "VQAR_all/VQAR/data/dataset/scene_graph/gqa_train_formatted_scene_graph.pkl"
feat_f = "/VL/space/zhan1624/VQAR-launcher/VQAR_all/VQAR/data/features.npy"

obj_feats = np.load(feat_f, allow_pickle=True).item()
cooked_sgs = pickle.load(open(train_path, 'rb'))
id_to_name = [item for sg in cooked_sgs for item in sg['relations'].items() if item[0] in obj_feats]
obj_ids, name_idxes = map(list, zip(*id_to_name))

out_dict= {}
for i in name_idxes:
    tmp = set(i.values())
    for j in tmp:
        if j not in out_dict:
            out_dict[j] = 1
        else:
            out_dict[j] += 1


result = dict(sorted(out_dict.items(), key=lambda item: item[1], reverse=True))
print('yue')


