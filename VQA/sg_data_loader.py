import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SceneGraphDataset(Dataset):
    def __init__(self, meta_info, data_f, obj_feats, type):
        cooked_sgs = pickle.load(open(data_f, 'rb'))

        self.type = type
        if type == 'name':
            id_to_name = [item for sg in cooked_sgs for item in sg['names'].items() if item[0] in obj_feats]
            _, self.name_idxes = map(list, zip(*id_to_name))
            hierachy_id_to_name =  [item for sg in cooked_sgs for item in sg['hierachy_names'].items() if item[0] in obj_feats]
            self.obj_ids, self.hierachy_name_idxes = map(list, zip(*hierachy_id_to_name))
            self.name_idxes1, self.name_idxes2, self.name_idxes3, self.name_idxes4 = map(list, zip(*self.hierachy_name_idxes))
            # self.obj_ids, self.name_idxes = map(list, zip(*id_to_name))
            self.name_idxes = torch.tensor(self.name_idxes)
            self.name_idxes1 = torch.tensor(self.name_idxes1)
            self.name_idxes2 = torch.tensor(self.name_idxes2)
            self.name_idxes3 = torch.tensor(self.name_idxes3)
            self.name_idxes4 = torch.tensor(self.name_idxes4)
        elif type == 'relation':
            n_rels = meta_info['rel']['num']
            sub_to_obj_rel = [item for sg in cooked_sgs for item in sg['relations'].items()]
            # sub_obj_rel = [(sub_id, obj_id, obj_rel[obj_id])]
            self.bboxes = {k: torch.tensor(v) for sg in cooked_sgs for k, v in sg['bboxes'].items()}
            # filtered_id_to_rel = [item for item in id_to_rel if item[0][0] in self.bboxes and item[0][1] in self.bboxes
            #                       and item[0][0] in obj_feats and item[0][1] in obj_feats]
            # pair_ids, self.rel_idxes = map(list, zip(*filtered_id_to_rel))
            # self.rel_idxes = torch.tensor(self.rel_idxes)
            # self.sub_ids, self.obj_ids = map(list, zip(*pair_ids))
            # TODO: replace with more efficient map operation
            self.sub_ids = []
            self.obj_ids = []
            self.rel_idxes = []
            for sub_id, obj_to_rel_dict in sub_to_obj_rel:
                for obj_id in obj_to_rel_dict:
                    rel_idx = obj_to_rel_dict[obj_id]
                    if rel_idx < 0:
                        rel_idx = n_rels
                    self.sub_ids.append(sub_id)
                    self.obj_ids.append(obj_id)
                    self.rel_idxes.append(rel_idx)
            self.rel_idxes = torch.tensor(self.rel_idxes)

        else:       # type == 'attribute'
            n_attrs = meta_info['attr']['num']
            id_to_attr = {id: attr_list for sg in cooked_sgs for id, attr_list in sg['attributes'].items()}
            attr_labels = []
            for id in id_to_attr:
                # TODO: replace with more efficient torch scatter
                attr_labels_i = torch.zeros(n_attrs, dtype=torch.float)
                for pos_idx in id_to_attr[id]:
                    attr_labels_i[pos_idx] = 1.
                attr_labels.append(attr_labels_i)
            self.obj_ids = list(id_to_attr.keys())
            self.attr_labels = torch.stack(attr_labels)

        self.feats = obj_feats
        print('%d datapoints loaded' % self.__len__())

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, index):
        obj_feat = torch.from_numpy(self.feats[self.obj_ids[index]])
        if self.type == 'name':
            return obj_feat, self.name_idxes[index], self.name_idxes1[index], self.name_idxes2[index], self.name_idxes3[index], self.name_idxes4[index]
        elif self.type == 'relation':
            sub_feat = torch.from_numpy(self.feats[self.sub_ids[index]])
            feat = torch.cat([sub_feat, obj_feat, self.bboxes[self.sub_ids[index]], self.bboxes[self.obj_ids[index]]])
            return feat, self.rel_idxes[index]
        else:       # self.type == 'attribute'
            return obj_feat, self.attr_labels[index]

class SceneGraphLoader(DataLoader):
    def __init__(self, meta_info, data_f, obj_feats, type, batch_size, drop_last=True, shuffle=True):
        self.dataset = SceneGraphDataset(meta_info, data_f, obj_feats, type)
        self.batch_size = batch_size
        super(SceneGraphLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

if __name__ == '__main__':
    import os

    # DATA_ROOT = os.getenv('HOME') + '/project_data/CRIC/scene_graph_data'
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
    DATA_ROOT = os.path.join(data_dir, "scene_graph_data")

    train_data_f = DATA_ROOT + '/train_split.pkl'
    val_data_f = DATA_ROOT + '/val_split.pkl'
    feat_f = DATA_ROOT + '/features.npy'

    # if not os.path.exists(train_data_f):
    #     train_ratio = 0.8
    #     data_f = DATA_ROOT + '/train_dataset_cooked.pkl'
    #     cooked_sgs = pickle.load(open(data_f, 'rb'))
    #     idxes = list(cooked_sgs.keys())
    #     random.seed(1234)
    #     train_idxes = random.sample(idxes, k=int(train_ratio*len(idxes)))
    #     val_idxes = [idx for idx in idxes if idx not in train_idxes]
    #     train_sgs = {idx: sg for idx, sg in cooked_sgs.items() if idx in train_idxes}
    #     val_sgs = {idx: sg for idx, sg in cooked_sgs.items() if idx in val_idxes}
    #
    #     print(len(train_sgs), len(val_sgs), len(cooked_sgs))
    #     pickle.dump(train_sgs, open(train_data_f, 'wb'))
    #     pickle.dump(val_sgs, open(val_data_f, 'wb'))

    feat_f = np.load(feat_f, allow_pickle=True).item()
    d = SceneGraphDataset(val_data_f, feat_f, type='attr:0')

    print(d)