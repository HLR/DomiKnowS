import torch
import numpy as np

from domiknows.data.reader import RegrReader
from sg_data_loader import SceneGraphLoader
import json


class VQAReader(RegrReader):
    def getrepval(self, item):
        return torch.tensor(item["reps"])

    def getlevel1val(self, item):
        return torch.tensor(item['levels1']['logit'])
        
    def getlevel1_labelval(self, item):
        return torch.tensor(item['levels1']['label'])

    def getlevel2val(self, item):
        return torch.tensor(item['levels2']['logit'])
        
    def getlevel2_labelval(self, item):
        return torch.tensor(item['levels2']['label'])

    def getlevel3val(self, item):
        return torch.tensor(item['levels3']['logit'])
        
    def getlevel3_labelval(self, item):
        return torch.tensor(item['levels3']['label'])

    def getlevel4val(self, item):
        return torch.tensor(item['level4']['logit'])
        
    def getlevel4_labelval(self, item):
        return torch.tensor(item['levels4']['label'])

    def parse_file(self):
        data = np.load(self.file, allow_pickle=True)
        print("here!")
        print("yes again here!")


import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class VQADataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        x1 = torch.tensor(self.data['levels1']['logit'][index])
        x2 = torch.tensor(self.data['levels2']['logit'][index])
        x3 = torch.tensor(self.data['levels3']['logit'][index])
        x4 = torch.tensor(self.data['levels4']['logit'][index])

        y1 = torch.tensor(self.data['levels1']['labels'][index])
        y2 = torch.tensor(self.data['levels2']['labels'][index])
        y3 = torch.tensor(self.data['levels3']['labels'][index])
        y4 = torch.tensor(self.data['levels4']['labels'][index])
        
        return {
            "level1": x1, "level2": x2, "level3": x3, "level4": x4, 
            "level1_label": y1, "level2_label": y2, "level3_label": y3, "level4_label": y4,
            "reps": torch.randn((2))
            }
    
    def __len__(self):
        return len(self.data['levels1']['logit'])


class VQADatasetSG(Dataset):
    def __init__(self, data):
        meta_info = json.load(open('../../data/gqa_info.json', 'r'))

        obj_feats = np.load('../../data/features.npy', allow_pickle=True).item()

        self.sg_test_loader = SceneGraphLoader(
            meta_info=meta_info,
            data_f='./rescources/new_label_data/val.pickle',
            obj_feats=obj_feats,
            type='name',
            batch_size=1,
            drop_last=False
        )
        
    def __getitem__(self, index):
        x1 = torch.tensor(self.data['levels1']['logit'][index])

        y1 = torch.tensor(self.data['levels1']['labels'][index])
        y2 = torch.tensor(self.data['levels2']['labels'][index])
        y3 = torch.tensor(self.data['levels3']['labels'][index])
        y4 = torch.tensor(self.data['levels4']['labels'][index])
        
        return {
            "level1": x1, "level2": x2, "level3": x3, "level4": x4, 
            "level1_label": y1, "level2_label": y2, "level3_label": y3, "level4_label": y4,
            "reps": torch.randn((2))
            }
    
    def __len__(self):
        return len(self.data['levels1']['logit'])