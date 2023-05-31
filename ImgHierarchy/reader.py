import torch
import numpy as np

from domiknows.data.reader import RegrReader


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
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class VQADataset(Dataset):
    def __init__(self, data):
        self.data = data
        

    def change_vector(self, vector):
        signs = vector - torch.mean(vector)
        signs[signs < 0] = -1
        signs[signs >= 0] = +1
        adjustment = signs * torch.pow(vector - torch.mean(vector), 2)
        vector = torch.clamp(vector, min=1e-12, max=1 - 1e-12)
        entropy = torch.distributions.Categorical(torch.log(vector)).entropy() / vector.shape[0]
        return (vector/torch.mean(vector))
        # return (1/entropy.item()) * vector
        # return (1/entropy.item()) * (vector/torch.mean(vector))
        # return (1/entropy.item()) * (vector) + adjustment
        # return vector
    
    def __getitem__(self, index):
        x1 = torch.tensor(self.data['levels1']['logit'][index]).softmax(dim=-1)
        x2 = torch.tensor(self.data['levels2']['logit'][index]).softmax(dim=-1)
        x3 = torch.tensor(self.data['levels3']['logit'][index]).softmax(dim=-1)
        x4 = torch.tensor(self.data['levels4']['logit'][index]).softmax(dim=-1)

        ### Normalize the values
        # x1 = x1 / torch.mean(x1)
        # x2 = x2 / torch.mean(x2)
        # x3 = x3 / torch.mean(x3)
        # x4 = x4 / torch.mean(x4)

        ### normalize with sign and std
        x1 = self.change_vector(x1)
        x2 = self.change_vector(x2)
        x3 = self.change_vector(x3)
        x4 = self.change_vector(x4)

        # x1 = torch.std(x1) * x1
        # x2 = torch.std(x2) * x2
        # x3 = torch.std(x3) * x3
        # x4 = torch.std(x4) * x4

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

