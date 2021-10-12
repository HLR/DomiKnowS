import json

from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import torch
import random
from download_dataset import download


class Butterfly(Dataset):

    def __init__(self, transform=None, mode="val"):

        self.data = []
        self.targets = []
        self.transform = transform
        x = open(f"./ETHEC_dataset/splits/{mode}.json").read()
        # structure = {}
        for index, (key, values) in enumerate(json.loads(x).items()):
            # if index > 100:
            #     break
            # print(index, values)
            # print(values)
            family = values["family"].lower()
            subfamily = values["subfamily"].lower()
            # structure[family] = list(set(structure.get(family, []) + [subfamily]))

            # print(dictData)
            file_path = os.path.join(
                f"./ETHEC_dataset/IMAGO_build_test_resized/{values['image_path']}/{values['image_name']}")
            self.data.append(np.array(Image.open(file_path)))
            self.targets.append({
                'subFamily': ['dismorphiinae', 'pierinae', 'coliadinae', 'polyommatinae', 'theclinae', 'aphnaeinae',
                              'lycaeninae',
                              'limenitidinae', 'apaturinae', 'danainae', 'satyrinae', 'nymphalinae', 'libytheinae',
                              'heliconiinae',
                              'charaxinae', 'heteropterinae', 'pyrginae', 'hesperiinae', 'parnassiinae',
                              'papilioninae',
                              'nemeobiinae'].index(subfamily),
                'family': ['pieridae', 'lycaenidae', 'nymphalidae', 'hesperiidae', 'papilionidae', "riodinidae"].index(
                    family)
            })
        # # print(structure)
        #
        # data = []
        # for value in structure.values():
        #     data += value
        # for value in structure.keys():
        #     data.append(value)
        # print(len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict:
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        img = img.unsqueeze(0)

        return {**target, **{'pixels': img}}


def loadDataset(size=100, mode="train", batch_size=32):
    download(root="./")
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
    )
    data_set = Butterfly(transform=transform, mode=mode)

    def concat_images(input_dict):
        new_dict = {}
        for i_name, i_value in input_dict.items():
            if i_name == "pixels":
                new_dict[i_name] = torch.stack(i_value, dim=0)
                new_dict[i_name] = new_dict[i_name].permute(1, 0, 2, 3, 4)
                # print(new_dict[i_name].shape)
            else:
                new_dict[i_name] = "@@".join([str(j) for j in i_value])
        return new_dict

    random.seed(777)
    data_set = list(data_set)
    random.shuffle(data_set)
    final_reader = []
    tmp = 0
    cur_dict = {'pixels': [], 'family': [], 'subFamily': []}
    for i in data_set:
        for i_name, i_value in i.items():
            cur_dict[i_name].append(i_value)
        tmp += 1
        if tmp == batch_size:
            final_reader.append(concat_images(cur_dict))
            cur_dict = {'pixels': [], 'family': [], 'subFamily': []}
            tmp = 0
    final_reader.append(concat_images(cur_dict))
    print("last batch size:", tmp)
    return final_reader

# Butterfly(mode="train")
# Butterfly(mode="val")
# Butterfly(mode="test")

# x = loadDataset(mode="val")
# print(x)
# print(x[0]["pixels"].shape)
# #
# data = ['coliadinae', 'dismorphiinae', 'pierinae', 'polyommatinae', 'theclinae', 'lycaeninae', 'aphnaeinae',
#                     'charaxinae', 'limenitidinae', 'libytheinae', 'danainae', 'nymphalinae', 'apaturinae', 'satyrinae',
#                     'heliconiinae', 'pyrginae', 'hesperiinae', 'heteropterinae', 'parnassiinae', 'papilioninae',
#                     'nemeobiinae', 'pieridae', 'lycaenidae', 'nymphalidae', 'hesperiidae', 'papilionidae', 'riodinidae']
#
# print(
#     [f'image_group["{x}_group"]' for x in data]
# )
