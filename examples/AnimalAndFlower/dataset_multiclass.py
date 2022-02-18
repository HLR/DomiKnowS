from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import torch


class AnimalAndFlowers(Dataset):

    def __init__(self, args, root, transform=None):

        self.root = root
        self.data = []
        self.targets = []
        self.transform = transform
        index = 0
        if args.verbose:
            print("current directory", os.getcwd())
        for folder_name in os.listdir(root):
            if "." in folder_name or "food" in folder_name:
                continue
            for category_name in os.listdir(f"{root}/{folder_name}"):
                print(category_name)
                for file_name in os.listdir(f"{root}/{folder_name}/{category_name}/"):
                    file_path = os.path.join(self.root, folder_name, category_name, file_name)
                    with open(file_path, 'rb') as f:
                        self.data.append(np.array(Image.open(file_path).resize((100, 100))))
                        target = {
                            'tag': [["cat", "dog", "monkey", "squirrel", "daisy", "dandelion", "rose", "sunflower","tulip"].index(category_name)],
                            'category': [["animals", "flowers"].index(folder_name)]
                        }

                        self.targets.append(target)
                        # self.targets.append({
                        #     'monkey': 1 if category_name == 'monkey' else 0,
                        #     'cat': 1 if category_name == 'cat' else 0,
                        #     'squirrel': 1 if category_name == 'squirrel' else 0,
                        #     'dog': 1 if category_name == 'dog' else 0,
                        #     'daisy': 1 if category_name == 'daisy' else 0,
                        #     'dandelion': 1 if category_name == 'dandelion' else 0,
                        #     'rose': 1 if category_name == 'rose' else 0,
                        #     'tulip': 1 if category_name == 'tulip' else 0,
                        #     'sunflower': 1 if category_name == 'sunflower' else 0,
                        #     'flower': 1 if folder_name == 'flowers' else 0,
                        #     'animal': 1 if folder_name == 'animals' else 0,
                        # })
                        index += 1
        self.data = np.vstack(self.data).reshape(-1, 3, 100, 100)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

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


def load_animals_and_flowers(args, root='./data/', size=100, ):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=round(size // 8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    )
    data_set = AnimalAndFlowers(args, root=root, transform=transform)
    import random
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
    batch_size = 32
    tmp = 0
    cur_dict = {'tag': [], 'category': [], 'pixels': []}
    for i in data_set:
        for i_name, i_value in i.items():
            cur_dict[i_name].append(i_value)
        tmp += 1
        if tmp == batch_size:
            final_reader.append(concat_images(cur_dict))
            cur_dict = {'tag': [], 'category': [], 'pixels': []}
            tmp = 0
    final_reader.append(concat_images(cur_dict))
    print("last batch size:", tmp)

    return final_reader
