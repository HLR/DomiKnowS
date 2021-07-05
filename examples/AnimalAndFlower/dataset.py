from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import transforms
import numpy as np


class AnimalAndFlowers(Dataset):

    def __init__(self, root, transform=None):

        self.root = root
        self.data = []
        self.targets = []
        self.transform = transform
        index = 0
        for folder_name in os.listdir(root):
            for category_name in os.listdir(f"{root}/{folder_name}"):
                for file_name in os.listdir(f"{root}/{folder_name}/{category_name}/"):
                    file_path = os.path.join(self.root, folder_name, category_name, file_name)
                    with open(file_path, 'rb') as f:
                        self.data.append(np.array(Image.open(file_path).resize((100, 100))))
                        self.targets.append({
                            'monkey': 1 if category_name == 'monkey' else 0,
                            'cat': 1 if category_name == 'cat' else 0,
                            'squirrel': 1 if category_name == 'squirrel' else 0,
                            'dog': 1 if category_name == 'dog' else 0,
                            'daisy': 1 if category_name == 'daisy' else 0,
                            'dandelion': 1 if category_name == 'dandelion' else 0,
                            'rose': 1 if category_name == 'rose' else 0,
                            'tulip': 1 if category_name == 'tulip' else 0,
                            'sunflower': 1 if category_name == 'sunflower' else 0,
                            'flower': 1 if folder_name == 'flowers' else 0,
                            'animal': 1 if folder_name == 'animals' else 0,
                        })
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


def load_animals_and_flowers(root='./data/', size=100):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=round(size // 8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
    )

    return AnimalAndFlowers(root=root, transform=transform)