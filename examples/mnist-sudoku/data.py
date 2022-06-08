from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import Flatten
import torch
import random
import numpy as np

import config

solutions_9by9 = np.zeros((10000, 81), np.int32)
for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
    if i >= solutions_9by9.shape[0]:
        break

    quiz, solution = line.split(",")
    for j, q_s in enumerate(zip(quiz, solution)):
        q, s = q_s
        solutions_9by9[i, j] = s

solutions_9by9 = solutions_9by9.reshape((-1, 9, 9))
solutions_6by6 = solutions_9by9[:,:6,:6]

if config.size == 9:
    solutions = solutions_9by9
elif config.size == 6:
    solutions = solutions_6by6

transform = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,)),
                          Flatten(0)
                          ])
trainset = datasets.MNIST('data', download=True, train=True, transform=transform)
testset = datasets.MNIST('data', download=True, train=False, transform=transform)

train_ids = random.sample(range(0, 50000), config.train_sample)
valid_ids = random.sample(range(50000, 60000), config.valid_sample)

class SudokuDataset(Dataset):
    def __init__(self, dataset, digit_ids, solutions):
        self.dataset = dataset
        
        self.digit_to_id = {}

        for d in range(10):
            self.digit_to_id[d] = []

        for d_id in digit_ids:
            self.digit_to_id[dataset[d_id][1]].append(d_id)
    
        print('num digit_ids: ', len(digit_ids))
        print('solutions: ', len(solutions))
        
        
        self.solution_image_ids = torch.empty((solutions.shape[0],
                                               solutions.shape[1] ** 2), dtype=torch.int32) # n x 36 or 81
        for i, sol in enumerate(solutions):
            for j, digit in enumerate(sol.flatten()):
                self.solution_image_ids[i, j] = self.get_id_from_digit(digit)

    def get_id_from_digit(self, digit):
        return random.sample(self.digit_to_id[digit], 1)[0]
    
    def __len__(self):
        return self.solution_image_ids.shape[0]
    
    def __getitem__(self, idx):
        digit_ids = self.solution_image_ids[idx]
        
        images = torch.empty((self.solution_image_ids.shape[1], 28 * 28))
        labels = torch.empty((self.solution_image_ids.shape[1],))
        
        for i, d_id in enumerate(digit_ids):
            images[i] = self.dataset[d_id][0]
            labels[i] = self.dataset[d_id][1]
        
        images = torch.unsqueeze(images, dim=0)
        labels = torch.unsqueeze(labels, dim=0)
        
        return {'images': images, 'labels': labels}
    
s_data = SudokuDataset(trainset, train_ids, solutions[:config.num_train])
s_data_valid = SudokuDataset(trainset, valid_ids, solutions[:config.num_valid])

trainloader = DataLoader(
    s_data,
    shuffle=False
)

validloader = DataLoader(
    s_data_valid,
    shuffle=False
)
