import torch
import numpy as np
import argparse
from dataset import load_annodata
from model import model_declaration

def main():
    # if torch.cuda.is_available():
    #     device="cuda:0"
    # else:
    #     device='cpu'
    device='cpu'
    print("selected device is:",device)

    data = load_annodata("data/LowContextAnnoData.csv")

    train_data, val_data, test_data = np.split(data, [int(len(data)*0.8), int(len(data)*0.9)])

    program = model_declaration(device)
     
    program.train(train_data, valid_set=val_data, train_epoch_num=4,
                  Optim=lambda param: torch.optim.Adam(param, lr=5e-5), device=device)

    program.test(test_data)

main()