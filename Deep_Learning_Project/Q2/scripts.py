import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd(), 'Q2/skeleton_code'))
from utils_w_masking import load_rnacompete_data

def see_valid(split, missing):
    train_dataset = load_rnacompete_data(protein_name='RBFOX1', split=split)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    exists25 = 0
    exists0 = 0
    invalid_x = 0
    masked_x = 0

    for batch in train_loader:
        x, x_mask, y, mask = batch
        for element in x:
            if (element == torch.tensor([0.25,0.25,0.25,0.25])).all(axis = 1).sum() > 0:
                exists25 += 1
            if (element == torch.tensor([0,0,0,0])).all(axis = 1).sum() > 0:
                exists0 += 1
            if (element == torch.tensor([0,0,0,0])).all(axis = 1).sum() > 3:
                invalid_x += 1

        for masked_element in x_mask:
            if (masked_element == 0).sum() > missing:
                masked_x += 1

    print("-----------------------------------------")
    print("Split -",split)
    print("-----------------------------------------")
    print("Number of Ns one-hot encoded as [0.25,0.25,0.25,0.25]:", exists25)
    print("Number of Ns one-hot encoded as [0,0,0,0]:", exists0)
    print("Number of invalid Xs:", invalid_x)
    print("Number of Xs:", (len(train_loader)-1) * 64 + len(x))
    print("Number of invalid Xs (X_mask):", masked_x)
    print("-----------------------------------------")

see_valid('train', 10)
see_valid('val', 10)
see_valid('test', 10)