import os
import pandas as pd
from torch.utils.data import Dataset
from scipy.io import loadmat
import torch
def readmat(filename, device=None):
    m = loadmat(filename)
    L = list(m.keys())
    # print(L)
    cube = m[L[3]]
    cube_tensor = torch.unsqueeze(torch.tensor(cube),0) # make another dimension for channels
    if device:
        cube_tensor=cube_tensor.to(device, dtype=torch.float)    
    else:
        cube_tensor=cube_tensor.to(dtype=torch.float)
    return cube_tensor

def readmat_2(filename, device=None): # one mat file contains two cubes
    m = loadmat(filename)
    L = list(m.keys())
    # print(L)
    cube1 = m[L[3]]
    cube2 = m[L[4]]

    cube_tensor1 = torch.unsqueeze(torch.tensor(cube1),0) # make another dimension for channels
    cube_tensor2 = torch.unsqueeze(torch.tensor(cube2),0) # make another dimension for channels
    if device:
        cube_tensor1=cube_tensor1.to(device, dtype=torch.float)
        cube_tensor2=cube_tensor2.to(device, dtype=torch.float)
    else:
        cube_tensor1=cube_tensor1.to(dtype=torch.float)
        cube_tensor2=cube_tensor2.to(dtype=torch.float)
    return cube_tensor1, cube_tensor2

class PE_dataset(Dataset):
    def __init__(self, label_file, img_dir, transform=None, target_transform=None, return_filename=False):
        self.img_labels = pd.read_csv(label_file, header=None) # list file
        self.img_dir   = img_dir # data folder
        self.transform = transform
        self.target_transform = target_transform
        self.return_filename = return_filename

    def __len__(self): # total data number
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        filename = self.img_labels.iloc[idx, 0]
        print(img_path)
        volume = readmat(img_path)
        label = self.img_labels.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.float)

        if self.transform:
            volume = self.transform(volume)
        if self.target_transform:
            label = self.target_transform(label)
        if self.return_filename:
            return volume, label, filename
        return volume, label

class PE_dataset_2(Dataset):
    def __init__(self, label_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(label_file, header=None) # list file
        self.img_dir   = img_dir # data folder
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self): # total data number
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        print(img_path)
        volume1, volume2 = readmat_2(img_path)
        # print(torch.min(volume1), torch.max(volume1))
        # assert False
        label = self.img_labels.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.float)

        if self.transform:
            volume1 = self.transform(volume1)
            volume2 = self.transform(volume2)
        if self.target_transform:
            label = self.target_transform(label)

        return volume1, volume2, label

class PE_perc_dataset(Dataset):
    def __init__(self, label_file, img_dir):
        self.img_labels = pd.read_csv(label_file, header=None) # list file
        self.img_dir   = img_dir # data folder

    def __len__(self): # total data number
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        pt00, pt50 = readmat_2(img_path)
        label = self.img_labels.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.float)
        return pt00, pt50, label