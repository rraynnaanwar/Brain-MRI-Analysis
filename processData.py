import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

class BrainMRIData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    
data_dir = r'C:\Users\rrayn\OneDrive\Desktop\Personal-Project\Brain MRI Analysis\Brain-MRI-Analysis\Br35H-Mask-RCNN\TRAIN'

def processData():
    
    # Get a dictionary associating target values with folder names
    #holds a dictionary mapping 0-no, 1-yes
    target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
    transform = transforms.Compose([transforms.Resize((225,225)), transforms.ToTensor()])
    #dataset now converted to tensors
    dataset = BrainMRIData(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    

if __name__=="__main__":
    processData()