import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sys
sys.path.append('../')
from CustomDataclass import BrainMRIData
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image



def processData(dir):
    transform = transforms.Compose([transforms.Resize((200,200)), transforms.ToTensor(), transforms.Normalize(mean=[0.2802, 0.2804, 0.2806], std=[0.2664, 0.2664, 0.2665])])
    dataSet = BrainMRIData(dir, transform)
    return dataSet

    
