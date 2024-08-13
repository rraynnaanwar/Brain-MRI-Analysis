import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sys
sys.path.append('../')
from CustomDataclass import BrainMRIData
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image



def processData(dir):
    transform = transforms.Compose([transforms.Resize((225,225)), transforms.ToTensor()])
    dataSet = BrainMRIData(dir, transform)
    return dataSet

    
