from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append('../')

class BrainMRIData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
        self.imagePaths=  [item[0] for item in self.data.imgs]
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image,label =self.data[idx]
        imageName = os.path.basename(self.imagePaths[idx])
        return image,label, imageName
    
    @property
    def classes(self):
        return self.data.classes