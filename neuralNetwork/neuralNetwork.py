import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
    
class BrainMRIClassifier(nn.Module):
    #where we define all the parts of the model
    def __init__(self):
        
