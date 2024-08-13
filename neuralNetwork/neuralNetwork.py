import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
    
class BrainMRIClassifier(nn.Module):
    #where we define all the parts of the model
    def __init__(self, numClasses):
        super(BrainMRIClassifier, self).__init__()

        #first convolutional layer, takes in 3 input channels (RGB)
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.conv_layer4 = nn.Conv2d(in_channels =192, out_channels=384, kernel_size=3, padding = 1)
        self.relu4 = nn.ReLU()
        self.max_pool4 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.conv_layer5 = nn.Conv2d(in_channels = 384, out_channels= 256, kernel_size=3, padding = 1)
        self.relu5  = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride = 2)

        self.conv_layer6 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, padding = 1)
        self.relu6 = nn.ReLU()
        self.max_pool6 = nn.MaxPool2d(kernel_size=3, stride = 2)

        self.dropout6 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()

        self.fc7 = nn.Linear(in_features=1024,out_features=512)
        self.relu7 = nn.ReLU()
        
        self.fc8 = nn.Linear(in_features=512, out_features=256)
        self.relu8 = nn.ReLU()

        self.dropout9 = nn.Dropout(p=0.5)
        self.fc9 = nn.Linear(in_features=256, out_features=numClasses)

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv_layer1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        
        x = self.conv_layer2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv_layer3(x)
        x = self.relu3(x)
        x = self.max_pool3(x)

        x = self.conv_layer4(x)
        x = self.relu4(x)
        x = self.max_pool4(x)

        x = self.conv_layer5(x)
        x = self.relu5(x)
        x = self.max_pool5(x)

        x = self.conv_layer6(x)
        x = self.relu6(x)
        x = self.max_pool6(x)

        # Flatten the output for the fully connected layers
        x = self.flatten(x)
        
        # Forward pass through the fully connected layers
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)
        x = self.relu8(x)
        x = self.dropout9(x)
        x = self.fc9(x)

        return x



