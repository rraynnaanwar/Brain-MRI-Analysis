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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.dropout3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.dropout4 = nn.Dropout(p=0.2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=25600, out_features=512)  # Adjust the input size based on your image size and pooling
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=numClasses)

    def forward(self, x):
        x = self.conv1(x)
        #print("Shape after conv1:", x.shape)
        x = torch.relu(x)
        x = self.pool(x)
        #print("Shape after pool1:", x.shape)
        x = self.dropout1(x)

        x = self.conv2(x)
        #print("Shape after conv2:", x.shape)
        x = torch.relu(x)
        x = self.pool(x)
        #print("Shape after pool2:", x.shape)
        x = self.dropout2(x)

        x = self.conv3(x)
       # print("Shape after conv3:", x.shape)
        x = torch.relu(x)
        x = self.pool(x)
       # print("Shape after pool3:", x.shape)
        x = self.dropout3(x)

        x = self.conv4(x)
        #print("Shape after conv4:", x.shape)
        x = torch.relu(x)
        x = self.pool(x)
       # print("Shape after pool4:", x.shape)
        x = self.dropout4(x)

        x = self.flatten(x)
        #print("Shape after flatten:", x.shape)
        x = self.fc1(x)
        #print("Shape after fc1:", x.shape)
        x = torch.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        #print("Shape after fc2:", x.shape)
        return x




