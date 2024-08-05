import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sys
sys.path.append('../')
from CustomDataclass import BrainMRIData
from torch.utils.data import Dataset, DataLoader

data_dir = r'C:\Users\rrayn\OneDrive\Desktop\Personal-Project\Brain MRI Analysis\Brain-MRI-Analysis\Br35H-Mask-RCNN\VAL'

def processData():
    transform = transforms.Compose([transforms.Resize((225,225)), transforms.ToTensor()])
    #dataset now converted to tensors
    dataset = BrainMRIData(data_dir, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset
