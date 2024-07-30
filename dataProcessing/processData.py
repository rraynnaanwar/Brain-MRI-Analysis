import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from CustomDataclass import BrainMRIData
from torch.utils.data import Dataset, DataLoader
    
data_dir = r'C:\Users\rrayn\OneDrive\Desktop\Personal-Project\Brain MRI Analysis\Brain-MRI-Analysis\Br35H-Mask-RCNN\TRAIN'

def processData():
    
    # Get a dictionary associating target values with folder names
    #holds a dictionary mapping 0-no, 1-yes
    target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
    transform = transforms.Compose([transforms.Resize((225,225)), transforms.ToTensor()])
    #dataset now converted to tensors
    dataset = BrainMRIData(data_dir, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataLoader
