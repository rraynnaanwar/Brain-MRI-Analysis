from dataProcessing import processData
from CustomDataclass import BrainMRIData
from torch.utils.data import Dataset, DataLoader
from neuralNetwork import BrainMRIClassifier
from torch import nn
import logging
import torch 
train_dir = r'C:\Users\rrayn\OneDrive\Desktop\Personal-Project\Brain MRI Analysis\Brain-MRI-Analysis\Br35H-Mask-RCNN\TRAIN'
val_dir = r'C:\Users\rrayn\OneDrive\Desktop\Personal-Project\Brain MRI Analysis\Brain-MRI-Analysis\Br35H-Mask-RCNN\VAL'
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
def main():
    trainSet = processData(train_dir)
    valSet = processData(val_dir)
    trainLoader = DataLoader(trainSet, batch_size = 64, shuffle=True)
    valLoader = DataLoader(valSet, batch_size = 64, shuffle=True)
    model = BrainMRIClassifier(numClasses=2).to('cuda')
    lossFunction = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for images, labels in trainLoader:
            images, labels = images.to('cuda'), labels.to('cuda')  # Move to GPU
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            outputs = model(images)  # Forward pass
            loss = lossFunction(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainLoader)
        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # No gradient calculation during validation
            for images, labels in valLoader:
                images, labels = images.to('cuda'), labels.to('cuda')
                
                outputs = model(images)
                loss = lossFunction(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(valLoader)
        val_accuracy = 100 * correct / total
        
        # Log training and validation information
        logging.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

if __name__=='__main__':
    main()
    