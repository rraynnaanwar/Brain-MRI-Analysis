from dataProcessing import processData
from CustomDataclass import BrainMRIData
from torch.utils.data import Dataset, DataLoader
from neuralNetwork import BrainMRIClassifier
from torch import nn
import logging
import torch 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
train_dir = r'C:\Users\rrayn\OneDrive\Desktop\Personal-Project\Brain MRI Analysis\Brain-MRI-Analysis\Br35H-Mask-RCNN\TRAIN'
val_dir = r'C:\Users\rrayn\OneDrive\Desktop\Personal-Project\Brain MRI Analysis\Brain-MRI-Analysis\Br35H-Mask-RCNN\VAL'
test_dir = r'C:\Users\rrayn\OneDrive\Desktop\Personal-Project\Brain MRI Analysis\Brain-MRI-Analysis\Br35H-Mask-RCNN\TEST'

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    trainSet = processData(train_dir)
    valSet = processData(val_dir)
    trainLoader = DataLoader(trainSet, batch_size = 64, shuffle=True)
    valLoader = DataLoader(valSet, batch_size = 64, shuffle=True)
    
    
    # this trains the model for the first time
    #trainingLoop(trainLoader, valLoader)


    #load trained model
    model = BrainMRIClassifier(numClasses=2)
    model.load_state_dict(torch.load('BrainMRIClassifier.pt'))
    model.eval().to('cuda')

    testSet = processData(test_dir)
    testLoader = DataLoader(testSet, batch_size = 64, shuffle=True)
    classifyImages(model, testLoader)


def trainingLoop(trainLoader ,valLoader):
    model = BrainMRIClassifier(numClasses=2).to('cuda')
    lossFunction = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    num_epochs = 50

    # training loop
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
        torch.save(model.state_dict(), 'BrainMRIClassifier.pt')



def classifyImages(model, testLoader):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []
    image_names = []  # To store image file names

    with torch.no_grad():  # No need to track gradients during inference
        for batch in testLoader:
            images, labels, names = batch
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_predictions.extend(predicted.cpu().numpy())  # Store predicted labels
            image_names.extend(names)  # Store image names

    # Print each image's name, prediction, and actual label
    for img_name, pred, actual in zip(image_names, all_predictions, all_labels):
        pred_label = 'yes' if pred == 1 else 'no'
        actual_label = 'yes' if actual == 1 else 'no'
        print(f"Image: {img_name} | Predicted: {pred_label} | Actual: {actual_label}")

    # Calculate accuracy
    correct = sum(1 for true, pred in zip(all_labels, all_predictions) if true == pred)
    total = len(all_labels)
    test_accuracy = 100 * correct / total
    logging.info(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Build and display confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no', 'yes'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    return all_predictions


if __name__=='__main__':
    main()
    