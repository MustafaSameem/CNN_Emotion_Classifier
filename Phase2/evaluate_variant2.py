import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from cnn_variant2 import CNNVariant2  

#Seed to reproduce consistent results
seed = 40
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = '../imageClasses'
model_path = 'best_model_variant2.pth'
num_classes = 4

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def evaluate_model():
    #Use variant2
    model = CNNVariant2()  
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    #Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    #Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, ['Angry', 'Engaged', 'Happy', 'Neutral'], rotation=45)
    plt.yticks(tick_marks, ['Angry', 'Engaged', 'Happy', 'Neutral'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', color='black')
            
    plt.tight_layout()
    plt.show()

    #Calculate and print accuracy
    accuracy = np.trace(cm) / float(np.sum(cm))
    print(f'Accuracy of the model on the test images: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    evaluate_model()
