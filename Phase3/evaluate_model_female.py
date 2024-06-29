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
from sklearn.metrics import precision_recall_fscore_support

# Seed to reproduce consistent results
seed = 40
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = '../imageBiasFemaleAugmented'
model_path = 'best_model_female_augmented.pth'
num_classes = 4

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8 * 8 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

def evaluate_on_dataset():
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())
    
    print(f'Test Accuracy of the model on the test images: {100 * correct / total} %')

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, ['Angry', 'Engaged', 'Happy', 'Neutral'], rotation=45)
    plt.yticks(tick_marks, ['Angry', 'Engaged', 'Happy', 'Neutral'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', color='black')

    plt.tight_layout()
    plt.show()

    # Calculate and print precision, recall, and F1-score
    precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    print(f'Macro-averaged Precision: {precision:.4f}')
    print(f'Macro-averaged Recall: {recall:.4f}')
    print(f'Macro-averaged F1-score: {fscore:.4f}')
    precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_predictions, average='micro')
    print(f'Micro-averaged Precision: {precision:.4f}')
    print(f'Micro-averaged Recall: {recall:.4f}')
    print(f'Micro-averaged F1-score: {fscore:.4f}')

if __name__ == '__main__':
    evaluate_on_dataset()

    # Evaluate on a single image
    # image_path = './path_to_image.jpg'  # Specify the path to your image
    # evaluate_on_single_image(image_path)
