import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import random

# Seed to reproduce consistent results
seed = 40
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_epochs = 10
num_classes = 4
learning_rate = 0.001
patience = 3

data_dir = '../imageClasses'

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(root=data_dir, transform=transform)


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


def train_and_evaluate(train_loader, val_loader, test_loader):
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = 'best_model_kfold.pth'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch + 1 >= 10 and epochs_no_improve >= patience:
            break

    model.load_state_dict(torch.load(best_model_path))

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_micro, recall_micro, fscore_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

    return accuracy, precision_macro, recall_macro, fscore_macro, precision_micro, recall_micro, fscore_micro


if __name__ == "__main__":
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'fscore_macro': [],
        'precision_micro': [],
        'recall_micro': [],
        'fscore_micro': []
    }

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        train_val_set = Subset(dataset, train_val_idx)
        test_set = Subset(dataset, test_idx)

        train_size = int(0.85 * len(train_val_set))
        val_size = len(train_val_set) - train_size

        train_set, val_set = random_split(train_val_set, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)

        accuracy, precision_macro, recall_macro, fscore_macro, precision_micro, recall_micro, fscore_micro = train_and_evaluate(
            train_loader, val_loader, test_loader)

        results['accuracy'].append(accuracy)
        results['precision_macro'].append(precision_macro)
        results['recall_macro'].append(recall_macro)
        results['fscore_macro'].append(fscore_macro)
        results['precision_micro'].append(precision_micro)
        results['recall_micro'].append(recall_micro)
        results['fscore_micro'].append(fscore_micro)

        print(
            f'Fold {fold + 1}: Accuracy = {accuracy:.4f}, Precision (Macro) = {precision_macro:.4f}, Recall (Macro) = {recall_macro:.4f}, F1-score (Macro) = {fscore_macro:.4f}')
        print(
            f'Fold {fold + 1}: Precision (Micro) = {precision_micro:.4f}, Recall (Micro) = {recall_micro:.4f}, F1-score (Micro) = {fscore_micro:.4f}')

    print('10-Fold Cross-Validation Results:')
    print(f"Average Accuracy: {np.mean(results['accuracy']):.4f}")
    print(f"Average Precision (Macro): {np.mean(results['precision_macro']):.4f}")
    print(f"Average Recall (Macro): {np.mean(results['recall_macro']):.4f}")
    print(f"Average F1-score (Macro): {np.mean(results['fscore_macro']):.4f}")
    print(f"Average Precision (Micro): {np.mean(results['precision_micro']):.4f}")
    print(f"Average Recall (Micro): {np.mean(results['recall_micro']):.4f}")
    print(f"Average F1-score (Micro): {np.mean(results['fscore_micro']):.4f}")