import wandb
wandb.login()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torchvision import datasets
import zipfile
from torch.cuda.amp import autocast, GradScaler
try:
    import torchmetrics
except ImportError:
    !pip install torchmetrics
from torchmetrics.classification import Accuracy
import torch, gc
from tqdm import tqdm

gc.collect()
torch.cuda.empty_cache()

# 1 data preparation

# importing zip file from drive
from google.colab import drive
drive.mount('/content/drive')

# dataset extraction
zip_file_path = '/content/drive/MyDrive/nature_12K.zip'
extract_dir = '/content/nature_12K/'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Dataset extracted to {extract_dir}")


# data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# splitting dataset
from torch.utils.data import Subset

dataset= ImageFolder(root='/content/nature_12K/inaturalist_12K/train', transform=train_transform)
targets = [sample[1] for sample in dataset]
num_classes = len(dataset.classes)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X=targets, y=targets))

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#######

## 2 CNN Model

## CNN model with batch normalization

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.gap(x)
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x

######

## 3 defining loss and optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
acc_metric = Accuracy(task="multiclass", num_classes=10).to(device)

######

wandb.init(project = 'cnn_inaturalist', config={
   'epochs':10,
   'batch_size':64,
   'learning_rate':0.001,
   'optimizer': 'Adam'
})



## 4 training the model

epochs = 10  # (to be changed for better acc)

# Training the model
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc='Training', total=len(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc
######

_## 5 evaluation on validation data

def validate(model, val_loader, criterion, acc_metric, device):
    model.eval()
    val_loss = 0.0
    acc_metric.reset()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", total=len(val_loader)):
            images, labels = images.to(device),labels.to(device)
            outputs = model(images)
            loss= criterion(outputs,labels)
            val_loss += loss.item()
            acc_metric.update(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_val_loss = val_loss/len(val_loader)
    val_accuracy = acc_metric.compute().item()
    return avg_val_loss, val_accuracy
for epoch in range(wandb.config.epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, acc_metric, device)
    print(f"Epoch [{epoch+1}/{wandb.config.epochs}] Train_Loss: {train_loss:.4f} Train_Accuracy: {train_acc:.2f}%, Val_Loss: {val_loss:.2f}%, Val_Accuracy: {val_acc:.2f}%")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })
wandb.finish()