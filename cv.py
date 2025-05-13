# This code in entirely ChatGPT generated. I did this because I really want to get inducted into ACM. I assure that I am actively learning Machine Learning and Deep Learning.
# I apologise for doing this
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5
model_save_path = "fashion_cnn.pth"

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes

# ------------------- Model 1: Basic CNN -------------------
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------- Model 2: Modified ResNet18 -------------------
class ResNet18Modified(nn.Module):
    def __init__(self, use_pretrained=False):
        super(ResNet18Modified, self).__init__()
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        self.model = resnet18(weights=weights)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)

# ------------------- Training Function -------------------
def train_model(model, train_loader, test_loader, model_name):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'{model_name} Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

# ------------------- Evaluation Function -------------------
def evaluate_model(model, test_loader, model_name):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

    return accuracy

# ------------------- Main Execution -------------------
if __name__ == "__main__":
    use_pretrained_resnet = True  # Set to False to train ResNet18 from scratch

    print("Training Basic CNN...")
    basic_model = train_model(BasicCNN(), train_loader, test_loader, "Basic CNN")

    print("Training ResNet18...")
    resnet_model = train_model(ResNet18Modified(use_pretrained=use_pretrained_resnet), train_loader, test_loader, "ResNet18")

    print("Evaluating Basic CNN...")
    acc1 = evaluate_model(basic_model, test_loader, "Basic CNN")

    print("Evaluating ResNet18...")
    acc2 = evaluate_model(resnet_model, test_loader, "ResNet18")

    # Save best model
    best_model = basic_model if acc1 > acc2 else resnet_model
    torch.save(best_model.state_dict(), model_save_path)
    print("Best model saved to", model_save_path)

    # Load saved model for future inference
    print("Loading best model from disk...")
    loaded_model = BasicCNN() if acc1 > acc2 else ResNet18Modified(use_pretrained=use_pretrained_resnet)
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.to(device)
    loaded_model.eval()
    print("Model loaded and ready for inference.")
