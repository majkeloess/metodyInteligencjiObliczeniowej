import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

# Sprawdzenie dostępności GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

# Przygotowanie danych
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Nazwy klas
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Definicje trzech różnych architektury sieci CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResidualCNN(nn.Module):
    def __init__(self):
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pierwszy blok rezydualny
        identity = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(identity)))
        out = out + identity
        out = self.pool(out)
        
        # Drugi blok rezydualny
        identity = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(identity)))
        out = out + identity
        out = self.pool(out)
        
        out = out.view(-1, 64 * 7 * 7)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Funkcja do trenowania modelu
def train_model(model, criterion, optimizer, num_epochs=10):
    model.to(device)
    train_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    return train_losses

# Funkcja do testowania modelu
def test_model(model):
    model.eval()
    y_true = []
    y_pred = []
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Zapisz rzeczywiste i przewidywane etykiety
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            # Znajdź źle sklasyfikowane obrazki
            mask = (predicted != labels)
            if torch.any(mask):
                misclassified_idx = torch.where(mask)[0]
                for idx in misclassified_idx:
                    if len(misclassified_images) < 5:  # Ograniczenie liczby zapisanych obrazków
                        misclassified_images.append(inputs[idx].cpu())
                        misclassified_labels.append(labels[idx].item())
                        misclassified_preds.append(predicted[idx].item())
    
    # Oblicz dokładność
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    total = len(y_true)
    accuracy = correct / total
    
    print(f'Dokładność na zbiorze testowym: {accuracy:.4f}')
    
    # Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred)
    
    return cm, misclassified_images, misclassified_labels, misclassified_preds

# Funkcja do wizualizacji
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title('Macierz pomyłek')
    plt.show()

def plot_loss(losses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.grid(True)
    plt.show()

def show_misclassified(images, true_labels, pred_labels):
    if not images:
        print("Brak błędnie sklasyfikowanych obrazów.")
        return
    
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axs = [axs]
    
    for i in range(len(images)):
        img = images[i].squeeze().numpy()
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f'True: {classes[true_labels[i]]}\nPred: {classes[pred_labels[i]]}')
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Trenowanie i testowanie modeli
models = {
    "Simple CNN": SimpleCNN(),
    "Deep CNN": DeepCNN(),
    "Residual CNN": ResidualCNN()
}

num_epochs = 5
criterion = nn.CrossEntropyLoss()

for model_name, model in models.items():
    print(f"\nTrenowanie modelu: {model_name}")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = train_model(model, criterion, optimizer, num_epochs)
    
    print(f"\nTestowanie modelu: {model_name}")
    cm, misclassified_images, misclassified_labels, misclassified_preds = test_model(model)
    
    print("\nMacierz pomyłek:")
    plot_confusion_matrix(cm, classes)
    
    print("\nWykres funkcji straty:")
    plot_loss(losses, f'Funkcja straty - {model_name}')
    
    print("\nŹle sklasyfikowane obrazki:")
    show_misclassified(misclassified_images, misclassified_labels, misclassified_preds)

# Porównanie czasu uczenia na CPU vs GPU dla wybranej architektury (ResidualCNN)
print("\n\n" + "="*50)
print("PORÓWNANIE CZASU UCZENIA NA CPU VS GPU - ResidualCNN")
print("="*50)

selected_model = ResidualCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(selected_model.parameters(), lr=0.001)
num_epochs = 2  # Mniejsza liczba epok dla porównania czasu

# Trenowanie na CPU
device_cpu = torch.device("cpu")
selected_model_cpu = ResidualCNN().to(device_cpu)
optimizer_cpu = optim.Adam(selected_model_cpu.parameters(), lr=0.001)

print("\nRozpoczynam trenowanie na CPU...")
start_time_cpu = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device_cpu), labels.to(device_cpu)
        
        optimizer_cpu.zero_grad()
        outputs = selected_model_cpu(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cpu.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

cpu_time = time.time() - start_time_cpu
print(f"Czas trenowania na CPU: {cpu_time:.2f} sekund")

# Trenowanie na GPU (jeśli dostępne)
if torch.cuda.is_available():
    device_gpu = torch.device("cuda:0")
    selected_model_gpu = ResidualCNN().to(device_gpu)
    optimizer_gpu = optim.Adam(selected_model_gpu.parameters(), lr=0.001)
    
    print("\nRozpoczynam trenowanie na GPU...")
    start_time_gpu = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device_gpu), labels.to(device_gpu)
            
            optimizer_gpu.zero_grad()
            outputs = selected_model_gpu(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_gpu.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    gpu_time = time.time() - start_time_gpu
    print(f"Czas trenowania na GPU: {gpu_time:.2f} sekund")
    
    # Porównanie czasów
    speedup = cpu_time / gpu_time
    print(f"\nPrzyśpieszenie GPU względem CPU: {speedup:.2f}x")
else:
    print("\nGPU nie jest dostępne. Uruchom notebook w środowisku z dostępem do GPU.")
    print("W Google Colab: Runtime->Change Runtime Type i wybierz opcję 'T4 GPU'")

# Wykres porównania czasów (jeśli GPU dostępne)
if torch.cuda.is_available():
    plt.figure(figsize=(10, 5))
    devices = ['CPU', 'GPU']
    times = [cpu_time, gpu_time]
    plt.bar(devices, times, color=['blue', 'orange'])
    plt.title('Porównanie czasu trenowania ResidualCNN na CPU vs GPU')
    plt.ylabel('Czas [s]')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Dodanie wartości czasów nad słupkami
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    plt.show()