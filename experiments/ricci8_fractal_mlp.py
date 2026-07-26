# ==============================================
# Eksperimen #008: FractalMLP vs RegularMLP (AdamW) on MNIST
# ==============================================

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time, numpy as np, matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
train_set = datasets.MNIST('.', train=True, download=True, transform=transform)
test_set = datasets.MNIST('.', train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512)

class RegularMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256); self.fc2 = nn.Linear(256, 128); self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = self.fc3(x); return x

class FractalLinear(nn.Module):
    def __init__(self, in_features, out_features, freeze_detail=True):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.shape = (out_features, in_features)
        self.LL = nn.Parameter(torch.randn(out_features//2, in_features//2)*0.1)
        self.LH = nn.Parameter(torch.randn(out_features//2, in_features//2)*0.1)
        self.HL = nn.Parameter(torch.randn(out_features//2, in_features//2)*0.1)
        self.HH = nn.Parameter(torch.randn(out_features//2, in_features//2)*0.1)
        if freeze_detail:
            self.LH.requires_grad = False; self.HL.requires_grad = False; self.HH.requires_grad = False
        self.bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        top = torch.cat([self.LL, self.LH], dim=1)
        bottom = torch.cat([self.HL, self.HH], dim=1)
        W = torch.cat([top, bottom], dim=0)
        return F.linear(x, W, self.bias)

class FractalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FractalLinear(784, 256); self.fc2 = FractalLinear(256, 128); self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = self.fc3(x); return x

def train(model, train_loader, test_loader, epochs=10, lr=1e-3):
    model.train(); optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss':[], 'test_acc':[], 'time':[]}
    for ep in range(epochs):
        t0 = time.time(); total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        model.eval(); correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
        acc = correct / len(test_loader.dataset)
        t_epoch = time.time() - t0
        history['train_loss'].append(train_loss); history['test_acc'].append(acc); history['time'].append(t_epoch)
        print(f'Epoch {ep+1:2d}: Loss={train_loss:.4f}, Test Acc={acc:.4f}, Time={t_epoch:.3f}s')
    return history

def count_params(model, trainable_only=True):
    if trainable_only: return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

print("Regular MLP:")
model_reg = RegularMLP().to(device)
print(f"  Trainable params: {count_params(model_reg)}")
hist_reg = train(model_reg, train_loader, test_loader, epochs=10)

print("\nFractal MLP (details frozen):")
model_frac = FractalMLP().to(device)
print(f"  Trainable params: {count_params(model_frac)} (total: {count_params(model_frac, False)})")
hist_frac = train(model_frac, train_loader, test_loader, epochs=10)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hist_reg['test_acc'], label='Regular MLP')
plt.plot(hist_frac['test_acc'], label='Fractal MLP')
plt.xlabel('Epoch'); plt.ylabel('Test Accuracy'); plt.title('Accuracy'); plt.legend(); plt.grid(alpha=0.3)
plt.subplot(1,2,2)
plt.plot(hist_reg['time'], label='Regular MLP')
plt.plot(hist_frac['time'], label='Fractal MLP')
plt.xlabel('Epoch'); plt.ylabel('Time (s)'); plt.title('Training Speed'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('exp008_fractal_adamw.png'); plt.show()
