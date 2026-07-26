# ==============================================
# Eksperimen #009: FractalMLP v2 — Koreksi DWT & Perbandingan Adil
# ==============================================

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time, numpy as np, matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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
        x = x.view(-1, 784); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = self.fc3(x); return x

class RegularMLP_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64); self.fc2 = nn.Linear(64, 32); self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = x.view(-1, 784); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = self.fc3(x); return x

class FractalLinear(nn.Module):
    def __init__(self, in_features, out_features, freeze_detail=True):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.h_half = out_features//2; self.w_half = in_features//2
        self.LL = nn.Parameter(torch.randn(self.h_half, self.w_half)*0.1)
        self.LH = nn.Parameter(torch.randn(self.h_half, self.w_half)*0.1)
        self.HL = nn.Parameter(torch.randn(self.h_half, self.w_half)*0.1)
        self.HH = nn.Parameter(torch.randn(self.h_half, self.w_half)*0.1)
        if freeze_detail:
            self.LH.requires_grad=False; self.HL.requires_grad=False; self.HH.requires_grad=False
        self.bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        top = torch.empty(self.h_half, self.w_half*2, device=x.device)
        top[:,0::2]=self.LL; top[:,1::2]=self.LH
        bottom = torch.empty(self.h_half, self.w_half*2, device=x.device)
        bottom[:,0::2]=self.HL; bottom[:,1::2]=self.HH
        W = torch.empty(self.h_half*2, self.w_half*2, device=x.device)
        W[0::2,:]=top; W[1::2,:]=bottom
        return F.linear(x, W, self.bias)

class FractalMLP(nn.Module):
    def __init__(self, freeze_detail=True):
        super().__init__()
        self.fc1 = FractalLinear(784, 256, freeze_detail)
        self.fc2 = FractalLinear(256, 128, freeze_detail)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 784); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = self.fc3(x); return x

def train_model(model, epochs=20, lr=1e-3):
    model.to(device); optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(); history = {'train_loss':[], 'test_acc':[], 'time':[]}
    for ep in range(epochs):
        model.train(); t0=time.time(); total_loss=0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad()
            loss=criterion(model(x),y); loss.backward(); optimizer.step()
            total_loss+=loss.item()*x.size(0)
        train_loss=total_loss/len(train_loader.dataset)
        model.eval(); correct=0
        with torch.no_grad():
            for x,y in test_loader:
                x,y=x.to(device),y.to(device); pred=model(x).argmax(dim=1)
                correct+=(pred==y).sum().item()
        acc=correct/len(test_loader.dataset); elapsed=time.time()-t0
        history['train_loss'].append(train_loss); history['test_acc'].append(acc); history['time'].append(elapsed)
        print(f"Epoch {ep+1:2d} | Loss: {train_loss:.4f} | Test Acc: {acc:.4f} | Time: {elapsed:.2f}s")
    return history

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n=== 1. Regular MLP (235K) ===")
model_reg=RegularMLP().to(device); print(f"Trainable params: {count_trainable(model_reg)}")
hist_reg=train_model(model_reg)

print("\n=== 2. Regular MLP Small (52K) ===")
model_small=RegularMLP_Small().to(device); print(f"Trainable params: {count_trainable(model_small)}")
hist_small=train_model(model_small)

print("\n=== 3. Fractal MLP (details frozen, 60K) ===")
model_frac_frozen=FractalMLP(freeze_detail=True).to(device)
print(f"Trainable params: {count_trainable(model_frac_frozen)}")
hist_frac_frozen=train_model(model_frac_frozen)

print("\n=== 4. Fractal MLP (all trainable, 235K) ===")
model_frac_full=FractalMLP(freeze_detail=False).to(device)
print(f"Trainable params: {count_trainable(model_frac_full)}")
hist_frac_full=train_model(model_frac_full)

epochs=range(1,21)
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(epochs,hist_reg['test_acc'],label='Regular 235K')
plt.plot(epochs,hist_small['test_acc'],label='Regular 52K')
plt.plot(epochs,hist_frac_frozen['test_acc'],label='Fractal 60K (frozen)')
plt.plot(epochs,hist_frac_full['test_acc'],label='Fractal 235K')
plt.xlabel('Epoch'); plt.ylabel('Test Accuracy'); plt.title('MNIST Accuracy'); plt.legend(); plt.grid(True,alpha=0.3)
plt.subplot(1,2,2)
labels=['Reg 235K','Reg 52K','Frac 60K','Frac 235K']
times=[np.mean(hist_reg['time']),np.mean(hist_small['time']),np.mean(hist_frac_frozen['time']),np.mean(hist_frac_full['time'])]
accs=[hist_reg['test_acc'][-1],hist_small['test_acc'][-1],hist_frac_frozen['test_acc'][-1],hist_frac_full['test_acc'][-1]]
colors=['blue','cyan','orange','green']
bars=plt.bar(labels,times,color=colors)
for bar,t in zip(bars,times): plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,f'{t:.1f}s',ha='center')
plt.ylabel('Avg Time/Epoch (s)'); plt.title('Training Speed')
for i,(bar,acc) in enumerate(zip(bars,accs)):
    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()/2,f'Acc:{acc*100:.1f}%',ha='center',color='white',fontweight='bold')
plt.tight_layout(); plt.savefig('exp009_fractal_v2.png'); plt.show()
