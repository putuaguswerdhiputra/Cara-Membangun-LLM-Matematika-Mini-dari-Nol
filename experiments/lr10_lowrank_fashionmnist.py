# ==============================================
# Eksperimen #010: Low-Rank MLP pada Fashion-MNIST
# ==============================================

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time, numpy as np, matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_set = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST('.', train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=512, num_workers=2)

class RegularMLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden); self.fc2 = nn.Linear(hidden, 10)
    def forward(self, x):
        x = x.view(-1, 784); x = F.relu(self.fc1(x)); x = self.fc2(x); return x

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features=in_features; self.out_features=out_features; self.rank=rank
        self.U=nn.Parameter(torch.randn(out_features,rank)*0.1)
        self.V=nn.Parameter(torch.randn(rank,in_features)*0.1)
        self.bias=nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        W=self.U@self.V
        return F.linear(x, W, self.bias)

class LowRankMLP(nn.Module):
    def __init__(self, hidden=256, rank=16):
        super().__init__()
        self.fc1=LowRankLinear(784, hidden, rank); self.fc2=nn.Linear(hidden, 10)
    def forward(self, x):
        x=x.view(-1,784); x=F.relu(self.fc1(x)); x=self.fc2(x); return x

def train_model(model, epochs=20, lr=1e-3):
    model.to(device); opt=optim.AdamW(model.parameters(), lr=lr)
    crit=nn.CrossEntropyLoss(); history={'train_loss':[],'test_acc':[],'time':[]}
    for ep in range(epochs):
        model.train(); t0=time.time(); total_loss=0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            loss=crit(model(x),y); loss.backward(); opt.step()
            total_loss+=loss.item()*x.size(0)
        train_loss=total_loss/len(train_loader.dataset)
        model.eval(); correct=0
        with torch.no_grad():
            for x,y in test_loader:
                x,y=x.to(device),y.to(device); pred=model(x).argmax(dim=1)
                correct+=(pred==y).sum().item()
        acc=correct/len(test_loader.dataset); t_ep=time.time()-t0
        history['train_loss'].append(train_loss); history['test_acc'].append(acc); history['time'].append(t_ep)
        print(f"Ep {ep+1:2d} | Loss: {train_loss:.4f} | Test Acc: {acc:.4f} | Time: {t_ep:.2f}s")
    return history

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

configs = {
    "Regular 256": (RegularMLP(hidden=256), "blue"),
    "Regular 64":  (RegularMLP(hidden=64), "cyan"),
    "LowRank r=8": (LowRankMLP(hidden=256, rank=8), "orange"),
    "LowRank r=16": (LowRankMLP(hidden=256, rank=16), "green"),
    "LowRank r=32": (LowRankMLP(hidden=256, rank=32), "red"),
}
histories={}; param_counts={}
for name,(model,color) in configs.items():
    pcount=count_params(model); param_counts[name]=pcount
    print(f"\n=== {name} (params: {pcount}) ===")
    histories[name]=train_model(model)

fig,axes=plt.subplots(1,3,figsize=(18,5))
for name,(_,color) in configs.items():
    axes[0].plot(histories[name]['test_acc'],label=f"{name} ({param_counts[name]} params)",color=color)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test Accuracy'); axes[0].set_title('Fashion-MNIST Accuracy'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
for name,(_,color) in configs.items():
    axes[1].plot(histories[name]['time'],label=name,color=color)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Time (s)'); axes[1].set_title('Training Time'); axes[1].legend(); axes[1].grid(True,alpha=0.3)
names=list(configs.keys()); accs=[histories[n]['test_acc'][-1] for n in names]; params=[param_counts[n] for n in names]
colors=[c for _,c in configs.values()]
bars=axes[2].bar(names,params,color=colors,alpha=0.6)
axes[2].set_ylabel('Total Parameters'); axes[2].set_title('Parameter Count')
for bar,p,acc in zip(bars,params,accs):
    axes[2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+500,f'{p}\nAcc:{acc*100:.1f}%',ha='center',fontsize=8)
plt.tight_layout(); plt.savefig('exp010_lowrank_fashionmnist.png'); plt.show()
