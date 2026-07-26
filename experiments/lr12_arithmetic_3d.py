# ==============================================
# Eksperimen #012: Penjumlahan 3-Digit (0-999)
# ==============================================

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, time, matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def generate_data(n_samples=60000):
    a=np.random.randint(0,1000,(n_samples,)); b=np.random.randint(0,1000,(n_samples,)); c=a+b
    def to_digits(x,n_digits):
        return np.stack([(x//10**i)%10 for i in range(n_digits-1,-1,-1)],axis=1)
    X=np.concatenate([to_digits(a,3),to_digits(b,3)],axis=1); Y=to_digits(c,4)
    return X,Y

X,Y=generate_data(60000)
train_set=TensorDataset(torch.tensor(X[:50000],dtype=torch.long),torch.tensor(Y[:50000],dtype=torch.long))
test_set=TensorDataset(torch.tensor(X[50000:],dtype=torch.long),torch.tensor(Y[50000:],dtype=torch.long))
train_loader=DataLoader(train_set,batch_size=256,shuffle=True)
test_loader=DataLoader(test_set,batch_size=512)

class LowRankLinear(nn.Module):
    def __init__(self,in_features,out_features,rank):
        super().__init__()
        self.U=nn.Parameter(torch.randn(out_features,rank)*0.1)
        self.V=nn.Parameter(torch.randn(rank,in_features)*0.1)
        self.bias=nn.Parameter(torch.zeros(out_features))
    def forward(self,x):
        return F.linear(x,self.U@self.V,self.bias)

class ArithmeticMLP(nn.Module):
    def __init__(self,hidden_dims=[256,256],use_lowrank=False,rank=16):
        super().__init__()
        self.embed=nn.Embedding(10,32); in_dim=6*32
        layers=[]; prev_dim=in_dim
        for h_dim in hidden_dims:
            if use_lowrank: layers.append(LowRankLinear(prev_dim,h_dim,rank))
            else: layers.append(nn.Linear(prev_dim,h_dim))
            layers.append(nn.ReLU()); prev_dim=h_dim
        self.body=nn.Sequential(*layers)
        self.head=nn.Linear(prev_dim,4*10)
    def forward(self,x):
        embeds=self.embed(x); flat=embeds.view(x.size(0),-1)
        out=self.body(flat); logits=self.head(out)
        return logits.view(-1,4,10)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model,epochs=30,lr=1e-3,clip=1.0):
    model.to(device); opt=optim.AdamW(model.parameters(),lr=lr)
    crit=nn.CrossEntropyLoss(); hist={'train_loss':[],'test_acc':[],'time':[]}
    for ep in range(epochs):
        model.train(); t0=time.time(); total_loss=0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            logits=model(x); loss=sum(crit(logits[:,i,:],y[:,i]) for i in range(4))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),clip); opt.step()
            total_loss+=loss.item()*x.size(0)
        train_loss=total_loss/len(train_loader.dataset)
        model.eval(); correct=0
        with torch.no_grad():
            for x,y in test_loader:
                x,y=x.to(device),y.to(device); logits=model(x); preds=logits.argmax(dim=2)
                correct+=(preds==y).all(dim=1).sum().item()
        acc=correct/len(test_loader.dataset); t_ep=time.time()-t0
        hist['train_loss'].append(train_loss); hist['test_acc'].append(acc); hist['time'].append(t_ep)
        print(f"Ep {ep+1:2d} | Loss: {train_loss:.4f} | Test Acc: {acc:.4f} | Time: {t_ep:.2f}s")
    return hist

model_reg256=ArithmeticMLP([256,256],False); model_lr16=ArithmeticMLP([256,256],True,16)
model_lr32=ArithmeticMLP([256,256],True,32); model_reg128=ArithmeticMLP([128,128],False)
model_reg64=ArithmeticMLP([64,64],False)
configs={
    "Regular 256": (model_reg256,"blue"), "LowRank r=32": (model_lr32,"green"),
    "LowRank r=16": (model_lr16,"orange"), "Regular 128": (model_reg128,"purple"),
    "Regular 64": (model_reg64,"red"),
}
histories={}; param_counts={}
for name,(model,color) in configs.items():
    pcount=count_params(model); param_counts[name]=pcount
    print(f"\n=== {name} (params: {pcount}) ===")
    histories[name]=train_model(model)

fig,axes=plt.subplots(1,3,figsize=(18,5))
for name,(_,color) in configs.items():
    axes[0].plot(histories[name]['test_acc'],label=f"{name} ({param_counts[name]} params)",color=color)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Full-Sequence Accuracy'); axes[0].set_title('3-Digit Addition'); axes[0].legend(); axes[0].grid(alpha=0.3)
for name,(_,color) in configs.items():
    axes[1].plot(histories[name]['time'],label=name,color=color)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Time (s)'); axes[1].set_title('Training Time'); axes[1].legend(); axes[1].grid(alpha=0.3)
names=list(configs.keys()); accs=[histories[n]['test_acc'][-1] for n in names]; params=[param_counts[n] for n in names]
bars=axes[2].bar(names,params,color=[c for _,c in configs.values()],alpha=0.6)
for bar,p,acc in zip(bars,params,accs):
    axes[2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+500,f'{p}\nAcc:{acc*100:.1f}%',ha='center',fontsize=8)
axes[2].set_ylabel('Total Parameters'); axes[2].set_title('Parameter Count')
plt.tight_layout(); plt.savefig('exp012_add3digit.png'); plt.show()
