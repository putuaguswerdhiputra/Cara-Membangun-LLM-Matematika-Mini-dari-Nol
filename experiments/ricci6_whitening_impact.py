# ==============================================
# Eksperimen #006: Dampak Whitening pada Optimasi
# ==============================================

import torch, torch.nn as nn, numpy as np, time, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

def gen_data(n=1000):
    x = (torch.rand(n,2)*2-1)*np.pi
    y = torch.sin(x[:,0])*torch.cos(x[:,1]) + torch.randn(n)*0.05
    return x, y.unsqueeze(1)

X, Y = gen_data(); X, Y = X.to(device), Y.to(device)
mean = X.mean(0); Xc = X - mean
cov = (Xc.T @ Xc) / (len(X)-1)
L = torch.linalg.cholesky(cov)
W_whiten = torch.linalg.inv(L).to(device)
X_white = Xc @ W_whiten.T

n_train = int(0.8*len(X))
train_raw = DataLoader(TensorDataset(X[:n_train], Y[:n_train]), batch_size=64, shuffle=True)
val_raw = DataLoader(TensorDataset(X[n_train:], Y[n_train:]), batch_size=64)
train_white = DataLoader(TensorDataset(X_white[:n_train], Y[:n_train]), batch_size=64, shuffle=True)
val_white = DataLoader(TensorDataset(X_white[n_train:], Y[n_train:]), batch_size=64)

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=16, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def train(model, train_ldr, val_ldr, opt_class, opt_kwargs, epochs=100):
    model.train(); opt = opt_class(model.parameters(), **opt_kwargs)
    hist = {'train_loss':[], 'val_loss':[], 'time':[]}
    for ep in range(epochs):
        t0 = time.time(); total_loss = 0.0
        for xb,yb in train_ldr:
            def closure(): return torch.mean((model(xb)-yb)**2)
            if isinstance(opt, torch.optim.Optimizer):
                opt.zero_grad(); loss = closure(); loss.backward(); opt.step()
                total_loss += loss.item()*xb.size(0)
            else:
                loss = opt.step(closure); total_loss += loss.item()*xb.size(0)
        train_loss = total_loss/len(train_ldr.dataset)
        model.eval()
        with torch.no_grad():
            vloss = sum(torch.mean((model(xb)-yb)**2).item()*xb.size(0) for xb,yb in val_ldr)/len(val_ldr.dataset)
        hist['train_loss'].append(train_loss); hist['val_loss'].append(vloss)
        hist['time'].append(time.time()-t0)
        if ep%20==0: print(f'{ep:3d} | Train: {train_loss:.5f} | Val: {vloss:.5f} | Time: {hist["time"][-1]:.3f}s')
    return hist

class SPSA:
    def __init__(self, params, a=1e-3, c=1e-3, alpha=0.602, gamma=0.101, A=10):
        self.params = list(params); self.a=a; self.c=c; self.alpha=alpha; self.gamma=gamma; self.A=A
        self.t=0; self.dim=sum(p.numel() for p in self.params)
    def step(self, loss_fn):
        self.t+=1; at=self.a/((self.t+self.A)**self.alpha); ct=self.c/(self.t**self.gamma)
        theta=torch.cat([p.data.view(-1) for p in self.params])
        delta=torch.randint(0,2,(self.dim,),device=theta.device)*2-1
        self._set(theta+ct*delta); lp=loss_fn()
        self._set(theta-ct*delta); lm=loss_fn()
        g=(lp-lm)/(2*ct)*delta; self._set(theta - at*g)
        return (lp+lm)/2
    def _set(self, vec):
        idx=0
        for p in self.params: n=p.numel(); p.data=vec[idx:idx+n].view_as(p); idx+=n

epochs=150
configs = {
    'AdamW Raw': (MLP().to(device), train_raw, val_raw, torch.optim.AdamW, {'lr':1e-3}),
    'AdamW White': (MLP().to(device), train_white, val_white, torch.optim.AdamW, {'lr':1e-3}),
    'SPSA Raw': (MLP().to(device), train_raw, val_raw, SPSA, {'a':1e-3,'c':1e-3}),
    'SPSA White': (MLP().to(device), train_white, val_white, SPSA, {'a':1e-3,'c':1e-3}),
}
results = {}
for name, (model, tr_ldr, v_ldr, opt_cls, kwargs) in configs.items():
    print(f'\n=== {name} ===')
    results[name] = train(model, tr_ldr, v_ldr, opt_cls, kwargs, epochs=epochs)

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
for name, hist in results.items(): plt.plot(hist['val_loss'], label=name)
plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Val Loss'); plt.title('Convergence'); plt.legend(); plt.grid(alpha=0.3)
plt.subplot(1,2,2)
mean_t = [np.mean(h['time']) for h in results.values()]
plt.bar(results.keys(), mean_t, color=['blue','cyan','orange','yellow'])
plt.ylabel('Avg time/epoch (s)'); plt.title('Speed'); plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig('exp006_optimization_impact.png'); plt.show()
