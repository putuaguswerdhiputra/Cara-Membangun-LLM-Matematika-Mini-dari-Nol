# ==============================================
# Eksperimen #007: Grid Search SPSA Hyperparameters
# ==============================================

import torch, torch.nn as nn, numpy as np, itertools, time
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_data(n=1000):
    torch.manual_seed(42)
    x = (torch.rand(n, 2) * 2 - 1) * np.pi
    y = torch.sin(x[:, 0]) * torch.cos(x[:, 1]) + torch.randn(n) * 0.05
    return x, y.unsqueeze(1)

X, Y = generate_data(); X, Y = X.to(device), Y.to(device)
mean = X.mean(dim=0); X_centered = X - mean
cov = (X_centered.T @ X_centered) / (len(X) - 1)
L = torch.linalg.cholesky(cov)
W_whiten = torch.linalg.inv(L).to(device)
X_white = X_centered @ W_whiten.T

n_train = int(0.8 * len(X))
train_loader = DataLoader(TensorDataset(X_white[:n_train], Y[:n_train]), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_white[n_train:], Y[n_train:]), batch_size=64)

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=16, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x)); x = self.fc2(x); return x

class SPSA:
    def __init__(self, params, a=1e-3, c=1e-3, alpha=0.602, gamma=0.101, A=10):
        self.params = list(params); self.a=a; self.c=c; self.alpha=alpha; self.gamma=gamma; self.A=A
        self.t=0; self.total_dim=sum(p.numel() for p in self.params)
    def _pack(self): return torch.cat([p.data.view(-1) for p in self.params])
    def _unpack(self, vec):
        idx=0
        for p in self.params: n=p.numel(); p.data=vec[idx:idx+n].view_as(p); idx+=n
    def step(self, loss_fn):
        self.t+=1; a_t=self.a/((self.t+self.A)**self.alpha); c_t=self.c/(self.t**self.gamma)
        theta=self._pack(); delta=torch.randint(0,2,(self.total_dim,),device=theta.device)*2-1
        self._unpack(theta+c_t*delta); lp=loss_fn()
        self._unpack(theta-c_t*delta); lm=loss_fn()
        g=(lp-lm)/(2*c_t)*delta; self._unpack(theta-a_t*g)
        return (lp+lm)/2

def evaluate_spsa_config(a, c, epochs=100, seeds=[0,1,2]):
    val_losses = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = MLP().to(device)
        optimizer = SPSA(model.parameters(), a=a, c=c)
        best_val = float('inf')
        for ep in range(epochs):
            model.train()
            for xb, yb in train_loader:
                loss_fn = lambda: torch.mean((model(xb)-yb)**2)
                optimizer.step(loss_fn)
            model.eval()
            with torch.no_grad():
                total_val = sum(torch.sum((model(xb)-yb)**2).item() for xb,yb in val_loader)
                val_loss = total_val / len(val_loader.dataset)
            best_val = min(best_val, val_loss)
            if seed==seeds[0] and ep%20==0: print(f"  seed {seed} epoch {ep:3d}: val={val_loss:.5f}, best={best_val:.5f}")
        val_losses.append(best_val)
        print(f"  seed {seed}: best val={best_val:.5f}")
    return np.mean(val_losses), np.std(val_losses)

print("\n=== Starting Grid Search ===")
a_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
c_values = [1e-2, 1e-3, 1e-4]
results = {}; best_config = None; best_loss = float('inf')
start_time = time.time()
for a, c in itertools.product(a_values, c_values):
    print(f"\nTesting a={a:.0e}, c={c:.0e}")
    mean_val, std_val = evaluate_spsa_config(a, c, epochs=100, seeds=[0,1,2])
    results[(a,c)] = (mean_val, std_val)
    if mean_val < best_loss: best_loss = mean_val; best_config = (a,c)
    print(f"  => Mean best val loss: {mean_val:.4f} ± {std_val:.4f}")
print(f"\nGrid search finished in {(time.time()-start_time)/60:.1f} minutes.")
print(f"Best: a={best_config[0]}, c={best_config[1]} with val loss={best_loss:.4f}")
print("\nAll results (sorted):")
for (a,c), (mv,sv) in sorted(results.items(), key=lambda x: x[1][0]):
    print(f"a={a:.0e}, c={c:.0e} : {mv:.4f} ± {sv:.4f}")
