# ==============================================
# Eksperimen #004: SynapticLinear – Diagonal Fisher Geometry
# ==============================================

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_dataset(n_samples=2000, noise_std=0.01):
    x = (torch.rand(n_samples, 2) * 2 - 1) * np.pi
    y = torch.sin(x[:, 0]) * torch.cos(x[:, 1]) + torch.randn(n_samples) * noise_std
    return x, y.unsqueeze(1)

class SynapticLinear(nn.Module):
    def __init__(self, in_features, out_features, init_gain=1.0):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.a = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.phi = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.gain = nn.Parameter(torch.ones(out_features) * init_gain)
    def forward(self, x):
        W_real = torch.exp(self.a) * torch.cos(self.phi)
        out = F.linear(x, W_real) * self.gain
        return out

class SynapticMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=1):
        super().__init__()
        self.fc1 = SynapticLinear(in_dim, hidden)
        self.fc2 = SynapticLinear(hidden, hidden)
        self.fc3 = SynapticLinear(hidden, out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x)); x = self.act(self.fc2(x)); x = self.fc3(x); return x

class RegularMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim); self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x)); x = self.act(self.fc2(x)); x = self.fc3(x); return x

class SPSA_Optimizer:
    def __init__(self, params, a=1e-3, c=1e-3, alpha=0.602, gamma=0.101, A=10):
        self.params = list(params); self.a = a; self.c = c
        self.alpha = alpha; self.gamma = gamma; self.A = A; self.t = 0
        self.total_dim = sum(p.numel() for p in self.params)
    def _pack(self):
        return torch.cat([p.data.view(-1) for p in self.params])
    def _unpack(self, vec):
        idx = 0
        for p in self.params: n = p.numel(); p.data = vec[idx:idx+n].view_as(p); idx += n
    def step(self, loss_fn):
        self.t += 1
        a_t = self.a / ((self.t + self.A) ** self.alpha)
        c_t = self.c / (self.t ** self.gamma)
        theta = self._pack()
        delta = torch.randint(0, 2, (self.total_dim,), device=theta.device, dtype=theta.dtype) * 2 - 1
        theta_plus = theta + c_t * delta; theta_minus = theta - c_t * delta
        self._unpack(theta_plus); loss_plus = loss_fn()
        self._unpack(theta_minus); loss_minus = loss_fn()
        g_hat = (loss_plus - loss_minus) / (2 * c_t) * delta
        theta_new = theta - a_t * g_hat
        self._unpack(theta_new)
        return (loss_plus + loss_minus) / 2

def compute_mse(model, x, y):
    return torch.mean((model(x) - y) ** 2)

def train_spsa(model, params, train_loader, val_loader, epochs=100, a=1e-3, c=1e-3):
    opt = SPSA_Optimizer(params, a=a, c=c)
    history = {'train_loss':[], 'val_loss':[], 'time':[]}
    for epoch in range(epochs):
        t0 = time.time(); model.train(); total_loss = 0.0
        for xb, yb in train_loader:
            loss_fn = lambda: compute_mse(model, xb, yb)
            loss_val = opt.step(loss_fn)
            total_loss += loss_val.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            val_loss = sum(compute_mse(model, xb, yb).item() * xb.size(0) for xb, yb in val_loader) / len(val_loader.dataset)
        t_epoch = time.time() - t0
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss); history['time'].append(t_epoch)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {t_epoch:.4f}s')
    return history

def train_adamw(model, train_loader, val_loader, epochs=100, lr=1e-3):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {'train_loss':[], 'val_loss':[], 'time':[]}
    for epoch in range(epochs):
        t0 = time.time(); total_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = compute_mse(model, xb, yb)
            loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            val_loss = sum(compute_mse(model, xb, yb).item() * xb.size(0) for xb, yb in val_loader) / len(val_loader.dataset)
        t_epoch = time.time() - t0
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss); history['time'].append(t_epoch)
        if epoch % 10 == 0: print(f'AdamW Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {t_epoch:.4f}s')
    return history

x_data, y_data = generate_dataset(); x_data, y_data = x_data.to(device), y_data.to(device)
train_size = int(0.8 * len(x_data))
train_loader = DataLoader(TensorDataset(x_data[:train_size], y_data[:train_size]), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(x_data[train_size:], y_data[train_size:]), batch_size=128)
EPOCHS = 150; HIDDEN = 32

print("=== AdamW (RegularMLP) ===")
hist_adam = train_adamw(RegularMLP(in_dim=2, hidden=HIDDEN, out_dim=1).to(device), train_loader, val_loader, epochs=EPOCHS)

print("\n=== SPSA (RegularMLP) ===")
hist_reg = train_spsa(RegularMLP(in_dim=2, hidden=HIDDEN, out_dim=1).to(device),
                      RegularMLP(in_dim=2, hidden=HIDDEN, out_dim=1).to(device).parameters(),
                      train_loader, val_loader, epochs=EPOCHS)

print("\n=== SPSA (SynapticMLP) ===")
model_syn = SynapticMLP(in_dim=2, hidden=HIDDEN, out_dim=1).to(device)
hist_syn = train_spsa(model_syn, model_syn.parameters(), train_loader, val_loader, epochs=EPOCHS)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(hist_adam['val_loss'], label='AdamW (Regular)', linewidth=2)
plt.plot(hist_reg['val_loss'], label='SPSA (Regular)', linewidth=2)
plt.plot(hist_syn['val_loss'], label='SPSA (Synaptic)', linewidth=2)
plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Val Loss')
plt.title('Convergence Comparison'); plt.legend(); plt.grid(alpha=0.3)
plt.subplot(1,2,2)
mean_times = [np.mean(hist_adam['time']), np.mean(hist_reg['time']), np.mean(hist_syn['time'])]
plt.bar(['AdamW Reg','SPSA Reg','SPSA Syn'], mean_times, color=['blue','orange','green'])
for i, v in enumerate(mean_times): plt.text(i, v, f'{v:.4f}s', ha='center')
plt.ylabel('Avg time/epoch (s)'); plt.title('Speed')
plt.tight_layout(); plt.savefig('exp004_synaptic_geometry.png'); plt.show()
