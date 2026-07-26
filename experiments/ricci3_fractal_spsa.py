# ==============================================
# Eksperimen #003: Fractal Subspace SPSA
# ==============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_dataset(n_samples=2000, noise_std=0.01):
    x = (torch.rand(n_samples, 2) * 2 - 1) * np.pi
    y = torch.sin(x[:, 0]) * torch.cos(x[:, 1]) + torch.randn(n_samples) * noise_std
    return x, y.unsqueeze(1)

class RegularMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

def haar_dwt_1d(w):
    w_even = w[:, 0::2]; w_odd = w[:, 1::2]
    L = (w_even + w_odd) / np.sqrt(2)
    H = (w_even - w_odd) / np.sqrt(2)
    return L, H

def inverse_haar_dwt_1d(L, H):
    w_even = (L + H) / np.sqrt(2)
    w_odd = (L - H) / np.sqrt(2)
    w = torch.stack([w_even, w_odd], dim=2).reshape(L.size(0), -1)
    return w

class FractalLinear(nn.Module):
    def __init__(self, in_features, out_features, freeze_H=True):
        super().__init__()
        assert in_features % 2 == 0
        self.in_features = in_features; self.out_features = out_features
        self.half_in = in_features // 2
        self.L = nn.Parameter(torch.randn(out_features, self.half_in) * 0.1)
        self.H = nn.Parameter(torch.randn(out_features, self.half_in) * 0.1)
        if freeze_H: self.H.requires_grad = False
        self.bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        W = inverse_haar_dwt_1d(self.L, self.H)
        return F.linear(x, W, self.bias)

class FractalMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=1):
        super().__init__()
        self.fc1 = FractalLinear(in_dim, hidden)
        self.fc2 = FractalLinear(hidden, hidden)
        self.fc3 = FractalLinear(hidden, out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

class SPSA_Optimizer:
    def __init__(self, params_to_optimize, a=1e-3, c=1e-3, alpha=0.602, gamma=0.101, A=10):
        self.params = list(params_to_optimize)
        self.a = a; self.c = c; self.alpha = alpha; self.gamma = gamma; self.A = A
        self.t = 0; self.total_dim = sum(p.numel() for p in self.params)
    def _pack(self):
        return torch.cat([p.view(-1).float() for p in self.params])
    def _unpack(self, vec):
        idx = 0
        for p in self.params:
            num = p.numel(); p.data = vec[idx:idx+num].view_as(p).to(p.dtype); idx += num
    def step(self, loss_fn):
        self.t += 1
        a_t = self.a / ((self.t + self.A) ** self.alpha)
        c_t = self.c / (self.t ** self.gamma)
        theta = self._pack()
        delta = torch.randint(0, 2, (self.total_dim,), device=theta.device, dtype=theta.dtype) * 2 - 1
        theta_plus = theta + c_t * delta
        theta_minus = theta - c_t * delta
        self._unpack(theta_plus); loss_plus = loss_fn()
        self._unpack(theta_minus); loss_minus = loss_fn()
        g_hat = (loss_plus - loss_minus) / (2 * c_t) * delta
        theta_new = theta - a_t * g_hat
        self._unpack(theta_new)
        return (loss_plus + loss_minus) / 2

def compute_mse(model, x, y):
    pred = model(x)
    return torch.mean((pred - y) ** 2)

def train_adamw(model, train_loader, val_loader, epochs=100, lr=1e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'time': []}
    for epoch in range(epochs):
        t0 = time.time(); total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = compute_mse(model, xb, yb)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            val_loss = sum(compute_mse(model, xb, yb).item() * xb.size(0) for xb, yb in val_loader) / len(val_loader.dataset)
        t_epoch = time.time() - t0
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss); history['time'].append(t_epoch)
        if epoch % 10 == 0:
            print(f'AdamW Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {t_epoch:.4f}s')
    return history

def train_spsa_regular(model, train_loader, val_loader, epochs=100, a=1e-3, c=1e-3):
    params_all = list(model.parameters())
    opt = SPSA_Optimizer(params_all, a=a, c=c)
    history = {'train_loss': [], 'val_loss': [], 'time': []}
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
            print(f'SPSA Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {t_epoch:.4f}s')
    return history

def train_fractal_spsa(model, train_loader, val_loader, epochs=100, a=1e-3, c=1e-3):
    params_opt = []
    for name, p in model.named_parameters():
        if 'L' in name or 'bias' in name:
            params_opt.append(p)
    opt = SPSA_Optimizer(params_opt, a=a, c=c)
    history = {'train_loss': [], 'val_loss': [], 'time': []}
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
            print(f'FracSPSA Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {t_epoch:.4f}s')
    return history

if __name__ == '__main__':
    hidden = 32; epochs = 150
    x_data, y_data = generate_dataset(2000)
    x_data, y_data = x_data.to(device), y_data.to(device)
    train_size = int(0.8 * len(x_data))
    x_train, x_val = x_data[:train_size], x_data[train_size:]
    y_train, y_val = y_data[:train_size], y_data[train_size:]
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    print("Training AdamW...")
    model_adamw = RegularMLP(in_dim=2, hidden=hidden, out_dim=1).to(device)
    history_adamw = train_adamw(model_adamw, train_loader, val_loader, epochs=epochs)

    print("\nTraining SPSA (all params)...")
    model_spsa = RegularMLP(in_dim=2, hidden=hidden, out_dim=1).to(device)
    history_spsa = train_spsa_regular(model_spsa, train_loader, val_loader, epochs=epochs)

    print("\nTraining Fractal SPSA (low-frequency only)...")
    model_fractal = FractalMLP(in_dim=2, hidden=hidden, out_dim=1).to(device)
    history_fractal = train_fractal_spsa(model_fractal, train_loader, val_loader, epochs=epochs)
