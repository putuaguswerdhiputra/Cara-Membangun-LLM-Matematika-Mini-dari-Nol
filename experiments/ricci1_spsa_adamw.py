# ==============================================
# Eksperimen #001: Fondasi Synaptic Geometry
# Validasi Prinsip Aliran Holomorfik via SPSA
# ==============================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Perangkat: {device}')
if device.type == 'cuda':
    print(f'GPU terdeteksi: {torch.cuda.get_device_name(0)}')

def generate_complex_dataset(num_samples=1000, x_range=(-2*np.pi, 2*np.pi)):
    x = torch.linspace(x_range[0], x_range[1], num_samples).unsqueeze(1)
    y_complex = torch.exp(1j * x)
    return x, y_complex

x_data, y_data = generate_complex_dataset(num_samples=2000)
x_data, y_data = x_data.to(device), y_data.to(device)
train_size = int(0.8 * len(x_data))
x_train, x_val = x_data[:train_size], x_data[train_size:]
y_train, y_val = y_data[:train_size], y_data[train_size:]

train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

class ComplexLinearMLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, dtype=torch.cfloat)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.cfloat)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=torch.cfloat)
    def forward(self, x):
        x = x.type(torch.cfloat)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def compute_loss_complex(model, x, y):
    pred = model(x)
    diff = pred - y
    loss = torch.mean(torch.abs(diff)**2)
    return loss

def train_adamw(model, train_loader, val_loader, epochs=100, lr=1e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'time_per_epoch': []}
    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = compute_loss_complex(model, xb, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            val_loss = sum(compute_loss_complex(model, xb, yb).item() * xb.size(0)
                           for xb, yb in val_loader) / len(val_loader.dataset)
        elapsed = time.time() - start
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['time_per_epoch'].append(elapsed)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e} | Time: {elapsed:.4f}s')
    return history

class SPSAOptimizer:
    def __init__(self, model, a=1e-3, c=1e-4, alpha=0.602, gamma=0.101, A=10):
        self.model = model
        self.a = a; self.c = c; self.alpha = alpha; self.gamma = gamma; self.A = A
        self.t = 0
        self.shapes = [p.shape for p in model.parameters()]
        self.sizes = [p.numel() for p in model.parameters()]
        self.total_params = sum(self.sizes)

    def _pack_parameters(self):
        vecs = []
        for p in self.model.parameters():
            vecs.append(p.data.view(-1).real)
            vecs.append(p.data.view(-1).imag)
        return torch.cat(vecs)

    def _unpack_parameters(self, vec):
        idx = 0
        for p in self.model.parameters():
            num = p.numel()
            real_part = vec[idx:idx+num].view_as(p.real); idx += num
            imag_part = vec[idx:idx+num].view_as(p.imag); idx += num
            p.data = torch.complex(real_part, imag_part)

    def compute_loss_flat(self, vec, x, y):
        original_vec = self._pack_parameters()
        self._unpack_parameters(vec)
        loss = compute_loss_complex(self.model, x, y)
        self._unpack_parameters(original_vec)
        return loss

    def step(self, x_batch, y_batch):
        self.t += 1
        a_t = self.a / ((self.t + self.A) ** self.alpha)
        c_t = self.c / (self.t ** self.gamma)
        theta = self._pack_parameters()
        delta = torch.randint(0, 2, (self.total_params*2,), device=theta.device, dtype=theta.dtype) * 2 - 1
        theta_plus = theta + c_t * delta
        theta_minus = theta - c_t * delta
        loss_plus = self.compute_loss_flat(theta_plus, x_batch, y_batch)
        loss_minus = self.compute_loss_flat(theta_minus, x_batch, y_batch)
        g_hat = (loss_plus - loss_minus) / (2 * c_t) * delta
        theta_new = theta - a_t * g_hat
        self._unpack_parameters(theta_new)
        return loss_plus.item(), loss_minus.item()

def train_spsa(model, train_loader, val_loader, epochs=100, a=1e-3, c=1e-3):
    model.train()
    optimizer = SPSAOptimizer(model, a=a, c=c)
    history = {'train_loss': [], 'val_loss': [], 'time_per_epoch': []}
    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            loss_plus, loss_minus = optimizer.step(xb, yb)
            total_loss += (loss_plus + loss_minus) / 2 * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            val_loss = sum(compute_loss_complex(model, xb, yb).item() * xb.size(0)
                           for xb, yb in val_loader) / len(val_loader.dataset)
        elapsed = time.time() - start
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['time_per_epoch'].append(elapsed)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e} | Time: {elapsed:.4f}s')
    return history

EPOCHS = 200; HIDDEN_DIM = 64
print("="*60)
print("MELATIH DENGAN ADAMW (BACKPROP)")
print("="*60)
model_adam = ComplexLinearMLP(hidden_dim=HIDDEN_DIM).to(device)
hist_adam = train_adamw(model_adam, train_loader, val_loader, epochs=EPOCHS, lr=1e-3)

print("\n" + "="*60)
print("MELATIH DENGAN SPSA (FORWARD-ONLY)")
print("="*60)
model_spsa = ComplexLinearMLP(hidden_dim=HIDDEN_DIM).to(device)
hist_spsa = train_spsa(model_spsa, train_loader, val_loader, epochs=EPOCHS, a=5e-3, c=1e-3)

fig, axes = plt.subplots(1, 3, figsize=(18,5))
axes[0].plot(hist_adam['val_loss'], label='AdamW', linewidth=2)
axes[0].plot(hist_spsa['val_loss'], label='SPSA', linewidth=2)
axes[0].set_yscale('log'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val Loss')
axes[0].set_title('Konvergensi'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(hist_adam['time_per_epoch'], label='AdamW', linewidth=2)
axes[1].plot(hist_spsa['time_per_epoch'], label='SPSA', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Waktu (s)')
axes[1].set_title('Time per Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
mean_time_adam = np.mean(hist_adam['time_per_epoch'])
mean_time_spsa = np.mean(hist_spsa['time_per_epoch'])
axes[2].bar(['AdamW', 'SPSA'], [mean_time_adam, mean_time_spsa], color=['blue', 'orange'])
axes[2].set_ylabel('Rata-rata Waktu per Epoch (s)')
plt.tight_layout()
plt.savefig('eksperimen_001_hasil.png', dpi=150)
plt.show()
print(f"\nRata-rata waktu AdamW: {mean_time_adam:.4f} detik/epoch")
print(f"Rata-rata waktu SPSA : {mean_time_spsa:.4f} detik/epoch")
