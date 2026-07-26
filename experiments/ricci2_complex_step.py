# ==============================================
# Eksperimen #002: RCSSG – Random Coordinate Stochastic Subspace Gradient
# Estimasi Gradien via Complex-Step Differentiation
# ==============================================

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Perangkat: {device}')

class ComplexLinearMLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, dtype=torch.cfloat)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.cfloat)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=torch.cfloat)
    def forward(self, x):
        x = x.type(torch.cfloat)
        x = self.fc1(x); x = self.fc2(x); x = self.fc3(x)
        return x

def compute_loss_complex(model, x, y):
    pred = model(x); diff = pred - y
    return torch.mean(torch.abs(diff)**2)

def train_adamw(model, train_loader, val_loader, epochs=100, lr=1e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'time_per_epoch': []}
    for epoch in range(epochs):
        start = time.time(); model.train(); total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = compute_loss_complex(model, xb, yb)
            loss.backward(); optimizer.step()
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

def analytic_loss(pred, target):
    diff = pred - target
    loss = torch.mean(diff**2)
    return loss

class RCSSG_Optimizer:
    def __init__(self, model, lr=1e-3, h=1e-4, num_directions=2, beta=0.9):
        self.model = model; self.lr = lr; self.h = h
        self.num_directions = num_directions; self.beta = beta
        self.momentum_dirs = None

    def _pack_params_real(self):
        vec = []
        for p in self.model.parameters():
            vec.append(p.data.real.view(-1)); vec.append(p.data.imag.view(-1))
        return torch.cat(vec)

    def _unpack_params_real(self, vec):
        idx = 0
        for p in self.model.parameters():
            num = p.numel()
            p.data.real = vec[idx:idx+num].view_as(p.data.real); idx += num
            p.data.imag = vec[idx:idx+num].view_as(p.data.imag); idx += num

    def _generate_random_directions(self):
        total_dim = sum(p.numel()*2 for p in self.model.parameters())
        directions = []
        for _ in range(self.num_directions):
            d = torch.randn(total_dim, device=device)
            for existing in directions:
                d -= torch.dot(d, existing) * existing
            d = d / torch.norm(d); directions.append(d)
        return directions

    def _directional_derivative(self, x, y, d_vec_real):
        original_vec = self._pack_params_real()
        idx = 0; d_complex_list = []
        for p in self.model.parameters():
            num = p.numel()
            d_real_part = d_vec_real[idx:idx+num].view_as(p.data.real); idx += num
            d_imag_part = d_vec_real[idx:idx+num].view_as(p.data.imag); idx += num
            d_complex_list.append(torch.complex(d_real_part, d_imag_part))
        for p, d in zip(self.model.parameters(), d_complex_list):
            p.data = p.data + 1j * self.h * d
        pred = self.model(x)
        loss = analytic_loss(pred, y)
        self._unpack_params_real(original_vec)
        return loss.imag / self.h

    def step(self, x_batch, y_batch):
        dirs = self._generate_random_directions()
        grad_approx = torch.zeros_like(self._pack_params_real())
        for d in dirs:
            deriv = self._directional_derivative(x_batch, y_batch, d)
            grad_approx += deriv * d
        theta = self._pack_params_real()
        theta_new = theta - self.lr * grad_approx
        self._unpack_params_real(theta_new)
        return None

def train_rcssg(model, train_loader, val_loader, epochs=100, lr=1e-3, h=1e-4, num_dirs=2):
    optimizer = RCSSG_Optimizer(model, lr=lr, h=h, num_directions=num_dirs)
    history = {'train_loss': [], 'val_loss': [], 'time_per_epoch': []}
    for epoch in range(epochs):
        start = time.time(); model.train(); total_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss_real = torch.mean(torch.abs(pred - yb)**2)
            total_loss += loss_real.item() * xb.size(0)
            optimizer.step(xb, yb)
        train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            val_loss = sum(torch.mean(torch.abs(model(xb) - yb)**2).item() * xb.size(0)
                           for xb, yb in val_loader) / len(val_loader.dataset)
        elapsed = time.time() - start
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['time_per_epoch'].append(elapsed)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e} | Time: {elapsed:.4f}s')
    return history

EPOCHS = 200; HIDDEN_DIM = 64
model_adam = ComplexLinearMLP(hidden_dim=HIDDEN_DIM).to(device)
print("=== AdamW ===")
hist_adam = train_adamw(model_adam, train_loader, val_loader, epochs=EPOCHS, lr=1e-3)

model_rcssg = ComplexLinearMLP(hidden_dim=HIDDEN_DIM).to(device)
print("\n=== RCSSG (num_dirs=2) ===")
hist_rcssg = train_rcssg(model_rcssg, train_loader, val_loader, epochs=EPOCHS, lr=1e-3, h=1e-4, num_dirs=2)

xb, yb = next(iter(train_loader))
model_temp = ComplexLinearMLP(hidden_dim=HIDDEN_DIM).to(device)
model_temp.load_state_dict(model_rcssg.state_dict())
for p in model_temp.parameters(): p.grad = None
pred = model_temp(xb)
loss = torch.mean(torch.abs(pred - yb)**2)
loss.backward()
true_grads = torch.cat([p.grad.real.view(-1) for p in model_temp.parameters()] +
                       [p.grad.imag.view(-1) for p in model_temp.parameters()])
optimizer = RCSSG_Optimizer(model_rcssg, lr=1e-3, h=1e-4, num_directions=2)
dirs = optimizer._generate_random_directions()
est_grad = torch.zeros_like(true_grads)
for d in dirs:
    deriv = optimizer._directional_derivative(xb, yb, d)
    est_grad += deriv * d
cosine_sim = torch.nn.functional.cosine_similarity(est_grad.unsqueeze(0), true_grads.unsqueeze(0))
print(f"Cosine similarity antara gradien estimasi (2 arah) dan gradien asli: {cosine_sim.item():.4f}")
