# ==============================================
# Eksperimen #005: Fisher Diagonalization Probe
# ==============================================

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

N = 500; D_in = 4; D_out = 2
X_raw = torch.randn(N, D_in, device=device)
mean = X_raw.mean(dim=0)
X_centered = X_raw - mean
cov = (X_centered.T @ X_centered) / (N-1)
L = torch.linalg.cholesky(cov)
W_whiten = torch.linalg.inv(L).to(device)
X_white = X_centered @ W_whiten.T
Y = torch.randn(N, D_out, device=device)

class LinearModel(nn.Module):
    def __init__(self, whiten=False):
        super().__init__()
        self.W = nn.Parameter(torch.randn(D_out, D_in, device=device)*0.1)
        self.b = nn.Parameter(torch.zeros(D_out, device=device))
        self.whiten = whiten
        if whiten:
            self.register_buffer('W_whiten', W_whiten)
            self.register_buffer('mean', mean)
    def forward(self, x):
        if self.whiten: x = (x - self.mean) @ self.W_whiten.T
        return F.linear(x, self.W, self.b)

def compute_fisher(model, X, Y, loss_fn):
    model.train(); fisher = None
    for i in range(X.size(0)):
        model.zero_grad()
        pred = model(X[i:i+1])
        loss = loss_fn(pred, Y[i:i+1])
        loss.backward()
        grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.requires_grad])
        if fisher is None: fisher = torch.outer(grad, grad)
        else: fisher += torch.outer(grad, grad)
    fisher /= X.size(0)
    return fisher.detach()

loss_func = lambda pred, target: 0.5 * torch.sum((pred - target)**2)
model_raw = LinearModel(whiten=False).to(device)
model_white = LinearModel(whiten=True).to(device)
F_raw = compute_fisher(model_raw, X_raw, Y, loss_func)
F_white = compute_fisher(model_white, X_raw, Y, loss_func)

def off_diag_ratio(F):
    diag = torch.diag(F); off_diag = F - torch.diag(diag)
    return torch.norm(off_diag) / torch.norm(F)

print(f"Off-diagonal ratio (Raw): {off_diag_ratio(F_raw):.4f}")
print(f"Off-diagonal ratio (Whitened): {off_diag_ratio(F_white):.4f}")

fig, axes = plt.subplots(1,2, figsize=(12,5))
vmax = max(F_raw.abs().max(), F_white.abs().max())
axes[0].imshow(F_raw.cpu(), cmap='bwr', vmin=-vmax, vmax=vmax)
axes[0].set_title(f'FIM Raw (off-diag ratio: {off_diag_ratio(F_raw):.3f})')
axes[1].imshow(F_white.cpu(), cmap='bwr', vmin=-vmax, vmax=vmax)
axes[1].set_title(f'FIM Whitened (off-diag ratio: {off_diag_ratio(F_white):.3f})')
plt.tight_layout(); plt.savefig('fisher_diagonalization.png'); plt.show()
