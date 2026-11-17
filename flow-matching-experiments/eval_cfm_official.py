# -*- coding: utf-8 -*-
"""
CFM MNIST 官方配置: 生成 64 个样本 -> 分类器打分 -> 选出最优 16 个 -> 与 GT 真实样本对比拼图
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"设备: {DEVICE}")


# ============================================================================
# 模型 (与训练脚本 train_cfm_official.py 完全一致)
# ============================================================================
class ConvBlock(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True, gn=False, gn_groups=8):
        if padding == 'same':
            padding = kernel_size // 2 * dilation
        super(ConvBlock, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation,
                                        groups=groups, bias=bias)
        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None

    def forward(self, x, time_embedding=None, residual=False):
        if residual:
            x = x + time_embedding
            y = x
            x = super(ConvBlock, self).forward(x)
            y = y + x
        else:
            y = super(ConvBlock, self).forward(x)
        y = self.group_norm(y) if self.group_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        return y


class Model(nn.Module):
    def __init__(self, image_resolution, hidden_dims=[256, 256], sigma_min=0.):
        super(Model, self).__init__()
        _, _, img_C = image_resolution
        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)
        self.time_project = nn.Sequential(
            ConvBlock(1, hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[1], kernel_size=1))
        self.convs = nn.ModuleList([ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)])
        for idx in range(1, len(hidden_dims)):
            self.convs.append(ConvBlock(hidden_dims[idx-1], hidden_dims[idx], kernel_size=3,
                                        dilation=3**((idx-1)//2), activation_fn=True, gn=True, gn_groups=8))
        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3)
        self.sigma_min = sigma_min

    def get_velocity(self, x_0, x_1):
        return (x_1 - (1 - self.sigma_min) * x_0)

    def interpolate(self, x_0, x_1, t):
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    def forward(self, x, t):
        time_embedding = self.time_project(t)
        y = self.in_project(x)
        for i in range(len(self.convs)):
            y = self.convs[i](y, time_embedding, residual=True)
        v_t = self.out_project(y)
        return v_t

    @torch.no_grad()
    def sample(self, t_steps, shape, DEVICE):
        """Euler ODE 积分: 从噪声 x_0 走到数据 x_1"""
        B, C, W, H = shape
        x_0 = torch.randn(size=shape, device=DEVICE)
        t_vals = torch.linspace(0, 1, t_steps, device=DEVICE)
        delta = 1.0 / (t_steps - 1)
        x_1_hat = x_0
        for i in range(t_steps - 1):
            t_cur = t_vals[i].view(-1, 1, 1, 1)
            velocity_pred = self(x_1_hat, t_cur)
            x_1_hat = x_1_hat + velocity_pred * delta
        return x_1_hat


# ============================================================================
# 快速数字分类器 (用于挑选"最清晰的生成样本")
# ============================================================================
class ClsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 数据
# ============================================================================
transform = transforms.Compose([transforms.ToTensor()])
dataset_path = os.path.join(CURRENT_DIR, 'cfm_datasets')
train_set = MNIST(dataset_path, train=True, download=False, transform=transform)
test_set = MNIST(dataset_path, train=False, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

# ============================================================================
# 1. 加载最优模型并生成 64 个样本
# ============================================================================
model = Model((28, 28, 1), hidden_dims=[256] * 8, sigma_min=0).to(DEVICE)
ck = torch.load(os.path.join(CURRENT_DIR, 'model_cfm_official.pth'),
                map_location=DEVICE, weights_only=False)
model.load_state_dict(ck['model_state_dict'])
model.eval()

with torch.no_grad():
    x_hats = model.sample(100, [64, 1, 28, 28], DEVICE).clamp(0, 1)
print(f"已生成 64 个样本 (Euler T=100)")

# ============================================================================
# 2. 训练分类器 (训练集 2 epochs, 快速)
# ============================================================================
cls = ClsNet().to(DEVICE)
opt = torch.optim.Adam(cls.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
cls.train()
for ep in range(2):
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = ce(cls(xb), yb)
        loss.backward()
        opt.step()
# 评估分类器
cls.eval()
correct = total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred_c = cls(xb).argmax(dim=1)
        correct += (pred_c == yb).sum().item()
        total += yb.shape[0]
print(f"分类器准确率: {correct / total * 100:.2f}%")

# ============================================================================
# 3. 打分: 分类置信度 + 前景占比过滤 (空图/糊图降权)
# ============================================================================
with torch.no_grad():
    logits = cls(x_hats)
    probs = torch.softmax(logits, dim=-1)
    top_p, pred = probs.max(dim=-1)

fg = x_hats.view(64, -1).mean(dim=1)              # 前景占比 (0~1)
valid = (fg > 0.03) & (fg < 0.55)                 # 合理数字的像素占比
score = top_p.clone()
score[~valid] = 0.0

# 按分数降序, 每类最多挑 2 个, 凑 16 个
order = torch.argsort(score, descending=True)
picked = []
cnt = {}
for i in order.tolist():
    d = pred[i].item()
    if cnt.get(d, 0) >= 2:
        continue
    picked.append(i)
    cnt[d] = cnt.get(d, 0) + 1
    if len(picked) == 16:
        break

print(f"选出的 16 个生成样本 (类别分布: {dict(sorted(cnt.items()))}):")
for gi in picked:
    print(f"  gen[{gi:2d}] pred={pred[gi].item()} p={top_p[gi].item():.3f} fg={fg[gi].item():.3f}")

# ============================================================================
# 4. GT: 与预测类别一一对应的真实测试样本
# ============================================================================
test_imgs = test_set.data.float() / 255.0
test_labels = test_set.targets
gt_pick = []
used = set()
for i in picked:
    d = pred[i].item()
    idxs = (test_labels == d).nonzero().flatten().tolist()
    cand = [j for j in idxs if j not in used]
    j = cand[0]
    used.add(j)
    gt_pick.append(j)

# ============================================================================
# 5. 拼图: 上排生成 Top-16 / 下排 GT 真实 (一一对应)
# ============================================================================
fig, axes = plt.subplots(2, 16, figsize=(21, 5.2))
fig.suptitle('CFM MNIST Official (best weight): Top-16 Generated vs Ground Truth', fontsize=14, y=1.0)
for col, gi in enumerate(picked):
    ax = axes[0, col]
    ax.imshow(x_hats[gi, 0].cpu().numpy(), cmap='gray')
    ax.set_title(f'{pred[gi].item()} ({top_p[gi].item():.2f})', fontsize=9)
    ax.axis('off')
for col, j in enumerate(gt_pick):
    ax = axes[1, col]
    ax.imshow(test_imgs[j].numpy(), cmap='gray')
    ax.set_title(f'{test_labels[j].item()}', fontsize=9)
    ax.axis('off')
axes[0, 0].set_ylabel('Generated\nTop-16', fontsize=11, rotation=0, labelpad=45)
axes[1, 0].set_ylabel('GT Real', fontsize=11, rotation=0, labelpad=30)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'cfm_official_top16_compare.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"对比图已保存: {out}")
