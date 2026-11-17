"""
=============================================================================
Pytorch Conditional Flow Matching (CFM) Tutorial — MNIST 复现
=============================================================================
来源: Pytorch-Conditional-Flow-Matching-Tutorial/01_conditional-flow-matching.ipynb
作者: Minsu Kang

修改点:
  1. GPU id: 2 -> 0 (单卡 RTX 4060)
  2. 增加训练日志保存 (logs/)
  3. 增加中间过程图片保存 (plots/)
  4. 增加模型保存 (model_cfm.pth)
=============================================================================
"""

import os, sys, time, json, logging, traceback
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from torch.optim import AdamW
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 路径和日志
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
CKPT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
OUT_DIR = os.path.join(CURRENT_DIR, 'generated_cfm')
for d in [PLOTS_DIR, LOGS_DIR, CKPT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'cfm_mnist_log_{TIMESTAMP}.txt')

logger = logging.getLogger('CFM_MNIST')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(ch)

def flush_logs():
    sys.stdout.flush()
    for h in logger.handlers:
        if hasattr(h, 'flush'):
            h.flush()

# ============================================================================
# 配置（来自 notebook）
# ============================================================================
dataset_path = os.path.join(CURRENT_DIR, 'cfm_datasets')
dataset = "MNIST"
gpu_id = 0          # 修改: 2 -> 0 (单卡)
cuda = True

hidden_dim = 128
n_layers = 4

lr = 5e-5
sigma_min = 0

n_epochs = 70
train_batch_size = 128
inference_batch_size = 64

seed = 1234

os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpu_id)
DEVICE = torch.device("cuda:0" if cuda else "cpu")
img_size = (32, 32, 3) if dataset == "CIFAR10" else (28, 28, 1)  # (width, height, channels)
hidden_dims = [hidden_dim for _ in range(n_layers)]

torch.manual_seed(seed)
np.random.seed(seed)

logger.info(f"{'='*60}")
logger.info(f"CFM MNIST 复现实验")
logger.info(f"设备: {DEVICE} ({torch.cuda.get_device_name(0) if cuda else 'CPU'})")
logger.info(f"模型: hidden={hidden_dim} x {n_layers}层")
logger.info(f"训练: {n_epochs} epochs, batch={train_batch_size}, lr={lr}")
logger.info(f"{'='*60}")

# ============================================================================
# 数据加载
# ============================================================================
transform = transforms.Compose([transforms.ToTensor()])
kwargs = {'num_workers': 0, 'pin_memory': True}   # Windows 用 num_workers=0

train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)

logger.info(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")

# ============================================================================
# 模型（来自 notebook）
# ============================================================================
class ConvBlock(nn.Conv2d):
    """
        Conv2D Block
            Args:
                x: (N, C_in, H, W)
            Returns:
                y: (N, C_out, H, W)
    """
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
            # timestep embedding was only applied to residual blocks of U-Net
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
        """Target velocity: 把噪声映射到数据的向量场"""
        return (x_1 - (1 - self.sigma_min) * x_0)

    def interpolate(self, x_0, x_1, t):
        """条件流: 噪声与数据的线性插值 (OT 条件向量场)"""
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    def forward(self, x, t):
        """估计向量场 v(x_t, t)"""
        time_embedding = self.time_project(t)
        y = self.in_project(x)
        for i in range(len(self.convs)):
            y = self.convs[i](y, time_embedding, residual=True)
        v_t = self.out_project(y)
        return v_t

    @torch.no_grad()
    def sample(self, t_steps, shape, DEVICE):
        """Euler ODE 积分: 从 x_0 (噪声) 走到 x_1 (数据)"""
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


model = Model(image_resolution=img_size, hidden_dims=hidden_dims, sigma_min=sigma_min).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
logger.info(f"模型参数量: {n_params:,} ({n_params/1e6:.1f}M)")

optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))

# ============================================================================
# 断点续训
# ============================================================================
start_epoch = 0
latest_ckpt = None
if os.path.exists(CKPT_DIR):
    ckpt_files = [f for f in os.listdir(CKPT_DIR) if f.startswith('cfm_epoch_') and f.endswith('.pt')]
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=lambda f: int(f.split('_')[2].split('.')[0]))
if latest_ckpt:
    ckpt_path = os.path.join(CKPT_DIR, latest_ckpt)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    logger.info(f"✅ 从断点恢复: {latest_ckpt} (epoch {start_epoch})")
else:
    logger.info(f"无断点，从头训练")

# ============================================================================
# 训练
# ============================================================================
logger.info(f"\n--- 开始训练 CFM ({n_epochs} epochs) ---")
model.train()
t0 = time.time()
epoch_losses = []

for epoch in range(start_epoch, n_epochs):
    total_loss = 0
    n_batch = 0
    ep_start = time.time()
    for batch_idx, (x_1, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_1 = x_1.to(DEVICE)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, 1, 1, device=DEVICE)

        x_t = model.interpolate(x_0, x_1, t)
        velocity_target = model.get_velocity(x_0, x_1)
        velocity_pred = model(x_t, t)

        loss = ((velocity_pred - velocity_target) ** 2).mean()
        total_loss += loss.item()
        n_batch += 1

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / n_batch
    epoch_losses.append(avg_loss)
    ep_time = time.time() - ep_start

    logger.info(f"  Epoch {epoch+1:3d}/{n_epochs} | CFM Loss: {avg_loss:.6f} | {ep_time:.0f}s")
    flush_logs()

    # 每 10 epochs 生成一次并保存（断点）
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            samples = model.sample(50, [16, 1, 28, 28], DEVICE)
        grid = make_grid(samples, nrow=4, padding=2, normalize=True)
        save_image(grid, os.path.join(PLOTS_DIR, f'cfm_gen_epoch_{epoch+1:03d}.png'))
        logger.info(f"    ✓ 保存: plots/cfm_gen_epoch_{epoch+1:03d}.png")
        model.train()

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(CKPT_DIR, f'cfm_epoch_{epoch+1}.pt'))
        logger.info(f"    ✓ 断点: checkpoints/cfm_epoch_{epoch+1}.pt")

t1 = time.time()
logger.info(f"\n--- 训练完成 ---")
logger.info(f"  耗时: {t1-t0:.0f}s ({(t1-t0)/60:.1f}min)")
logger.info(f"  CFM Loss: {epoch_losses[0]:.6f} → {epoch_losses[-1]:.6f}")

# 保存模型
model_path = os.path.join(CURRENT_DIR, 'model_cfm.pth')
torch.save({'model_state_dict': model.state_dict()}, model_path)
logger.info(f"  模型: {model_path}")

# Loss 曲线
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epoch_losses, linewidth=1.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('CFM Loss')
ax.set_title('Conditional Flow Matching Loss 曲线 (MNIST)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'cfm_loss.png'), dpi=150)
plt.close()

# ============================================================================
# 生成测试
# ============================================================================
logger.info(f"\n--- 生成测试 ---")
model.eval()

# 不同 T 值（流步数）对比
for n_flows, name in [(2, 'T2'), (5, 'T5'), (10, 'T10'), (25, 'T25'), (50, 'T50')]:
    with torch.no_grad():
        x_hat = model.sample(n_flows, [1, 1, 28, 28], DEVICE)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(x_hat[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy(), cmap='gray')
    ax.axis('off')
    ax.set_title(f'T={n_flows}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cfm_flow_{name}.png'), dpi=150)
    plt.close()
logger.info(f"  T=2/5/10/25/50 单样本对比图已保存")

# 64 张批量生成 (T=50)
B = inference_batch_size
with torch.no_grad():
    x_hats = model.sample(50, [B, 1, 28, 28], DEVICE)
grid = make_grid(x_hats, nrow=8, padding=2, normalize=True)
save_image(grid, os.path.join(PLOTS_DIR, 'cfm_generated_64.png'))
logger.info(f"  64 张生成: plots/cfm_generated_64.png")

# 真实样本对比
real_batch = next(iter(test_loader))[0][:inference_batch_size]
grid_real = make_grid(real_batch, nrow=8, padding=2, normalize=True)
save_image(grid_real, os.path.join(PLOTS_DIR, 'cfm_real_64.png'))
logger.info(f"  64 张真实: plots/cfm_real_64.png")

# 保存生成样本（单张）
for i in range(min(16, len(x_hats))):
    save_image(x_hats[i], os.path.join(OUT_DIR, f'cfm_sample_{i:02d}.png'))

# 总结
with open(os.path.join(CURRENT_DIR, 'cfm_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'model': 'CFM MNIST (ConvBlock x8, hidden=256)',
        'params_m': n_params / 1e6,
        'n_epochs': n_epochs,
        'batch_size': train_batch_size,
        'final_loss': epoch_losses[-1],
        'time_min': (t1 - t0) / 60,
    }, f, ensure_ascii=False, indent=2)

logger.info(f"\n✅ CFM MNIST 复现完成！")
