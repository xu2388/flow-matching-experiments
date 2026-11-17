# -*- coding: utf-8 -*-
"""
=============================================================================
CFM MNIST 官方配置版训练 (Pytorch Conditional Flow Matching Tutorial)
=============================================================================
来源: Pytorch-Conditional-Flow-Matching-Tutorial/01_conditional-flow-matching.ipynb

本脚本 = 官方原始模型配置 (hidden=256 x 8 层) + 工程增强:
  1. 断点续训: checkpoint 保存 model/optimizer/epoch/历史loss, 自动恢复
  2. 训练中测试: 每 5 epochs 在测试集上计算验证速度场 loss + 采样生成图
  3. 日志/图片/模型自动保存
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
from torch.optim.lr_scheduler import CosineAnnealingLR
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
LOG_FILE = os.path.join(LOGS_DIR, f'cfm_official_log_{TIMESTAMP}.txt')

logger = logging.getLogger('CFM_OFFICIAL')
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
# 配置（官方 notebook 原始值）
# ============================================================================
dataset_path = os.path.join(CURRENT_DIR, 'cfm_datasets')
dataset = "MNIST"
gpu_id = 0          # 官方为 2，单卡改为 0
cuda = True

hidden_dim = 256    # 官方
n_layers = 8        # 官方
lr = 5e-5           # 首次从头训练学习率
sigma_min = 0       # 官方
FINE_TUNE_LR = 2e-5     # 断点续训(微调)阶段学习率
FINE_TUNE_EPOCHS = 50   # 微调阶段最多再训练轮数 (从断点起)
EMA_DECAY = 0.999       # EMA 指数滑动平均衰减
RESUME_FROM_BEST = True # 从验证 loss 最优权重启动微调 (True) / 从最近断点续训 (False)
n_epochs = 200      # 官方
train_batch_size = 128
inference_batch_size = 64

seed = 1234
CKPT_PREFIX = 'official_best_'     # 最优模型前缀 (只保留前 BEST_KEEP 个)
RESUME_CKPT = 'official_resume.pt' # 续训断点 (只保留最新一个, 覆盖保存)
BEST_KEEP = 5                      # 保存验证 loss 最优的模型个数
TEST_INTERVAL = 5                  # 训练中测试间隔 (epochs)

os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpu_id)
DEVICE = torch.device("cuda:0" if cuda else "cpu")
img_size = (32, 32, 3) if dataset == "CIFAR10" else (28, 28, 1)
hidden_dims = [hidden_dim for _ in range(n_layers)]

torch.manual_seed(seed)
np.random.seed(seed)

logger.info(f"{'='*60}")
logger.info(f"CFM MNIST 官方配置版训练")
logger.info(f"设备: {DEVICE} ({torch.cuda.get_device_name(0) if cuda else 'CPU'})")
logger.info(f"模型: hidden={hidden_dim} x {n_layers}层 (官方)")
logger.info(f"训练: {n_epochs} epochs, batch={train_batch_size}, lr={lr}, sigma_min={sigma_min}")
logger.info(f"测试: 每 {TEST_INTERVAL} epochs 验证 loss + 采样图")
logger.info(f"{'='*60}")

# ============================================================================
# 数据加载
# ============================================================================
transform = transforms.Compose([transforms.ToTensor()])
kwargs = {'num_workers': 0, 'pin_memory': False}   # Windows + 8G 显存

train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)

logger.info(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")

# ============================================================================
# 模型（官方 notebook 原样）
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


class EMA:
    """指数滑动平均: 平滑模型权重, 提升生成质量与稳定性"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for k, v in model.state_dict().items():
            if v.is_floating_point() and 'num_batches_tracked' not in k:
                self.shadow[k] = v.detach().clone()

    def update(self, model):
        for k, v in self.shadow.items():
            self.shadow[k].data.mul_(self.decay).add_(model.state_dict()[k].data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        for k, v in self.shadow.items():
            model.state_dict()[k].data.copy_(v.data)

    def restore(self, model):
        for k, v in self.backup.items():
            model.state_dict()[k].data.copy_(v.data)


model = Model(image_resolution=img_size, hidden_dims=hidden_dims, sigma_min=sigma_min).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
logger.info(f"模型参数量: {n_params:,} ({n_params/1e6:.1f}M)")

optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))

# ============================================================================
# 断点续训
# ============================================================================
start_epoch = 0
epoch_losses = []       # 训练 loss 历史
val_losses = []         # 验证 loss 历史
ema_shadow_loaded = None
scheduler = None
resume_from_best = False
resume_path = os.path.join(CKPT_DIR, RESUME_CKPT)

# 1) 优先: 从最优权重启动 (验证 loss 最小)
if RESUME_FROM_BEST:
    best_files = [f for f in os.listdir(CKPT_DIR)
                  if f.startswith(CKPT_PREFIX) and f.endswith('.pt')]
    if best_files:
        best_records = []
        for f in best_files:
            ck = torch.load(os.path.join(CKPT_DIR, f), map_location=DEVICE, weights_only=False)
            best_records.append((ck['val_loss'], f, ck))
        best_records.sort(key=lambda x: x[0])
        best_v, best_name, best_ckpt = best_records[0]
        model.load_state_dict(best_ckpt['model_state_dict'])
        start_epoch = best_ckpt['epoch']
        resume_from_best = True
        logger.info(f"[OK] 从最优权重启动: {best_name} (epoch {start_epoch}, val_loss {best_v:.6f})")
    else:
        logger.warning("未找到 best 权重, 回退断点续训")

# 2) 其次: 从最近断点续训
if not resume_from_best and os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    epoch_losses = list(ckpt.get('epoch_losses', []))
    val_losses = list(ckpt.get('val_losses', []))
    ema_shadow_loaded = ckpt.get('ema_shadow')
    logger.info(f"[OK] 从断点恢复: {RESUME_CKPT} (epoch {start_epoch}, 历史训练loss {len(epoch_losses)} 个)")
if start_epoch == 0 and not resume_from_best and not os.path.exists(resume_path):
    logger.info("无断点，从头训练")

# EMA 必须在权重加载完成之后初始化, 否则影子权重会从随机初始值开始 (导致验证/生成全乱)
ema = EMA(model, EMA_DECAY)
if ema_shadow_loaded is not None:
    ema.shadow = {k: v.to(DEVICE) for k, v in ema_shadow_loaded.items()}
    logger.info("  已加载 EMA 影子权重")
else:
    logger.info("  EMA 初始化为当前(加载后)权重")

# 微调模式: 降低学习率 + 余弦退火 + EMA 精调
if start_epoch > 0:
    lr = FINE_TUNE_LR
    for g in optimizer.param_groups:
        g['lr'] = lr
    if resume_from_best:
        n_epochs_target = start_epoch + FINE_TUNE_EPOCHS   # 从最优权重精确微调 50 轮
    else:
        n_epochs_target = min(n_epochs, start_epoch + FINE_TUNE_EPOCHS)
    scheduler = CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS, eta_min=0)
    logger.info(f"[FT] 微调模式: lr={lr}, CosineAnnealing(T_max={FINE_TUNE_EPOCHS}, eta_min=0), EMA decay={EMA_DECAY}")
    logger.info(f"  本次运行目标: {start_epoch} -> {n_epochs_target} epochs")
else:
    n_epochs_target = n_epochs
    logger.info(f"  首次从头训练: {n_epochs} epochs")


# ============================================================================
# 训练中测试: 验证集速度场 loss + 采样生成图
# ============================================================================
@torch.no_grad()
def evaluate_val_loss():
    """在测试集上计算平均速度场 MSE (训练中测试指标)"""
    model.eval()
    total = 0.0
    n = 0
    for x_1, _ in test_loader:
        x_1 = x_1.to(DEVICE)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, 1, 1, device=DEVICE)
        x_t = model.interpolate(x_0, x_1, t)
        v_t = model.get_velocity(x_0, x_1)
        v_pred = model(x_t, t)
        total += ((v_pred - v_t) ** 2).mean().item() * x_1.shape[0]
        n += x_1.shape[0]
    model.train()
    return total / n


@torch.no_grad()
def save_test_samples(epoch):
    """采样 16 张生成图并保存 (训练中测试)"""
    model.eval()
    samples = model.sample(50, [16, 1, 28, 28], DEVICE)
    grid = make_grid(samples, nrow=4, padding=2, normalize=True)
    save_image(grid, os.path.join(PLOTS_DIR, f'cfm_official_gen_epoch_{epoch:03d}.png'))
    model.train()
    return grid


# ============================================================================
# 训练主循环
# ============================================================================
logger.info(f"\n--- 开始训练 CFM 官方版 ({n_epochs} epochs) ---")
model.train()
t0 = time.time()

for epoch in range(start_epoch, n_epochs_target):
    total_loss = 0.0
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
    eta = ep_time * (n_epochs_target - epoch - 1) / 60
    lr_now = optimizer.param_groups[0]['lr']
    logger.info(f"  Epoch {epoch+1:3d}/{n_epochs_target} | CFM Loss: {avg_loss:.6f} | lr={lr_now:.2e} | {ep_time:.0f}s | ETA {eta:.1f}min")
    flush_logs()

    # 每轮结束: 更新 EMA 滑动平均; 微调阶段推进学习率调度
    ema.update(model)
    if scheduler is not None:
        scheduler.step()

    # 训练中测试: 每 TEST_INTERVAL epochs (用 EMA 权重评估与保存)
    if (epoch + 1) % TEST_INTERVAL == 0:
        ema.apply_shadow(model)   # 切换到 EMA 权重
        val_loss = evaluate_val_loss()
        val_losses.append((epoch + 1, val_loss))
        logger.info(f"    [OK] 验证 loss (EMA): {val_loss:.6f}")
        save_test_samples(epoch + 1)
        logger.info(f"    [OK] 采样图: plots/cfm_official_gen_epoch_{epoch+1:03d}.png")

        # 最优模型: 按验证 loss 保留前 BEST_KEEP 个 (EMA 权重), 自动清理多余
        best_records = []
        for f in os.listdir(CKPT_DIR):
            if f.startswith(CKPT_PREFIX) and f.endswith('.pt'):
                ck = torch.load(os.path.join(CKPT_DIR, f), map_location='cpu', weights_only=False)
                best_records.append((ck['val_loss'], f, ck))
        best_records.append((val_loss, f'{CKPT_PREFIX}{epoch+1}.pt', {
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'ema': True,
            'model_state_dict': model.state_dict(),
            'config': {'hidden_dim': hidden_dim, 'n_layers': n_layers, 'n_epochs': n_epochs},
        }))
        best_records.sort(key=lambda x: x[0])   # val_loss 升序
        keep = best_records[:BEST_KEEP]
        for _, fname, ck in keep:
            torch.save(ck, os.path.join(CKPT_DIR, fname))
        keep_names = {fname for _, fname, _ in keep}
        for f in os.listdir(CKPT_DIR):
            if f.startswith(CKPT_PREFIX) and f.endswith('.pt') and f not in keep_names:
                os.remove(os.path.join(CKPT_DIR, f))
        logger.info(f"    [OK] 最优模型 Top-{BEST_KEEP} (EMA): " + ", ".join(sorted(keep_names)))

        ema.restore(model)        # 恢复训练权重

        # 续训断点 (训练权重 + EMA 影子, 覆盖保存)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': {k: v.detach().cpu() for k, v in ema.shadow.items()},
            'epoch_losses': epoch_losses,
            'val_losses': val_losses,
            'config': {'hidden_dim': hidden_dim, 'n_layers': n_layers,
                       'lr': lr, 'sigma_min': sigma_min, 'n_epochs': n_epochs},
        }, os.path.join(CKPT_DIR, RESUME_CKPT))
        logger.info(f"    [OK] 续训断点: checkpoints/{RESUME_CKPT}")
        flush_logs()

t1 = time.time()
logger.info(f"\n--- 训练完成 ---")
logger.info(f"  耗时: {t1-t0:.0f}s ({(t1-t0)/60:.1f}min)")
logger.info(f"  CFM Loss: {epoch_losses[0]:.6f} -> {epoch_losses[-1]:.6f}")

# 保存最终模型 (取验证 loss 最优的一个)
best_path = None
best_v = float('inf')
for f in os.listdir(CKPT_DIR):
    if f.startswith(CKPT_PREFIX) and f.endswith('.pt'):
        ck = torch.load(os.path.join(CKPT_DIR, f), map_location='cpu', weights_only=False)
        if ck['val_loss'] < best_v:
            best_v = ck['val_loss']
            best_path = os.path.join(CKPT_DIR, f)
model_path = os.path.join(CURRENT_DIR, 'model_cfm_official.pth')
if best_path is not None:
    best_state = torch.load(best_path, map_location=DEVICE, weights_only=False)['model_state_dict']
    torch.save({'model_state_dict': best_state}, model_path)
    logger.info(f"  模型(最优, val_loss={best_v:.6f}): {model_path}")
else:
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    logger.info(f"  模型: {model_path}")

# Loss 曲线 (训练 + 验证)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=1.5, label='Train CFM Loss')
if val_losses:
    vx, vy = zip(*val_losses)
    ax.plot(vx, vy, 'o-', linewidth=1.5, label='Val CFM Loss', markersize=4)
ax.set_xlabel('Epoch'); ax.set_ylabel('CFM Loss')
ax.set_title('Conditional Flow Matching Loss 曲线 (MNIST 官方配置)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'cfm_official_loss.png'), dpi=150)
plt.close()
logger.info(f"  Loss 曲线: plots/cfm_official_loss.png")

# ============================================================================
# 生成测试 (最终)
# ============================================================================
logger.info(f"\n--- 生成测试 ---")
model.eval()

# 不同 T 值对比
for n_flows, name in [(2, 'T2'), (5, 'T5'), (10, 'T10'), (25, 'T25'), (50, 'T50')]:
    with torch.no_grad():
        x_hat = model.sample(n_flows, [1, 1, 28, 28], DEVICE)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(x_hat[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy(), cmap='gray')
    ax.axis('off')
    ax.set_title(f'T={n_flows}')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cfm_official_flow_{name}.png'), dpi=150)
    plt.close()
logger.info(f"  T=2/5/10/25/50 单样本对比图已保存")

# 64 张批量生成 (T=100)
B = inference_batch_size
with torch.no_grad():
    x_hats = model.sample(100, [B, 1, 28, 28], DEVICE)
grid = make_grid(x_hats, nrow=8, padding=2, normalize=True)
save_image(grid, os.path.join(PLOTS_DIR, 'cfm_official_generated_64.png'))
logger.info(f"  64 张生成: plots/cfm_official_generated_64.png")

real_batch = next(iter(test_loader))[0][:inference_batch_size]
grid_real = make_grid(real_batch, nrow=8, padding=2, normalize=True)
save_image(grid_real, os.path.join(PLOTS_DIR, 'cfm_official_real_64.png'))
logger.info(f"  64 张真实: plots/cfm_official_real_64.png")

for i in range(min(16, len(x_hats))):
    save_image(x_hats[i], os.path.join(OUT_DIR, f'cfm_official_sample_{i:02d}.png'))

with open(os.path.join(CURRENT_DIR, 'cfm_official_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'model': 'CFM MNIST 官方配置 (ConvBlock x8, hidden=256)',
        'params_m': n_params / 1e6,
        'n_epochs': n_epochs_target,
        'batch_size': train_batch_size,
        'lr': lr,
        'ema_decay': EMA_DECAY,
        'final_loss': epoch_losses[-1],
        'val_losses': val_losses,
        'time_min': (t1 - t0) / 60,
        'model_path': model_path,
    }, f, ensure_ascii=False, indent=2)
logger.info(f"  总结: cfm_official_summary.json")
logger.info(f"\n[OK] CFM MNIST 官方配置版训练完成！")
