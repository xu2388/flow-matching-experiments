# -*- coding: utf-8 -*-
"""
=============================================================================
CFM IMBR 图像生成版训练 (随机配对 / 随机采样, 标准 Conditional Flow Matching)
=============================================================================
思路转变:
  之前失败: 点云拼字母 (无序点集 + 速度场回归, 表示不适配)
  本次方案: IMBR 渲染为 32x128 图像, 完全复用 MNIST 上验证成功的 CFM 流程
      x_0 ~ N(0,1) 高斯噪声图, x_1 = 随机采样一张 IMBR 增强图像 (随机配对)
      直线调度 x_t = (1-t)x_0 + t*x_1, 回归速度场 v = x_1 - x_0
工程功能 (与 train_cfm_official.py 相同):
  断点续训 / 最优 Top-5 / EMA / cosine / 训练中采样测试
=============================================================================
"""

import os, sys, time, json, logging
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
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 路径和日志
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
CKPT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
OUT_DIR = os.path.join(CURRENT_DIR, 'generated_imbr')
for d in [PLOTS_DIR, LOGS_DIR, CKPT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'cfm_imbr_log_{TIMESTAMP}.txt')

logger = logging.getLogger('CFM_IMBR')
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
# 配置
# ============================================================================
gpu_id = 0
cuda = True

IMG_H, IMG_W = 32, 128            # IMBR 图像尺寸 (4 字母 x 32x32)
hidden_dim = 256
n_layers = 8
lr = 5e-5
sigma_min = 0
FINE_TUNE_LR = 2e-5
FINE_TUNE_EPOCHS = 50
EMA_DECAY = 0.999
RESUME_FROM_BEST = False          # IMBR: 默认从最近断点续训 (首次从头)
n_epochs = 80
STEPS_PER_EPOCH = 125             # 每 epoch 125 步 x batch 32 = 4000 样本
train_batch_size = 32
inference_batch_size = 16

seed = 1234
CKPT_PREFIX = 'imbr_best_'
RESUME_CKPT = 'imbr_resume.pt'
BEST_KEEP = 5
TEST_INTERVAL = 5

os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpu_id)
DEVICE = torch.device("cuda:0" if cuda else "cpu")
img_size = (IMG_H, IMG_W, 1)
hidden_dims = [hidden_dim for _ in range(n_layers)]

torch.manual_seed(seed)
np.random.seed(seed)

logger.info("=" * 60)
logger.info("CFM IMBR 图像生成版训练 (随机配对, 标准 CFM)")
logger.info(f"设备: {DEVICE} ({torch.cuda.get_device_name(0) if cuda else 'CPU'})")
logger.info(f"图像: {IMG_H}x{IMG_W} | 模型: hidden={hidden_dim} x {n_layers}层")
logger.info(f"训练: {n_epochs} epochs x {STEPS_PER_EPOCH} 步, batch={train_batch_size}, lr={lr}")
logger.info("=" * 60)

# ============================================================================
# 数据: 在线随机渲染 IMBR 图像 (无限增强数据, 防过拟合)
# ============================================================================
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",     # Arial Bold
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\consolab.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
if FONT_PATH is None:
    raise RuntimeError("未找到可用字体 (arial/consola 等), 请检查 C:\\Windows\\Fonts")
logger.info(f"字体: {FONT_PATH}")


def render_imbr(rng: np.random.RandomState) -> Image.Image:
    """渲染一张随机增强的 IMBR 图像 (32x128, 白字黑底)"""
    canvas = Image.new('L', (IMG_W, IMG_H), 0)
    font_size = int(rng.randint(22, 28))          # 字号抖动 = 笔画粗细变化
    font = ImageFont.truetype(FONT_PATH, font_size)

    for i, ch in enumerate('IMBR'):
        cell = Image.new('L', (32, 32), 0)
        d = ImageDraw.Draw(cell)
        # 字母居中绘制
        bbox = d.textbbox((0, 0), ch, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((32 - tw) / 2 - bbox[0], (32 - th) / 2 - bbox[1]), ch, fill=255, font=font)
        # 每字母独立: 小位移 + 小旋转
        ang = rng.uniform(-10, 10)
        cell = cell.rotate(ang, resample=Image.BILINEAR, fillcolor=0)
        dx, dy = int(rng.randint(-2, 3)), int(rng.randint(-1, 2))
        canvas.paste(cell, (i * 32 + dx, dy), cell)

    # 全局: 模糊 (笔画软化) + 亮度
    if rng.rand() < 0.8:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    bright = rng.uniform(0.75, 1.0)
    canvas = canvas.point(lambda v: int(v * bright))
    return canvas


class IMBRDataset(Dataset):
    """在线渲染数据集; fixed_seed 时为固定验证集 (loss 可比)"""
    def __init__(self, length, fixed_seed=None):
        self.length = length
        self.fixed_seed = fixed_seed
        if fixed_seed is not None:
            self.rng = np.random.RandomState(fixed_seed)
            self.cache = [np.asarray(render_imbr(self.rng), dtype=np.float32) / 255.0
                          for _ in range(length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.fixed_seed is not None:
            arr = self.cache[idx]
        else:
            arr = np.asarray(render_imbr(np.random.RandomState()), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)   # [1, 32, 128]


train_dataset = IMBRDataset(STEPS_PER_EPOCH * train_batch_size)   # 每 epoch 重新渲染
val_dataset = IMBRDataset(256, fixed_seed=999)                     # 固定验证集

kwargs = {'num_workers': 0, 'pin_memory': False}
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
val_loader = DataLoader(val_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)

logger.info(f"数据: 在线渲染 IMBR (每 epoch {STEPS_PER_EPOCH * train_batch_size} 张) | 验证集 256 张 (固定)")

# ============================================================================
# 模型 (与 MNIST 成功版本完全相同: ConvBlock 残差堆叠)
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
epoch_losses = []
val_losses = []
ema_shadow_loaded = None
scheduler = None
resume_from_best = False
resume_path = os.path.join(CKPT_DIR, RESUME_CKPT)

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

if not resume_from_best and os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    epoch_losses = list(ckpt.get('epoch_losses', []))
    val_losses = list(ckpt.get('val_losses', []))
    # 旧断点的 EMA 影子是"每 epoch 更新"时代的坏权重, 丢弃 (重置为当前模型权重, 5 epochs 内重新收敛)
    ema_shadow_loaded = None
    logger.info(f"[OK] 从断点恢复: {RESUME_CKPT} (epoch {start_epoch}, 旧 EMA 影子已重置)")
if start_epoch == 0 and not resume_from_best and not os.path.exists(resume_path):
    logger.info("无断点, 从头训练")

# EMA 必须在权重加载之后初始化 (MNIST 踩过的坑)
ema = EMA(model, EMA_DECAY)
if ema_shadow_loaded is not None:
    ema.shadow = {k: v.to(DEVICE) for k, v in ema_shadow_loaded.items()}
    logger.info("  已加载 EMA 影子权重")
else:
    logger.info("  EMA 初始化为当前(加载后)权重")

if start_epoch > 0:
    lr = FINE_TUNE_LR
    for g in optimizer.param_groups:
        g['lr'] = lr
    n_epochs_target = min(n_epochs, start_epoch + FINE_TUNE_EPOCHS)
    scheduler = CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS, eta_min=0)
    logger.info(f"[FT] 微调模式: lr={lr}, cosine, EMA decay={EMA_DECAY}")
    logger.info(f"  本次运行目标: {start_epoch} -> {n_epochs_target} epochs")
else:
    n_epochs_target = n_epochs
    logger.info(f"  首次从头训练: {n_epochs} epochs")


# ============================================================================
# 训练中测试
# ============================================================================
@torch.no_grad()
def evaluate_val_loss():
    model.eval()
    total = 0.0
    n = 0
    for x_1 in val_loader:
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
    model.eval()
    samples = model.sample(50, [4, 1, IMG_H, IMG_W], DEVICE)
    grid = make_grid(samples, nrow=2, padding=2, normalize=True)
    save_image(grid, os.path.join(PLOTS_DIR, f'cfm_imbr_gen_epoch_{epoch:03d}.png'))
    model.train()


# ============================================================================
# 训练主循环
# ============================================================================
logger.info(f"\n--- 开始训练 CFM IMBR ({n_epochs_target} epochs, 随机配对) ---")
model.train()
t0 = time.time()

for epoch in range(start_epoch, n_epochs_target):
    total_loss = 0.0
    n_batch = 0
    ep_start = time.time()
    for x_1 in train_loader:
        optimizer.zero_grad()
        x_1 = x_1.to(DEVICE)
        x_0 = torch.randn_like(x_1)                        # 随机噪声起点
        t = torch.rand(x_1.shape[0], 1, 1, 1, device=DEVICE)  # 随机时间

        x_t = model.interpolate(x_0, x_1, t)
        velocity_target = model.get_velocity(x_0, x_1)
        velocity_pred = model(x_t, t)

        loss = ((velocity_pred - velocity_target) ** 2).mean()
        total_loss += loss.item()
        n_batch += 1

        loss.backward()
        optimizer.step()
        ema.update(model)          # EMA 每 step 更新 (标准做法; 每 epoch 更新会导致影子收敛过慢)

    avg_loss = total_loss / n_batch
    epoch_losses.append(avg_loss)
    ep_time = time.time() - ep_start
    eta = ep_time * (n_epochs_target - epoch - 1) / 60
    lr_now = optimizer.param_groups[0]['lr']
    logger.info(f"  Epoch {epoch+1:3d}/{n_epochs_target} | CFM Loss: {avg_loss:.6f} | lr={lr_now:.2e} | {ep_time:.0f}s | ETA {eta:.1f}min")
    flush_logs()

    if scheduler is not None:
        scheduler.step()

    if (epoch + 1) % TEST_INTERVAL == 0:
        ema.apply_shadow(model)
        val_loss = evaluate_val_loss()
        val_losses.append((epoch + 1, val_loss))
        logger.info(f"    [OK] 验证 loss (EMA): {val_loss:.6f}")
        save_test_samples(epoch + 1)
        logger.info(f"    [OK] 采样图: plots/cfm_imbr_gen_epoch_{epoch+1:03d}.png")

        # 最优 Top-5 (EMA 权重)
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
            'config': {'hidden_dim': hidden_dim, 'n_layers': n_layers, 'img': (IMG_H, IMG_W)},
        }))
        best_records.sort(key=lambda x: x[0])
        keep = best_records[:BEST_KEEP]
        for _, fname, ck in keep:
            torch.save(ck, os.path.join(CKPT_DIR, fname))
        keep_names = {fname for _, fname, _ in keep}
        for f in os.listdir(CKPT_DIR):
            if f.startswith(CKPT_PREFIX) and f.endswith('.pt') and f not in keep_names:
                os.remove(os.path.join(CKPT_DIR, f))
        logger.info(f"    [OK] 最优模型 Top-{BEST_KEEP} (EMA): " + ", ".join(sorted(keep_names)))

        ema.restore(model)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': {k: v.detach().cpu() for k, v in ema.shadow.items()},
            'epoch_losses': epoch_losses,
            'val_losses': val_losses,
            'config': {'hidden_dim': hidden_dim, 'n_layers': n_layers,
                       'img': (IMG_H, IMG_W), 'lr': lr, 'sigma_min': sigma_min},
        }, os.path.join(CKPT_DIR, RESUME_CKPT))
        logger.info(f"    [OK] 续训断点: checkpoints/{RESUME_CKPT}")
        flush_logs()

t1 = time.time()
logger.info(f"\n--- 训练完成 ---")
logger.info(f"  耗时: {t1-t0:.0f}s ({(t1-t0)/60:.1f}min)")
logger.info(f"  CFM Loss: {epoch_losses[0]:.6f} -> {epoch_losses[-1]:.6f}")

# 最终模型 = 验证最优
best_path, best_v = None, float('inf')
for f in os.listdir(CKPT_DIR):
    if f.startswith(CKPT_PREFIX) and f.endswith('.pt'):
        ck = torch.load(os.path.join(CKPT_DIR, f), map_location='cpu', weights_only=False)
        if ck['val_loss'] < best_v:
            best_v, best_path = ck['val_loss'], os.path.join(CKPT_DIR, f)
model_path = os.path.join(CURRENT_DIR, 'model_cfm_imbr.pth')
if best_path is not None:
    best_state = torch.load(best_path, map_location=DEVICE, weights_only=False)['model_state_dict']
    torch.save({'model_state_dict': best_state}, model_path)
    logger.info(f"  模型(最优, val_loss={best_v:.6f}): {model_path}")
else:
    torch.save({'model_state_dict': model.state_dict()}, model_path)

# Loss 曲线
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=1.5, label='Train CFM Loss')
if val_losses:
    vx, vy = zip(*val_losses)
    ax.plot(vx, vy, 'o-', linewidth=1.5, label='Val CFM Loss', markersize=4)
ax.set_xlabel('Epoch'); ax.set_ylabel('CFM Loss')
ax.set_title('CFM Loss (IMBR, random pairing)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'cfm_imbr_loss.png'), dpi=150)
plt.close()
logger.info("  Loss 曲线: plots/cfm_imbr_loss.png")

# ============================================================================
# 最终生成: 16 生成 + 16 真实 对比
# ============================================================================
logger.info(f"\n--- 生成测试 ---")
# 用最优 EMA 权重生成 (而非最后一轮训练权重)
if best_path is not None:
    model.load_state_dict(best_state)
    logger.info(f"  已加载最优权重 (val_loss={best_v:.6f}) 用于生成")
model.eval()
B = inference_batch_size
with torch.no_grad():
    x_hats = model.sample(100, [B, 1, IMG_H, IMG_W], DEVICE).clamp(0, 1)
grid = make_grid(x_hats, nrow=4, padding=2)
save_image(grid, os.path.join(PLOTS_DIR, 'cfm_imbr_generated_16.png'))

reals = torch.stack([val_dataset[i] for i in range(B)])
grid_r = make_grid(reals, nrow=4, padding=2)
save_image(grid_r, os.path.join(PLOTS_DIR, 'cfm_imbr_real_16.png'))

# 上下拼接对比图 (Generated / GT)
from PIL import Image as PILImage
g1 = PILImage.open(os.path.join(PLOTS_DIR, 'cfm_imbr_generated_16.png'))
g2 = PILImage.open(os.path.join(PLOTS_DIR, 'cfm_imbr_real_16.png'))
w, h = g1.size
comb = PILImage.new('RGB', (w, h * 2 + 30), 'white')
comb.paste(g1, (0, 0))
comb.paste(g2, (0, h + 30))
comb.save(os.path.join(PLOTS_DIR, 'cfm_imbr_compare_16.png'))
logger.info("  16 生成 vs 16 真实对比: plots/cfm_imbr_compare_16.png")

for i in range(B):
    save_image(x_hats[i], os.path.join(OUT_DIR, f'imbr_sample_{i:02d}.png'))

with open(os.path.join(CURRENT_DIR, 'cfm_imbr_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'model': 'CFM IMBR image generation (ConvBlock x8, hidden=256, random pairing)',
        'params_m': n_params / 1e6,
        'img': [IMG_H, IMG_W],
        'n_epochs': n_epochs_target,
        'batch_size': train_batch_size,
        'lr': lr,
        'ema_decay': EMA_DECAY,
        'final_loss': epoch_losses[-1],
        'val_losses': val_losses,
        'time_min': (t1 - t0) / 60,
        'model_path': model_path,
    }, f, ensure_ascii=False, indent=2)
logger.info(f"\n[OK] CFM IMBR 训练完成!")
