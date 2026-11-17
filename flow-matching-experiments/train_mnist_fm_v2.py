"""
=============================================================================
MNIST Flow Matching v2 — 改进版训练
=============================================================================
改进点（相对 v1）：
  1. 更多 epochs: 20 → 60
  2. LR warmup + cosine: 避免 LR 过早降到 0 导致训练停滞
  3. EMA (指数移动平均): 提升生成质量与稳定性
  4. 更大 batch_size: 128 → 256（充分利用 8G 显存）
  5. 生成用 EMA 权重 + 50 步 ODE
=============================================================================
"""

import os, sys, time, json, logging, traceback, copy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import make_grid

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
CKPT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
DATA_DIR = os.path.join(CURRENT_DIR, 'mnist_data')
OUT_DIR = os.path.join(CURRENT_DIR, 'generated_mnist_v2')
for d in [PLOTS_DIR, LOGS_DIR, CKPT_DIR, DATA_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'mnist_fm_v2_log_{TIMESTAMP}.txt')

logger = logging.getLogger('MNIST_FM_V2')
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"{'='*60}")
logger.info(f"MNIST Flow Matching v2（改进版）")
logger.info(f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
logger.info(f"{'='*60}")

# ============================================================================
# UNet 架构（与 v1 相同，复用成熟项目）
# ============================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embedding_l = nn.Linear(3, dim)
    def forward(self, time):
        time = torch.cat([time, torch.cos(time), torch.sin(time)], dim=-1)
        return self.embedding_l(time)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, c=64):
        super().__init__()
        self.time_mlp1 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c)
        )
        self.time_mlp2 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 2*c), nn.SiLU(), nn.Linear(2*c, 2*c)
        )
        self.time_mlp3 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 4*c), nn.SiLU(), nn.Linear(4*c, 4*c)
        )
        self.conv1 = DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(c, 2*c)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(2*c, 4*c)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(4*c, 8*c)
        self.upconv3 = nn.ConvTranspose2d(8*c, 4*c, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(12*c, 4*c)
        self.upconv2 = nn.ConvTranspose2d(4*c, 2*c, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(6*c, 2*c)
        self.upconv1 = nn.ConvTranspose2d(2*c, c, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(3*c, c)
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = t.view(-1, 1)
        if t.numel() == 1:
            t = t.expand(x.shape[0], -1)
        t1 = self.time_mlp1(t)
        t2 = self.time_mlp2(t)
        t3 = self.time_mlp3(t)

        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)

        def pad_if_needed(u, s):
            if (u.shape[-1] + 1) == s.shape[-1]:
                return F.pad(u, (0, 1, 0, 1), mode='replicate')
            return u

        up3 = pad_if_needed(self.upconv3(conv4), conv3)
        t3b = t3[:,:,None,None].expand(-1,-1,up3.size(2),up3.size(3))
        conv5 = self.conv5(torch.cat([up3, conv3, t3b], dim=1))

        up2 = pad_if_needed(self.upconv2(conv5), conv2)
        t2b = t2[:,:,None,None].expand(-1,-1,up2.size(2),up2.size(3))
        conv6 = self.conv6(torch.cat([up2, conv2, t2b], dim=1))

        up1 = pad_if_needed(self.upconv1(conv6), conv1)
        t1b = t1[:,:,None,None].expand(-1,-1,up1.size(2),up1.size(3))
        conv7 = self.conv7(torch.cat([up1, conv1, t1b], dim=1))

        return self.final_conv(conv7)

# ============================================================================
# EMA 实现
# ============================================================================
class EMA:
    """指数移动平均，稳定训练并提升采样质量"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

# ============================================================================
# 生成函数
# ============================================================================
class FMVelocity(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, x, t, **extras):
        if t.numel() == 1:
            t = t.expand(x.shape[0]).to(x.device)
        return self.unet(x, t)

@torch.no_grad()
def generate(unet, n_samples=32, num_steps=50, device='cuda'):
    """用 ODESolver 生成，返回 [0,1] 图像"""
    vel = FMVelocity(unet).to(device)
    solver = ODESolver(velocity_model=vel)
    x_init = torch.randn(n_samples, 1, 28, 28, device=device)
    time_grid = torch.tensor([0.0, 1.0], device=device)
    step_size = 1.0 / num_steps
    gen = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid, method='euler')
    gen = (gen + 1) / 2
    return torch.clamp(gen, 0.0, 1.0)

def save_generated_grid(images, path, nrow=8):
    grid = make_grid(images, nrow=nrow, normalize=False, value_range=(0,1))
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid.permute(1,2,0).cpu().numpy())
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ============================================================================
# 主训练
# ============================================================================
def main():
    try:
        CONFIG = {
            'c_unet': 64,
            'batch_size': 256,
            'n_epochs': 60,
            'lr': 1e-4,
            'num_workers': 0,
            'warmup_epochs': 5,
            'ema_decay': 0.999,
            'gen_interval': 5,
        }

        logger.info(f"\n--- 配置 (v2) ---")
        for k, v in CONFIG.items():
            logger.info(f"  {k}: {v}")

        # 数据
        logger.info(f"\n--- 步骤1: 加载 MNIST 数据集 ---")
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
        logger.info(f"  训练集: {len(train_ds)} 张 | 测试集: {len(test_ds)} 张")

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                                  shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'],
                                 shuffle=False, num_workers=CONFIG['num_workers'])

        real_imgs = next(iter(test_loader))[0][:32].to(device)
        save_generated_grid(real_imgs, os.path.join(PLOTS_DIR, 'mnist_real_samples.png'))
        logger.info(f"  保存真实样本: plots/mnist_real_samples.png")

        # 模型 + EMA
        logger.info(f"\n--- 步骤2: 初始化模型 ---")
        unet = UNet(in_channels=1, out_channels=1, c=CONFIG['c_unet']).to(device)
        ema = EMA(unet, decay=CONFIG['ema_decay'])
        n_params = sum(p.numel() for p in unet.parameters())
        logger.info(f"  参数量: {n_params:,} ({n_params/1e6:.1f}M) | EMA decay: {CONFIG['ema_decay']}")

        # 组件
        cond_ot_path = CondOTProbPath()
        criterion = nn.MSELoss()
        opt = torch.optim.AdamW(unet.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
        # warmup + cosine
        total_steps = CONFIG['n_epochs'] * len(train_loader)
        warmup_steps = CONFIG['warmup_epochs'] * len(train_loader)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
            ],
            milestones=[warmup_steps]
        )
        scaler = torch.amp.GradScaler('cuda')

        # 训练
        logger.info(f"\n--- 步骤3: 训练 {CONFIG['n_epochs']} epochs ---")
        t0 = time.time()
        train_losses, val_losses = [], []
        global_step = 0

        for ep in range(CONFIG['n_epochs']):
            unet.train()
            ep_loss = 0
            nb = 0
            ep_start = time.time()

            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                imgs = imgs * 2.0 - 1.0  # [0,1] -> [-1,1]

                x0 = torch.randn_like(imgs)
                t = torch.rand(imgs.shape[0], device=device)
                pb = cond_ot_path.sample(x0, imgs, t)

                with torch.amp.autocast('cuda'):
                    pred = unet(pb.x_t, pb.t)
                    loss = criterion(pred, pb.dx_t)

                opt.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

                ema.update()
                ep_loss += loss.item()
                nb += 1
                global_step += 1

            avg_train = ep_loss / nb
            train_losses.append(avg_train)

            # 验证（用 EMA 权重）
            ema.apply_shadow()
            unet.eval()
            vloss, vcount = 0, 0
            with torch.no_grad():
                for imgs, _ in test_loader:
                    imgs = (imgs.to(device) * 2.0) - 1.0
                    x0 = torch.randn_like(imgs)
                    t = torch.rand(imgs.shape[0], device=device)
                    pb = cond_ot_path.sample(x0, imgs, t)
                    pred = unet(pb.x_t, pb.t)
                    vloss += criterion(pred, pb.dx_t).item()
                    vcount += 1
            avg_val = vloss / vcount
            val_losses.append(avg_val)
            ema.restore()

            ep_time = time.time() - ep_start
            logger.info(
                f"  Epoch {ep+1:3d}/{CONFIG['n_epochs']} | "
                f"TrainLoss: {avg_train:.5f} | ValLoss: {avg_val:.5f} | "
                f"LR: {sched.get_last_lr()[0]:.2e} | {ep_time:.0f}s"
            )
            flush_logs()

            # 定期生成（用 EMA）
            if (ep+1) % CONFIG['gen_interval'] == 0 or ep == 0:
                ema.apply_shadow()
                unet.eval()
                with torch.no_grad():
                    gen_imgs = generate(unet, n_samples=64, num_steps=50, device=device)
                ema.restore()
                save_generated_grid(gen_imgs,
                                   os.path.join(PLOTS_DIR, f'mnist_v2_gen_epoch_{ep+1:03d}.png'))
                logger.info(f"    ✓ 生成: plots/mnist_v2_gen_epoch_{ep+1:03d}.png")

                torch.save({
                    'epoch': ep+1,
                    'model_state_dict': unet.state_dict(),
                    'ema_state_dict': ema.shadow,
                    'train_loss': avg_train,
                    'val_loss': avg_val,
                }, os.path.join(CKPT_DIR, f'mnist_fm_v2_epoch_{ep+1}.pt'))

        t1 = time.time()
        logger.info(f"\n--- 训练完成 ---")
        logger.info(f"  耗时: {t1-t0:.0f}s ({(t1-t0)/60:.1f}min)")
        logger.info(f"  Train Loss: {train_losses[0]:.5f} → {train_losses[-1]:.5f}")
        logger.info(f"  Val   Loss: {val_losses[0]:.5f} → {val_losses[-1]:.5f}")

        # 保存最终模型（EMA）
        ema.apply_shadow()
        model_path = os.path.join(CURRENT_DIR, 'model_mnist_fm_v2.pth')
        torch.save({'model_state_dict': unet.state_dict()}, model_path)
        ema.restore()
        logger.info(f"  模型(EMA): {model_path}")

        # Loss 曲线
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label='Train Loss', linewidth=1.5)
        ax.plot(val_losses, label='Val Loss (EMA)', linewidth=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title('MNIST Flow Matching v2 Loss 曲线'); ax.grid(True, alpha=0.3)
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'mnist_v2_loss.png'), dpi=150); plt.close()

        # 最终生成
        logger.info(f"\n--- 步骤4: 最终生成 ---")
        ema.apply_shadow()
        unet.eval()
        for steps, name in [(10, '10步'), (50, '50步'), (100, '100步')]:
            with torch.no_grad():
                gen_imgs = generate(unet, n_samples=100, num_steps=steps, device=device)
            save_generated_grid(gen_imgs,
                               os.path.join(PLOTS_DIR, f'mnist_v2_gen_final_{steps}steps.png'), nrow=10)
            logger.info(f"  ODE {name}: plots/mnist_v2_gen_final_{steps}steps.png")
        ema.restore()

        # 总结
        with open(os.path.join(CURRENT_DIR, 'mnist_fm_v2_summary.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'model': 'MNIST Flow Matching v2 (UNet + CondOT + EMA)',
                'params_m': n_params / 1e6,
                'n_epochs': CONFIG['n_epochs'],
                'batch_size': CONFIG['batch_size'],
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'time_min': (t1-t0)/60,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ MNIST Flow Matching v2 完成！")

    except Exception as e:
        logger.error(f"失败: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()