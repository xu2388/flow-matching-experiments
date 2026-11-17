"""
=============================================================================
MNIST Flow Matching — 复现 flow-matching-mnist-main 实验
=============================================================================
核心组件（来自成熟项目）：
  1. UNet 速度场架构（复用 flow-matching-mnist-main 的 unet.py）
  2. Meta flow_matching 库：CondOTProbPath（条件 OT 配对）
  3. Meta flow_matching 库：ODESolver（dopri5 ODE 求解）

任务：从高斯噪声生成 MNIST 手写数字
=============================================================================
"""

import os, sys, time, json, logging, traceback
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
from torchvision.utils import make_grid, save_image

# Meta Flow Matching 库核心
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 路径和日志
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
CKPT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
DATA_DIR = os.path.join(CURRENT_DIR, 'mnist_data')
OUT_DIR = os.path.join(CURRENT_DIR, 'generated_mnist')
for d in [PLOTS_DIR, LOGS_DIR, CKPT_DIR, DATA_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'mnist_fm_log_{TIMESTAMP}.txt')

logger = logging.getLogger('MNIST_FM')
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
logger.info(f"MNIST Flow Matching 实验")
logger.info(f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
logger.info(f"{'='*60}")

# ============================================================================
# UNet 架构（直接复用 flow-matching-mnist-main 项目）
# ============================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embedding_l = nn.Linear(3, dim)
    def forward(self, time):
        device = time.device
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
        # Encoder
        self.conv1 = DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(c, 2*c)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(2*c, 4*c)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(4*c, 8*c)
        # Decoder
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
# 生成函数
# ============================================================================
class FMVelocity(nn.Module):
    """包装 UNet 成 velocity field：v(x_t, t)"""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, x, t, **extras):
        # t 可能是标量或 (1,) tensor，需要扩展到 batch 维
        if t.numel() == 1:
            t = t.expand(x.shape[0]).to(x.device)
        return self.unet(x, t)

@torch.no_grad()
def generate(unet, n_samples=32, num_steps=10, device='cuda', method='dopri5'):
    """用 ODESolver 从 t=0 积分到 t=1，生成 MNIST 图像"""
    vel = FMVelocity(unet).to(device)
    solver = ODESolver(velocity_model=vel)
    x_init = torch.randn(n_samples, 1, 28, 28, device=device)
    
    if method == 'euler':
        # Euler：用 step_size 固定步长
        time_grid = torch.tensor([0.0, 1.0], device=device)
        step_size = 1.0 / num_steps
        gen = solver.sample(
            x_init=x_init, step_size=step_size, time_grid=time_grid, method='euler'
        )
    else:
        # dopri5：自适应步长，step_size=None
        time_grid = torch.tensor([0.0, 1.0], device=device)
        gen = solver.sample(
            x_init=x_init, step_size=None, time_grid=time_grid,
            method='dopri5', atol=1e-4, rtol=1e-4
        )
    
    # 反归一化 [-1,1] -> [0,1]
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
# 主训练流程
# ============================================================================
def main():
    try:
        CONFIG = {
            'c_unet': 64,
            'batch_size': 128,
            'n_epochs': 20,
            'lr': 1e-4,
            'num_workers': 0,
        }

        logger.info(f"\n--- 配置 ---")
        for k, v in CONFIG.items():
            logger.info(f"  {k}: {v}")

        # 数据
        logger.info(f"\n--- 步骤1: 加载 MNIST 数据集 ---")
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
        logger.info(f"  训练集: {len(train_ds)} 张")
        logger.info(f"  测试集: {len(test_ds)} 张")

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                                  shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'],
                                 shuffle=False, num_workers=CONFIG['num_workers'])

        # 保存一些真实样本用于对比
        real_imgs = next(iter(test_loader))[0][:32].to(device)
        save_generated_grid(real_imgs, os.path.join(PLOTS_DIR, 'mnist_real_samples.png'))
        logger.info(f"  保存: plots/mnist_real_samples.png")

        # 模型
        logger.info(f"\n--- 步骤2: 初始化 UNet ---")
        unet = UNet(in_channels=1, out_channels=1, c=CONFIG['c_unet']).to(device)
        n_params = sum(p.numel() for p in unet.parameters())
        logger.info(f"  参数量: {n_params:,} ({n_params/1e6:.1f}M)")

        # 组件
        cond_ot_path = CondOTProbPath()
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(unet.parameters(), lr=CONFIG['lr'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG['n_epochs'], eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')

        # 训练
        logger.info(f"\n--- 步骤3: 训练 {CONFIG['n_epochs']} epochs ---")
        t0 = time.time()
        train_losses = []
        val_losses = []
        normalize = True

        for ep in range(CONFIG['n_epochs']):
            unet.train()
            ep_loss = 0
            n_batches = 0
            ep_start = time.time()

            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                bs = imgs.shape[0]
                if normalize:
                    imgs = imgs * 2.0 - 1.0  # [0,1] -> [-1,1]

                # 噪声
                x0 = torch.randn_like(imgs)
                t = torch.rand(bs, device=device)

                # CondOT 路径采样
                path_batch = cond_ot_path.sample(x0, imgs, t)
                xt = path_batch.x_t
                ut = path_batch.dx_t

                with torch.amp.autocast('cuda'):
                    pred = unet(xt, t)
                    loss = criterion(pred, ut)

                opt.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                ep_loss += loss.item()
                n_batches += 1

            avg_train = ep_loss / n_batches
            train_losses.append(avg_train)

            # 验证
            unet.eval()
            vloss = 0
            vcount = 0
            with torch.no_grad():
                for imgs, _ in test_loader:
                    imgs = imgs.to(device)
                    if normalize:
                        imgs = imgs * 2.0 - 1.0
                    x0 = torch.randn_like(imgs)
                    t = torch.rand(imgs.shape[0], device=device)
                    pb = cond_ot_path.sample(x0, imgs, t)
                    pred = unet(pb.x_t, pb.t)
                    vloss += criterion(pred, pb.dx_t).item()
                    vcount += 1
            avg_val = vloss / vcount
            val_losses.append(avg_val)

            sched.step()
            ep_time = time.time() - ep_start

            logger.info(
                f"  Epoch {ep+1:3d}/{CONFIG['n_epochs']} | "
                f"TrainLoss: {avg_train:.5f} | ValLoss: {avg_val:.5f} | "
                f"LR: {sched.get_last_lr()[0]:.6f} | {ep_time:.0f}s"
            )
            flush_logs()

            # 定期生成样本
            if (ep+1) % 5 == 0 or ep == 0:
                unet.eval()
                with torch.no_grad():
                    gen_imgs = generate(unet, n_samples=64, num_steps=10, device=device)
                save_generated_grid(gen_imgs,
                                   os.path.join(PLOTS_DIR, f'mnist_gen_epoch_{ep+1:03d}.png'))
                logger.info(f"    ✓ 生成样本: plots/mnist_gen_epoch_{ep+1:03d}.png")

                # 保存检查点
                ckpt_path = os.path.join(CKPT_DIR, f'mnist_fm_epoch_{ep+1}.pt')
                torch.save({
                    'epoch': ep+1,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'train_loss': avg_train,
                    'val_loss': avg_val,
                }, ckpt_path)

        t1 = time.time()
        logger.info(f"\n--- 训练完成 ---")
        logger.info(f"  耗时: {t1-t0:.0f}s ({(t1-t0)/60:.1f}min)")
        logger.info(f"  Train Loss: {train_losses[0]:.5f} → {train_losses[-1]:.5f}")
        logger.info(f"  Val   Loss: {val_losses[0]:.5f} → {val_losses[-1]:.5f}")

        # 保存最终模型
        model_path = os.path.join(CURRENT_DIR, 'model_mnist_fm.pt')
        torch.save(unet.state_dict(), model_path)
        logger.info(f"  模型: {model_path}")

        # Loss 曲线
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label='Train Loss', linewidth=1.5)
        ax.plot(val_losses, label='Val Loss', linewidth=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title('MNIST Flow Matching Loss 曲线'); ax.grid(True, alpha=0.3)
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'mnist_fm_loss.png'), dpi=150); plt.close()

        # 最终生成：用更多步
        logger.info(f"\n--- 步骤4: 最终生成 ---")
        unet.eval()
        for steps, name in [(10, '10步'), (50, '50步'), (100, '100步')]:
            with torch.no_grad():
                gen_imgs = generate(unet, n_samples=100, num_steps=steps, device=device)
            save_generated_grid(gen_imgs,
                               os.path.join(PLOTS_DIR, f'mnist_gen_final_{steps}steps.png'), nrow=10)
            # 保存单独图像
            for i, img in enumerate(gen_imgs[:32]):
                save_image(img, os.path.join(OUT_DIR, f'sample_steps{steps}_{i:02d}.png'))
            logger.info(f"  ODE {name}: plots/mnist_gen_final_{steps}steps.png")

        # 保存总结
        summary = {
            'model': 'MNIST Flow Matching (UNet + CondOT)',
            'params_m': n_params / 1e6,
            'n_epochs': CONFIG['n_epochs'],
            'batch_size': CONFIG['batch_size'],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'time_min': (t1-t0)/60,
        }
        with open(os.path.join(CURRENT_DIR, 'mnist_fm_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ MNIST Flow Matching 实验完成！")
        logger.info(f"  查看生成结果: plots/mnist_gen_final_*.png")

    except Exception as e:
        logger.error(f"失败: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()