"""
=============================================================================
IMBR Flow Matching — 图像级生成（优化版）
=============================================================================
架构（复用 MNIST 成功方案）：
  1. UNet 速度场（替换之前的点云 MLP）
  2. Meta flow_matching 库：CondOTProbPath（精确条件OT配对）
  3. Meta flow_matching 库：ODESolver（自适应 dopri5）
  4. 数据增强：随机平移 + 抖动（解决单一样本过拟合问题）

生成格式：64x256 单通道灰度图 → I M B R 四个字母
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
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
CKPT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
OUT_DIR = os.path.join(CURRENT_DIR, 'generated_imbr')
for d in [PLOTS_DIR, LOGS_DIR, CKPT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'imbr_img_fm_log_{TIMESTAMP}.txt')

logger = logging.getLogger('IMBR_FM')
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
logger.info(f"IMBR 图像级 Flow Matching")
logger.info(f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
logger.info(f"{'='*60}")

# ============================================================================
# 1. IMBR 图像生成 + 数据增强
# ============================================================================
IMG_H, IMG_W = 64, 256  # 高度64，宽度256（4个字母）

def generate_imbr_image():
    """生成 IMBR 64x256 灰度图"""
    try:
        font = ImageFont.truetype("arial.ttf", 56)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 56)
        except Exception:
            font = ImageFont.load_default()
    img = Image.new('L', (IMG_W, IMG_H), 0)
    draw = ImageDraw.Draw(img)
    # 居中绘制 I M B R
    text = "IMBR"
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (IMG_W - tw) // 2 - bbox[0]
        y = (IMG_H - th) // 2 - bbox[1]
    except Exception:
        x, y = 10, 4
    draw.text((x, y), text, fill=255, font=font)
    return np.array(img, dtype=np.float32) / 255.0  # [0,1]


class IMBRAugDataset(Dataset):
    """
    通过数据增强生成多样本：
      - 随机水平/垂直平移（像素级）
      - 随机抖动噪声
      - 轻微缩放（如果空间允许）
    """
    def __init__(self, base_img, n_augments=100000):
        self.base = base_img  # (H, W) float32 [0,1]
        self.n = n_augments
        logger.info(f"  数据集: {n_augments} 个增强样本 (基础: {IMG_H}x{IMG_W})")
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        img = self.base.copy()
        H, W = img.shape
        
        # 1. 随机平移（±3像素）
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        if dx != 0 or dy != 0:
            src_x0 = max(0, -dx); dst_x0 = max(0, dx)
            src_y0 = max(0, -dy); dst_y0 = max(0, dy)
            w = min(W - src_x0, W - dst_x0)
            h = min(H - src_y0, H - dst_y0)
            result = np.zeros_like(img)
            result[dst_y0:dst_y0+h, dst_x0:dst_x0+w] = img[src_y0:src_y0+h, src_x0:src_x0+w]
            img = result
        
        # 2. 添加小噪声（σ<=0.05）
        img += np.random.randn(*img.shape).astype(np.float32) * 0.03
        img = np.clip(img, 0.0, 1.0)
        
        return torch.tensor(img.reshape(1, H, W), dtype=torch.float32)

# ============================================================================
# 2. UNet 架构（适配 64x256，复用 MNIST 成功结构）
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

class UNet64x256(nn.Module):
    """
    适配 64x256 输入的 UNet：
      Level 0:  64 x 256    (c)
      Level 1:  32 x 128   (2c)
      Level 2:  16 x 64    (4c)
      Level 3:  8 x 32     (8c)
    """
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
        self.pool1 = nn.MaxPool2d(2)  # 64x256 -> 32x128
        self.conv2 = DoubleConv(c, 2*c)
        self.pool2 = nn.MaxPool2d(2)  # 32x128 -> 16x64
        self.conv3 = DoubleConv(2*c, 4*c)
        self.pool3 = nn.MaxPool2d(2)  # 16x64 -> 8x32
        self.conv4 = DoubleConv(4*c, 8*c)
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(8*c, 4*c, kernel_size=2, stride=2)  # 8x32 -> 16x64
        self.conv5 = DoubleConv(12*c, 4*c)
        self.upconv2 = nn.ConvTranspose2d(4*c, 2*c, kernel_size=2, stride=2)  # 16x64 -> 32x128
        self.conv6 = DoubleConv(6*c, 2*c)
        self.upconv1 = nn.ConvTranspose2d(2*c, c, kernel_size=2, stride=2)     # 32x128 -> 64x256
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
            # u: 上采样后  s: 对应encoder层
            dh = s.size(2) - u.size(2)
            dw = s.size(3) - u.size(3)
            if dh > 0 or dw > 0:
                return F.pad(u, (0, dw, 0, dh), mode='replicate')
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
# 3. 生成函数
# ============================================================================
class VelocityWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x, t, **extras):
        if t.numel() == 1:
            t = t.expand(x.shape[0]).to(x.device)
        return self.net(x, t)

@torch.no_grad()
def generate_images(unet, n_samples=4, device='cuda'):
    vel = VelocityWrapper(unet).to(device)
    solver = ODESolver(velocity_model=vel)
    x_init = torch.randn(n_samples, 1, IMG_H, IMG_W, device=device)
    time_grid = torch.tensor([0.0, 1.0], device=device)
    gen = solver.sample(
        x_init=x_init, step_size=None, time_grid=time_grid,
        method='dopri5', atol=1e-4, rtol=1e-4
    )
    gen = (gen + 1) / 2  # [-1,1] -> [0,1]
    return torch.clamp(gen, 0.0, 1.0)

def save_grid(images, path, figsize=(20, 8)):
    """保存多张图像到网格"""
    n = len(images)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False)
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze(0).cpu().numpy()
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def save_single(img, path):
    img = img.squeeze(0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# 4. 主训练
# ============================================================================
def main():
    try:
        CONFIG = {
            'c_unet': 96,
            'batch_size': 32,
            'n_epochs': 100,
            'n_augments': 20000,
            'lr': 1e-4,
            'save_interval': 20,
            'gen_interval': 10,
        }

        logger.info(f"\n--- 配置 ---")
        for k, v in CONFIG.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"  图像尺寸: {IMG_H}x{IMG_W}")

        # 1. 生成基础图像并创建数据集
        logger.info(f"\n--- 步骤1: 创建 IMBR 数据集（增强版） ---")
        base_img = generate_imbr_image()
        save_single(torch.tensor(base_img.reshape(1, IMG_H, IMG_W)),
                   os.path.join(PLOTS_DIR, 'imbr_img_target.png'))
        logger.info(f"  保存: plots/imbr_img_target.png")

        dataset = IMBRAugDataset(base_img, n_augments=CONFIG['n_augments'])
        loader = DataLoader(dataset, batch_size=CONFIG['batch_size'],
                           shuffle=True, num_workers=0, pin_memory=True)

        # 2. 创建模型
        logger.info(f"\n--- 步骤2: 初始化 UNet ---")
        unet = UNet64x256(in_channels=1, out_channels=1, c=CONFIG['c_unet']).to(device)
        n_params = sum(p.numel() for p in unet.parameters())
        logger.info(f"  参数量: {n_params:,} ({n_params/1e6:.1f}M)")

        # 3. 组件
        cond_ot_path = CondOTProbPath()
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(unet.parameters(), lr=CONFIG['lr'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG['n_epochs'], eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')

        # 4. 训练
        logger.info(f"\n--- 步骤3: 训练 {CONFIG['n_epochs']} epochs ---")
        t0 = time.time()
        losses = []

        for ep in range(CONFIG['n_epochs']):
            unet.train()
            ep_loss = 0
            nb = 0
            ep_start = time.time()

            for imgs in loader:
                imgs = imgs.to(device)
                bs = imgs.shape[0]
                imgs = imgs * 2.0 - 1.0  # [0,1] -> [-1,1]

                x0 = torch.randn_like(imgs)
                t = torch.rand(bs, device=device)

                pb = cond_ot_path.sample(x0, imgs, t)
                xt = pb.x_t
                ut = pb.dx_t

                with torch.amp.autocast('cuda'):
                    pred = unet(xt, t)
                    loss = criterion(pred, ut)

                opt.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                ep_loss += loss.item()
                nb += 1

            avg_loss = ep_loss / nb
            losses.append(avg_loss)
            sched.step()
            ep_time = time.time() - ep_start

            logger.info(
                f"  Epoch {ep+1:3d}/{CONFIG['n_epochs']} | "
                f"Loss: {avg_loss:.6f} | LR: {sched.get_last_lr()[0]:.6f} | {ep_time:.0f}s"
            )
            flush_logs()

            # 生成样本
            if (ep+1) % CONFIG['gen_interval'] == 0 or ep == 0:
                unet.eval()
                with torch.no_grad():
                    imgs_gen = generate_images(unet, n_samples=4, device=device)
                save_grid(imgs_gen,
                         os.path.join(PLOTS_DIR, f'imbr_img_gen_epoch_{ep+1:03d}.png'))
                logger.info(f"    ✓ 生成: plots/imbr_img_gen_epoch_{ep+1:03d}.png")

            # 检查点
            if (ep+1) % CONFIG['save_interval'] == 0:
                torch.save({
                    'epoch': ep+1,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(CKPT_DIR, f'imbr_fm_epoch_{ep+1}.pt'))

        t1 = time.time()
        logger.info(f"\n--- 训练完成 ---")
        logger.info(f"  耗时: {t1-t0:.0f}s ({(t1-t0)/60:.1f}min)")
        logger.info(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f}")

        # 保存模型
        model_path = os.path.join(CURRENT_DIR, 'model_imbr_img_fm.pt')
        torch.save(unet.state_dict(), model_path)
        logger.info(f"  模型: {model_path}")

        # Loss 曲线
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, linewidth=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title('IMBR Flow Matching Loss (图像级 UNet)')
        ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'imbr_img_fm_loss.png'), dpi=150); plt.close()

        # 最终生成
        logger.info(f"\n--- 步骤4: 最终生成 ---")
        unet.eval()
        with torch.no_grad():
            imgs_gen = generate_images(unet, n_samples=8, device=device)
        save_grid(imgs_gen, os.path.join(PLOTS_DIR, 'imbr_img_gen_final.png'), figsize=(20, 14))
        # 单独保存最好的一张
        best_idx = 0
        for i, img in enumerate(imgs_gen):
            save_single(img, os.path.join(OUT_DIR, f'imbr_final_{i:02d}.png'))
        logger.info(f"  最终: plots/imbr_img_gen_final.png")

        # 总结
        with open(os.path.join(CURRENT_DIR, 'imbr_img_fm_summary.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'model': 'IMBR Image Flow Matching (UNet + CondOT)',
                'params_m': n_params / 1e6,
                'n_epochs': CONFIG['n_epochs'],
                'batch_size': CONFIG['batch_size'],
                'final_loss': losses[-1],
                'time_min': (t1-t0)/60,
                'image_size': f'{IMG_W}x{IMG_H}'
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ IMBR 图像级 Flow Matching 完成！")

    except Exception as e:
        logger.error(f"失败: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()