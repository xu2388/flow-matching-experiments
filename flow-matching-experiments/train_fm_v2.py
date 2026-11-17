"""
=============================================================================
Flow Matching V2: 大模型 + 大 Batch + 混合精度 + 热重启调度
=============================================================================
目标：在 8G 显卡上充分训练，生成清晰的 IMBR 字母

改进点：
  1. 更大模型：2048 hidden × 16 blocks × 512 pos/time dim (137M 参数)
  2. 更大 batch：16384，充分利用显存（约 6.8GB）
  3. 混合精度训练（AMP）加速
  4. CosineAnnealingWarmRestarts 学习率调度（周期性重启）
  5. 梯度裁剪防止爆炸
  6. 每 200 个 epoch 保存检查点
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
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'fm_v2_log_{TIMESTAMP}.txt')
CKPT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

logger = logging.getLogger('FMV2')
logger.setLevel(logging.DEBUG)
# 使用 flush=True 确保日志立即写入文件
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(ch)

# 添加自定义 flush handler
class FlushHandler(logging.Handler):
    def emit(self, record):
        for h in logger.handlers:
            if hasattr(h, 'flush'):
                h.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"{'='*60}")
logger.info(f"Flow Matching V2: 大模型训练")
logger.info(f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
logger.info(f"{'='*60}")

# ============================================================================
# 1. IMBR 数据生成
# ============================================================================
def generate_imbr(n=5000, size=64):
    from PIL import Image, ImageDraw, ImageFont
    try: font = ImageFont.truetype("arial.ttf", int(size*0.85))
    except: font = ImageFont.load_default()
    w, h = size*4, size
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    for i, ch in enumerate('IMBR'):
        draw.text((i*size+size//4, 0), ch, fill=255, font=font)
    coords = np.argwhere(np.array(img) > 128)[:, [1, 0]].astype(np.float32)
    coords[:, 0] = (coords[:, 0]/w)*2 - 1
    coords[:, 1] = (coords[:, 1]/h)*2 - 1
    if len(coords) > n: coords = coords[np.random.choice(len(coords), n, replace=False)]
    else: coords = coords[np.random.choice(len(coords), n, replace=True)]
    return coords

# ============================================================================
# 2. 随机投影 OT 配对
# ============================================================================
def random_project_ot(source, target, n_directions=8):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    n = min(len(source), len(target))
    thetas = np.random.uniform(0, 2*np.pi, n_directions)
    for theta in thetas:
        proj_s = source[:n, 0]*np.cos(theta) + source[:n, 1]*np.sin(theta)
        proj_t = target[:n, 0]*np.cos(theta) + target[:n, 1]*np.sin(theta)
        idx_s = np.argsort(proj_s)
        idx_t = np.argsort(proj_t)
    paired = np.zeros_like(source)
    paired[:n] = target[idx_t[np.argsort(idx_s)]]
    if n < len(source):
        paired[n:] = target[np.random.randint(len(target), size=len(source)-n)]
    return paired

# ============================================================================
# 3. 数据集
# ============================================================================
class EpochOTDataset(Dataset):
    def __init__(self, target, n_epoch=20000, noise_std=0.6):
        self.target = target
        self.n_epoch = n_epoch
        self.noise_std = noise_std
        self._gen()
        logger.info(f"  Dataset: {n_epoch} points/epoch, σ={noise_std}")
    
    def _gen(self):
        noise = np.random.randn(self.n_epoch, 2).astype(np.float32) * self.noise_std
        t_idx = np.random.randint(0, len(self.target), self.n_epoch)
        target_sub = self.target[t_idx].copy()
        try:
            paired = random_project_ot(noise, target_sub, n_directions=8)
            target_sub = paired
        except:
            pass
        self.x0 = torch.tensor(noise, dtype=torch.float32)
        self.x1 = torch.tensor(target_sub, dtype=torch.float32)
    
    def __len__(self):
        return self.n_epoch
    
    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx]

# ============================================================================
# 4. 模型（更大版本）
# ============================================================================
class PosEnc(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.register_buffer('f', 2.0**torch.linspace(0, 10, dim//4))
    def forward(self, x):
        f = self.f
        xe, ye = x[:,0:1]*f[None,:]*np.pi, x[:,1:2]*f[None,:]*np.pi
        return torch.cat([torch.sin(xe), torch.cos(xe), torch.sin(ye), torch.cos(ye)], dim=-1)

class TimeEmb(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim))
    def forward(self, t):
        half = self.dim//2
        emb = torch.exp(torch.arange(half, device=t.device)*-np.log(10000.0)/(half-1))
        emb = t*emb[None,:]
        return self.net(torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1))

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d), nn.SiLU(), nn.Linear(d,d))
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        return self.norm(x + self.net(x))

class FMNet(nn.Module):
    def __init__(self, h=2048, pd=512, td=512, nb=16):
        super().__init__()
        self.pe = PosEnc(pd); self.te = TimeEmb(td)
        self.inp = nn.Sequential(nn.Linear(pd+td, h), nn.SiLU())
        self.blocks = nn.ModuleList([ResBlock(h) for _ in range(nb)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))
        self.apply(lambda m: nn.init.xavier_uniform_(m.weight, gain=0.5) if isinstance(m, nn.Linear) and m.weight.dim()>1 else None)
    
    def forward(self, x, t):
        h = torch.cat([self.pe(x), self.te(t)], dim=-1)
        h = self.inp(h)
        for b in self.blocks:
            h = b(h)
        return self.out(h)

# ============================================================================
# 5. 采样
# ============================================================================
@torch.no_grad()
def sample(model, x_init, n_steps=500, device='cpu'):
    model.eval()
    x = x_init.clone().to(device)
    dt = 1.0/n_steps
    traj = [x.cpu().numpy()]
    for i in range(n_steps):
        t = torch.ones(x.shape[0], 1).to(device)*(i/n_steps)
        k1 = model(x, t)
        k2 = model(x+0.5*dt*k1, t+0.5*dt)
        k3 = model(x+0.5*dt*k2, t+0.5*dt)
        k4 = model(x+dt*k3, t+dt)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
        if (i+1)%(n_steps//5)==0 or i==0:
            traj.append(x.cpu().numpy())
    traj.append(x.cpu().numpy())
    return traj

# ============================================================================
# 6. 可视化
# ============================================================================
def plot_traj(traj, title, path):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    steps = np.linspace(0, len(traj)-1, 6, dtype=int)
    for i, (ax, idx) in enumerate(zip(axes, steps)):
        pts = traj[idx]
        ax.scatter(pts[:,0], pts[:,1], s=4, alpha=0.7, c=plt.cm.viridis(idx/(len(traj)-1)))
        ax.set_title(f't={idx/(len(traj)-1):.1f}', fontsize=12)
        ax.set_xlim(-2,2); ax.set_ylim(-2,2)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def plot_comp(target, results, titles, path):
    n = len(results)+1
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    axes[0].scatter(target[:,0], target[:,1], s=5, c='red', alpha=0.6)
    axes[0].set_title('目标: IMBR', fontsize=12)
    axes[0].set_xlim(-1.5,1.5); axes[0].set_ylim(-1.5,1.5)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    for i, (pts, t) in enumerate(zip(results, titles)):
        axes[i+1].scatter(pts[:,0], pts[:,1], s=5, c='green', alpha=0.6)
        axes[i+1].set_title(t, fontsize=10)
        axes[i+1].set_xlim(-1.5,1.5); axes[i+1].set_ylim(-1.5,1.5)
        axes[i+1].set_aspect('equal'); axes[i+1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_checkpoint(model, opt, sched, scaler, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss
    }, path)

def load_checkpoint(path, model, opt, sched, scaler):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        sched.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        logger.info(f"  恢复断点: {path}, epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}")
        return ckpt['epoch'], ckpt['loss']
    return 0, float('inf')

# ============================================================================
# 7. 主训练流程
# ============================================================================
def main():
    try:
        # 配置
        CONFIG = {
            'hidden': 2048,
            'pos_dim': 512,
            'time_dim': 512,
            'n_blocks': 16,
            'batch_size': 4096,
            'n_epochs': 1300,
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'noise_std': 0.6,
            'noise_points': 100000,
            'target_points': 5000,
            'save_interval': 200,
            'log_interval': 10,
        }
        
        logger.info(f"\n--- 配置 ---")
        for k, v in CONFIG.items():
            logger.info(f"  {k}: {v}")
        
        # 生成目标
        logger.info(f"\n--- 步骤1: 生成 IMBR 目标 ---")
        target = generate_imbr(n=CONFIG['target_points'])
        logger.info(f"  目标: {len(target)} 点, x=[{target[:,0].min():.3f},{target[:,0].max():.3f}]")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(target[:,0], target[:,1], s=5, c='red', alpha=0.6)
        ax.set_title('IMBR 目标'); ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'fm_v2_target.png'), dpi=150); plt.close()
        
        # 创建数据集
        logger.info(f"\n--- 步骤2: 创建数据集 ---")
        dataset = EpochOTDataset(target, n_epoch=CONFIG['noise_points'], noise_std=CONFIG['noise_std'])
        loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        
        # 初始化模型
        logger.info(f"\n--- 步骤3: 初始化模型 ---")
        model = FMNet(h=CONFIG['hidden'], pd=CONFIG['pos_dim'], td=CONFIG['time_dim'], nb=CONFIG['n_blocks']).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  参数量: {n_params:,} ({n_params/1e6:.1f}M)")
        
        # 优化器、调度器、AMP
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=500, T_mult=2, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')
        
        # 尝试恢复断点
        ckpt_path = os.path.join(CKPT_DIR, 'fm_v2_latest.pt')
        start_epoch, best_loss = load_checkpoint(ckpt_path, model, opt, sched, scaler)
        if start_epoch > 0:
            logger.info(f"  从 epoch {start_epoch} 继续训练")
        
        # 训练
        logger.info(f"\n--- 步骤4: 训练 {CONFIG['n_epochs']} epochs ---")
        t0 = time.time()
        losses = []
        lrs = []
        model.train()
        
        for ep in range(start_epoch, CONFIG['n_epochs']):
            total_loss = 0
            n_batches = 0
            
            for x0, x1 in loader:
                x0, x1 = x0.to(device), x1.to(device)
                bs = x0.shape[0]
                t = torch.rand(bs, 1).to(device)
                xt = (1-t)*x0 + t*x1
                
                # 混合精度前向
                with torch.amp.autocast('cuda'):
                    v_pred = model(xt, t)
                    loss = nn.functional.mse_loss(v_pred, x1-x0)
                
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                
                total_loss += loss.item()
                n_batches += 1
            
            sched.step()
            avg_loss = total_loss / n_batches
            cur_lr = sched.get_last_lr()[0]
            losses.append(avg_loss)
            lrs.append(cur_lr)
            
            # 每个 epoch 重新生成数据
            loader.dataset._gen()
            
            if (ep+1) % CONFIG['log_interval'] == 0 or ep == 0:
                elapsed = time.time() - t0
                logger.info(f"  Epoch {ep+1:5d}/{CONFIG['n_epochs']} | Loss: {avg_loss:.6f} | LR: {cur_lr:.6f} | {elapsed:.0f}s")
                sys.stdout.flush()
                # 刷新文件日志
                for h in logger.handlers:
                    if hasattr(h, 'flush'):
                        h.flush()
            
            # 保存检查点
            if (ep+1) % CONFIG['save_interval'] == 0:
                save_checkpoint(model, opt, sched, scaler, ep+1, avg_loss, ckpt_path)
                # 同时保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_checkpoint(model, opt, sched, scaler, ep+1, avg_loss, 
                                  os.path.join(CKPT_DIR, 'fm_v2_best.pt'))
                logger.info(f"  ✓ 保存检查点 @ epoch {ep+1}")
        
        t1 = time.time()
        logger.info(f"\n--- 训练完成 ---")
        logger.info(f"  耗时: {t1-t0:.0f}s ({(t1-t0)/60:.1f}min)")
        logger.info(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
        
        # 保存最终模型
        model_path = os.path.join(CURRENT_DIR, 'model_fm_v2.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"  模型: {model_path}")
        
        # Loss 曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(losses, linewidth=1); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Loss 曲线'); ax1.grid(True, alpha=0.3)
        ax2.plot(lrs, linewidth=1); ax2.set_ylabel('LR')
        ax2.set_title('学习率调度 (WarmRestarts)'); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'fm_v2_loss.png'), dpi=150); plt.close()
        logger.info(f"  保存: plots/fm_v2_loss.png")
        
        # 验证采样
        logger.info(f"\n--- 步骤5: 验证 ---")
        model.eval()
        np.random.seed(42); torch.manual_seed(42)
        
        seeds = [
            ('高斯 σ=0.3', np.random.randn(3000,2)*0.3),
            ('高斯 σ=0.5', np.random.randn(3000,2)*0.5),
            ('高斯 σ=0.6', np.random.randn(3000,2)*0.6),
            ('高斯 σ=0.8', np.random.randn(3000,2)*0.8),
            ('均匀分布', np.random.uniform(-1.5,1.5,(3000,2))),
        ]
        results, titles = [], []
        for name, noise in seeds:
            init = torch.tensor(noise, dtype=torch.float32)
            traj = sample(model, init, n_steps=500, device=device)
            final = traj[-1]; results.append(final); titles.append(name)
            logger.info(f"  {name}: x=[{final[:,0].min():.3f},{final[:,0].max():.3f}]")
        
        # 轨迹图
        plot_traj(traj, 'FM V2: 高斯 σ=0.5 → IMBR', os.path.join(PLOTS_DIR, 'fm_v2_traj.png'))
        # 对比图
        plot_comp(target, results, titles, os.path.join(PLOTS_DIR, 'fm_v2_comp.png'))
        logger.info(f"  保存: fm_v2_traj.png, fm_v2_comp.png")
        
        # 10 种子图
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        torch.manual_seed(999)
        for i in range(10):
            noise = torch.randn(3000,2)*0.5
            final = sample(model, torch.tensor(noise, dtype=torch.float32), n_steps=500, device=device)[-1]
            r, c = i//5, i%5
            axes[r,c].scatter(final[:,0], final[:,1], s=4, alpha=0.7, c='green')
            axes[r,c].set_title(f'种子 {i+1}', fontsize=10)
            axes[r,c].set_xlim(-1.5,1.5); axes[r,c].set_ylim(-1.5,1.5)
            axes[r,c].set_aspect('equal'); axes[r,c].grid(True, alpha=0.3)
        plt.suptitle('FM V2: 10 种子 → IMBR', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'fm_v2_10_seeds.png'), dpi=150); plt.close()
        logger.info(f"  保存: fm_v2_10_seeds.png")
        
        # 保存总结
        summary = {
            'timestamp': TIMESTAMP,
            'device': str(device),
            'params': n_params,
            'config': CONFIG,
            'epochs': CONFIG['n_epochs'],
            'final_loss': losses[-1],
            'best_loss': min(losses),
            'time': t1-t0,
            'losses': [float(x) for x in losses[::100]],  # 每100个采样
            'lrs': [float(x) for x in lrs[::100]],
        }
        with open(os.path.join(CURRENT_DIR, 'fm_v2_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ 完成! 模型: {model_path}")
        
    except Exception as e:
        logger.error(f"失败: {e}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()