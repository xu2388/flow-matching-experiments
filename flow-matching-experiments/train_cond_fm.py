"""
=============================================================================
条件 Flow Matching：从噪声生成 IMBR 字母
=============================================================================
改进点：
  1. 条件生成：将目标 IMBR 形状作为条件输入
  2. 精确 OT 配对：使用 scipy 匈牙利算法
  3. 固定目标点：保持配对一致性
  4. 更大模型 + 充分训练
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
CKPT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'cond_fm_log_{TIMESTAMP}.txt')

logger = logging.getLogger('CondFM')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(ch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"{'='*60}")
logger.info(f"条件 Flow Matching: 生成 IMBR")
logger.info(f"设备: {device}")
logger.info(f"{'='*60}")

# ============================================================================
# 1. IMBR 数据生成
# ============================================================================
def generate_imbr(n=5000, size=64):
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
    return coords, (w, h)

# ============================================================================
# 2. 精确 OT 配对
# ============================================================================
def exact_ot_pairing(source, target):
    """使用匈牙利算法进行精确 OT 配对"""
    cost = cdist(source, target)
    row_ind, col_ind = linear_sum_assignment(cost)
    return target[col_ind]

# ============================================================================
# 3. 条件数据集
# ============================================================================
class CondOTDataset(Dataset):
    def __init__(self, target, n_samples=50000, noise_std=0.5):
        self.target = target
        self.n_samples = n_samples
        self.noise_std = noise_std
        
        # 预计算所有样本
        logger.info(f"  生成 {n_samples} 个样本 (精确 OT 配对)...")
        t0 = time.time()
        
        # 生成所有噪声
        self.x0_all = np.random.randn(n_samples, 2).astype(np.float32) * noise_std
        
        # 为每个噪声点计算精确 OT 配对
        # 分批处理以避免内存问题
        batch_size = 1000
        self.x1_all = np.zeros_like(self.x0_all)
        
        for i in range(0, n_samples, batch_size):
            end = min(i+batch_size, n_samples)
            batch_noise = self.x0_all[i:end]
            
            # 从目标中随机采样相同数量的点
            t_idx = np.random.choice(len(target), size=len(batch_noise), replace=False)
            batch_target = target[t_idx]
            
            # 精确 OT 配对
            paired = exact_ot_pairing(batch_noise, batch_target)
            self.x1_all[i:end] = paired
        
        self.x0_all = torch.tensor(self.x0_all, dtype=torch.float32)
        self.x1_all = torch.tensor(self.x1_all, dtype=torch.float32)
        
        logger.info(f"  完成 ({time.time()-t0:.1f}s)")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.x0_all[idx], self.x1_all[idx]

# ============================================================================
# 4. 条件 Flow Matching 模型
# ============================================================================
class PosEnc(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.register_buffer('f', 2.0**torch.linspace(0, 10, dim//4))
    def forward(self, x):
        f = self.f
        xe, ye = x[:,0:1]*f[None,:]*np.pi, x[:,1:2]*f[None,:]*np.pi
        return torch.cat([torch.sin(xe), torch.cos(xe), torch.sin(ye), torch.cos(ye)], dim=-1)

class TimeEmb(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim))
    def forward(self, t):
        half = self.dim//2
        emb = torch.exp(torch.arange(half, device=t.device)*-np.log(10000.0)/(half-1))
        emb = t*emb[None,:]
        return self.net(torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1))

class TargetEncoder(nn.Module):
    """将目标点云编码为条件向量"""
    def __init__(self, pos_dim=256, out_dim=512):
        super().__init__()
        self.pe = PosEnc(pos_dim)
        self.encoder = nn.Sequential(
            nn.Linear(pos_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(self, target_points):
        """
        target_points: (N, 2) - 目标点云
        返回: (out_dim,) - 条件向量（通过池化）
        """
        h = self.pe(target_points)  # (N, pos_dim)
        h = self.encoder(h)  # (N, out_dim)
        # 平均池化得到全局条件向量
        return h.mean(dim=0)  # (out_dim,)

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d), nn.SiLU(), nn.Linear(d,d))
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        return self.norm(x + self.net(x))

class CondFMNet(nn.Module):
    """条件 Flow Matching 网络"""
    def __init__(self, h=1024, pd=256, td=256, nb=8, cond_dim=512):
        super().__init__()
        self.pe = PosEnc(pd)
        self.te = TimeEmb(td)
        self.target_encoder = TargetEncoder(pd, cond_dim)
        
        # 条件调制：将条件向量注入每个残差块
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, h),
            nn.SiLU(),
            nn.Linear(h, h * nb),  # 为每个残差块生成调制参数
        )
        
        self.inp = nn.Sequential(nn.Linear(pd+td, h), nn.SiLU())
        self.blocks = nn.ModuleList([ResBlock(h) for _ in range(nb)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))
        
        self.nb = nb
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight, gain=0.5)
    
    def forward(self, x, t, target):
        """
        x: (B, 2) - 当前点位置
        t: (B, 1) - 时间
        target: (M, 2) - 目标点云（条件）
        """
        # 编码位置和时间
        h = torch.cat([self.pe(x), self.te(t)], dim=-1)  # (B, pd+td)
        h = self.inp(h)  # (B, h)
        
        # 编码目标条件
        cond = self.target_encoder(target)  # (cond_dim,)
        cond_params = self.cond_proj(cond)  # (nb * h,)
        cond_params = cond_params.view(self.nb, -1)  # (nb, h)
        
        # 条件调制的残差块
        for i, block in enumerate(self.blocks):
            # 用条件向量调制特征
            h = h + cond_params[i:i+1]  # 广播到 batch 维度
            h = block(h)
        
        return self.out(h)

# ============================================================================
# 5. 采样
# ============================================================================
@torch.no_grad()
def sample(model, x_init, target, n_steps=500, device='cpu'):
    model.eval()
    x = x_init.clone().to(device)
    target = target.to(device)
    dt = 1.0/n_steps
    traj = [x.cpu().numpy()]
    for i in range(n_steps):
        t = torch.ones(x.shape[0], 1).to(device)*(i/n_steps)
        k1 = model(x, t, target)
        k2 = model(x+0.5*dt*k1, t+0.5*dt, target)
        k3 = model(x+0.5*dt*k2, t+0.5*dt, target)
        k4 = model(x+dt*k3, t+dt, target)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
        if (i+1)%(n_steps//5)==0 or i==0:
            traj.append(x.cpu().numpy())
    traj.append(x.cpu().numpy())
    return traj

# ============================================================================
# 6. 可视化
# ============================================================================
def plot_traj(traj, target, title, path):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    steps = np.linspace(0, len(traj)-1, 6, dtype=int)
    for i, (ax, idx) in enumerate(zip(axes, steps)):
        pts = traj[idx]
        ax.scatter(target[:,0], target[:,1], s=3, c='red', alpha=0.3, label='target' if i==0 else '')
        ax.scatter(pts[:,0], pts[:,1], s=4, alpha=0.7, c='blue')
        ax.set_title(f't={idx/(len(traj)-1):.1f}', fontsize=12)
        ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=8)
    plt.suptitle(title, fontsize=14); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def plot_comp(target, results, titles, path):
    n = len(results)+1
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    axes[0].scatter(target[:,0], target[:,1], s=5, c='red', alpha=0.6, label='target')
    axes[0].set_title('目标', fontsize=12)
    axes[0].set_xlim(-1.5,1.5); axes[0].set_ylim(-1.5,1.5)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    for i, (pts, t) in enumerate(zip(results, titles)):
        axes[i+1].scatter(target[:,0], target[:,1], s=3, c='red', alpha=0.2)
        axes[i+1].scatter(pts[:,0], pts[:,1], s=5, c='green', alpha=0.6)
        axes[i+1].set_title(t, fontsize=10)
        axes[i+1].set_xlim(-1.5,1.5); axes[i+1].set_ylim(-1.5,1.5)
        axes[i+1].set_aspect('equal'); axes[i+1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ============================================================================
# 7. 主训练流程
# ============================================================================
def main():
    try:
        CONFIG = {
            'hidden': 1024,
            'pos_dim': 256,
            'time_dim': 256,
            'cond_dim': 512,
            'n_blocks': 8,
            'batch_size': 2048,
            'n_epochs': 500,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'noise_std': 0.5,
            'n_samples': 50000,
            'target_points': 5000,
            'save_interval': 100,
            'log_interval': 10,
        }
        
        logger.info(f"\n--- 配置 ---")
        for k, v in CONFIG.items():
            logger.info(f"  {k}: {v}")
        
        # 生成目标
        logger.info(f"\n--- 步骤1: 生成 IMBR 目标 ---")
        target, (img_w, img_h) = generate_imbr(n=CONFIG['target_points'])
        target_tensor = torch.tensor(target, dtype=torch.float32).to(device)
        logger.info(f"  目标: {len(target)} 点")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(target[:,0], target[:,1], s=5, c='red', alpha=0.6)
        ax.set_title('IMBR 目标'); ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'cond_fm_target.png'), dpi=150); plt.close()
        
        # 创建数据集（精确 OT 配对）
        logger.info(f"\n--- 步骤2: 创建数据集 (精确 OT 配对) ---")
        dataset = CondOTDataset(target, n_samples=CONFIG['n_samples'], noise_std=CONFIG['noise_std'])
        loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        
        # 初始化模型
        logger.info(f"\n--- 步骤3: 初始化条件模型 ---")
        model = CondFMNet(
            h=CONFIG['hidden'], 
            pd=CONFIG['pos_dim'], 
            td=CONFIG['time_dim'], 
            nb=CONFIG['n_blocks'],
            cond_dim=CONFIG['cond_dim']
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  参数量: {n_params:,} ({n_params/1e6:.1f}M)")
        
        # 优化器、调度器、AMP
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')
        
        # 训练
        logger.info(f"\n--- 步骤4: 训练 {CONFIG['n_epochs']} epochs ---")
        t0 = time.time()
        losses = []
        model.train()
        
        for ep in range(CONFIG['n_epochs']):
            total_loss = 0
            n_batches = 0
            
            for x0, x1 in loader:
                x0, x1 = x0.to(device), x1.to(device)
                bs = x0.shape[0]
                t = torch.rand(bs, 1).to(device)
                xt = (1-t)*x0 + t*x1
                
                with torch.amp.autocast('cuda'):
                    v_pred = model(xt, t, target_tensor)
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
            losses.append(avg_loss)
            
            if (ep+1) % CONFIG['log_interval'] == 0 or ep == 0:
                elapsed = time.time() - t0
                logger.info(f"  Epoch {ep+1:4d}/{CONFIG['n_epochs']} | Loss: {avg_loss:.6f} | {elapsed:.0f}s")
                sys.stdout.flush()
                for h in logger.handlers:
                    if hasattr(h, 'flush'):
                        h.flush()
            
            # 保存检查点
            if (ep+1) % CONFIG['save_interval'] == 0:
                save_path = os.path.join(CKPT_DIR, f'cond_fm_epoch_{ep+1}.pt')
                torch.save({
                    'epoch': ep+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': avg_loss
                }, save_path)
                logger.info(f"  ✓ 保存检查点 @ epoch {ep+1}")
        
        t1 = time.time()
        logger.info(f"\n--- 训练完成 ---")
        logger.info(f"  耗时: {t1-t0:.0f}s ({(t1-t0)/60:.1f}min)")
        logger.info(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
        
        # 保存模型
        model_path = os.path.join(CURRENT_DIR, 'model_cond_fm.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'target': target
        }, model_path)
        logger.info(f"  模型: {model_path}")
        
        # Loss 曲线
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, linewidth=1)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('条件 Flow Matching Loss 曲线'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'cond_fm_loss.png'), dpi=150); plt.close()
        
        # 验证
        logger.info(f"\n--- 步骤5: 验证 ---")
        model.eval()
        np.random.seed(42); torch.manual_seed(42)
        
        # 测试不同噪声
        for noise_std in [0.3, 0.5, 0.6]:
            noise = np.random.randn(3000, 2).astype(np.float32) * noise_std
            init = torch.tensor(noise, dtype=torch.float32)
            traj = sample(model, init, target_tensor, n_steps=500, device=device)
            final = traj[-1]
            
            logger.info(f"  高斯 σ={noise_std}: x=[{final[:,0].min():.3f},{final[:,0].max():.3f}], "
                       f"y=[{final[:,1].min():.3f},{final[:,1].max():.3f}]")
            
            # 绘制轨迹
            plot_traj(traj, target, 
                     f'条件FM: σ={noise_std} → IMBR',
                     os.path.join(PLOTS_DIR, f'cond_fm_traj_sigma{noise_std}.png'))
        
        # 多种子对比
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        torch.manual_seed(999)
        for i in range(10):
            noise = torch.randn(3000, 2) * 0.5
            traj = sample(model, noise, target_tensor, n_steps=500, device=device)
            final = traj[-1]
            r, c = i//5, i%5
            axes[r,c].scatter(target[:,0], target[:,1], s=3, c='red', alpha=0.2)
            axes[r,c].scatter(final[:,0], final[:,1], s=4, c='green', alpha=0.6)
            axes[r,c].set_title(f'种子 {i+1}', fontsize=10)
            axes[r,c].set_xlim(-1.5,1.5); axes[r,c].set_ylim(-1.5,1.5)
            axes[r,c].set_aspect('equal'); axes[r,c].grid(True, alpha=0.3)
        plt.suptitle('条件 Flow Matching: 10 种子 → IMBR', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'cond_fm_10_seeds.png'), dpi=150); plt.close()
        
        logger.info(f"\n✅ 完成!")
        
    except Exception as e:
        logger.error(f"失败: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()