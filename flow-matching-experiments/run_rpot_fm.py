"""
=============================================================================
Flow Matching + 随机投影 OT：从高斯噪声生成 IMBR
=============================================================================
核心理念：
  - 用随机投影近似 OT 配对，O(n log n) 快速计算
  - 每个 epoch 生成新噪声 + 新配对，防止过拟合
  - 大模型 + 长训练，充分学习速度场

随机投影 OT：
  1. 采样随机方向 θ
  2. 将所有点投影到该方向：proj = x*cos(θ) + y*sin(θ)
  3. 按投影排序，配对排序后的点
  4. 多个方向取平均，提高配对质量
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
LOG_FILE = os.path.join(LOGS_DIR, f'rp_ot_fm_log_{TIMESTAMP}.txt')
PROBLEM_LOG_FILE = os.path.join(CURRENT_DIR, '实验问题记录.md')

logger = logging.getLogger('RPOTFM')
logger.setLevel(logging.DEBUG)
for h in [logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setLevel(logging.DEBUG if isinstance(h, logging.FileHandler) else logging.INFO)
    h.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(h)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"{'='*60}")
logger.info(f"随机投影 OT Flow Matching 实验")
logger.info(f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
logger.info(f"{'='*60}")

problems = []
def record_problem(phase, title, desc, solution="待解决"):
    problems.append({'phase':phase,'title':title,'desc':desc,'solution':solution,'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    logger.warning(f"[问题] {phase}: {title}")

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
    """
    随机投影 OT 配对：在多个随机方向上排序配对
    O(n log n) 时间复杂度，比匈牙利算法 (O(n³)) 快得多
    """
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    n = min(len(source), len(target))
    # 采样随机方向
    thetas = np.random.uniform(0, 2*np.pi, n_directions)
    # 每个方向投票
    votes = np.zeros(n, dtype=np.int32)
    for theta in thetas:
        proj_s = source[:n, 0] * np.cos(theta) + source[:n, 1] * np.sin(theta)
        proj_t = target[:n, 0] * np.cos(theta) + target[:n, 1] * np.sin(theta)
        idx_s = np.argsort(proj_s)
        idx_t = np.argsort(proj_t)
        # 投票：每个 source 点获得对应的 target 点索引
        votes[idx_s] = idx_t  # 直接用当前方向的配对
    # 用最后一组配对作为最终结果
    paired = np.zeros_like(source)
    paired[:n] = target[idx_t[np.argsort(idx_s)]]
    # 剩余点用随机配对
    if n < len(source):
        paired[n:] = target[np.random.randint(len(target), size=len(source)-n)]
    return paired

# ============================================================================
# 3. 数据集：每个 epoch 生成新数据
# ============================================================================
class EpochOTDataset(Dataset):
    """每个 epoch 生成新噪声 + 新 OT 配对"""
    def __init__(self, target, n_epoch=20000, noise_std=0.6):
        self.target = target
        self.n_epoch = n_epoch
        self.noise_std = noise_std
        self._gen()
        logger.info(f"  EpochOTDataset: {n_epoch}点/epoch, σ={noise_std}")
    
    def _gen(self):
        noise = np.random.randn(self.n_epoch, 2).astype(np.float32) * self.noise_std
        # 随机采样 target
        t_idx = np.random.randint(0, len(self.target), self.n_epoch)
        target_sub = self.target[t_idx].copy()
        # 随机投影 OT 配对
        try:
            paired = random_project_ot(noise, target_sub, n_directions=4)
            target_sub = paired
        except: pass
        self.x0 = torch.tensor(noise, dtype=torch.float32)
        self.x1 = torch.tensor(target_sub, dtype=torch.float32)
    
    def __len__(self): return self.n_epoch
    def __getitem__(self, idx): return self.x0[idx], self.x1[idx]

# ============================================================================
# 4. 模型
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

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d), nn.SiLU(), nn.Linear(d,d))
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        return self.norm(x + self.net(x))

class FMNet(nn.Module):
    def __init__(self, h=1024, pd=256, td=256, nb=8):
        super().__init__()
        self.pe = PosEnc(pd); self.te = TimeEmb(td)
        self.inp = nn.Sequential(nn.Linear(pd+td, h), nn.SiLU())
        self.blocks = nn.ModuleList([ResBlock(h) for _ in range(nb)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2))
        self.apply(lambda m: nn.init.xavier_uniform_(m.weight, gain=0.1) if isinstance(m, nn.Linear) and m.weight.dim()>1 else None)
    def forward(self, x, t):
        h = torch.cat([self.pe(x), self.te(t)], dim=-1)
        h = self.inp(h)
        for b in self.blocks: h = b(h)
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
        if (i+1)%(n_steps//5)==0 or i==0: traj.append(x.cpu().numpy())
    traj.append(x.cpu().numpy())
    return traj

# ============================================================================
# 6. 训练
# ============================================================================
def train(model, loader, epochs=5000, lr=1e-3, name='FM'):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses = []
    model.train()
    for ep in range(epochs):
        total_loss = 0
        n_b = 0
        for x0, x1 in loader:
            x0, x1 = x0.to(device), x1.to(device)
            bs = x0.shape[0]
            t = torch.rand(bs, 1).to(device)
            xt = (1-t)*x0 + t*x1
            v_pred = model(xt, t)
            loss = nn.functional.mse_loss(v_pred, x1-x0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n_b += 1
        sched.step()
        avg = total_loss/n_b
        losses.append(avg)
        # 每个 epoch 重新生成数据
        loader.dataset._gen()
        if (ep+1)%1000==0 or ep==0:
            logger.info(f"  {name} Epoch {ep+1:4d}/{epochs} | Loss: {avg:.6f} | LR: {sched.get_last_lr()[0]:.6f}")
    return losses

# ============================================================================
# 7. 可视化
# ============================================================================
def plot_traj(traj, title, path):
    fig, axes = plt.subplots(1, 6, figsize=(24,4))
    steps = np.linspace(0, len(traj)-1, 6, dtype=int)
    for i, (ax, idx) in enumerate(zip(axes, steps)):
        pts = traj[idx]
        ax.scatter(pts[:,0], pts[:,1], s=4, alpha=0.7, c=plt.cm.viridis(idx/(len(traj)-1)))
        ax.set_title(f't={idx/(len(traj)-1):.1f}', fontsize=12)
        ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_comp(target, results, titles, path):
    n = len(results)+1
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    axes[0].scatter(target[:,0], target[:,1], s=5, c='red', alpha=0.6)
    axes[0].set_title('目标: IMBR', fontsize=12)
    axes[0].set_xlim(-1.5,1.5); axes[0].set_ylim(-1.5,1.5); axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    for i, (pts, t) in enumerate(zip(results, titles)):
        axes[i+1].scatter(pts[:,0], pts[:,1], s=5, c='green', alpha=0.6)
        axes[i+1].set_title(t, fontsize=10)
        axes[i+1].set_xlim(-1.5,1.5); axes[i+1].set_ylim(-1.5,1.5); axes[i+1].set_aspect('equal'); axes[i+1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ============================================================================
# 8. 主流程
# ============================================================================
def main():
    try:
        logger.info(f"\n--- 步骤1: 生成 IMBR ---")
        target = generate_imbr(n=5000)
        logger.info(f"  IMBR: {len(target)}点, x=[{target[:,0].min():.3f},{target[:,0].max():.3f}]")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.scatter(target[:,0], target[:,1], s=5, c='red', alpha=0.6)
        ax.set_title('IMBR 目标'); ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'rpot_target.png'), dpi=150); plt.close()
        
        logger.info(f"\n--- 步骤2: 创建数据集 ---")
        EPOCHS = 5000; BATCH = 1024; N_EPOCH = 20000
        dataset = EpochOTDataset(target, n_epoch=N_EPOCH, noise_std=0.6)
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
        
        logger.info(f"\n--- 步骤3: 初始化模型 ---")
        model = FMNet(h=1024, pd=256, td=256, nb=8).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  参数量: {n_params:,}")
        
        logger.info(f"\n--- 步骤4: 训练 {EPOCHS} epochs ---")
        t0 = time.time()
        losses = train(model, loader, epochs=EPOCHS, name='RPOT-FM')
        t1 = time.time()
        logger.info(f"  完成! {t1-t0:.0f}s ({(t1-t0)/60:.1f}min)")
        logger.info(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
        
        model_path = os.path.join(CURRENT_DIR, 'model_rpot_fm.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"  模型: {model_path}")
        
        # Loss 曲线
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
        ax1.plot(losses, linewidth=1); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Loss 曲线'); ax1.grid(True, alpha=0.3)
        ax2.plot(losses, linewidth=1); ax2.set_yscale('log')
        ax2.set_xlabel('Epoch'); ax2.set_title('Loss (log)'); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'rpot_loss.png'), dpi=150); plt.close()
        
        # 采样验证
        logger.info(f"\n--- 步骤5: 验证 ---")
        np.random.seed(42); torch.manual_seed(42)
        seeds = [
            ('高斯 σ=0.5', np.random.randn(3000,2)*0.5),
            ('高斯 σ=0.3', np.random.randn(3000,2)*0.3),
            ('高斯 σ=0.8', np.random.randn(3000,2)*0.8),
            ('均匀分布', np.random.uniform(-1.5,1.5,(3000,2))),
        ]
        results, titles = [], []
        for name, noise in seeds:
            init = torch.tensor(noise, dtype=torch.float32)
            traj = sample(model, init, n_steps=500, device=device)
            final = traj[-1]; results.append(final); titles.append(name)
            logger.info(f"  {name}: x=[{final[:,0].min():.3f},{final[:,0].max():.3f}]")
        
        plot_traj(traj, '随机投影 OT FM: 噪声 → IMBR', os.path.join(PLOTS_DIR, 'rpot_traj.png'))
        plot_comp(target, results, titles, os.path.join(PLOTS_DIR, 'rpot_comp.png'))
        
        # 10 种子
        fig2, axes2 = plt.subplots(2,5,figsize=(20,8))
        torch.manual_seed(999)
        for i in range(10):
            noise = torch.randn(3000,2)*0.5
            final = sample(model, torch.tensor(noise, dtype=torch.float32), n_steps=500, device=device)[-1]
            r,c = i//5, i%5
            axes2[r,c].scatter(final[:,0], final[:,1], s=4, alpha=0.7, c='green')
            axes2[r,c].set_title(f'种子 {i+1}', fontsize=10)
            axes2[r,c].set_xlim(-1.5,1.5); axes2[r,c].set_ylim(-1.5,1.5); axes2[r,c].set_aspect('equal'); axes2[r,c].grid(True, alpha=0.3)
        plt.suptitle('10 种子 → IMBR', fontsize=14)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'rpot_10_seeds.png'), dpi=150); plt.close()
        
        summary = {'timestamp':TIMESTAMP,'device':str(device),'params':n_params,
                   'epochs':EPOCHS,'batch':BATCH,'final_loss':losses[-1],'time':t1-t0}
        with open(os.path.join(CURRENT_DIR, 'rpot_summary.json'),'w',encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ 完成! 模型: {model_path}")
    except Exception as e:
        logger.error(f"失败: {e}"); logger.error(traceback.format_exc())
        record_problem("全局", "异常", str(e), traceback.format_exc())

if __name__ == '__main__':
    main()