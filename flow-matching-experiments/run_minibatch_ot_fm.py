"""
=============================================================================
Flow Matching + Mini-Batch OT：从高斯噪声生成 IMBR 字母
=============================================================================
核心理念：
  每个 epoch 生成新的噪声点，对每个 mini-batch 做 OT 配对训练。
  这样模型看到无限多的不同配对，防止过拟合，同时保持 OT 的干净速度场。

关键改进：
  1. Mini-Batch OT：每 batch 重新做 OT 配对，防止固定配对过拟合
  2. 每 epoch 生成新噪声：模型看到无限多的不同噪声点
  3. 更大的模型容量：2048 隐层维度，16 个残差块
  4. 更长的训练：5000 epochs
=============================================================================
"""

import os, sys, time, json, logging, traceback, threading
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, IterableDataset
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'minibatch_ot_fm_log_{TIMESTAMP}.txt')
PROBLEM_LOG_FILE = os.path.join(CURRENT_DIR, '实验问题记录.md')

logger = logging.getLogger('MiniOTFM')
logger.setLevel(logging.DEBUG)
for h in [logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setLevel(logging.DEBUG if isinstance(h, logging.FileHandler) else logging.INFO)
    h.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(h)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"{'='*60}")
logger.info(f"Mini-Batch OT Flow Matching 实验")
logger.info(f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
logger.info(f"{'='*60}")

problems = []
def record_problem(phase, title, desc, solution="待解决"):
    problems.append({'phase': phase, 'title': title, 'description': desc,
                     'solution': solution, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    logger.warning(f"[问题] {phase}: {title}")

# ============================================================================
# 1. 数据生成
# ============================================================================
def generate_imbr_cloud(n_points=5000, canvas_size=64):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", int(canvas_size * 0.85))
    except:
        font = ImageFont.load_default()
    w, h = canvas_size * 4, canvas_size
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    for i, ch in enumerate('IMBR'):
        draw.text((i * canvas_size + canvas_size // 4, 0), ch, fill=255, font=font)
    coords = np.argwhere(np.array(img) > 128)[:, [1, 0]].astype(np.float32)
    coords[:, 0] = (coords[:, 0] / w) * 2 - 1
    coords[:, 1] = (coords[:, 1] / h) * 2 - 1
    if len(coords) > n_points:
        coords = coords[np.random.choice(len(coords), n_points, replace=False)]
    else:
        coords = coords[np.random.choice(len(coords), n_points, replace=True)]
    return coords

# ============================================================================
# 2. 可迭代数据集：每步生成新噪声 + OT 配对
# ============================================================================
class InfiniteOTDataset(IterableDataset):
    """
    无限数据流：每次迭代生成新噪声点，做 OT 配对
    
    关键设计：
    - 每次迭代随机生成新噪声 → 防止过拟合
    - 每 BATCH_SIZE 个点做一次 OT 配对 → 干净速度场
    - 无限数据 → 模型学习泛化，而非记忆
    """
    def __init__(self, target_points, batch_size=2048, noise_std=0.6):
        self.target = torch.tensor(target_points, dtype=torch.float32)
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.n_target = len(target_points)
        logger.info(f"  InfiniteOTDataset: batch={batch_size}, noise_std={noise_std}")

    def __iter__(self):
        while True:
            # 生成噪声
            noise = torch.randn(self.batch_size, 2) * self.noise_std
            # 随机采样目标点
            t_idx = torch.randint(0, self.n_target, (self.batch_size,))
            target = self.target[t_idx].clone()
            # OT 配对
            try:
                n_ot = min(1000, self.batch_size)  # 控制 OT 开销
                idx_s = torch.randperm(self.batch_size)[:n_ot]
                idx_t = torch.randperm(self.batch_size)[:n_ot]
                s_sub = noise[idx_s].numpy()
                t_sub = target[idx_t].numpy()
                cost = torch.cdist(torch.tensor(s_sub), torch.tensor(t_sub)).numpy()
                _, col_ind = linear_sum_assignment(cost)
                for i, j in zip(range(n_ot), col_ind):
                    target[idx_s[i]] = torch.tensor(t_sub[j])
            except Exception as e:
                pass  # OT 失败就用随机配对
            # 生成样本
            for i in range(self.batch_size):
                yield noise[i], target[i]

# ============================================================================
# 3. 模型
# ============================================================================
class PositionalEncoding2D(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.register_buffer('freqs', 2.0 ** torch.linspace(0, 10, dim // 4))
    def forward(self, x):
        f = self.freqs
        xe, ye = x[:, 0:1]*f[None,:]*np.pi, x[:, 1:2]*f[None,:]*np.pi
        return torch.cat([torch.sin(xe), torch.cos(xe), torch.sin(ye), torch.cos(ye)], dim=-1)

class TimeEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim))
    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=t.device) * -np.log(10000.0)/(half-1))
        emb = t * emb[None, :]
        return self.net(torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1))

class BigResNet(nn.Module):
    def __init__(self, hidden=1024, pos_dim=256, time_dim=256, n_blocks=8):
        super().__init__()
        self.pos_enc = PositionalEncoding2D(pos_dim)
        self.time_emb = TimeEmbedding(time_dim)
        self.input_proj = nn.Sequential(nn.Linear(pos_dim+time_dim, hidden), nn.SiLU())
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)))
            self.blocks.append(nn.LayerNorm(hidden))
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.apply(lambda m: nn.init.xavier_uniform_(m.weight, gain=0.1) if isinstance(m, nn.Linear) and m.weight.dim() > 1 else None)
    def forward(self, x, t):
        h = torch.cat([self.pos_enc(x), self.time_emb(t)], dim=-1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h) if isinstance(block, nn.LayerNorm) else h + block(h)
        return self.output(h)

# ============================================================================
# 4. 采样
# ============================================================================
@torch.no_grad()
def sample_rk4(model, x_init, n_steps=500, device='cpu'):
    model.eval()
    x = x_init.clone().to(device)
    dt = 1.0 / n_steps
    traj = [x.cpu().numpy()]
    for i in range(n_steps):
        t = torch.ones(x.shape[0], 1).to(device) * (i / n_steps)
        k1 = model(x, t)
        k2 = model(x + 0.5*dt*k1, t + 0.5*dt)
        k3 = model(x + 0.5*dt*k2, t + 0.5*dt)
        k4 = model(x + dt*k3, t + dt)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
        if (i+1) % (n_steps//5) == 0 or i == 0:
            traj.append(x.cpu().numpy())
    traj.append(x.cpu().numpy())
    return traj

# ============================================================================
# 5. 训练
# ============================================================================
def train_infinite(model, dataset_iter, n_steps=50000, batch_size=2048, lr=5e-4):
    """在无限数据流上训练"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    losses = []
    data_iter = iter(dataset_iter)
    
    model.train()
    for step in range(n_steps):
        # 收集一个 batch
        x0_list, x1_list = [], []
        for _ in range(batch_size):
            x0, x1 = next(data_iter)
            x0_list.append(x0)
            x1_list.append(x1)
        x0 = torch.stack(x0_list).to(device)
        x1 = torch.stack(x1_list).to(device)
        
        t = torch.rand(batch_size, 1).to(device)
        x_t = (1-t)*x0 + t*x1
        v_pred = model(x_t, t)
        loss = nn.functional.mse_loss(v_pred, x1 - x0)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        
        if (step+1) % 5000 == 0 or step == 0:
            logger.info(f"  Step {step+1:5d}/{n_steps} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return losses

# ============================================================================
# 6. 可视化
# ============================================================================
def plot_trajectory(traj, title, save_path):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    steps = np.linspace(0, len(traj)-1, 6, dtype=int)
    for i, (ax, idx) in enumerate(zip(axes, steps)):
        pts = traj[idx]
        ax.scatter(pts[:,0], pts[:,1], s=4, alpha=0.7, c=plt.cm.viridis(idx/(len(traj)-1)))
        ax.set_title(f't={idx/(len(traj)-1):.1f}', fontsize=12)
        ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

def plot_comparison(target, results, titles, save_path):
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    axes[0].scatter(target[:,0], target[:,1], s=5, c='red', alpha=0.6)
    axes[0].set_title('目标: IMBR', fontsize=12)
    axes[0].set_xlim(-1.5,1.5); axes[0].set_ylim(-1.5,1.5); axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    for i, (pts, t) in enumerate(zip(results, titles)):
        axes[i+1].scatter(pts[:,0], pts[:,1], s=5, c='green', alpha=0.6)
        axes[i+1].set_title(t, fontsize=10)
        axes[i+1].set_xlim(-1.5,1.5); axes[i+1].set_ylim(-1.5,1.5); axes[i+1].set_aspect('equal'); axes[i+1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

# ============================================================================
# 7. 主流程
# ============================================================================
def main():
    try:
        # 生成目标
        logger.info(f"\n--- 步骤 1: 生成 IMBR ---")
        target = generate_imbr_cloud(n_points=5000)
        logger.info(f"  IMBR: {len(target)}点, x=[{target[:,0].min():.3f},{target[:,0].max():.3f}]")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.scatter(target[:,0], target[:,1], s=5, c='red', alpha=0.6)
        ax.set_title('IMBR 目标点云'); ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'mbot_target.png'), dpi=150); plt.close()
        
        # 创建无限数据流
        logger.info(f"\n--- 步骤 2: 创建无限数据流 ---")
        BATCH_SIZE = 1024
        N_STEPS = 50000
        dataset = InfiniteOTDataset(target, batch_size=BATCH_SIZE, noise_std=0.6)
        
        # 初始化模型
        logger.info(f"\n--- 步骤 3: 初始化模型 ---")
        model = BigResNet(hidden=1024, pos_dim=256, time_dim=256, n_blocks=8).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  参数量: {n_params:,}")
        
        # 训练
        logger.info(f"\n--- 步骤 4: 开始训练 ({N_STEPS} steps) ---")
        t0 = time.time()
        losses = train_infinite(model, dataset, n_steps=N_STEPS, batch_size=BATCH_SIZE)
        t1 = time.time()
        logger.info(f"  训练完成! 耗时: {t1-t0:.1f}s ({(t1-t0)/60:.1f}min)")
        logger.info(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
        
        # 保存模型
        model_path = os.path.join(CURRENT_DIR, 'model_minibatch_ot_fm.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"  模型保存: {model_path}")
        
        # Loss 曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
        ax1.plot(losses, linewidth=1); ax1.set_xlabel('Step'); ax1.set_ylabel('Loss')
        ax1.set_title('Loss 曲线'); ax1.grid(True, alpha=0.3)
        ax2.plot(losses, linewidth=1); ax2.set_yscale('log')
        ax2.set_xlabel('Step'); ax2.set_title('Loss (log)'); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'mbot_loss.png'), dpi=150); plt.close()
        
        # 采样验证
        logger.info(f"\n--- 步骤 5: 多种子采样验证 ---")
        np.random.seed(42)
        torch.manual_seed(42)
        
        seeds = [
            ('高斯 σ=0.5', np.random.randn(3000, 2) * 0.5),
            ('高斯 σ=0.3', np.random.randn(3000, 2) * 0.3),
            ('高斯 σ=0.8', np.random.randn(3000, 2) * 0.8),
            ('均匀分布', np.random.uniform(-1.5, 1.5, (3000, 2))),
        ]
        results, titles = [], []
        for name, noise in seeds:
            init = torch.tensor(noise, dtype=torch.float32)
            traj = sample_rk4(model, init, n_steps=500, device=device)
            final = traj[-1]
            results.append(final); titles.append(name)
            logger.info(f"  {name}: x=[{final[:,0].min():.3f},{final[:,0].max():.3f}] y=[{final[:,1].min():.3f},{final[:,1].max():.3f}]")
        
        # 轨迹图
        plot_trajectory(traj, 'Mini-Batch OT Flow Matching: 噪声 → IMBR',
                       os.path.join(PLOTS_DIR, 'mbot_trajectory.png'))
        
        # 对比图
        plot_comparison(target, results, titles,
                       os.path.join(PLOTS_DIR, 'mbot_comparison.png'))
        
        # 10 种子
        fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
        torch.manual_seed(999)
        for i in range(10):
            noise = torch.randn(3000, 2) * 0.5
            final = sample_rk4(model, torch.tensor(noise, dtype=torch.float32), n_steps=500, device=device)[-1]
            r, c = i // 5, i % 5
            axes2[r,c].scatter(final[:,0], final[:,1], s=4, alpha=0.7, c='green')
            axes2[r,c].set_title(f'种子 {i+1}', fontsize=10)
            axes2[r,c].set_xlim(-1.5,1.5); axes2[r,c].set_ylim(-1.5,1.5)
            axes2[r,c].set_aspect('equal'); axes2[r,c].grid(True, alpha=0.3)
        plt.suptitle('10 种子 → IMBR', fontsize=14)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'mbot_10_seeds.png'), dpi=150); plt.close()
        
        # 保存总结
        summary = {'timestamp': TIMESTAMP, 'device': str(device),
                   'model_params': n_params, 'n_steps': N_STEPS,
                   'batch_size': BATCH_SIZE, 'final_loss': losses[-1],
                   'training_time': t1-t0, 'problems': problems}
        with open(os.path.join(CURRENT_DIR, 'minibatch_ot_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ 实验完成! 模型: {model_path}")
        logger.info(f"图像: {PLOTS_DIR}")
        
    except Exception as e:
        logger.error(f"实验失败: {e}")
        logger.error(traceback.format_exc())
        record_problem("全局", "实验异常", str(e), traceback.format_exc())
        raise
    finally:
        with open(PROBLEM_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n\n## Mini-Batch OT FM 实验 ({TIMESTAMP})\n\n")
            if not problems: f.write("✅ 本次实验未出现重大问题\n")
            else:
                f.write(f"问题数: {len(problems)}\n")
                for p in problems: f.write(f"- [{p['phase']}] {p['title']}: {p['description']} → {p['solution']}\n")

if __name__ == '__main__':
    main()