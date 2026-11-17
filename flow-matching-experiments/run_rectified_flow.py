"""
=============================================================================
Rectified Flow + OT 配对：从高斯噪声生成 IMBR 字母
=============================================================================
核心理念：
  Rectified Flow 的关键创新：用 OT 配对代替随机配对，消除速度场的随机性。

训练流程：
  Step 1: OT 配对 → 训练模型 F1 (噪声→IMBR)
  Step 2: Reflow → 用 F1 生成新配对，训练模型 F2 (路径更直)
  Step 3: 推理 → 从任意噪声出发，沿 ODE 生成 IMBR

验证步骤（训练前确保数据正确）：
  [V1] IMBR 点云可视化确认
  [V2] OT 配对可视化确认
  [V3] 小数据过拟合测试
  [V4] 正式训练
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 路径配置
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'rectified_fm_log_{TIMESTAMP}.txt')
PROBLEM_LOG_FILE = os.path.join(CURRENT_DIR, '实验问题记录.md')

logger = logging.getLogger('RectFM')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"{'='*60}")
logger.info(f"Rectified Flow + OT 实验")
logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
logger.info(f"日志: {LOG_FILE}")
logger.info(f"图像: {PLOTS_DIR}")
logger.info(f"{'='*60}")

problems = []
def record_problem(phase, title, desc, solution="待解决"):
    problems.append({'phase': phase, 'title': title, 'description': desc,
                     'solution': solution, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    logger.warning(f"[问题] {phase}: {title}")

# ============================================================================
# 1. 数据生成与验证
# ============================================================================
def generate_imbr_cloud(n_points=3000, canvas_size=64):
    """生成 IMBR 点云"""
    from PIL import Image, ImageDraw, ImageFont
    font_size = int(canvas_size * 0.85)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    w = canvas_size * 4
    h = canvas_size
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    for i, ch in enumerate('IMBR'):
        draw.text((i * canvas_size + canvas_size // 4, 0), ch, fill=255, font=font)
    img_arr = np.array(img)
    coords = np.argwhere(img_arr > 128)
    if len(coords) == 0:
        raise ValueError("IMBR 生成失败！")
    coords = coords[:, [1, 0]].astype(np.float32)
    coords[:, 0] = (coords[:, 0] / w) * 2 - 1
    coords[:, 1] = (coords[:, 1] / h) * 2 - 1
    if len(coords) > n_points:
        idx = np.random.choice(len(coords), n_points, replace=False)
        coords = coords[idx]
    else:
        idx = np.random.choice(len(coords), n_points, replace=True)
        coords = coords[idx]
    return coords

def verify_imbr_data(target, save_path):
    """[V1] 验证 IMBR 点云是否清晰可辨"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(target[:, 0], target[:, 1], s=5, c='red', alpha=0.6)
    ax.set_title(f'IMBR 点云 ({len(target)} 点)', fontsize=14)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    stats = {
        'n_points': len(target),
        'x_range': (float(target[:,0].min()), float(target[:,0].max())),
        'y_range': (float(target[:,1].min()), float(target[:,1].max())),
        'center': (float(target[:,0].mean()), float(target[:,1].mean())),
    }
    logger.info(f"  [V1] IMBR 验证: 点={stats['n_points']}, "
                f"x=[{stats['x_range'][0]:.3f},{stats['x_range'][1]:.3f}], "
                f"y=[{stats['y_range'][0]:.3f},{stats['y_range'][1]:.3f}], "
                f"中心=({stats['center'][0]:.3f},{stats['center'][1]:.3f})")
    return stats

def verify_ot_pairs(source, target, pairs, save_path):
    """[V2] 验证 OT 配对是否合理"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(source[:, 0], source[:, 1], s=4, c='blue', alpha=0.5)
    axes[0].set_title('源: 高斯噪声', fontsize=12); axes[0].set_aspect('equal')
    axes[0].set_xlim(-2, 2); axes[0].set_ylim(-2, 2); axes[0].grid(True, alpha=0.3)
    axes[1].scatter(target[:, 0], target[:, 1], s=4, c='red', alpha=0.5)
    axes[1].set_title('目标: IMBR', fontsize=12); axes[1].set_aspect('equal')
    axes[1].set_xlim(-2, 2); axes[1].set_ylim(-2, 2); axes[1].grid(True, alpha=0.3)
    # 绘制部分配对线
    subset = min(500, len(pairs))
    for i in range(subset):
        axes[2].plot([source[i, 0], pairs[i, 0]], [source[i, 1], pairs[i, 1]], 
                     'gray', alpha=0.15, linewidth=0.5)
    axes[2].scatter(source[:subset, 0], source[:subset, 1], s=4, c='blue', alpha=0.5)
    axes[2].scatter(pairs[:subset, 0], pairs[:subset, 1], s=4, c='red', alpha=0.5)
    axes[2].set_title(f'OT 配对 (显示前{subset}条)', fontsize=12)
    axes[2].set_aspect('equal'); axes[2].set_xlim(-2, 2)
    axes[2].set_ylim(-2, 2); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    avg_dist = np.mean(np.sqrt(np.sum((source - pairs)**2, axis=1)))
    logger.info(f"  [V2] OT 配对验证: 平均配对距离={avg_dist:.4f}")
    return avg_dist

# ============================================================================
# 2. OT 配对数据集
# ============================================================================
def optimal_transport_pairing(source, target):
    """用匈牙利算法做最优传输配对"""
    # 下采样到 min(N, 5000) 避免 OOM
    n = min(len(source), len(target), 5000)
    idx_s = np.random.choice(len(source), n, replace=False)
    idx_t = np.random.choice(len(target), n, replace=False)
    s_sub = source[idx_s]
    t_sub = target[idx_t]
    cost = torch.cdist(torch.tensor(s_sub, dtype=torch.float32), torch.tensor(t_sub, dtype=torch.float32)).numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    paired = np.zeros_like(source)
    # 已配对的用 OT 结果
    for i, j in zip(row_ind, col_ind):
        paired[idx_s[i]] = target[idx_t[j]]
    # 未配对的用随机邻居
    unmatched = np.setdiff1d(np.arange(len(source)), idx_s[row_ind])
    for i in unmatched:
        paired[i] = target[np.random.randint(len(target))]
    return paired

class MiniBatchOTDataset(Dataset):
    """
    迷你批 OT 配对数据集：每个 epoch 随机生成新噪声点并做 OT 配对
    
    关键改进：
    - 每个 epoch 生成不同的噪声点 → 防止过拟合
    - 每个 batch 做 OT 配对 → 干净的速度场
    - 模型看到无限多的不同配对 → 泛化到新噪声
    """
    def __init__(self, target_points, n_per_epoch=10000, noise_std=0.6):
        self.target = torch.tensor(target_points, dtype=torch.float32)
        self.n_per_epoch = n_per_epoch
        self.noise_std = noise_std
        # 预生成当前 epoch 的数据
        self._generate_epoch()
        logger.info(f"  数据集: {n_per_epoch} 对/epoch, Mini-Batch OT 配对")

    def _generate_epoch(self):
        """生成新 epoch 的噪声点并做 OT 配对"""
        n = min(self.n_per_epoch, len(self.target), 3000)
        # 生成噪声
        noise = np.random.randn(self.n_per_epoch, 2).astype(np.float32) * self.noise_std
        # 随机采样目标点
        target_idx = np.random.choice(len(self.target), self.n_per_epoch, replace=True)
        target_sub = self.target[target_idx].numpy().copy()
        # 对前 n 个点做 OT 配对
        idx_s = np.random.choice(self.n_per_epoch, n, replace=False)
        idx_t = np.random.choice(self.n_per_epoch, n, replace=False)
        s_sub = noise[idx_s]
        t_sub = target_sub[idx_t]
        try:
            cost = torch.cdist(torch.tensor(s_sub), torch.tensor(t_sub)).numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                target_sub[idx_s[i]] = t_sub[col_ind[j]]
        except Exception as e:
            logger.warning(f"  OT 配对失败: {e}, 使用随机配对")
        
        self.source = torch.tensor(noise, dtype=torch.float32)
        self.target = torch.tensor(target_sub, dtype=torch.float32)
        self.on_epoch_end()

    def on_epoch_end(self):
        self._generate_epoch()

    def __len__(self):
        return self.n_per_epoch

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]

# ============================================================================
# 3. ResNet 模型（与之前相同，验证过有效）
# ============================================================================
class PositionalEncoding2D(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.register_buffer('freqs', 2.0 ** torch.linspace(0, 10, dim // 4))
    def forward(self, x):
        freqs = self.freqs
        x_enc = x[:, 0:1] * freqs[None, :] * np.pi
        y_enc = x[:, 1:2] * freqs[None, :] * np.pi
        return torch.cat([torch.sin(x_enc), torch.cos(x_enc),
                          torch.sin(y_enc), torch.cos(y_enc)], dim=-1)

class TimeEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim))
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * 
                        -np.log(10000.0) / (half_dim - 1))
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.net(emb)

class ResidualBlockFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x + self.net(x))

class ResNetFlowMatching(nn.Module):
    def __init__(self, hidden_dim=1024, pos_dim=256, time_dim=256, n_blocks=8):
        super().__init__()
        self.pos_enc = PositionalEncoding2D(pos_dim)
        self.time_emb = TimeEmbedding(time_dim)
        self.input_proj = nn.Sequential(nn.Linear(pos_dim + time_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList([ResidualBlockFM(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x, t):
        h = torch.cat([self.pos_enc(x), self.time_emb(t)], dim=-1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)

# ============================================================================
# 4. ODE 采样器
# ============================================================================
@torch.no_grad()
def sample_rk4(model, x_init, n_steps=500, device='cpu'):
    model.eval()
    x = x_init.clone().to(device)
    dt = 1.0 / n_steps
    trajectory = [x.cpu().numpy()]
    for i in range(n_steps):
        t = torch.ones(x.shape[0], 1).to(device) * (i / n_steps)
        k1 = model(x, t)
        k2 = model(x + 0.5*dt*k1, t + 0.5*dt)
        k3 = model(x + 0.5*dt*k2, t + 0.5*dt)
        k4 = model(x + dt*k3, t + dt)
        x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        record_freq = max(1, n_steps // 5)
        if (i + 1) % record_freq == 0 or i == 0:
            trajectory.append(x.cpu().numpy())
    trajectory.append(x.cpu().numpy())
    return trajectory

# ============================================================================
# 5. 训练
# ============================================================================
def train_flow_matching(model, dataloader, epochs=2000, lr=1e-3, exp_name='FM'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x0, x1 in dataloader:
            x0, x1 = x0.to(device), x1.to(device)
            bs = x0.shape[0]
            t = torch.rand(bs, 1).to(device)
            x_t = (1 - t) * x0 + t * x1
            target_v = x1 - x0
            pred_v = model(x_t, t)
            loss = nn.functional.mse_loss(pred_v, target_v)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        if (epoch + 1) % 500 == 0 or epoch == 0:
            logger.info(f"  {exp_name} Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    return loss_history

# ============================================================================
# 6. 验证采样
# ============================================================================
def verify_sampling(model, n_points=3000, save_prefix='verify'):
    """从不同的噪声种子采样，验证生成效果"""
    results = []
    seeds = [
        ('标准高斯 σ=0.5', np.random.randn(n_points, 2) * 0.5),
        ('标准高斯 σ=0.3', np.random.randn(n_points, 2) * 0.3),
        ('标准高斯 σ=0.8', np.random.randn(n_points, 2) * 0.8),
        ('均匀分布 [-1.5,1.5]', np.random.uniform(-1.5, 1.5, (n_points, 2))),
    ]
    final_results = []
    final_titles = []
    for name, noise in seeds:
        x_init = torch.tensor(noise, dtype=torch.float32)
        traj = sample_rk4(model, x_init, n_steps=500, device=device)
        final = traj[-1]
        final_results.append(final)
        final_titles.append(name)
        x_r = (final[:,0].min(), final[:,0].max())
        y_r = (final[:,1].min(), final[:,1].max())
        logger.info(f"  {name}: x=[{x_r[0]:.3f},{x_r[1]:.3f}] y=[{y_r[0]:.3f},{y_r[1]:.3f}]")
        results.append({'name': name, 'x_range': x_r, 'y_range': y_r})
    return final_results, final_titles, results

# ============================================================================
# 7. 可视化
# ============================================================================
def plot_trajectory(trajectory, title, save_path):
    n_plots = 6
    fig, axes = plt.subplots(1, n_plots, figsize=(24, 4))
    steps = np.linspace(0, len(trajectory)-1, n_plots, dtype=int)
    for i, (ax, idx) in enumerate(zip(axes, steps)):
        pts = trajectory[idx]
        t_val = idx / (len(trajectory)-1)
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.7, c=plt.cm.viridis(t_val))
        ax.set_title(f't={t_val:.1f}', fontsize=12)
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_final_comparison(target, generated_list, titles, save_path):
    n = len(generated_list) + 1
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    axes[0].scatter(target[:, 0], target[:, 1], s=5, c='red', alpha=0.6)
    axes[0].set_title('目标: IMBR', fontsize=12)
    axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    for i, (pts, title) in enumerate(zip(generated_list, titles)):
        axes[i+1].scatter(pts[:, 0], pts[:, 1], s=5, c='green', alpha=0.6)
        axes[i+1].set_title(title, fontsize=10)
        axes[i+1].set_xlim(-1.5, 1.5); axes[i+1].set_ylim(-1.5, 1.5)
        axes[i+1].set_aspect('equal'); axes[i+1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# 8. 主流程
# ============================================================================
def main():
    try:
        # ====================================================================
        # 步骤 0: 生成 IMBR 目标
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 0: 生成 IMBR 目标点云")
        logger.info(f"{'='*60}")
        N_POINTS = 5000
        target_imbr = generate_imbr_cloud(n_points=N_POINTS, canvas_size=64)
        verify_imbr_data(target_imbr, os.path.join(PLOTS_DIR, 'rect_v0_target_imbr.png'))
        
        # ====================================================================
        # 步骤 1: 生成噪声 + OT 配对
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 1: 噪声生成 + OT 最优传输配对")
        logger.info(f"{'='*60}")
        np.random.seed(42)
        torch.manual_seed(42)
        
        N_NOISE = 5000
        noise_source = np.random.randn(N_NOISE, 2) * 0.5
        
        logger.info(f"  噪声: {N_NOISE} 点, 范围 x:[{noise_source[:,0].min():.3f},{noise_source[:,0].max():.3f}]")
        logger.info(f"  目标: {len(target_imbr)} 点")
        logger.info(f"  正在计算 OT 配对...")
        
        t0 = time.time()
        paired_target = optimal_transport_pairing(noise_source, target_imbr)
        t1 = time.time()
        logger.info(f"  OT 配对完成, 耗时: {t1-t0:.1f}s")
        
        verify_ot_pairs(noise_source, target_imbr, paired_target,
                       os.path.join(PLOTS_DIR, 'rect_v1_ot_pairs.png'))
        
        # ====================================================================
        # 步骤 2: 创建数据集
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 2: 创建训练数据集")
        logger.info(f"{'='*60}")
        dataset = OTPairedDataset(noise_source, paired_target)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)
        
        # ====================================================================
        # 步骤 3: [V3] 小数据过拟合测试
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"[V3] 小数据过拟合测试 (500 点, 500 epochs)")
        logger.info(f"{'='*60}")
        tiny_source = noise_source[:500]
        tiny_target = paired_target[:500]
        tiny_dataset = OTPairedDataset(tiny_source, tiny_target)
        tiny_loader = DataLoader(tiny_dataset, batch_size=128, shuffle=True)
        test_model = ResNetFlowMatching(hidden_dim=512, pos_dim=128, time_dim=128, n_blocks=4).to(device)
        tiny_losses = train_flow_matching(test_model, tiny_loader, epochs=500, exp_name='过拟合测试')
        final_tiny_loss = tiny_losses[-1]
        logger.info(f"  [V3] 过拟合测试: Loss = {tiny_losses[0]:.6f} → {final_tiny_loss:.6f}")
        if final_tiny_loss < 0.01:
            logger.info(f"  ✅ [V3] 通过! 模型可以过拟合小数据, Loss={final_tiny_loss:.6f}")
        else:
            logger.warning(f"  ⚠️ [V3] Loss({final_tiny_loss:.6f}) > 0.01, 模型可能需要更强")
            record_problem("V3", "过拟合测试 Loss 不够低", f"{final_tiny_loss:.6f}", "增大模型容量或增加 epochs")
        
        # 清理测试模型
        del test_model, tiny_dataset, tiny_loader
        torch.cuda.empty_cache()
        
        # ====================================================================
        # 步骤 4: 正式训练 - 阶段 1 (OT 配对)
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 4: 正式训练阶段 1 — OT 配对 (2000 epochs)")
        logger.info(f"{'='*60}")
        model_f1 = ResNetFlowMatching(hidden_dim=1024, pos_dim=256, time_dim=256, n_blocks=8).to(device)
        n_params = sum(p.numel() for p in model_f1.parameters())
        logger.info(f"  模型参数量: {n_params:,}")
        
        t0 = time.time()
        losses_f1 = train_flow_matching(model_f1, dataloader, epochs=2000, exp_name='阶段1-OT')
        t1 = time.time()
        logger.info(f"  阶段1 完成! 耗时: {t1-t0:.1f}s")
        logger.info(f"  Loss: {losses_f1[0]:.6f} → {losses_f1[-1]:.6f}")
        
        # 保存阶段1模型
        model_f1_path = os.path.join(CURRENT_DIR, 'model_rectflow_f1_ot.pth')
        torch.save(model_f1.state_dict(), model_f1_path)
        logger.info(f"  模型保存: {model_f1_path}")
        
        # 阶段1采样验证
        logger.info(f"\n  --- 阶段1 采样验证 ---")
        f1_results, f1_titles, f1_stats = verify_sampling(model_f1, n_points=3000, save_prefix='f1')
        plot_final_comparison(target_imbr, f1_results, f1_titles,
                            os.path.join(PLOTS_DIR, 'rect_f1_comparison.png'))
        
        # 单轨迹可视化
        noise_demo = np.random.randn(3000, 2) * 0.5
        traj_f1 = sample_rk4(model_f1, torch.tensor(noise_demo, dtype=torch.float32), 
                            n_steps=500, device=device)
        plot_trajectory(traj_f1, '阶段1: OT 配对流 → IMBR',
                       os.path.join(PLOTS_DIR, 'rect_f1_trajectory.png'))
        
        # ====================================================================
        # 步骤 5: Reflow — 生成新配对
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 5: Reflow — 生成新配对 (路径矫直)")
        logger.info(f"{'='*60}")
        
        # 用阶段1模型生成新的 (x0, x1) 对
        logger.info(f"  正在用阶段1模型生成 {N_NOISE} 对新配对...")
        model_f1.eval()
        with torch.no_grad():
            new_source = np.random.randn(N_NOISE, 2) * 0.5
            x_init = torch.tensor(new_source, dtype=torch.float32).to(device)
            new_target = sample_rk4(model_f1, x_init, n_steps=500, device=device)[-1]
        logger.info(f"  新配对: 源={N_NOISE}点, 目标范围 x:[{new_target[:,0].min():.3f},{new_target[:,0].max():.3f}]")
        
        # 验证新配对
        verify_ot_pairs(new_source, target_imbr, new_target,
                       os.path.join(PLOTS_DIR, 'rect_v2_reflow_pairs.png'))
        
        # 创建 reflow 数据集
        reflow_dataset = OTPairedDataset(new_source, new_target)
        reflow_loader = DataLoader(reflow_dataset, batch_size=1024, shuffle=True, num_workers=0)
        
        # ====================================================================
        # 步骤 6: 正式训练 — 阶段 2 (Reflow)
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 6: 正式训练阶段 2 — Reflow (2000 epochs)")
        logger.info(f"{'='*60}")
        model_f2 = ResNetFlowMatching(hidden_dim=1024, pos_dim=256, time_dim=256, n_blocks=8).to(device)
        n_params_f2 = sum(p.numel() for p in model_f2.parameters())
        logger.info(f"  模型参数量: {n_params_f2:,}")
        
        t0 = time.time()
        losses_f2 = train_flow_matching(model_f2, reflow_loader, epochs=2000, exp_name='阶段2-Reflow')
        t1 = time.time()
        logger.info(f"  阶段2 完成! 耗时: {t1-t0:.1f}s")
        logger.info(f"  Loss: {losses_f2[0]:.6f} → {losses_f2[-1]:.6f}")
        
        # 保存阶段2模型
        model_f2_path = os.path.join(CURRENT_DIR, 'model_rectflow_f2_reflow.pth')
        torch.save(model_f2.state_dict(), model_f2_path)
        logger.info(f"  模型保存: {model_f2_path}")
        
        # ====================================================================
        # 步骤 7: 最终验证
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 7: 最终验证 — 多种子生成 IMBR")
        logger.info(f"{'='*60}")
        
        # 阶段2采样验证
        f2_results, f2_titles, f2_stats = verify_sampling(model_f2, n_points=3000, save_prefix='f2')
        plot_final_comparison(target_imbr, f2_results, f2_titles,
                            os.path.join(PLOTS_DIR, 'rect_f2_comparison.png'))
        
        # 阶段2单轨迹
        traj_f2 = sample_rk4(model_f2, torch.tensor(noise_demo, dtype=torch.float32),
                            n_steps=500, device=device)
        plot_trajectory(traj_f2, '阶段2: Reflow → IMBR',
                       os.path.join(PLOTS_DIR, 'rect_f2_trajectory.png'))
        
        # 阶段1 vs 阶段2 对比
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        steps = np.linspace(0, len(traj_f1)-1, 6, dtype=int)
        for i, idx in enumerate(steps):
            axes[0, i].scatter(traj_f1[idx][:, 0], traj_f1[idx][:, 1], s=4, alpha=0.7, c='blue')
            axes[0, i].set_title(f'阶段1 t={idx/(len(traj_f1)-1):.1f}', fontsize=10)
            axes[0, i].set_xlim(-2, 2); axes[0, i].set_ylim(-2, 2)
            axes[0, i].set_aspect('equal'); axes[0, i].grid(True, alpha=0.3)
            axes[1, i].scatter(traj_f2[idx][:, 0], traj_f2[idx][:, 1], s=4, alpha=0.7, c='green')
            axes[1, i].set_title(f'阶段2 t={idx/(len(traj_f2)-1):.1f}', fontsize=10)
            axes[1, i].set_xlim(-2, 2); axes[1, i].set_ylim(-2, 2)
            axes[1, i].set_aspect('equal'); axes[1, i].grid(True, alpha=0.3)
        plt.suptitle('阶段1 (OT) vs 阶段2 (Reflow) 轨迹对比', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'rect_f1_vs_f2.png'), dpi=150)
        plt.close()
        logger.info(f"  阶段对比: plots/rect_f1_vs_f2.png")
        
        # 10 个随机种子最终结果
        fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
        torch.manual_seed(999)
        for i in range(10):
            noise = torch.randn(3000, 2) * 0.5
            traj = sample_rk4(model_f2, torch.tensor(noise, dtype=torch.float32),
                            n_steps=500, device=device)
            final = traj[-1]
            row, col = i // 5, i % 5
            axes2[row, col].scatter(final[:, 0], final[:, 1], s=4, alpha=0.7, c='green')
            axes2[row, col].set_title(f'种子 {i+1}', fontsize=10)
            axes2[row, col].set_xlim(-1.5, 1.5); axes2[row, col].set_ylim(-1.5, 1.5)
            axes2[row, col].set_aspect('equal'); axes2[row, col].grid(True, alpha=0.3)
        plt.suptitle('10 个不同噪声种子 → IMBR (Reflow 阶段2)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'rect_10_seeds_final.png'), dpi=150)
        plt.close()
        logger.info(f"  10种子对比: plots/rect_10_seeds_final.png")
        
        # Loss 曲线对比
        fig3, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses_f1, label='阶段1: OT 配对', linewidth=1.5)
        ax.plot(losses_f2, label='阶段2: Reflow', linewidth=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title('Rectified Flow Loss 曲线对比')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'rect_loss_comparison.png'), dpi=150)
        plt.close()
        logger.info(f"  Loss对比: plots/rect_loss_comparison.png")
        
        # 保存总结
        summary = {
            'timestamp': TIMESTAMP, 'device': str(device),
            'model_f1_params': n_params, 'model_f2_params': n_params_f2,
            'f1_final_loss': losses_f1[-1], 'f2_final_loss': losses_f2[-1],
            'f1_time': t1 - t0, 'problems': problems
        }
        with open(os.path.join(CURRENT_DIR, 'rectified_flow_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ 实验完成!")
        logger.info(f"阶段1模型: {model_f1_path}")
        logger.info(f"阶段2模型: {model_f2_path}")
        logger.info(f"日志: {LOG_FILE}")
        logger.info(f"图像: {PLOTS_DIR}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"实验失败: {e}")
        logger.error(traceback.format_exc())
        record_problem("全局", "实验异常", str(e), traceback.format_exc())
        raise
    finally:
        with open(PROBLEM_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n\n## Rectified Flow 实验 ({TIMESTAMP})\n\n")
            if not problems:
                f.write("✅ 本次实验未出现重大问题\n")
            else:
                f.write(f"问题数: {len(problems)}\n")
                for p in problems:
                    f.write(f"- [{p['phase']}] {p['title']}: {p['description']} → {p['solution']}\n")

if __name__ == '__main__':
    main()