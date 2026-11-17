"""
=============================================================================
DDPM: 去噪扩散概率模型生成 IMBR 字母
=============================================================================
核心理念：从纯噪声出发，通过反向去噪过程逐步恢复出目标形状。
模型架构：深度 ResNet + 时间嵌入，约 8M 参数，GPU 训练。

训练：预测添加到数据上的噪声
采样：DDPM 1000步 / DDIM 50步
=============================================================================
"""

import os, sys, time, json, logging, math
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
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
LOG_FILE = os.path.join(LOGS_DIR, f'ddpm_log_{TIMESTAMP}.txt')
PROBLEM_LOG_FILE = os.path.join(CURRENT_DIR, '实验问题记录.md')

logger = logging.getLogger('DDPM')
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
logger.info(f"DDPM 生成 IMBR 实验")
logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"设备: {device}")
logger.info(f"日志: {LOG_FILE}")
logger.info(f"图像: {PLOTS_DIR}")
logger.info(f"{'='*60}")

problems = []
def record_problem(phase, title, desc, solution="待解决"):
    problems.append({
        'phase': phase, 'title': title, 'description': desc,
        'solution': solution, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    logger.warning(f"[问题] {phase}: {title}")

# ============================================================================
# 1. 数据生成
# ============================================================================
def generate_IMBR(n_points=10000, canvas_size=64):
    """生成 IMBR 字母点云"""
    from PIL import Image, ImageDraw, ImageFont
    n_chars = 4
    w = canvas_size * n_chars
    h = canvas_size
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", int(canvas_size * 0.8))
    except:
        font = ImageFont.load_default()
    for i, ch in enumerate('IMBR'):
        draw.text((i * canvas_size + canvas_size // 4, 0), ch, fill=255, font=font)
    img = np.array(img)
    coords = np.argwhere(img > 0)
    if len(coords) == 0:
        logger.error("IMBR 无有效像素！")
        return np.random.randn(n_points, 2).astype(np.float32)
    coords = coords[:, [1, 0]]
    coords[:, 0] = (coords[:, 0] / w) * 2 - 1
    coords[:, 1] = (coords[:, 1] / h) * 2 - 1
    if len(coords) > n_points:
        idx = np.random.choice(len(coords), n_points, replace=False)
        coords = coords[idx]
    elif len(coords) < n_points:
        idx = np.random.choice(len(coords), n_points, replace=True)
        coords = coords[idx]
    return coords.astype(np.float32)

# ============================================================================
# 2. 深度 ResNet 模型（带时间嵌入）
# ============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResBlock(nn.Module):
    """带时间条件注入的残差块"""
    def __init__(self, dim, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim),
        )
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, t_emb):
        h = self.net(x)
        h = h + self.time_mlp(t_emb)
        h = self.norm(h)
        return x + h  # 残差连接

class DeepResNet(nn.Module):
    """深度 ResNet 用于噪声预测"""
    def __init__(self, input_dim=2, hidden_dim=768, time_dim=256, n_blocks=12):
        super().__init__()
        self.time_embed = SinusoidalPosEmb(time_dim)
        # 时间条件投影
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 残差块
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
    def forward(self, x, t):
        """x: (B, 2), t: (B, 1)"""
        t_emb = self.time_embed(t.float() / 1000.0)  # 归一化到 [0,1]
        t_emb = self.time_proj(t_emb)
        h = self.input_proj(x)
        h = h + t_emb
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)

# ============================================================================
# 3. DDPM 调度器
# ============================================================================
class DDPMScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        self.posterior_variance = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        
    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha_bar_prev = self.alpha_bar_prev.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_recip_alpha = self.sqrt_recip_alpha.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
    
    def q_sample(self, x_0, t, noise=None):
        """前向加噪：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise
    
    @torch.no_grad()
    def sample_ddpm(self, model, n_samples, device):
        """DDPM 采样（1000步）"""
        model.eval()
        x = torch.randn(n_samples, 2).to(device)
        trajectory = [x.cpu().numpy()]
        for t in reversed(range(self.T)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            eps_pred = model(x, t_tensor)
            # 预测 x_0
            x_0_pred = (x - self.sqrt_one_minus_alpha_bar[t] * eps_pred) / self.sqrt_alpha_bar[t]
            # 去噪
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = 0
            coef1 = self.sqrt_recip_alpha[t]
            coef2 = self.beta[t] / self.sqrt_one_minus_alpha_bar[t]
            x = coef1 * (x - coef2 * eps_pred) + torch.sqrt(self.posterior_variance[t]) * z
            
            if t % (self.T // 5) == 0:
                trajectory.append(x.cpu().numpy())
        trajectory.append(x.cpu().numpy())
        return trajectory
    
    @torch.no_grad()
    def sample_ddim(self, model, n_samples, device, n_steps=50, eta=0.0):
        """DDIM 采样（快速，50步）"""
        model.eval()
        step_ratio = self.T // n_steps
        timesteps = list(range(0, self.T, step_ratio))
        timesteps = timesteps[::-1]
        
        x = torch.randn(n_samples, 2).to(device)
        trajectory = [x.cpu().numpy()]
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            eps_pred = model(x, t_tensor)
            
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar_prev[t]
            
            # 预测 x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            
            # DDIM 更新
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            c1 = torch.sqrt(alpha_bar_prev)
            c2 = torch.sqrt(1 - alpha_bar_prev - sigma**2)
            if i < len(timesteps) - 1:
                x = c1 * x_0_pred + c2 * eps_pred + sigma * torch.randn_like(x)
            else:
                x = x_0_pred
            
            if i % (n_steps // 5) == 0 or i == 0:
                trajectory.append(x.cpu().numpy())
        trajectory.append(x.cpu().numpy())
        return trajectory
    
    @torch.no_grad()
    def sample_ddim_eta(self, model, n_samples, device, n_steps=50, eta=0.5):
        """DDIM with eta for stochasticity"""
        model.eval()
        step_ratio = self.T // n_steps
        timesteps = [i * step_ratio for i in range(n_steps)]
        timesteps = timesteps[::-1]
        
        x = torch.randn(n_samples, 2).to(device)
        trajectory = [x.cpu().numpy()]
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            eps_pred = model(x, t_tensor)
            
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar_prev[t]
            
            x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            c1 = torch.sqrt(alpha_bar_prev)
            c2 = torch.sqrt(1 - alpha_bar_prev - sigma**2)
            if i < len(timesteps) - 1:
                noise = sigma * torch.randn_like(x)
                x = c1 * x_0_pred + c2 * eps_pred + noise
            else:
                x = x_0_pred
            
            if i % (n_steps // 5) == 0:
                trajectory.append(x.cpu().numpy())
        trajectory.append(x.cpu().numpy())
        return trajectory

# ============================================================================
# 4. 训练
# ============================================================================
def train_ddpm(model, dataloader, scheduler, epochs=5000, lr=2e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []
    
    model.train()
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_0 in dataloader:
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            t = torch.randint(0, scheduler.T, (batch_size,), device=device).long()
            noise = torch.randn_like(x_0)
            x_t, _ = scheduler.q_sample(x_0, t, noise)
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            step += 1
        
        scheduler_lr.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 500 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler_lr.get_last_lr()[0]:.6f}")
    
    return loss_history

# ============================================================================
# 5. 可视化
# ============================================================================
def plot_generation_results(trajectories, titles, save_path, suptitle=''):
    """多轨迹生成对比"""
    n_traj = len(trajectories)
    fig, axes = plt.subplots(n_traj, 6, figsize=(24, 4*n_traj))
    if n_traj == 1:
        axes = axes.reshape(1, -1)
    
    for row, (traj, title) in enumerate(zip(trajectories, titles)):
        # 确保至少6个时间点
        n_frames = min(6, len(traj))
        indices = np.linspace(0, len(traj)-1, n_frames, dtype=int)
        for col, idx in enumerate(indices):
            pts = traj[idx]
            t_val = idx / (len(traj)-1) if len(traj) > 1 else 0
            axes[row, col].scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.7, c=plt.cm.viridis(t_val))
            axes[row, col].set_title(f'step {idx}', fontsize=10)
            axes[row, col].set_xlim(-2, 2); axes[row, col].set_ylim(-2, 2)
            axes[row, col].set_aspect('equal'); axes[row, col].grid(True, alpha=0.3)
        axes[row, 0].set_ylabel(title, fontsize=11, rotation=0, labelpad=40)
    
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  图已保存: {save_path}")

def plot_multi_seeds(model, scheduler, n_seeds=10, device='cpu', save_path=''):
    """10个不同随机种子生成 IMBR 的最终结果"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(n_seeds):
        torch.manual_seed(42 + i)
        traj = scheduler.sample_ddim(model, 2000, device, n_steps=50, eta=0.0)
        final = traj[-1]
        row, col = i // 5, i % 5
        axes[row, col].scatter(final[:, 0], final[:, 1], s=4, alpha=0.7, c='green')
        axes[row, col].set_title(f'种子 {i+1}', fontsize=10)
        axes[row, col].set_xlim(-1.5, 1.5); axes[row, col].set_ylim(-1.5, 1.5)
        axes[row, col].set_aspect('equal'); axes[row, col].grid(True, alpha=0.3)
    plt.suptitle('DDPM: 10个不同随机种子 → IMBR 生成结果', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  多种子对比: {save_path}")

# ============================================================================
# 6. 主流程
# ============================================================================
def main():
    try:
        # 生成 IMBR 数据
        logger.info(f"\n{'='*60}")
        logger.info(f"生成 IMBR 训练数据")
        logger.info(f"{'='*60}")
        imbr_pts = generate_IMBR(n_points=10000, canvas_size=64)
        logger.info(f"IMBR 点云: {imbr_pts.shape}")
        
        # 可视化目标
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(imbr_pts[:, 0], imbr_pts[:, 1], s=5, c='red', alpha=0.6)
        ax.set_title(f'目标分布: IMBR ({len(imbr_pts)}点)', fontsize=14)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'ddpm_target_imbr.png'), dpi=150)
        plt.close()
        logger.info(f"目标图: plots/ddpm_target_imbr.png")
        
        # 数据集
        imbr_tensor = torch.tensor(imbr_pts, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(imbr_tensor)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
        
        # 初始化模型
        logger.info(f"\n{'='*60}")
        logger.info(f"初始化深度 ResNet 模型")
        logger.info(f"{'='*60}")
        model = DeepResNet(input_dim=2, hidden_dim=768, time_dim=256, n_blocks=12).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数量: {n_params:,}")
        logger.info(f"  hidden_dim=768, n_blocks=12, time_dim=256")
        
        # 调度器
        scheduler = DDPMScheduler(T=1000, beta_start=1e-4, beta_end=0.02).to(device)
        logger.info(f"DDPM 调度器: T=1000, beta=[1e-4, 0.02]")
        
        # 训练
        logger.info(f"\n{'='*60}")
        logger.info(f"开始训练 (5000 epochs)")
        logger.info(f"{'='*60}")
        t0 = time.time()
        losses = train_ddpm(model, dataloader, scheduler, epochs=5000, lr=2e-4)
        t1 = time.time()
        logger.info(f"训练完成! 耗时: {t1-t0:.1f}s")
        logger.info(f"Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
        
        # 保存模型
        model_path = os.path.join(CURRENT_DIR, 'model_ddpm_imbr.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型保存: {model_path}")
        
        # Loss 曲线
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, linewidth=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title('DDPM 训练 Loss 曲线 (噪声预测)')
        ax.grid(True, alpha=0.3); ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'ddpm_loss_curve.png'), dpi=150)
        plt.close()
        logger.info(f"Loss曲线: plots/ddpm_loss_curve.png")
        
        # ====================================================================
        # 采样
        # ====================================================================
        logger.info(f"\n{'='*