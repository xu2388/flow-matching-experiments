"""
=============================================================================
生成式 Flow Matching：从高斯噪声生成 IMBR 字母
=============================================================================
核心理念：训练一个向量场，使得从任意噪声点出发，沿 ODE 积分都能生成目标形状。

训练方式：随机配对 (x0 ~ N(0,1), x1 ~ IMBR)，回归条件速度 v = x1 - x0。
采样方式：从任意噪声点出发，用 ODE 求解器积分到 t=1。
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

# ============================================================================
# 路径配置
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'generative_fm_log_{TIMESTAMP}.txt')
PROBLEM_LOG_FILE = os.path.join(CURRENT_DIR, '实验问题记录.md')

logger = logging.getLogger('GenFM')
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
logger.info(f"生成式 Flow Matching 实验")
logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"设备: {device}")
logger.info(f"日志保存: {LOG_FILE}")
logger.info(f"图像保存: {PLOTS_DIR}")
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
def generate_letter_cloud(letter='C', n_points=1000, canvas_size=64):
    """生成字母点云，支持多字母"""
    from PIL import Image, ImageDraw, ImageFont
    n_chars = len(letter)
    w = canvas_size * n_chars
    h = canvas_size
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", int(canvas_size * 0.8))
    except:
        font = ImageFont.load_default()
    if n_chars == 1:
        draw.text((canvas_size//4, 0), letter, fill=255, font=font)
    else:
        for i, ch in enumerate(letter):
            draw.text((i * canvas_size + canvas_size//4, 0), ch, fill=255, font=font)
    img = np.array(img)
    coords = np.argwhere(img > 0)
    if len(coords) == 0:
        logger.error(f"字母 '{letter}' 无有效像素！")
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
# 2. 生成式数据集：源 = 高斯噪声，目标 = IMBR
# ============================================================================
class GenerativePointDataset(Dataset):
    """源是高斯噪声，目标是 IMBR 点云，训练时随机配对"""
    def __init__(self, target_points, n_source=2000):
        self.target = torch.tensor(target_points, dtype=torch.float32)
        self.n_source = n_source
        logger.info(f"  生成式数据集: 源=高斯噪声({n_source}点), 目标={len(target_points)}点")

    def __len__(self):
        return self.n_source

    def __getitem__(self, idx):
        x0 = torch.randn(2) * 0.8  # 标准高斯噪声
        rand_idx = torch.randint(0, len(self.target), (1,)).item()
        x1 = self.target[rand_idx]
        return x0, x1

# ============================================================================
# 3. 更强大的模型（带残差连接）
# ============================================================================
class TimeEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.net(x)  # 残差连接

class GenerativeVectorField(nn.Module):
    """带残差连接和更深层的向量场网络"""
    def __init__(self, dim=512, time_dim=128, n_blocks=6):
        super().__init__()
        self.time_mlp = TimeEncoder(time_dim)
        self.input_proj = nn.Linear(2 + time_dim, dim)
        
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(dim))
            blocks.append(nn.SiLU())
        self.blocks = nn.Sequential(*blocks)
        
        self.output_proj = nn.Linear(dim, 2)
        
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = torch.cat([x, t_emb], dim=-1)
        h = self.input_proj(h)
        h = self.blocks(h)
        return self.output_proj(h)

# ============================================================================
# 4. ODE 采样器
# ============================================================================
@torch.no_grad()
def sample_ode(model, x_init, n_steps=200, method='rk4', device='cpu'):
    """从初始点出发，沿 ODE 积分到 t=1"""
    model.eval()
    x = x_init.clone().to(device)
    dt = 1.0 / n_steps
    trajectory = [x.cpu().numpy()]
    
    for i in range(n_steps):
        t = torch.ones(x.shape[0], 1).to(device) * (i / n_steps)
        
        if method == 'euler':
            v = model(x, t)
            x = x + dt * v
        elif method == 'rk4':
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
def train_generative_fm(model, dataloader, epochs=2000, lr=1e-3, exp_name='GenFM'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x0, x1 in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
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
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 200 == 0 or epoch == 0:
            logger.info(f"  {exp_name} Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return loss_history

# ============================================================================
# 6. 可视化
# ============================================================================
def plot_generation(trajectories, titles, save_path):
    """多起点生成对比"""
    n_traj = len(trajectories)
    fig, axes = plt.subplots(n_traj, 6, figsize=(24, 4*n_traj))
    if n_traj == 1:
        axes = axes.reshape(1, -1)
    
    steps = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for row, (traj, title) in enumerate(zip(trajectories, titles)):
        for col, t_val in enumerate(steps):
            idx = int(t_val * (len(traj)-1))
            pts = traj[idx]
            axes[row, col].scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.7, 
                                   c=plt.cm.viridis(t_val))
            axes[row, col].set_title(f't={t_val:.1f}', fontsize=12)
            axes[row, col].set_xlim(-2, 2); axes[row, col].set_ylim(-2, 2)
            axes[row, col].set_aspect('equal')
            axes[row, col].grid(True, alpha=0.3)
        axes[row, 0].set_ylabel(title, fontsize=12)
    
    plt.suptitle('生成式 Flow Matching: 多起点高斯噪声 → IMBR', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  多起点生成图: {save_path}")

def plot_target_distribution(target, save_path):
    """绘制目标分布"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target[:, 0], target[:, 1], s=5, c='red', alpha=0.6)
    ax.set_title('目标分布: IMBR (2000点)', fontsize=14)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  目标分布图: {save_path}")

# ============================================================================
# 7. 主流程
# ============================================================================
def main():
    try:
        # 生成 IMBR 目标
        logger.info(f"\n{'='*60}")
        logger.info(f"生成目标分布: IMBR")
        logger.info(f"{'='*60}")
        target_imbr = generate_letter_cloud('IMBR', n_points=2000, canvas_size=64)
        logger.info(f"IMBR 点云: {target_imbr.shape}")
        plot_target_distribution(target_imbr, os.path.join(PLOTS_DIR, 'gen_target_imbr.png'))
        
        # 创建数据集
        logger.info(f"\n{'='*60}")
        logger.info(f"创建生成式训练数据集")
        logger.info(f"{'='*60}")
        dataset = GenerativePointDataset(target_imbr, n_source=5000)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
        
        # 初始化模型
        logger.info(f"\n{'='*60}")
        logger.info(f"初始化模型")
        logger.info(f"{'='*60}")
        model = GenerativeVectorField(dim=512, time_dim=128, n_blocks=6).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数量: {n_params:,}")
        
        # 训练
        logger.info(f"\n{'='*60}")
        logger.info(f"开始训练 (2000 epochs)")
        logger.info(f"{'='*60}")
        t0 = time.time()
        losses = train_generative_fm(model, dataloader, epochs=2000, exp_name='GenFM')
        t1 = time.time()
        logger.info(f"训练完成，耗时: {t1-t0:.1f}s")
        logger.info(f"Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
        
        # 保存模型
        model_path = os.path.join(CURRENT_DIR, 'model_generative_fm_imbr.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型保存: {model_path}")
        
        # 绘制 Loss 曲线
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, linewidth=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title('生成式 Flow Matching Loss 曲线')
        ax.grid(True, alpha=0.3); ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'gen_loss_curve.png'), dpi=150)
        plt.close()
        logger.info(f"Loss曲线: plots/gen_loss_curve.png")
        
        # ====================================================================
        # 多起点采样验证
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"多起点采样验证")
        logger.info(f"{'='*60}")
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        seed_configs = [
            ('种子1: 标准高斯', np.random.randn(2000, 2) * 0.8),
            ('种子2: 收缩高斯', np.random.randn(2000, 2) * 0.5),
            ('种子3: 膨胀高斯', np.random.randn(2000, 2) * 1.2),
            ('种子4: 均匀分布', np.random.uniform(-1.5, 1.5, (2000, 2))),
        ]
        
        trajectories = []
        titles = []
        for name, noise in seed_configs:
            x_init = torch.tensor(noise, dtype=torch.float32)
            traj = sample_ode(model, x_init, n_steps=300, method='rk4', device=device)
            trajectories.append(traj)
            titles.append(name)
            # 计算最终点与目标分布的 Chamfer 距离
            final_pts = traj[-1]
            logger.info(f"  {name}: 生成 {len(final_pts)} 点, 范围 x:[{final_pts[:,0].min():.2f},{final_pts[:,0].max():.2f}] y:[{final_pts[:,1].min():.2f},{final_pts[:,1].max():.2f}]")
        
        # 多起点对比图
        plot_generation(trajectories, titles, 
                       os.path.join(PLOTS_DIR, 'gen_multi_seed_comparison.png'))
        
        # 额外采样：10个不同种子，只显示最终结果
        logger.info(f"\n  --- 10个不同种子最终结果对比 ---")
        fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
        torch.manual_seed(999)
        for i in range(10):
            noise = torch.randn(2000, 2) * 0.8
            traj = sample_ode(model, noise, n_steps=300, method='rk4', device=device)
            final = traj[-1]
            row, col = i // 5, i % 5
            axes2[row, col].scatter(final[:, 0], final[:, 1], s=4, alpha=0.7, c='green')
            axes2[row, col].set_title(f'种子 {i+1}', fontsize=10)
            axes2[row, col].set_xlim(-1.5, 1.5); axes2[row, col].set_ylim(-1.5, 1.5)
            axes2[row, col].set_aspect('equal'); axes2[row, col].grid(True, alpha=0.3)
        plt.suptitle('10个不同噪声种子 → IMBR 生成结果', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'gen_10_seeds_final.png'), dpi=150)
        plt.close()
        logger.info(f"10种子对比: plots/gen_10_seeds_final.png")
        
        # 保存总结
        summary = {
            'timestamp': TIMESTAMP, 'device': str(device),
            'model_params': n_params, 'final_loss': losses[-1],
            'training_time': t1-t0, 'problems': problems
        }
        with open(os.path.join(CURRENT_DIR, 'generative_fm_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"实验完成！")
        logger.info(f"模型: {model_path}")
        logger.info(f"日志: {LOG_FILE}")
        logger.info(f"图像: {PLOTS_DIR}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"实验失败: {e}")
        logger.error(traceback.format_exc())
        record_problem("全局", "实验异常", str(e), traceback.format_exc())
        raise
    finally:
        # 生成问题记录
        with open(PROBLEM_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n\n## 生成式 FM 实验 ({TIMESTAMP})\n\n")
            if not problems:
                f.write("✅ 本次实验未出现重大问题\n")
            else:
                f.write(f"问题数: {len(problems)}\n")
                for p in problems:
                    f.write(f"- [{p['phase']}] {p['title']}: {p['description']} → {p['solution']}\n")

if __name__ == '__main__':
    main()