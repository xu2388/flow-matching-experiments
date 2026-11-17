"""
=============================================================================
Flow Matching 实验完整运行脚本
=============================================================================
实验内容：
  实验一：圆环 → 月牙形 (随机配对)
  实验二：十字笔画 → 字母 'C' (随机配对)
  实验三：十字笔画 → 'IMBR' (最优传输 OT 配对)

所有日志、图像、模型均保存在当前文件夹下。
=============================================================================
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，确保不出图也能保存
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_moons
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 日志和输出目录配置
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
LOGS_DIR = os.path.join(CURRENT_DIR, 'logs')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOGS_DIR, f'experiment_log_{TIMESTAMP}.txt')
PROBLEM_LOG_FILE = os.path.join(CURRENT_DIR, '实验问题记录.md')

# 设置日志器
logger = logging.getLogger('FlowMatching')
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
logger.info(f"Flow Matching 实验启动")
logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"设备: {device}")
logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
logger.info(f"日志保存: {LOG_FILE}")
logger.info(f"图像保存: {PLOTS_DIR}")
logger.info(f"{'='*60}")

# 问题记录
problems = []

def record_problem(phase, title, description, solution="待解决"):
    """记录实验中出现的问题"""
    problems.append({
        'phase': phase,
        'title': title,
        'description': description,
        'solution': solution,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    logger.warning(f"[问题记录] {phase}: {title}")


# ============================================================================
# 1. 数据生成器
# ============================================================================

def generate_circle(n_samples=1000, radius=1.0):
    """生成圆环/高斯球数据"""
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    r = radius * np.sqrt(np.random.rand(n_samples))
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.stack([x, y], axis=1).astype(np.float32)

def generate_crescent(n_samples=1000):
    """生成月牙形数据"""
    X, _ = make_moons(n_samples=n_samples, noise=0.05)
    return X.astype(np.float32)

def generate_cross(n_samples_per_arm=100):
    """生成十字笔画点云"""
    points = []
    x_h = np.linspace(-1, 1, n_samples_per_arm)
    y_h = np.random.uniform(-0.1, 0.1, n_samples_per_arm)
    y_v = np.linspace(-1, 1, n_samples_per_arm)
    x_v = np.random.uniform(-0.1, 0.1, n_samples_per_arm)
    points.extend(zip(x_h, y_h))
    points.extend(zip(x_v, y_v))
    return np.array(points, dtype=np.float32)

def generate_letter(letter='C', n_samples=500, canvas_size=32):
    """
    使用 PIL 绘制字母并提取点云。
    支持多字母字符串（如 'IMBR'），自动扩展画布宽度并排绘制。
    """
    from PIL import Image, ImageDraw, ImageFont
    
    n_chars = len(letter)
    # 多字母时，画布宽度按比例扩展，保持高度不变
    w = canvas_size * n_chars
    h = canvas_size
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", int(canvas_size * 0.8))
    except Exception as e:
        logger.warning(f"字体加载失败: {e}，使用默认字体")
        font = ImageFont.load_default()
    
    if n_chars == 1:
        # 单字母：居中绘制
        try:
            draw.text((canvas_size // 4, 0), letter, fill=255, font=font)
        except Exception as e:
            logger.warning(f"文字绘制失败: {e}，尝试小尺寸")
            draw.text((2, 0), letter, fill=255, font=font)
    else:
        # 多字母：逐个绘制，每个字母占 canvas_size 宽度区域
        try:
            for i, ch in enumerate(letter):
                draw.text((i * canvas_size + canvas_size // 4, 0), ch, fill=255, font=font)
        except Exception as e:
            logger.warning(f"多字母绘制失败: {e}，使用默认字体逐个绘制")
            for i, ch in enumerate(letter):
                draw.text((i * canvas_size + 2, 0), ch, fill=255, font=font)
    
    img = np.array(img)
    coords = np.argwhere(img > 0)
    if len(coords) == 0:
        logger.error(f"字母 '{letter}' 绘制后无有效像素！")
        record_problem("数据生成", f"字母 '{letter}' 绘制失败", "无有效像素点", "尝试调整字体大小或使用系统字体路径")
        coords = np.random.rand(100, 2) * 20 + 10
    
    # 转换坐标：x 归一化到 [-1, 1]，y 归一化到 [-1, 1]
    coords = coords[:, [1, 0]]  # (y, x) -> (x, y)
    coords[:, 0] = (coords[:, 0] / w) * 2 - 1   # x: 0~w -> -1~1
    coords[:, 1] = (coords[:, 1] / h) * 2 - 1   # y: 0~h -> -1~1
    
    if len(coords) > n_samples:
        idx = np.random.choice(len(coords), n_samples, replace=False)
        coords = coords[idx]
    elif len(coords) < n_samples and len(coords) > 0:
        idx = np.random.choice(len(coords), n_samples, replace=True)
        coords = coords[idx]
    return coords.astype(np.float32)


# ============================================================================
# 2. 数据集
# ============================================================================

class PointCloudDataset(Dataset):
    def __init__(self, source_points, target_points, mode='random'):
        self.source = torch.tensor(source_points, dtype=torch.float32)
        self.target = torch.tensor(target_points, dtype=torch.float32)
        self.mode = mode
        if mode == 'ot':
            logger.info("  正在计算 OT 配对 (匈牙利算法)...")
            t0 = time.time()
            self.paired_target = self._optimal_transport_pairing()
            logger.info(f"  OT 配对完成，耗时 {time.time()-t0:.2f}s")
        else:
            logger.info(f"  使用随机配对模式")

    def _optimal_transport_pairing(self):
        cost = torch.cdist(self.source, self.target).numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        return self.target[col_ind]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        x0 = self.source[idx]
        if self.mode == 'ot':
            x1 = self.paired_target[idx]
        else:
            rand_idx = torch.randint(0, len(self.target), (1,)).item()
            x1 = self.target[rand_idx]
        return x0, x1


# ============================================================================
# 3. 模型
# ============================================================================

class TimeEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class VectorFieldMLP(nn.Module):
    def __init__(self, dim=128, time_dim=64):
        super().__init__()
        self.time_mlp = TimeEncoder(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2)
        )
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


# ============================================================================
# 4. 采样器
# ============================================================================

@torch.no_grad()
def sample_euler(model, x_init, n_steps=100, device='cpu'):
    model.eval()
    x = x_init.clone().to(device)
    step_size = 1.0 / n_steps
    trajectory = [x.cpu().numpy()]
    for i in range(n_steps):
        t = torch.ones(x.shape[0], 1).to(device) * (i / n_steps)
        v = model(x, t)
        x = x + step_size * v
        record_freq = max(1, n_steps // 5)
        if (i + 1) % record_freq == 0 or i == 0:
            trajectory.append(x.cpu().numpy())
    trajectory.append(x.cpu().numpy())
    return trajectory

@torch.no_grad()
def sample_rk4(model, x_init, n_steps=100, device='cpu'):
    """四阶龙格-库塔采样器，比欧拉方法更精确"""
    model.eval()
    x = x_init.clone().to(device)
    dt = 1.0 / n_steps
    trajectory = [x.cpu().numpy()]
    for i in range(n_steps):
        t = torch.ones(x.shape[0], 1).to(device) * (i / n_steps)
        
        # RK4 四个斜率
        k1 = model(x, t)
        k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = model(x + dt * k3, t + dt)
        
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        record_freq = max(1, n_steps // 5)
        if (i + 1) % record_freq == 0 or i == 0:
            trajectory.append(x.cpu().numpy())
    trajectory.append(x.cpu().numpy())
    return trajectory


# ============================================================================
# 5. 训练函数
# ============================================================================

def train_flow_matching(model, dataloader, epochs=500, lr=1e-3, exp_name='实验'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []
    epoch_times = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for x0, x1 in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            batch_size = x0.shape[0]
            t = torch.rand(batch_size, 1).to(device)
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
        epoch_times.append(time.time() - t0)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_time = np.mean(epoch_times[-50:]) if len(epoch_times) >= 50 else np.mean(epoch_times)
            logger.info(f"  {exp_name} Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
                        f"每Epoch: {avg_time:.2f}s")
    
    return loss_history


# ============================================================================
# 6. 可视化工具
# ============================================================================

def plot_data_distribution(source, target, title, save_path):
    """绘制源和目标分布"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(source[:, 0], source[:, 1], s=3, alpha=0.6, c='blue')
    axes[0].set_title('源分布 (Source)')
    axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_aspect('equal')
    axes[1].scatter(target[:, 0], target[:, 1], s=3, alpha=0.6, c='red')
    axes[1].set_title('目标分布 (Target)')
    axes[1].set_xlim(-1.5, 1.5); axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_aspect('equal')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  数据分布图已保存: {save_path}")

def plot_trajectory(trajectory, title, save_path, trajectory2=None, title2=None):
    """绘制变形轨迹的时间序列"""
    n_plots = 6
    fig, axes = plt.subplots(1, n_plots, figsize=(21, 3.5))
    steps = np.linspace(0, len(trajectory)-1, n_plots, dtype=int)
    for i, (ax, idx) in enumerate(zip(axes, steps)):
        points = trajectory[idx]
        ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.6, c='blue')
        ax.set_title(f"t={idx/(len(trajectory)-1):.1f}")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  轨迹图已保存: {save_path}")

def plot_loss_curves(losses_dict, save_path):
    """绘制多个实验的 Loss 对比曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Flow Matching 训练 Loss 对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Loss 对比图已保存: {save_path}")

def plot_ot_comparison(traj_random, traj_ot, save_path):
    """随机配对 vs OT 配对的对比图"""
    fig, axes = plt.subplots(2, 6, figsize=(21, 7))
    steps = np.linspace(0, len(traj_random)-1, 6, dtype=int)
    for i, idx in enumerate(steps):
        # 第一行：随机配对
        axes[0, i].scatter(traj_random[idx][:, 0], traj_random[idx][:, 1], s=3, alpha=0.6, c='blue')
        axes[0, i].set_title(f"Random t={idx/(len(traj_random)-1):.1f}")
        axes[0, i].set_xlim(-1.5, 1.5); axes[0, i].set_ylim(-1.5, 1.5)
        axes[0, i].set_aspect('equal')
        axes[0, i].grid(True, alpha=0.3)
        # 第二行：OT 配对
        axes[1, i].scatter(traj_ot[idx][:, 0], traj_ot[idx][:, 1], s=3, alpha=0.6, c='red')
        axes[1, i].set_title(f"OT t={idx/(len(traj_ot)-1):.1f}")
        axes[1, i].set_xlim(-1.5, 1.5); axes[1, i].set_ylim(-1.5, 1.5)
        axes[1, i].set_aspect('equal')
        axes[1, i].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel('随机配对')
    axes[1, 0].set_ylabel('OT 配对')
    plt.suptitle('随机配对 vs OT 最优传输配对对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  OT 对比图已保存: {save_path}")


# ============================================================================
# 7. 主实验流程
# ============================================================================

def run_experiment():
    all_losses = {}
    all_results = {}
    
    try:
        # ====================================================================
        # 实验一：圆环 → 月牙形
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"实验一：圆环 → 月牙形 (随机配对)")
        logger.info(f"{'='*60}")
        
        n_points = 2000
        source_data = generate_circle(n_points, radius=0.8)
        target_data = generate_crescent(n_points)
        logger.info(f"源数据: {source_data.shape}, 目标数据: {target_data.shape}")
        
        # 保存数据分布图
        plot_data_distribution(source_data, target_data, '实验一：圆环 → 月牙形',
                               os.path.join(PLOTS_DIR, 'exp1_data_distribution.png'))
        
        dataset = PointCloudDataset(source_data, target_data, mode='random')
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        model1 = VectorFieldMLP().to(device)
        logger.info(f"模型参数量: {sum(p.numel() for p in model1.parameters()):,}")
        
        t_start = time.time()
        losses1 = train_flow_matching(model1, dataloader, epochs=300, exp_name='实1')
        t_end = time.time()
        all_losses['圆环→月牙(随机)'] = losses1
        
        logger.info(f"实验一训练完成，耗时: {t_end-t_start:.1f}s")
        logger.info(f"  Loss 变化: {losses1[0]:.6f} → {losses1[-1]:.6f}")
        
        # 采样
        x_init = torch.tensor(source_data[:500], dtype=torch.float32)
        traj1 = sample_euler(model1, x_init, n_steps=100, device=device)
        plot_trajectory(traj1, '实验一：圆环 → 月牙形 (Flow Matching)',
                        os.path.join(PLOTS_DIR, 'exp1_trajectory.png'))
        all_results['实验一'] = {'source': '圆环', 'target': '月牙形', 'mode': '随机配对',
                                'final_loss': losses1[-1], 'time': t_end-t_start}
        
        # 保存模型
        model_path = os.path.join(CURRENT_DIR, 'model_exp1_circle_to_crescent.pth')
        torch.save(model1.state_dict(), model_path)
        logger.info(f"模型保存: {model_path}")
        
        # ====================================================================
        # 实验二：十字 → 字母 C
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"实验二：十字笔画 → 字母 'C' (随机配对)")
        logger.info(f"{'='*60}")
        
        source_cross = generate_cross(n_samples_per_arm=300)
        target_C = generate_letter('C', n_samples=600)
        logger.info(f"十字数据: {source_cross.shape}, 字母C数据: {target_C.shape}")
        
        plot_data_distribution(source_cross, target_C, '实验二：十字 → 字母C',
                               os.path.join(PLOTS_DIR, 'exp2_data_distribution.png'))
        
        dataset2 = PointCloudDataset(source_cross, target_C, mode='random')
        dataloader2 = DataLoader(dataset2, batch_size=256, shuffle=True)
        
        model2 = VectorFieldMLP().to(device)
        
        t_start = time.time()
        losses2 = train_flow_matching(model2, dataloader2, epochs=500, exp_name='实2')
        t_end = time.time()
        all_losses['十字→C(随机)'] = losses2
        
        logger.info(f"实验二训练完成，耗时: {t_end-t_start:.1f}s")
        logger.info(f"  Loss 变化: {losses2[0]:.6f} → {losses2[-1]:.6f}")
        
        x_init2 = torch.tensor(source_cross, dtype=torch.float32)
        traj2 = sample_euler(model2, x_init2, n_steps=150, device=device)
        plot_trajectory(traj2, '实验二：十字笔画 → 字母 C (Flow Matching, 随机配对)',
                        os.path.join(PLOTS_DIR, 'exp2_trajectory.png'))
        all_results['实验二'] = {'source': '十字', 'target': '字母C', 'mode': '随机配对',
                                'final_loss': losses2[-1], 'time': t_end-t_start}
        
        model_path2 = os.path.join(CURRENT_DIR, 'model_exp2_cross_to_c.pth')
        torch.save(model2.state_dict(), model_path2)
        logger.info(f"模型保存: {model_path2}")
        
        # ====================================================================
        # 实验三：十字 → IMBR (OT 配对)
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"实验三：十字笔画 → 'IMBR' (最优传输 OT 配对)")
        logger.info(f"{'='*60}")
        
        source_cross3 = generate_cross(n_samples_per_arm=500)  # 1000点
        # IMBR：4个字母，画布扩展为 256x64，用更多采样点
        # 注意：源和目标点数必须一致，否则 OT 配对会遗漏部分目标点
        n_imbr = 1000
        try:
            target_imbr = generate_letter('IMBR', n_samples=n_imbr, canvas_size=64)
        except Exception as e:
            logger.error(f"生成 IMBR 失败: {e}")
            record_problem("实验三", "IMBR 字母生成失败", str(e), 
                          "使用逐字母生成后拼接")
            # 备选方案：逐字母生成
            parts = []
            for ch in 'IMBR':
                pts = generate_letter(ch, n_samples=250, canvas_size=64)
                parts.append(pts)
            target_imbr = np.vstack(parts)[:n_imbr]
        
        logger.info(f"十字数据: {source_cross3.shape}, IMBR数据: {target_imbr.shape}")
        
        plot_data_distribution(source_cross3, target_imbr, '实验三：十字 → IMBR',
                               os.path.join(PLOTS_DIR, 'exp3_data_distribution.png'))
        
        # 先训练随机配对版本作为对比
        logger.info(f"\n  --- 实验三 A：随机配对对比 ---")
        dataset3_random = PointCloudDataset(source_cross3, target_imbr, mode='random')
        dataloader3_random = DataLoader(dataset3_random, batch_size=512, shuffle=True)
        model3_random = VectorFieldMLP(dim=256, time_dim=64).to(device)  # 更大容量
        logger.info(f"模型参数量: {sum(p.numel() for p in model3_random.parameters()):,}")
        t_start = time.time()
        losses3_random = train_flow_matching(model3_random, dataloader3_random, epochs=800, exp_name='实3随机')
        t_end = time.time()
        all_losses['十字→IMBR(随机)'] = losses3_random
        logger.info(f"实验三A训练完成，耗时: {t_end-t_start:.1f}s")
        logger.info(f"  Loss 变化: {losses3_random[0]:.6f} → {losses3_random[-1]:.6f}")
        
        x_init3 = torch.tensor(source_cross3, dtype=torch.float32)
        traj3_random = sample_rk4(model3_random, x_init3, n_steps=500, device=device)
        plot_trajectory(traj3_random, '实验三A：十字 → IMBR (随机配对)',
                        os.path.join(PLOTS_DIR, 'exp3a_trajectory_random.png'))
        
        # 正式训练 OT 版本
        logger.info(f"\n  --- 实验三 B：OT 最优传输配对 ---")
        dataset3_ot = PointCloudDataset(source_cross3, target_imbr, mode='ot')
        dataloader3_ot = DataLoader(dataset3_ot, batch_size=512, shuffle=True)
        model3_ot = VectorFieldMLP(dim=256, time_dim=64).to(device)  # 更大容量
        logger.info(f"模型参数量: {sum(p.numel() for p in model3_ot.parameters()):,}")
        t_start = time.time()
        losses3_ot = train_flow_matching(model3_ot, dataloader3_ot, epochs=1200, exp_name='实3OT')
        t_end = time.time()
        all_losses['十字→IMBR(OT)'] = losses3_ot
        logger.info(f"实验三B训练完成，耗时: {t_end-t_start:.1f}s")
        logger.info(f"  Loss 变化: {losses3_ot[0]:.6f} → {losses3_ot[-1]:.6f}")
        
        traj3_ot = sample_rk4(model3_ot, x_init3, n_steps=500, device=device)
        plot_trajectory(traj3_ot, '实验三B：十字 → IMBR (OT-CFM 最优传输)',
                        os.path.join(PLOTS_DIR, 'exp3b_trajectory_ot.png'))
        
        # OT 对比图
        plot_ot_comparison(traj3_random, traj3_ot,
                          os.path.join(PLOTS_DIR, 'exp3_ot_comparison.png'))
        
        all_results['实验三A'] = {'source': '十字', 'target': 'IMBR', 'mode': '随机配对',
                                 'final_loss': losses3_random[-1], 'time': t_end-t_start}
        all_results['实验三B'] = {'source': '十字', 'target': 'IMBR', 'mode': 'OT配对',
                                 'final_loss': losses3_ot[-1], 'time': t_end-t_start}
        
        model_path3 = os.path.join(CURRENT_DIR, 'model_exp3_cross_to_imbr_ot.pth')
        torch.save(model3_ot.state_dict(), model_path3)
        logger.info(f"模型保存: {model_path3}")
        
        # ====================================================================
        # 8. 综合可视化
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"生成综合可视化...")
        logger.info(f"{'='*60}")
        
        # 所有 Loss 对比
        plot_loss_curves(all_losses, os.path.join(PLOTS_DIR, 'all_loss_comparison.png'))
        
        # 综合实验总结
        logger.info(f"\n{'='*60}")
        logger.info(f"实验总结")
        logger.info(f"{'='*60}")
        for name, res in all_results.items():
            logger.info(f"  {name}: {res['source']} → {res['target']} | "
                       f"模式={res['mode']} | Loss={res['final_loss']:.6f} | 耗时={res['time']:.0f}s")
        
        logger.info(f"\n所有实验完成！")
        logger.info(f"日志文件: {LOG_FILE}")
        logger.info(f"图像目录: {PLOTS_DIR}")
        
        # 生成实验总结 JSON
        summary = {
            'timestamp': TIMESTAMP,
            'device': str(device),
            'results': all_results,
            'problems': problems
        }
        summary_path = os.path.join(CURRENT_DIR, 'experiment_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"总结保存: {summary_path}")
        
    except Exception as e:
        logger.error(f"实验运行出错: {e}")
        logger.error(traceback.format_exc())
        record_problem("全局", "实验运行异常", str(e), traceback.format_exc())
        raise
    finally:
        # 生成问题记录文档
        generate_problem_report()


def generate_problem_report():
    """生成问题记录文档"""
    with open(PROBLEM_LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("# Flow Matching 实验问题记录\n\n")
        f.write(f"## 实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 实验脚本：`run_flow_matching_experiments.py`\n\n")
        f.write("---\n\n")
        
        if not problems:
            f.write("## ✅ 本次实验未出现重大问题\n\n")
            f.write("所有实验均正常运行，未记录到异常。\n")
        else:
            f.write(f"## 问题汇总（共 {len(problems)} 个）\n\n")
            for i, p in enumerate(problems, 1):
                f.write(f"### {i}. [{p['phase']}] {p['title']}\n\n")
                f.write(f"- **时间**: {p['time']}\n")
                f.write(f"- **描述**: {p['description']}\n")
                f.write(f"- **解决方案**: {p['solution']}\n\n")
                f.write("---\n\n")
        
        f.write("\n## 文件清单\n\n")
        f.write(f"- 实验脚本: `run_flow_matching_experiments.py`\n")
        f.write(f"- 日志文件: `logs/experiment_log_{TIMESTAMP}.txt`\n")
        f.write(f"- 图像目录: `plots/`\n")
        f.write(f"- 模型权重: `model_exp1_*.pth`, `model_exp2_*.pth`, `model_exp3_*.pth`\n")
        f.write(f"- 实验总结: `experiment_summary.json`\n")
    
    logger.info(f"问题记录已保存: {PROBLEM_LOG_FILE}")


# ============================================================================
# 入口
# ============================================================================
if __name__ == '__main__':
    run_experiment()