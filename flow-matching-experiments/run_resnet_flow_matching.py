"""
=============================================================================
生成式 Flow Matching + ResNet：从高斯噪声生成 IMBR 字母
=============================================================================
核心理念：
  - 训练一个向量场 v_theta(x, t)，使得 ODE 从任何噪声点出发都收敛到 IMBR
  - 使用 ResNet 结构 + 位置编码，提升模型表达能力
  - 用 GPU 加速训练

关键改进：
  1. 位置编码 (Positional Encoding) - 让模型理解空间位置关系
  2. 残差网络 (ResNet) - 8 个残差块，1024 隐层维度
  3. 层归一化 (LayerNorm) - 稳定训练
  4. 长训练 (5000 epochs) - 充分收敛
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
LOG_FILE = os.path.join(LOGS_DIR, f'resnet_fm_log_{TIMESTAMP}.txt')
PROBLEM_LOG_FILE = os.path.join(CURRENT_DIR, '实验问题记录.md')

logger = logging.getLogger('ResNetFM')
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
logger.info(f"生成式 ResNet Flow Matching 实验")
logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
logger.info(f"日志保存: {LOG_FILE}")
logger.info(f"图像保存: {PLOTS_DIR}")
logger.info(f"{'='*60}")

problems = []
def record_problem(phase, title, desc, solution="待解决"):
    problems.append({'phase': phase, 'title': title, 'description': desc,
                     'solution': solution, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    logger.warning(f"[问题] {phase}: {title}")

# ============================================================================
# 1. 生成 IMBR 目标点云
# ============================================================================
def generate_imbr_cloud(n_points=3000, canvas_size=64):
    """生成清晰的 IMBR 点云"""
    from PIL import Image, ImageDraw, ImageFont
    
    font_size = int(canvas_size * 0.85)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 4 个字母并排，每个占 canvas_size 宽度
    w = canvas_size * 4
    h = canvas_size
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    
    # 逐个绘制字母，确保清晰
    for i, ch in enumerate('IMBR'):
        # 每个字母居中放置在其区域内
        x_offset = i * canvas_size + canvas_size // 4
        draw.text((x_offset, 0), ch, fill=255, font=font)
    
    img_arr = np.array(img)
    coords = np.argwhere(img_arr > 128)  # 阈值过滤
    if len(coords) == 0:
        logger.error("IMBR 生成失败，无有效像素")
        # 返回一个简单形状
        t = np.linspace(0, 2*np.pi, n_points)
        return np.column_stack([np.cos(t), np.sin(t)]).astype(np.float32)
    
    # 坐标转换
    coords = coords[:, [1, 0]]  # (y, x) -> (x, y)
    coords = coords.astype(np.float32)
    coords[:, 0] = (coords[:, 0] / w) * 2 - 1  # x: 0~w -> -1~1
    coords[:, 1] = (coords[:, 1] / h) * 2 - 1  # y: 0~h -> -1~1
    
    # 采样到目标点数
    if len(coords) > n_points:
        idx = np.random.choice(len(coords), n_points, replace=False)
        coords = coords[idx]
    elif len(coords) < n_points:
        idx = np.random.choice(len(coords), n_points, replace=True)
        coords = coords[idx]
    
    logger.info(f"IMBR 点云: {len(coords)} 点, 范围 x:[{coords[:,0].min():.3f},{coords[:,0].max():.3f}]")
    return coords

# ============================================================================
# 2. 生成式数据集
# ============================================================================
class GenerativePointDataset(Dataset):
    """源 = 高斯噪声，目标 = IMBR，训练时随机配对"""
    def __init__(self, target_points, n_source=10000):
        self.target = torch.tensor(target_points, dtype=torch.float32)
        self.n_source = n_source
        self.n_target = len(target_points)
        logger.info(f"  数据集: 高斯噪声({n_source}点/epoch) × IMBR({self.n_target}点)")

    def __len__(self):
        return self.n_source

    def __getitem__(self, idx):
        # 从高斯噪声采样，标准差 0.5 确保覆盖目标区域
        x0 = torch.randn(2) * 0.5
        # 随机配对目标点
        rand_idx = torch.randint(0, self.n_target, (1,)).item()
        x1 = self.target[rand_idx]
        return x0, x1

# ============================================================================
# 3. ResNet 架构 + 位置编码
# ============================================================================
class PositionalEncoding2D(nn.Module):
    """2D 位置编码 - 将输入坐标映射到高频特征空间"""
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.freqs = 2.0 ** torch.linspace(0, 10, dim // 4)
        
    def forward(self, x):
        # x: (N, 2)
        freqs = self.freqs.to(x.device)
        # 分别对 x 和 y 坐标编码
        x_enc = x[:, 0:1] * freqs[None, :] * np.pi  # (N, dim/4)
        y_enc = x[:, 1:2] * freqs[None, :] * np.pi  # (N, dim/4)
        # 拼接 sin/cos
        enc = torch.cat([
            torch.sin(x_enc), torch.cos(x_enc),
            torch.sin(y_enc), torch.cos(y_enc)
        ], dim=-1)  # (N, dim)
        return enc

class TimeEmbedding(nn.Module):
    """时间编码器"""
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        
    def forward(self, t):
        # t: (N, 1)
        half_dim = self.dim // 2
        emb = np.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.net(emb)

class ResidualBlockFM(nn.Module):
    """Flow Matching 残差块"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.norm(x + self.net(x))

class ResNetFlowMatching(nn.Module):
    """
    ResNet 架构的 Flow Matching 模型
    
    结构：
    - 位置编码: 2D 坐标 → 高频特征 (256维)
    - 时间编码: t → 时间特征 (256维)
    - 输入投影: 512维 → 1024维
    - 8 个残差块: 每个包含 Linear+SiLU+Linear + LayerNorm + 残差连接
    - 输出投影: 1024维 → 2维 (速度)
    """
    def __init__(self, hidden_dim=1024, pos_dim=256, time_dim=256, n_blocks=8):
        super().__init__()
        self.pos_enc = PositionalEncoding2D(pos_dim)
        self.time_emb = TimeEmbedding(time_dim)
        
        self.input_proj = nn.Sequential(
            nn.Linear(pos_dim + time_dim, hidden_dim),
            nn.SiLU(),
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlockFM(hidden_dim) for _ in range(n_blocks)
        ])
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x, t):
        # x: (N, 2), t: (N, 1)
        x_enc = self.pos_enc(x)           # (N, pos_dim)
        t_emb = self.time_emb(t)          # (N, time_dim)
        h = torch.cat([x_enc, t_emb], dim=-1)
        h = self.input_proj(h)
        
        for block in self.blocks:
            h = block(h)
        
        return self.output_proj(h)

# ============================================================================
# 4. ODE 采样器
# ============================================================================
@torch.no_grad()
def sample_ode(model, x_init, n_steps=500, method='rk4', device='cpu'):
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
def train_flow_matching(model, dataloader, epochs=5000, lr=1e-3, exp_name='ResNetFM'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x0, x1 in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            bs = x0.shape[0]
            
            # 随机采样时间 t
            t = torch.rand(bs, 1).to(device)
            
            # 线性插值：x_t = (1-t)*x0 + t*x1
            x_t = (1 - t) * x0 + t * x1
            
            # 目标速度：v = x1 - x0
            target_v = x1 - x0
            
            # 预测速度
            pred_v = model(x_t, t)
            
            # MSE 损失
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
            lr_now = scheduler.get_last_lr()[0]
            logger.info(f"  {exp_name} Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr_now:.6f}")
            
            # 每 1000 epoch 做一次快速采样检查
            if (epoch + 1) % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    test_noise = torch.randn(2000, 2).to(device) * 0.5
                    test_traj = sample_ode(model, test_noise, n_steps=100, method='rk4', device=device)
                    final_pts = test_traj[-1]
                    logger.info(f"    [采样检查] 范围 x:[{final_pts[:,0].min():.3f},{final_pts[:,0].max():.3f}] y:[{final_pts[:,1].min():.3f},{final_pts[:,1].max():.3f}]")
                model.train()
    
    return loss_history

# ============================================================================
# 6. 可视化
# ============================================================================
def plot_trajectory(trajectory, title, save_path):
    """绘制生成轨迹"""
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
    logger.info(f"  轨迹图: {save_path}")

def plot_final_comparison(target, generated_list, titles, save_path):
    """最终结果对比"""
    n = len(generated_list) + 1
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    
    axes[0].scatter(target[:, 0], target[:, 1], s=5, c='red', alpha=0.6)
    axes[0].set_title('目标: IMBR', fontsize=12)
    axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    
    for i, (pts, title) in enumerate(zip(generated_list, titles)):
        axes[i+1].scatter(pts[:, 0], pts[:, 1], s=5, c='green', alpha=0.6)
        axes[i+1].set_title(title, fontsize=12)
        axes[i+1].set_xlim(-1.5, 1.5); axes[i+1].set_ylim(-1.5, 1.5)
        axes[i+1].set_aspect('equal'); axes[i+1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  对比图: {save_path}")

# ============================================================================
# 7. 主流程
# ============================================================================
def main():
    try:
        # 生成目标分布
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 1/4: 生成 IMBR 目标点云")
        logger.info(f"{'='*60}")
        target_imbr = generate_imbr_cloud(n_points=3000, canvas_size=64)
        
        # 可视化目标
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(target_imbr[:, 0], target_imbr[:, 1], s=5, c='red', alpha=0.6)
        ax.set_title('目标分布: IMBR', fontsize=14)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'resnet_target_imbr.png'), dpi=150)
        plt.close()
        
        # 创建数据集
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 2/4: 创建训练数据集")
        logger.info(f"{'='*60}")
        dataset = GenerativePointDataset(target_imbr, n_source=20000)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0,
                                pin_memory=True if device.type == 'cuda' else False)
        
        # 初始化模型
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 3/4: 初始化 ResNet Flow Matching 模型")
        logger.info(f"{'='*60}")
        model = ResNetFlowMatching(
            hidden_dim=1024,
            pos_dim=256,
            time_dim=256,
            n_blocks=8
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  模型参数量: {n_params:,}")
        logger.info(f"  架构: ResNet(8块, 1024维) + 位置编码(256维)")
        
        # 训练
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤 4/4: 开始训练 (5000 epochs)")
        logger.info(f"{'='*60}")
        t0 = time.time()
        losses = train_flow_matching(model, dataloader, epochs=5000, exp_name='ResNetFM')
        t1 = time.time()
        logger.info(f"\n训练完成! 耗时: {t1-t0:.1f}s ({t1-t0:.1f}s)")
        logger.info(f"Loss 变化: {losses[0]:.6f} → {losses[-1]:.6f}")
        
        # 保存模型
        model_path = os.path.join(CURRENT_DIR, 'model_resnet_fm_imbr.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型保存: {model_path}")
        
        # 绘制 Loss 曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(losses, linewidth=1.5)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss')
        ax1.set_title('Loss 曲线'); ax1.grid(True, alpha=0.3)
        
        ax2.plot(losses, linewidth=1.5)
        ax2.set_yscale('log')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE Loss (log)')
        ax2.set_title('Loss 曲线 (对数)'); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'resnet_loss_curve.png'), dpi=150)
        plt.close()
        logger.info(f"Loss曲线: plots/resnet_loss_curve.png")
        
        # ====================================================================
        # 采样验证
        # ====================================================================
        logger.info(f"\n{'='*60}")
        logger.info(f"采样验证: 多起点生成 IMBR")
        logger.info(f"{'='*60}")
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        # 8 个不同种子
        seed_configs = [
            ('标准高斯 (σ=0.5)', np.random.randn(3000, 2) * 0.5),
            ('收缩高斯 (σ=0.3)', np.random.randn(3000, 2) * 0.3),
            ('膨胀高斯 (σ=0.8)', np.random.randn(3000, 2) * 0.8),
            ('均匀分布', np.random.uniform(-1.5, 1.5, (3000, 2))),
            ('环形分布', np.column_stack([
                np.cos(np.linspace(0, 2*np.pi, 3000)) * (0.5 + 0.3*np.random.rand(3000)),
                np.sin(np.linspace(0, 2*np.pi, 3000)) * (0.5 + 0.3*np.random.rand(3000))
            ])),
        ]
        
        # 单种子详细轨迹
        noise = np.random.randn(3000, 2) * 0.5
        x_init = torch.tensor(noise, dtype=torch.float32)
        traj = sample_ode(model, x_init, n_steps=500, method='rk4', device=device)
        plot_trajectory(traj, 'ResNet Flow Matching: 高斯噪声 → IMBR 生成过程',
                        os.path.join(PLOTS_DIR, 'resnet_trajectory.png'))
        
        # 所有种子最终结果
        final_results = []
        final_titles = []
        for name, noise in seed_configs:
            x_init = torch.tensor(noise, dtype=torch.float32)
            traj = sample_ode(model, x_init, n_steps=500, method='rk4', device=device)
            final = traj[-1]
            final_results.append(final)
            final_titles.append(name)
            logger.info(f"  {name}: 范围 x:[{final[:,0]:.3f},{final[:,0]:.3f}] y:[{final[:,1]:.3f},{final[:,1]:.3f}]")
        
        # 多种子对比图
        plot_final_comparison(target_imbr, final_results, final_titles,
                             os.path.join(PLOTS_DIR, 'resnet_multi_seed_comparison.png'))
        
        # 10 个随机种子最终结果网格
        fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
        torch.manual_seed(999)
        for i in range(10):
            noise = torch.randn(3000, 2) * 0.5
            traj = sample_ode(model, torch.tensor(noise, dtype=torch.float32), 
                            n_steps=500, method='rk4', device=device)
            final = traj[-1]
            row, col = i // 5, i % 5
            axes2[row, col].scatter(final[:, 0], final[:, 1], s=4, alpha=0.7, c='green')
            axes2[row, col].set_title(f'种子 {i+1}', fontsize=10)
            axes2[row, col].set_xlim(-1.5, 1.5); axes2[row, col].set_ylim(-1.5, 1.5)
            axes2[row, col].set_aspect('equal'); axes2[row, col].grid(True, alpha=0.3)
        plt.suptitle('10 个不同噪声种子 → IMBR 生成结果', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'resnet_10_seeds.png'), dpi=150)
        plt.close()
        logger.info(f"10种子对比: plots/resnet_10_seeds.png")
        
        # 保存总结
        summary = {
            'timestamp': TIMESTAMP, 'device': str(device),
            'model_params': n_params, 'final_loss': losses[-1],
            'training_time': t1-t0, 'problems': problems
        }
        with open(os.path.join(CURRENT_DIR, 'resnet_fm_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ 实验完成！")
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
        with open(PROBLEM_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n\n## ResNet FM 实验 ({TIMESTAMP})\n\n")
            if not problems:
                f.write("✅ 本次实验未出现重大问题\n")
            else:
                f.write(f"问题数: {len(problems)}\n")
                for p in problems:
                    f.write(f"- [{p['phase']}] {p['title']}: {p['description']} → {p['solution']}\n")

if __name__ == '__main__':
    main()