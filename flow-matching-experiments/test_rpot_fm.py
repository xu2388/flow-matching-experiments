"""
=============================================================================
随机投影 OT Flow Matching 模型测试脚本
=============================================================================
测试内容：
  1. 加载训练好的模型
  2. 从不同噪声种子生成 IMBR
  3. 计算定量指标（覆盖度、EMD、分布相似度）
  4. 生成详细的可视化报告
=============================================================================
"""

import os, sys, time, json
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"{'='*60}")
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"设备: {device}")
print(f"{'='*60}")

# ============================================================================
# 1. 模型定义（与训练时完全一致）
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
    def forward(self, x, t):
        h = torch.cat([self.pe(x), self.te(t)], dim=-1)
        h = self.inp(h)
        for b in self.blocks: h = b(h)
        return self.out(h)

# ============================================================================
# 2. 采样函数（RK4）
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
# 3. IMBR 目标数据
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
# 4. 指标计算
# ============================================================================
def compute_metrics(generated, target, name=""):
    """计算生成结果与目标之间的定量指标"""
    # 1. 覆盖度：生成点落在目标 bounding box 内的比例
    x_min, x_max = target[:,0].min(), target[:,0].max()
    y_min, y_max = target[:,1].min(), target[:,1].max()
    in_box = ((generated[:,0] >= x_min) & (generated[:,0] <= x_max) &
              (generated[:,1] >= y_min) & (generated[:,1] <= y_max))
    coverage = in_box.mean() * 100
    
    # 2. 最近邻距离（生成点到目标点的平均最近距离）
    target_tree = KDTree(target)
    dists, _ = target_tree.query(generated)
    mean_nn_dist = dists.mean()
    median_nn_dist = np.median(dists)
    std_nn_dist = dists.std()
    
    # 3. Wasserstein 距离（一维投影）
    wd_x = wasserstein_distance(generated[:,0], target[:,0])
    wd_y = wasserstein_distance(generated[:,1], target[:,1])
    wd_avg = (wd_x + wd_y) / 2
    
    # 4. 生成范围与目标范围对比
    gen_x_range = generated[:,0].max() - generated[:,0].min()
    gen_y_range = generated[:,1].max() - generated[:,1].min()
    tgt_x_range = x_max - x_min
    tgt_y_range = y_max - y_min
    
    print(f"  [{name}]")
    print(f"    Bounding Box 覆盖度: {coverage:.1f}%")
    print(f"    平均最近邻距离: {mean_nn_dist:.4f}")
    print(f"    中位数最近邻距离: {median_nn_dist:.4f}")
    print(f"    Wasserstein距离 (x): {wd_x:.4f}")
    print(f"    Wasserstein距离 (y): {wd_y:.4f}")
    print(f"    Wasserstein距离 (平均): {wd_avg:.4f}")
    print(f"    生成 x 范围: [{generated[:,0].min():.3f}, {generated[:,0].max():.3f}] (目标: [{x_min:.3f}, {x_max:.3f}])")
    print(f"    生成 y 范围: [{generated[:,1].min():.3f}, {generated[:,1].max():.3f}] (目标: [{y_min:.3f}, {y_max:.3f}])")
    
    return {
        'name': name,
        'coverage': coverage,
        'mean_nn_dist': mean_nn_dist,
        'median_nn_dist': median_nn_dist,
        'std_nn_dist': std_nn_dist,
        'wd_x': wd_x, 'wd_y': wd_y, 'wd_avg': wd_avg,
        'gen_x_min': float(generated[:,0].min()), 'gen_x_max': float(generated[:,0].max()),
        'gen_y_min': float(generated[:,1].min()), 'gen_y_max': float(generated[:,1].max()),
    }

# ============================================================================
# 5. 主测试流程
# ============================================================================
def main():
    try:
        # 加载模型
        model_path = os.path.join(CURRENT_DIR, 'model_rpot_fm.pth')
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return
        
        print(f"\n--- 加载模型: {model_path} ---")
        model = FMNet(h=1024, pd=256, td=256, nb=8).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {n_params:,}")
        
        # 生成目标
        print(f"\n--- 生成目标 IMBR ---")
        target = generate_imbr(n=5000)
        print(f"  目标点: {len(target)}点, x=[{target[:,0].min():.3f},{target[:,0].max():.3f}], y=[{target[:,1].min():.3f},{target[:,1].max():.3f}]")
        
        # 测试不同噪声
        print(f"\n--- 测试不同噪声的生成效果 ---")
        np.random.seed(42)
        torch.manual_seed(42)
        
        test_configs = [
            ('高斯 σ=0.3', np.random.randn(3000, 2) * 0.3),
            ('高斯 σ=0.5', np.random.randn(3000, 2) * 0.5),
            ('高斯 σ=0.6', np.random.randn(3000, 2) * 0.6),
            ('高斯 σ=0.8', np.random.randn(3000, 2) * 0.8),
            ('均匀 [-1,1]', np.random.uniform(-1, 1, (3000, 2))),
        ]
        
        all_metrics = []
        all_results = {}
        
        for name, noise in test_configs:
            init = torch.tensor(noise, dtype=torch.float32)
            traj = sample(model, init, n_steps=500, device=device)
            final = traj[-1]
            all_results[name] = {'traj': traj, 'final': final}
            metrics = compute_metrics(final, target, name)
            all_metrics.append(metrics)
        
        # ====================================================================
        # 可视化 1: 每种噪声的生成结果对比
        # ====================================================================
        print(f"\n--- 生成可视化 ---")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 目标
        axes[0].scatter(target[:,0], target[:,1], s=3, c='red', alpha=0.5, label=f'目标 ({len(target)}点)')
        axes[0].set_title('目标: IMBR', fontsize=14, fontweight='bold')
        axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)
        axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
        
        for i, (name, _) in enumerate(test_configs):
            ax = axes[i+1]
            final = all_results[name]['final']
            m = all_metrics[i]
            ax.scatter(final[:,0], final[:,1], s=3, alpha=0.5, c='green')
            ax.set_title(f'{name}\n覆盖度={m["coverage"]:.0f}%  WDist={m["wd_avg"]:.4f}  NN={m["mean_nn_dist"]:.3f}',
                         fontsize=11)
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        
        plt.suptitle('随机投影 OT Flow Matching: 不同噪声 → IMBR', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'test_comparison_{TIMESTAMP}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存: plots/test_comparison_{TIMESTAMP}.png")
        
        # ====================================================================
        # 可视化 2: 最佳噪声的轨迹
        # ====================================================================
        best_name = '高斯 σ=0.3'
        best_traj = all_results[best_name]['traj']
        fig2, axes2 = plt.subplots(1, 6, figsize=(24, 4))
        steps = np.linspace(0, len(best_traj)-1, 6, dtype=int)
        for i, (ax, idx) in enumerate(zip(axes2, steps)):
            pts = best_traj[idx]
            ax.scatter(pts[:,0], pts[:,1], s=4, alpha=0.7, c=plt.cm.viridis(idx/(len(best_traj)-1)))
            ax.set_title(f't={idx/(len(best_traj)-1):.1f}', fontsize=12)
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        plt.suptitle(f'生成轨迹 ({best_name} → IMBR)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'test_traj_{TIMESTAMP}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存: plots/test_traj_{TIMESTAMP}.png")
        
        # ====================================================================
        # 可视化 3: 10 个不同种子
        # ====================================================================
        fig3, axes3 = plt.subplots(2, 5, figsize=(20, 8))
        torch.manual_seed(999)
        for i in range(10):
            noise = torch.randn(3000, 2) * 0.3
            init = torch.tensor(noise, dtype=torch.float32)
            final = sample(model, init, n_steps=500, device=device)[-1]
            r, c = i//5, i%5
            axes3[r,c].scatter(final[:,0], final[:,1], s=3, alpha=0.6, c='green')
            axes3[r,c].set_title(f'种子 {i+1}', fontsize=10)
            axes3[r,c].set_xlim(-1.5, 1.5); axes3[r,c].set_ylim(-1.5, 1.5)
            axes3[r,c].set_aspect('equal'); axes3[r,c].grid(True, alpha=0.3)
        plt.suptitle('10 个随机种子 → IMBR (σ=0.3)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'test_10seeds_{TIMESTAMP}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存: plots/test_10seeds_{TIMESTAMP}.png")
        
        # ====================================================================
        # 可视化 4: 指标对比柱状图
        # ====================================================================
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
        names = [m['name'] for m in all_metrics]
        coverages = [m['coverage'] for m in all_metrics]
        wds = [m['wd_avg'] for m in all_metrics]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        axes4[0].bar(names, coverages, color=colors, alpha=0.8)
        axes4[0].set_ylabel('覆盖度 (%)', fontsize=12)
        axes4[0].set_title('Bounding Box 覆盖度', fontsize=13, fontweight='bold')
        axes4[0].set_ylim(0, 110)
        for i, v in enumerate(coverages):
            axes4[0].text(i, v+1, f'{v:.1f}%', ha='center', fontsize=10)
        axes4[0].tick_params(axis='x', rotation=15)
        axes4[0].grid(True, alpha=0.3, axis='y')
        
        axes4[1].bar(names, wds, color=colors, alpha=0.8)
        axes4[1].set_ylabel('Wasserstein 距离', fontsize=12)
        axes4[1].set_title('Wasserstein 距离 (越低越好)', fontsize=13, fontweight='bold')
        for i, v in enumerate(wds):
            axes4[1].text(i, v+0.001, f'{v:.4f}', ha='center', fontsize=10)
        axes4[1].tick_params(axis='x', rotation=15)
        axes4[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'test_metrics_{TIMESTAMP}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存: plots/test_metrics_{TIMESTAMP}.png")
        
        # ====================================================================
        # 输出总结
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"测试总结")
        print(f"{'='*60}")
        best_metric = min(all_metrics, key=lambda m: m['wd_avg'])
        print(f"最佳噪声: {best_metric['name']}")
        print(f"  Wasserstein 距离: {best_metric['wd_avg']:.4f}")
        print(f"  覆盖度: {best_metric['coverage']:.1f}%")
        print(f"  平均最近邻距离: {best_metric['mean_nn_dist']:.4f}")
        print(f"\n所有配置排名 (按 Wasserstein 距离):")
        sorted_metrics = sorted(all_metrics, key=lambda m: m['wd_avg'])
        for i, m in enumerate(sorted_metrics):
            print(f"  {i+1}. {m['name']}: WDist={m['wd_avg']:.4f}, 覆盖度={m['coverage']:.1f}%, NN={m['mean_nn_dist']:.3f}")
        
        # 保存结果
        results = {
            'timestamp': TIMESTAMP,
            'device': str(device),
            'model': model_path,
            'params': n_params,
            'metrics': all_metrics,
            'best': best_metric['name'],
            'ranking': [m['name'] for m in sorted_metrics],
        }
        with open(os.path.join(CURRENT_DIR, f'test_results_{TIMESTAMP}.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 测试完成! 结果已保存")
        
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    main()