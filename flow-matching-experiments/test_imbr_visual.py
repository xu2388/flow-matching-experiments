"""
直观测试：生成清晰的 IMBR 字母图
"""
import os, json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 模型定义
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

@torch.no_grad()
def sample(model, x_init, n_steps=500, device='cpu'):
    model.eval()
    x = x_init.clone().to(device)
    dt = 1.0/n_steps
    for i in range(n_steps):
        t = torch.ones(x.shape[0], 1).to(device)*(i/n_steps)
        k1 = model(x, t)
        k2 = model(x+0.5*dt*k1, t+0.5*dt)
        k3 = model(x+0.5*dt*k2, t+0.5*dt)
        k4 = model(x+dt*k3, t+dt)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
    return x.cpu().numpy()

# ============================================================================
# IMBR 目标
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
    else: coords = coords[np.random.choice(len(coords), n, replace=True)]
    return coords, (w, h)

# ============================================================================
# 主测试
# ============================================================================
print("加载模型...")
model = FMNet(h=1024, pd=256, td=256, nb=8).to(device)
model.load_state_dict(torch.load(os.path.join(CURRENT_DIR, 'model_rpot_fm.pth'), map_location=device, weights_only=True))

print("生成 IMBR 目标...")
target, (img_w, img_h) = generate_imbr(n=5000)

# 用小噪声生成（最容易成功）
print("\n开始生成...")
np.random.seed(42)
torch.manual_seed(42)

noise = np.random.randn(5000, 2).astype(np.float32) * 0.3
init = torch.tensor(noise, dtype=torch.float32)
generated = sample(model, init, n_steps=500, device=device)

print(f"生成完成!")
print(f"  生成点数: {len(generated)}")
print(f"  生成 x 范围: [{generated[:,0].min():.3f}, {generated[:,0].max():.3f}]")
print(f"  生成 y 范围: [{generated[:,1].min():.3f}, {generated[:,1].max():.3f}]")

# ============================================================================
# 可视化 1: 散点图直接对比
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(target[:,0], target[:,1], s=8, c='red', alpha=0.6)
axes[0].set_title('目标: IMBR (原始)', fontsize=14, fontweight='bold')
axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)
axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
axes[0].invert_yaxis()  # 翻转y轴，符合图像坐标

axes[1].scatter(generated[:,0], generated[:,1], s=8, c='green', alpha=0.6)
axes[1].set_title('生成: FM(σ=0.3)', fontsize=14, fontweight='bold')
axes[1].set_xlim(-1.5, 1.5); axes[1].set_ylim(-1.5, 1.5)
axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)
axes[1].invert_yaxis()

plt.suptitle('Flow Matching: 生成 IMBR 对比', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'imbr_comparison_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n保存: imbr_comparison_scatter.png")

# ============================================================================
# 可视化 2: 把生成的点云画成图像（类似字体渲染）
# ============================================================================
def points_to_image(points, img_w, img_h, title, save_path):
    """将归一化坐标的点云绘制为类似字体的二值图像"""
    # 从 [-1,1] 映射回像素坐标
    x_pixels = ((points[:,0] + 1) / 2 * img_w).astype(int)
    y_pixels = ((points[:,1] + 1) / 2 * img_h).astype(int)
    
    # 创建空白图像
    img = Image.new('L', (img_w, img_h), 0)
    # 画点（用小圆点）
    for x, y in zip(x_pixels, y_pixels):
        if 0 <= x < img_w and 0 <= y < img_h:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= x+dx < img_w and 0 <= y+dy < img_h:
                        img.putpixel((x+dx, y+dy), 255)
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(np.array(img), cmap='gray')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return img

# 生成的图像
gen_img = points_to_image(generated, img_w, img_h, '生成的 IMBR', 
                          os.path.join(PLOTS_DIR, 'imbr_generated_image.png'))
# 目标的图像
tgt_img = points_to_image(target, img_w, img_h, '目标 IMBR', 
                          os.path.join(PLOTS_DIR, 'imbr_target_image.png'))

print("保存: imbr_generated_image.png")
print("保存: imbr_target_image.png")

# ============================================================================
# 可视化 3: 并排对比图
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

axes[0].imshow(np.array(tgt_img), cmap='gray')
axes[0].set_title('目标 IMBR', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(np.array(gen_img), cmap='gray')
axes[1].set_title('生成的 IMBR', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.suptitle('IMBR 生成效果对比', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'imbr_side_by_side.png'), dpi=150, bbox_inches='tight')
plt.close()
print("保存: imbr_side_by_side.png")

# ============================================================================
# 定量评估
# ============================================================================
# 计算 IoU (Intersection over Union)
tgt_arr = np.array(tgt_img).astype(bool)
gen_arr = np.array(gen_img).astype(bool)
intersection = (tgt_arr & gen_arr).sum()
union = (tgt_arr | gen_arr).sum()
iou = intersection / union if union > 0 else 0

# 计算目标覆盖率
tgt_coverage = (tgt_arr & gen_arr).sum() / tgt_arr.sum() if tgt_arr.sum() > 0 else 0

print(f"\n{'='*50}")
print(f"定量评估")
print(f"{'='*50}")
print(f"IoU (交并比): {iou:.2%}")
print(f"目标覆盖率: {tgt_coverage:.2%}")
print(f"{'='*50}")

# 保存评估结果
result = {
    'iou': float(iou),
    'target_coverage': float(tgt_coverage),
    'generated_points': len(generated),
    'target_points': len(target),
    'model': 'model_rpot_fm.pth'
}
with open(os.path.join(CURRENT_DIR, 'imbr_eval_result.json'), 'w') as f:
    json.dump(result, f, indent=2)

print("\n✅ 测试完成！")