"""
MNIST Flow Matching v2 生成质量评估
定量指标：
  1. 生成图像与真实 MNIST 的分布对比（均值/方差）
  2. 像素激活率（判断是否生成的是"空白"或"纯噪声"）
  3. 数字清晰度（白色像素占比）
  4. 与 v1 的对比
"""
import os, sys, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mnist_fm_v2 import UNet, FMVelocity, generate

device = 'cuda'
print(f"设备: {device}")

# 1. 加载 v2 模型
unet = UNet(in_channels=1, out_channels=1, c=64).to(device)
ckpt = torch.load('model_mnist_fm_v2.pth', map_location=device, weights_only=True)
if 'model_state_dict' in ckpt:
    unet.load_state_dict(ckpt['model_state_dict'])
else:
    unet.load_state_dict(ckpt)
print("✅ v2 模型加载成功")

# 2. 生成 200 张
torch.manual_seed(0)
np.random.seed(0)
gen_imgs = generate(unet, n_samples=200, num_steps=50, device=device).cpu().numpy()
gen_imgs = gen_imgs.squeeze(1)  # (200, 28, 28)

# 3. 加载真实 MNIST 对比
transform = transforms.Compose([transforms.ToTensor()])
test_ds = datasets.MNIST('mnist_data', train=False, download=False, transform=transform)
real_imgs = torch.stack([test_ds[i][0] for i in range(200)]).numpy().squeeze(1)

print(f"\n=== 分布对比 ===")
print(f"真实: 均值={real_imgs.mean():.4f}, 方差={real_imgs.var():.4f}")
print(f"生成: 均值={gen_imgs.mean():.4f}, 方差={gen_imgs.var():.4f}")

# 4. 像素激活率（>0.5 视为前景）
real_active = (real_imgs > 0.5).mean()
gen_active = (gen_imgs > 0.5).mean()
print(f"\n=== 前景像素比例 ===")
print(f"真实: {real_active*100:.2f}%")
print(f"生成: {gen_active*100:.2f}%")

# 5. 空白图像比例（激活率 < 1% 视为空白/失败）
real_blank = ((real_imgs > 0.5).sum(axis=(1,2)) < 8).mean()
gen_blank = ((gen_imgs > 0.5).sum(axis=(1,2)) < 8).mean()
print(f"\n=== 空白图像比例 ===")
print(f"真实: {real_blank*100:.2f}%")
print(f"生成: {gen_blank*100:.2f}%")

# 6. 中心度（数字是否居中）
def center_ratio(imgs):
    # 中心 14x14 区域的像素占比
    center = imgs[:, 7:21, 7:21]
    total = (imgs > 0.5).sum(axis=(1,2)) + 1e-6
    c = (center > 0.5).sum(axis=(1,2))
    return (c/total).mean()

real_cr = center_ratio(real_imgs)
gen_cr = center_ratio(gen_imgs)
print(f"\n=== 中心度（数字是否居中）===")
print(f"真实: {real_cr:.4f}")
print(f"生成: {gen_cr:.4f}")

# 7. 可视化 20 张
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(real_imgs[i], cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(gen_imgs[i], cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('真实', fontsize=12)
axes[1, 0].set_ylabel('生成', fontsize=12)
plt.suptitle('MNIST v2: 真实 vs 生成', fontsize=14)
plt.tight_layout()
plt.savefig('plots/mnist_v2_eval_compare.png', dpi=150)
plt.close()
print("\n✅ 对比图: plots/mnist_v2_eval_compare.png")

# 8. 保存评估结果
result = {
    'real_mean': float(real_imgs.mean()),
    'gen_mean': float(gen_imgs.mean()),
    'real_var': float(real_imgs.var()),
    'gen_var': float(gen_imgs.var()),
    'real_active_pct': float(real_active*100),
    'gen_active_pct': float(gen_active*100),
    'real_blank_pct': float(real_blank*100),
    'gen_blank_pct': float(gen_blank*100),
    'real_center': float(real_cr),
    'gen_center': float(gen_cr),
}
with open('mnist_v2_eval.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("✅ 评估结果: mnist_v2_eval.json")
