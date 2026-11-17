好的，这是一份为你量身定制的 **Flow Matching 手把手实验教程**。

这份文档按照你认可的“三步走”战略设计，从极简的“圆环变月牙”开始，逐步进阶到“十字变字母”。我加入了大量的**原理回顾**、**代码注释**和**避坑指南**，确保你在动手的时候，能清楚地知道每一行代码对应着PPT里的哪个数学概念。

请将此内容保存为 `FlowMatching_Lab_Manual.md`，它将是你实验路上的地图和伙伴。

---

# Flow Matching 手把手实验教程：从点云变形到最优传输 (OT-CFM)

**实验目标**： 通过二维点云变形任务，彻底理解 Flow Matching 的核心思想——**“直接回归条件速度场”**。
**适用场景**： 学习理解 Flow Matching / 扩散模型底层逻辑。
**硬件要求**： CPU 即可（如使用 GPU 会更快）。

---

## 第一部分：实验哲学与核心公式回顾

在做实验之前，请在心里默念 Flow Matching 最核心的“交易”：

1.  **我们有什么？** 一堆来自分布 $p_{data}$ 的样本点（比如字母上的坐标）。
2.  **我们不知道什么？** $p_{data}$ 的数学公式。
3.  **Flow Matching 的骚操作**： 我们不去求公式，而是去学一个 **“速度场” (Vector Field, $u_t$)**。
4.  **怎么学？** 给定起点 $x_0$（噪声/笔画）和终点 $x_1$（目标字母），我们定义一条直线路径 $x_t = (1-t)x_0 + t x_1$。这条路径的速度是恒定的：$\frac{d}{dt}x_t = x_1 - x_0$。我们只要训练神经网络 $v_\theta$ 去预测这个速度即可。**Loss = MSE($v_\theta(x_t, t), x_1 - x_0$)**。

---

## 第二部分：准备工作 (Setup & Utils)

首先，我们需要生成实验用的“玩具数据”。我们将使用 `torch` 和 `matplotlib`。

### 1. 导入库与环境配置

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import linear_sum_assignment  # 用于OT配对（第三步用）
from sklearn.datasets import make_moons
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，保证实验结果可复现
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")
```

### 2. 核心工具：点云数据集 (PointCloudDataset)

我们要把“图像”变成“点云”。这里我封装了一个类，它接收起点坐标集和终点坐标集，并在训练时随机配对（或使用OT配对）。

```python
class PointCloudDataset(Dataset):
    def __init__(self, source_points, target_points, mode='random'):
        """
        source_points: numpy array (N, 2) - 初始分布的点（如高斯噪声）
        target_points: numpy array (M, 2) - 目标分布的点（如字母）
        mode: 'random' 或 'ot' (Optimal Transport)
        """
        self.source = torch.tensor(source_points, dtype=torch.float32)
        self.target = torch.tensor(target_points, dtype=torch.float32)
        self.mode = mode
        
        # 预处理配对 (如果是ot模式，预先算好配对矩阵)
        if mode == 'ot':
            self.paired_target = self._optimal_transport_pairing()
        else:
            # 随机模式不需要预先配对，在 __getitem__ 中随机采样
            pass

    def _optimal_transport_pairing(self):
        """使用匈牙利算法求解最优传输配对，让点走的路程总和最短"""
        # 计算代价矩阵 (欧氏距离)
        cost = torch.cdist(self.source, self.target).numpy()  # (N, M)
        row_ind, col_ind = linear_sum_assignment(cost)
        # 返回配对好的目标点 (保证 source[i] -> target[col_ind[i]])
        return self.target[col_ind]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        x0 = self.source[idx]  # 起点
        
        if self.mode == 'ot':
            x1 = self.paired_target[idx]  # 固定的最优配对终点
        else:
            # 随机配对：为当前起点随机挑选一个目标点
            rand_idx = torch.randint(0, len(self.target), (1,)).item()
            x1 = self.target[rand_idx]
            
        return x0, x1
```

### 3. 数据生成器：画出我们的“玩具”

这里定义了我们实验中用到的所有形状：月牙、圆环、十字、字母。

```python
def generate_crescent(n_samples=1000):
    """生成月牙形数据 (类似 make_moons)"""
    X, _ = make_moons(n_samples=n_samples, noise=0.05)
    return X.astype(np.float32)

def generate_circle(n_samples=1000, radius=1.0):
    """生成圆环/高斯球数据"""
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    r = radius * np.sqrt(np.random.rand(n_samples))  # 均匀分布在圆内
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.stack([x, y], axis=1).astype(np.float32)

def generate_cross(n_samples_per_arm=100):
    """生成十字笔画点云 (横线+竖线)"""
    points = []
    # 横线: y 在 -0.1 到 0.1 之间
    x_h = np.linspace(-1, 1, n_samples_per_arm)
    y_h = np.random.uniform(-0.1, 0.1, n_samples_per_arm)
    # 竖线: x 在 -0.1 到 0.1 之间
    y_v = np.linspace(-1, 1, n_samples_per_arm)
    x_v = np.random.uniform(-0.1, 0.1, n_samples_per_arm)
    
    points.extend(zip(x_h, y_h))
    points.extend(zip(x_v, y_v))
    return np.array(points, dtype=np.float32)

def generate_letter(letter='C', n_samples=500, canvas_size=32):
    """
    使用 PIL 绘制字母并提取点云
    """
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    # 创建画布
    img = Image.new('L', (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)
    # 加载默认字体，这里使用系统字体，如果报错请改为 'Arial.ttf' 或 'DejaVuSans.ttf'
    try:
        font = ImageFont.truetype("arial.ttf", int(canvas_size * 0.8))
    except:
        font = ImageFont.load_default()
    
    draw.text((canvas_size//4, 0), letter, fill=255, font=font)
    img = np.array(img)
    
    # 提取坐标 (非零像素)
    coords = np.argwhere(img > 0)  # (y, x)
    # 归一化到 [-1, 1] 并转换顺序为 (x, y)
    coords = coords[:, [1, 0]]  # 换成 (x, y)
    coords = (coords / canvas_size) * 2 - 1  # 归一化到 -1 ~ 1
    
    # 如果点太多，随机采样
    if len(coords) > n_samples:
        idx = np.random.choice(len(coords), n_samples, replace=False)
        coords = coords[idx]
    elif len(coords) < n_samples:
        # 如果点太少，重复采样补齐
        idx = np.random.choice(len(coords), n_samples, replace=True)
        coords = coords[idx]
        
    return coords.astype(np.float32)

# 快速可视化测试 (可以运行一下看看效果)
# plt.scatter(*generate_crescent().T); plt.show()
```

---

## 第三部分：核心模型 (Vector Field Network)

在 Flow Matching 中，我们的神经网络 $v_\theta(x, t)$ 输入是**坐标** $x$ 和**时间** $t$，输出是**速度** $v$。

重点在于时间 $t$ 的编码：我们不能直接把一个标量 $t$ 丢进网络，需要用 **正弦位置编码 (Positional Encoding)**，这在 Transformer 和扩散模型中都是标配，能帮助网络感知时间的细微变化。

```python
class TimeEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        # t: (batch, 1)
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
        
        # 输入: x(2) + time(64) = 66
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2)  # 输出二维速度 (vx, vy)
        )
        
    def forward(self, x, t):
        # x: (batch, 2), t: (batch, 1) 范围 0~1
        t_emb = self.time_mlp(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)
```

---

## 第四部分：训练与采样工具箱

### 1. 欧拉采样器 (Euler Sampler)
这对应我们之前学的 PPT：$X_{t+h} = X_t + h \cdot u_t(X_t)$。

```python
@torch.no_grad()
def sample_euler(model, x_init, n_steps=100, step_size=None):
    """
    从初始点 x_init 开始，利用欧拉方法推演 t=0 到 t=1
    """
    model.eval()
    x = x_init.clone().to(device)
    if step_size is None:
        step_size = 1.0 / n_steps
    
    trajectory = [x.cpu().numpy()]
    for i in range(n_steps):
        t = torch.ones(x.shape[0], 1).to(device) * (i / n_steps)
        v = model(x, t)
        x = x + step_size * v
        if (i + 1) % (n_steps // 5) == 0:  # 记录中间过程便于做动画
            trajectory.append(x.cpu().numpy())
    trajectory.append(x.cpu().numpy())
    return trajectory
```

### 2. 训练循环 (Training Loop)

这是 Flow Matching 最精髓的部分：**Simulation-free 训练**。我们不需要在训练时模拟 ODE，只需要随机采两个点 $x_0, x_1$ 和随机时间 $t$，计算直线速度目标 $(x_1 - x_0)$，然后让网络去回归它。

```python
def train_flow_matching(model, dataloader, epochs=500, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x0, x1 in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            batch_size = x0.shape[0]
            
            # 1. 随机采样时间 t ~ Uniform(0, 1)
            t = torch.rand(batch_size, 1).to(device)
            
            # 2. 线性插值路径: x_t = (1-t)*x0 + t*x1
            x_t = (1 - t) * x0 + t * x1
            
            # 3. 真实速度 (直线路径的导数)
            target_v = x1 - x0
            
            # 4. 网络预测速度
            pred_v = model(x_t, t)
            
            # 5. 计算 MSE Loss
            loss = nn.functional.mse_loss(pred_v, target_v)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
            loss_list.append(epoch_loss/len(dataloader))
            
    return loss_list
```

---

## 第五部分：三步走实验实操

### 实验一：初窥门径 (圆环 → 月牙形)

**实验目的**：验证 Flow Matching 能否将对称分布扭曲成非对称分布。这是最标准的 Toy Test。

```python
# 1. 准备数据
n_points = 2000
source_data = generate_circle(n_points, radius=0.8)  # 圆形
target_data = generate_crescent(n_points)            # 月牙

# 2. 加载数据集 (这里用随机配对即可)
dataset = PointCloudDataset(source_data, target_data, mode='random')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 3. 初始化模型
model = VectorFieldMLP().to(device)

# 4. 训练
print("开始训练 圆环 -> 月牙...")
train_flow_matching(model, dataloader, epochs=300)

# 5. 推理采样
x_init = torch.tensor(source_data[:500], dtype=torch.float32).to(device)  # 取500个点
traj = sample_euler(model, x_init, n_steps=100)

# 6. 可视化 (会在后面统一画出)
print("实验一完成！")
```

**预期结果**： 你将看到圆形的点云在 $t=0.5$ 时被拉伸，最终在 $t=1$ 时完全贴合月牙形状，且点云的密度会随月牙的曲率变化。

---

### 实验二：结构变形 (十字笔画 → 字母 'C')

**实验目的**：观察离散的、分离的笔画（十字）如何“融化”并重组成连续的曲线字母。

```python
# 1. 准备数据
source_cross = generate_cross(n_samples_per_arm=300)  # 约600个点
target_C = generate_letter('C', n_samples=600)

# 2. 加载数据集 (推荐用随机配对，看看乱配能跑到什么效果)
dataset = PointCloudDataset(source_cross, target_C, mode='random')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 3. 重新初始化模型 (防止污染)
model = VectorFieldMLP().to(device)

# 4. 训练
print("开始训练 十字 -> 字母C (随机配对)...")
train_flow_matching(model, dataloader, epochs=500)

# 5. 采样
x_init = torch.tensor(source_cross, dtype=torch.float32).to(device)
traj = sample_euler(model, x_init, n_steps=150)

print("实验二完成！")
```

**预期结果**： 随机配对下，你会发现十字的点确实覆盖了 C 的区域，但看起来有点“散”或“模糊”，像是一个毛茸茸的 C。这是因为横线上的某个点可能被随机配给了 C 的底部，产生了交叉的“拉力”。

---

### 实验三：终极进阶 (十字 → 'IMBR' 与 OT 配对)

**实验目的**：引入 **OT-CFM (最优传输)**，让配对变得有“逻辑”，从而实现清晰、笔直的变形轨迹，完美反映数据的几何结构。

```python
# 1. 准备数据
# 为了看清字母，我们把目标点增多
source_cross = generate_cross(n_samples_per_arm=400)  # 800点
target_imbr = generate_letter('IMBR', n_samples=800, canvas_size=64)  # 注意这里生成IMBR

# 2. 关键一步：使用 OT 配对模式！
dataset_ot = PointCloudDataset(source_cross, target_imbr, mode='ot')
dataloader_ot = DataLoader(dataset_ot, batch_size=256, shuffle=True)

# 3. 重新初始化模型
model_ot = VectorFieldMLP().to(device)

# 4. 训练
print("开始训练 十字 -> IMBR (Optimal Transport 配对)...")
train_flow_matching(model_ot, dataloader_ot, epochs=500)

# 5. 采样
x_init = torch.tensor(source_cross, dtype=torch.float32).to(device)
traj_ot = sample_euler(model_ot, x_init, n_steps=150)

print("实验三完成！")
```

**预期结果（最惊艳的效果）**：
你会看到横线并没有散开，而是整整齐齐地**平移**过去组成了 "I" 和 "M" 的顶部，竖线则**滑移**过去组成了 "B" 和 "R" 的躯干。路径几乎笔直，没有交叉混乱，清晰度极高。

---

## 第六部分：总可视化代码

把三个实验的结果画在一起，直观对比。

```python
def plot_trajectory(trajectory, title, save_path=None):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    steps = [0, 1, 2, 3, 4]  # 取 0%, 25%, 50%, 75%, 100% 的时间点
    for i, ax in enumerate(axes):
        idx = min(i * (len(trajectory)-1) // 4, len(trajectory)-1)
        points = trajectory[idx]
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.6)
        ax.set_title(f"t={idx/(len(trajectory)-1):.1f}")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 假设你保存了 trajectory_1, trajectory_2, trajectory_ot
# plot_trajectory(traj, "实验一：圆环 -> 月牙")
# plot_trajectory(traj_ot, "实验三：OT-CFM 十字 -> IMBR (笔直变形)")
```

---

## 第七部分：总结与思考题

完成实验后，请带着以下问题回顾代码：

1. **为什么训练时不模拟 ODE 也能收敛？** （答：因为回归目标 `x1-x0` 是解析的，我们逼迫网络在任意中间时刻 $t$ 都知道该往哪里加速，这叫“局部最优保证全局最优”。）
2. **为什么 OT 配对比随机配对好？** （答：随机配对让网络学到的是“期望速度”，有熵增；OT 配对让每个点都有明确的、最短路径的“唯一责任点”，极大减少了速度场的弯曲程度。）
3. **NFE (步数) 的影响**：试着把 `sample_euler` 的 `n_steps` 从 100 改成 10，看看效果是否依然清晰？这直接反映了 Flow Matching 支持**少步数快速生成**的优势。

祝实验顺利！如果在跑代码时遇到关于维度不匹配或者字体找不到的问题，随时告诉我，我来帮你修 bug。