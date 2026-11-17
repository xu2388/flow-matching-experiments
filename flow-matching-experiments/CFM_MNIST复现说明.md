# CFM MNIST 手写数字生成 · 复现说明

> 使用 **Conditional Flow Matching (CFM)** 在 MNIST 上训练生成模型，最终实现"从高斯噪声逐步演化出清晰数字"。
> 本项目基于开源教程 `Pytorch-Conditional-Flow-Matching-Tutorial` 复现，并针对单卡 8G 显存（RTX 4060 Laptop）做了训练策略优化。

---

## 1. 项目来源

- **开源教程仓库**：`Pytorch-Conditional-Flow-Matching-Tutorial`（作者 Minsu Kang）
  - 本机位置：`d:\大三上的资料\假期学习记录\Pytorch-Conditional-Flow-Matching-Tutorial\`
  - 参考文件：`01_conditional-flow-matching.ipynb`
- **原始配置**：`hidden=256 × 8 层`，`sigma_min=0`，`n_epochs=200`，`lr=5e-5`

---

## 2. 模型架构

模型为**时间条件卷积残差网络**（无下采样的全卷积网络，官方 notebook 原样，参数量 **4,805,633 / 4.8M**）：

```
输入 x:  [B, 1, 28, 28]       时间 t: [B, 1, 1, 1]
   │                                │
   ▼                                ▼
in_project (Conv 1→256, k=7)   time_project (Conv 1×1 ×2, 256 通道)
   │                                │
   └──────────────┬─────────────────┘
                  ▼
   8 × ConvBlock(残差):
   │   kernel=3, dilation=1,1,3,3,9,9,27,27
   │   SiLU 激活 + GroupNorm(8 组)
   │   加法注入时间嵌入: x = x + time_embedding
   ▼
out_project (Conv 256→1, k=3)
   │
   ▼
输出 v_t: [B, 1, 28, 28]   ← 预测速度场
```

**关键设计**：
| 组件 | 说明 |
|------|------|
| 时间条件 | `t` 通过 1×1 卷积投影为 256 通道嵌入，以**加法残差**方式注入每层 |
| 空洞卷积 | dilation 按 `3^((layer-1)//2)` 递增（1,1,3,3,9,9,27,27），逐层扩大感受野 |
| 归一化 | GroupNorm(groups=8)，适配小 batch |
| 激活 | SiLU（Swish） |
| 插值路径 | 直线调度：`x_t = (1-t)·x_0 + t·x_1`，`sigma_min=0` |
| 速度场 | `v = x_1 - x_0`（CFM 回归目标） |
| 采样 | **Euler 积分**：从高斯噪声 `x_0` 出发，T=100 步沿预测速度场积分 |

---

## 3. 关键文件与文件夹

| 路径 | 说明 |
|------|------|
| `流匹配\train_cfm_official.py` | **训练脚本**（含断点续训、最优 Top-5、EMA、微调） |
| `流匹配\eval_cfm_official.py` | 评估脚本：生成 64 → 分类器打分 → 选 Top-16 → 与 GT 对比 |
| `流匹配\model_cfm_official.pth` | **最终模型**（最优权重，18.7MB） |
| `流匹配\checkpoints\official_best_*.pt` | 验证 loss 最优的 Top-5 权重 |
| `流匹配\checkpoints\official_resume.pt` | 续训断点（模型+优化器+EMA+历史，覆盖保存） |
| `流匹配\plots\cfm_official_gen_epoch_*.png` | 每 5 轮采样图（全程 5→185） |
| `流匹配\plots\cfm_official_top16_compare.png` | 最优 16 张生成 vs GT 对比图 |
| `流匹配\plots\cfm_official_loss.png` | 训练+验证 loss 曲线 |
| `流匹配\cfm_datasets\` | MNIST 数据集缓存 |
| `流匹配\generated_cfm\` | 单张样本输出 |
| `Pytorch-Conditional-Flow-Matching-Tutorial\` | 官方 notebook 原始工程 |

---

## 4. 训练配置与策略

### 4.1 基础配置（官方原值）
```
hidden_dim=256, n_layers=8   → 4.8M 参数
sigma_min=0, n_epochs=200
首次训练 lr=5e-5, batch=128
单卡 RTX 4060 (8G), cuda:0, seed=1234
每 5 epochs: 验证 loss + 采样图 + 最优模型 Top-5 + 续训断点
```

### 4.2 本次采用的优化策略
| 策略 | 内容 | 作用 |
|------|------|------|
| 断点续训 | checkpoint 保存模型+优化器+EMA+epoch 历史 | 中断可无缝恢复 |
| 最优 Top-5 | 按验证 loss 保留前 5 个权重，自动清理 | 防磁盘膨胀 + 留多档 |
| **从最优权重微调** | 从验证 loss 最优的 epoch 135 启动，再训 50 轮 | 避免重头再跑 |
| 微调学习率 | `lr=2e-5` + CosineAnnealing(T_max=50, eta_min=0) | 精细收敛，避免震荡 |
| **EMA** | decay=0.999，验证/保存/采样均用 EMA 权重 | 生成更平滑、更稳定 |

### 4.3 踩坑记录（EMA 初始化 Bug）
> `ema = EMA(model)` 若写在 `load_state_dict` **之前**，影子权重会从随机初始化开始，
> 导致验证 loss 暴涨（0.085 → 1.38）、采样图变成噪声。
> **修复**：EMA 初始化必须放在权重加载完成之后；并移除断点中已污染的 ema_shadow。

---

## 5. 复现步骤

```powershell
cd "d:\大三上的资料\假期学习记录\流匹配"

# 1) 从头训练官方模型（200 epochs，约 9 小时；可 Ctrl+C 中断后重跑自动续训）
D:\anaconda3\python.exe train_cfm_official.py

# 2) 评估：生成 64 张 → 分类器选 Top-16 → 与 GT 拼图对比
D:\anaconda3\python.exe eval_cfm_official.py
```

脚本行为说明：
- **首次运行**：从头训练；每 5 epochs 保存断点 + 采样图 + 最优 Top-5。
- **再次运行**：若 `RESUME_FROM_BEST=True`，自动从验证 loss 最优权重启动微调（本次实现：epoch 135 → 185，50 轮）。
- 若 `RESUME_FROM_BEST=False`：从最近断点 `official_resume.pt` 恢复续训。
- 训练完成后自动生成：最终模型、loss 曲线、64 张生成/真实对照、T=2/5/10/25/50 步数对比。

---

## 6. 实验结果

| 阶段 | 模型 | 训练量 | 结果 |
|------|------|--------|------|
| 失败案例 | 128×4（0.6M） | 70 epochs / 34 min | loss 0.241→0.104，但生成数字模糊不成形 |
| **最终成果** | **官方 256×8（4.8M）** | 从头 135 epochs + 微调 50 | 最优 val_loss **0.08502** |
| 质量验证 | Top-16 生成样本 | 分类置信度 | **全部 ≥0.996**，笔画清晰 |

- **Loss 曲线**：`流匹配\plots\cfm_official_loss.png`
- **对比图**：`流匹配\plots\cfm_official_top16_compare.png`（上排生成 / 下排 GT 一一对应）
- **结论**：官方容量模型 + EMA + 最优权重微调 + cosine 调度后，CFM 生成质量已接近真实 MNIST。
