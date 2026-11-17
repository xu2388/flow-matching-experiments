# 流匹配（Flow Matching）系列生成实验

围绕流匹配与扩散模型在 MNIST 及图像上的一组生成实验，每个脚本对应一种方法。

## 包含的实验

| 脚本 | 方法 |
|------|------|
| `train_cfm_mnist.py` / `train_cfm_official.py` | Conditional Flow Matching 生成 MNIST 数字 |
| `train_mnist_fm.py` / `train_mnist_fm_v2.py` / `train_fm_v2.py` | 基础流匹配 MNIST |
| `run_rectified_flow.py` | Rectified Flow |
| `run_minibatch_ot_fm.py` | Mini-batch OT 流匹配 |
| `run_rpot_fm.py` | RPOT（递归最优传输） |
| `run_generative_flow_matching.py` | 生成式流匹配 |
| `train_imbr_image_fm.py` / `train_cfm_imbr.py` / `run_ddpm_imbr.py` | IMBR 相关实验 |
| `eval_cfm_official.py` / `eval_mnist_v2.py` | 评测脚本 |

## 文档

1. `CFM_MNIST复现说明.md`：CFM 复现细节（基于开源教程 Pytorch-Conditional-Flow-Matching-Tutorial）
2. `实验问题记录.md`：实验中遇到的问题与解决
3. `生成IMBR实验手册.md`

## 怎么跑

```bash
pip install torch torchvision matplotlib
python train_cfm_mnist.py
```

## 说明

`checkpoints/`（模型权重）、`mnist_data/`、`cfm_datasets/` 与上游教程代码 `flow-matching-mnist-main/` 未随仓库上传；`generated_*` 目录保留了生成结果图。
