# -*- coding: utf-8 -*-
"""检查生成图多样性：把 64 张网格拆开，统计每张的差异度"""
import numpy as np
from PIL import Image

def grid_to_cells(path, nrow=8, pad=2):
    img = Image.open(path).convert('L')
    a = np.asarray(img, dtype=np.float32) / 255.0
    H, W = a.shape
    cell = (H - pad*(nrow-1)) // nrow
    cells = []
    for i in range(nrow):
        for j in range(nrow):
            y0 = i*(cell+pad); x0 = j*(cell+pad)
            cells.append(a[y0:y0+cell, x0:x0+cell])
    return np.stack(cells)

gen = grid_to_cells(r'd:\大三上的资料\假期学习记录\流匹配\plots\cfm_generated_64.png')
real = grid_to_cells(r'd:\大三上的资料\假期学习记录\流匹配\plots\cfm_real_64.png')

def diversity_report(cells, name):
    print(f"--- {name} ---")
    # 每个 cell 的亮度（非零像素比例）
    nz = (cells > 0.05).mean(axis=(1,2))
    print(f"  数字占比: 均值 {nz.mean()*100:.1f}% | 最小 {nz.min()*100:.1f}% | 最大 {nz.max()*100:.1f}%")
    # 每张图之间的成对相关性（L2 距离），越小越多样
    flat = cells.reshape(len(cells), -1)
    nrm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-6)
    sim = nrm @ nrm.T
    off = sim[~np.eye(len(cells), dtype=bool)]
    print(f"  成对余弦相似度: 均值 {off.mean():.3f} | 最大 {off.max():.3f} (>0.9 说明同质化严重)")
    # 空白 cell 数量
    print(f"  空白(占比<5%)数量: {(nz < 0.05).sum()}/{len(cells)}")
    print()

diversity_report(gen, 'CFM 生成 64 张')
diversity_report(real, 'CFM 真实 64 张')
