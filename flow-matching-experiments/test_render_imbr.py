# -*- coding: utf-8 -*-
"""快速测试 IMBR 渲染: 画 8 张样例确认字母可辨认"""
import os
import numpy as np
import torch
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont, ImageFilter

IMG_H, IMG_W = 32, 128
FONT_PATH = r"C:\Windows\Fonts\arialbd.ttf"


def render_imbr(rng):
    canvas = Image.new('L', (IMG_W, IMG_H), 0)
    font_size = int(rng.randint(22, 28))
    font = ImageFont.truetype(FONT_PATH, font_size)
    for i, ch in enumerate('IMBR'):
        cell = Image.new('L', (32, 32), 0)
        d = ImageDraw.Draw(cell)
        bbox = d.textbbox((0, 0), ch, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((32 - tw) / 2 - bbox[0], (32 - th) / 2 - bbox[1]), ch, fill=255, font=font)
        ang = rng.uniform(-10, 10)
        cell = cell.rotate(ang, resample=Image.BILINEAR, fillcolor=0)
        dx, dy = int(rng.randint(-2, 3)), int(rng.randint(-1, 2))
        canvas.paste(cell, (i * 32 + dx, dy), cell)
    if rng.rand() < 0.8:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    bright = rng.uniform(0.75, 1.0)
    canvas = canvas.point(lambda v: int(v * bright))
    return canvas


rng = np.random.RandomState(42)
imgs = torch.stack([torch.from_numpy(np.asarray(render_imbr(rng), dtype=np.float32) / 255.0).unsqueeze(0)
                    for _ in range(8)])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'imbr_render_test.png')
save_image(imgs, out, nrow=4, padding=2)
print(f"saved: {out}")
