# -*- coding: utf-8 -*-
"""加载已有的 64 生成 + 64 真实，拼成一张上下对比图"""
import os
from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots')

gen = Image.open(os.path.join(PLOTS_DIR, 'cfm_official_generated_64.png'))
real = Image.open(os.path.join(PLOTS_DIR, 'cfm_official_real_64.png'))

# 确保尺寸一致
w, h = gen.size
real = real.resize((w, h))

# 上下拼接（留白给标签）
label_h = 36
combined = Image.new('RGB', (w, h * 2 + label_h * 2 + 10), 'white')
draw = ImageDraw.Draw(combined)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# 上半：生成，下半：真实
combined.paste(gen, (0, label_h))
combined.paste(real, (0, h + label_h + 10))

draw.text((w//2 - 120, 6), "Generated (64 samples, T=100)", fill=(40, 60, 120), font=font)
draw.text((w//2 - 120, h + label_h + 14), "Ground Truth (64 real MNIST)", fill=(40, 60, 120), font=font)

out = os.path.join(PLOTS_DIR, 'cfm_official_64_compare.png')
combined.save(out)
print(f"对比图已保存: {out}")