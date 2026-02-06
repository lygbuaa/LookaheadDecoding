#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# RTX-A4000
data1x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
data1y = [0.0127, 0.0133, 0.0133, 0.0131, 0.0133, 0.013, 0.0138, 0.0133, 0.0167, 0.0302, 0.0547]

# i7-11700K
data2x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
data2y = [0.0758, 0.0956, 0.1159, 0.187, 0.5873, 0.8814, 1.4508, 2.5646, 4.7655, 8.5757, 16.0264]

# J6
data3x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
data3y = [0.0594, 0.0381, 0.0391, 0.0398, 0.0395, 0.0458, 0.0519, 0.0733, 0.1217, 0.2635, 0.7447]

# J5
data4x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
data4y = [0.0555, 0.0344, 0.0345, 0.0353, 0.0371, 0.0463, 0.058, 0.1046, 0.192, 0.448, 1.0822]

# 创建2x2的子图
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# fig.subplots_adjust(top=0.95) 

# 展平axes以便于索引
axes = axes.flatten()

# 数据列表
all_x = [data1x, data2x, data3x, data4x]
all_y = [data1y, data2y, data3y, data4y]
titles = ['Llama3.2-1B-BF16@RTX-A4000, 19.2 TFLOPs, 19.2 TFLOPs', 'Llama3.2-1B-BF16@i7-11700K, 0.3 TFLOPs, 102.4 GB/s', 'Llama3.2-1B-W8A16@J6, 128 TOPS, 204.8 GB/s', 'Llama3.2-1B-W8A16@J5, 48 TOPS, 102.4 GB/s']

# 绘制每个子图
for i, (ax, x, y, title) in enumerate(zip(axes, all_x, all_y, titles)):
    # 绘制折线图
    ax.plot(x, y, 'b-o', linewidth=2, markersize=8, label=f'Series {i+1}')
    
    # 设置x轴为对数坐标
    ax.set_xscale('log')
    
    # 设置y轴为线性坐标（默认就是线性）
    ax.set_yscale('linear')
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('seq len', fontsize=12)
    ax.set_ylabel('infer time (sec)', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels([str(val) for val in x])
    
    # 设置y轴范围，留一些边距
    if i == 0:
        y_min = 0
        y_max = 0.1
    elif i == 1:
        y_min = 0
        y_max = 20
    else:
        y_min = 0
        y_max = 1

    # y_min, y_max = min(y), max(y)
    # y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
    y_margin = 0
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlim(1, 1024)
    
    # 添加图例
    # ax.legend(loc='best')

# 调整子图之间的间距
plt.tight_layout()

# 添加总标题
# fig.suptitle('Llama-3.2-1B-Instruct @4x Platform', fontsize=16, fontweight='bold', y=0.98)

# 显示图形
plt.show()