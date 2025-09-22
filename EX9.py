# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 21:05:04 2025

@author: Acer
"""

import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD

# === 中文 ===
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

"""
import pydmd
print(f"当前 PyDMD 版本: {pydmd.__version__}")
"""
def exercises():
    print("=== bzhan666 ===\n")
    
    # 复用之前的数据
    n_space = 128
    n_time = 100
    x_coords = np.linspace(-10, 10, n_space)
    t_coords = np.linspace(0, 8*np.pi, n_time)

    
    mode1_spatial = np.sin(x_coords)
    mode2_spatial = np.cos(3 * x_coords)
    dynamics1 = np.exp(1j * 1.0 * t_coords) 
    dynamics2 = np.exp(1j * 0.5 * t_coords) * np.exp(-0.2 * t_coords)
    data = (np.outer(mode1_spatial, dynamics1) + np.outer(mode2_spatial, dynamics2)).real
    
    # 添加一些噪声
    noise_level = 0.1
    noisy_data = data + noise_level * np.random.randn(*data.shape)
    
    # 1. 使用 PyDMD 的标准 DMD（兼容所有版本）
    dmd = DMD(svd_rank=5)
    dmd.fit(noisy_data)
    

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 提取前2个空间模态（实部）
    for i in range(2):
        mode = dmd.modes[:, i].real  # 取实部，因为原始数据是实数
        axes[0].plot(x_coords, mode, label=f'Mode {i+1}')
    axes[0].set_xlabel('空间坐标 x')
    axes[0].set_ylabel('模态幅值')
    axes[0].set_title('DMD 空间模态（前2个主导模态）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 提取前2个时间演化（实部）
    for i in range(2):
        dyn = dmd.dynamics[i, :].real  # 时间动态
        axes[1].plot(t_coords, dyn, label=f'Mode {i+1}')
    axes[1].set_xlabel('时间 t')
    axes[1].set_ylabel('模态动态幅值')
    axes[1].set_title('DMD 时间演化（前2个主导模态）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 获取特征值（复数）
    eigs = dmd.eigs
    
    # 绘制单位圆
    theta = np.linspace(0, 2*np.pi, 200)
    unit_circle = np.exp(1j * theta)
    ax.plot(unit_circle.real, unit_circle.imag, 'k--', linewidth=1.5, label='单位圆 |λ|=1')
    
    # 绘制特征值点
    ax.scatter(eigs.real, eigs.imag, c='red', s=80, edgecolors='black', linewidth=1.2, label='DMD特征值', zorder=5)
    
    # 绘制坐标轴
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8)
    
    # 设置范围和比例
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal', adjustable='box')
    
    # 标题和标签
    ax.set_title('DMD 动力学指纹图：复平面特征值分布', fontsize=14, fontweight='bold')
    ax.set_xlabel('实部 (增长/衰减率)', fontsize=12)
    ax.set_ylabel('虚部 (振荡频率)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加解读文字
    fig.text(0.02, 0.02,
        "💡 解读：\n"
        "• 单位圆上的点 → 稳定周期运动\n"
        "• 圆内点 → 衰减模态\n"
        "• 圆外点 → 不稳定增长\n"
        "• 靠近实轴 → 低频或非振荡\n"
        "• 远离实轴 → 高频振荡",
        fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    

    reconstructed_dmd = dmd.reconstructed_data
    
    error_dmd = np.linalg.norm(data - reconstructed_dmd) / np.linalg.norm(data)
    
    print(f"带噪声数据下, 标准DMD重构与'干净'数据的误差: {error_dmd:.4%}")
    
    print("\nPyDMD 极大简化了DMD的应用流程，并提供了更鲁棒的算法。")

if __name__ == "__main__":
    exercises()