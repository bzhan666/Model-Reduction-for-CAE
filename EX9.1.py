# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:08:25 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD, OptDMD

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
    
    # 1. 使用 PyDMD 的标准 DMD
    dmd = DMD(svd_rank=5)
    dmd.fit(noisy_data)
    
    # 2. 使用 PyDMD 的优化 DMD (更适合含噪声数据)
    try:
        optdmd = OptDMD(svd_rank=5)
        optdmd.fit(noisy_data)
        has_optdmd = True
    except Exception as e:
        print(f"OptDMD 运行出错: {e}")
        has_optdmd = False
    
    # 可视化标准DMD的结果 - 使用自定义可视化替代原有方法
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制前两个DMD模态
    plt.subplot(221)
    for i in range(min(2, dmd.modes.shape[1])):
        plt.plot(x_coords, dmd.modes.real[:, i], label=f'Mode {i+1}')
    plt.title('DMD Modes')
    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    
    # 2. 绘制DMD特征值
    plt.subplot(222)
    # 在单位圆上绘制参考圆
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2)
    # 绘制特征值
    plt.scatter(dmd.eigs.real, dmd.eigs.imag, c='r', marker='o')
    plt.axis('equal')
    plt.title('DMD Eigenvalues')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.grid(True)
    
    # 3. 绘制原始数据与重构数据的比较（第一个时间步）
    plt.subplot(223)
    plt.plot(x_coords, noisy_data[:, 0], 'b-', label='Noisy Data')
    plt.plot(x_coords, dmd.reconstructed_data[:, 0], 'r--', label='DMD Reconstruction')
    plt.title('Data vs DMD Reconstruction (t=0)')
    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    
    # 4. 绘制原始数据与重构数据的比较（中间时间步）
    mid_t = n_time // 2
    plt.subplot(224)
    plt.plot(x_coords, noisy_data[:, mid_t], 'b-', label='Noisy Data')
    plt.plot(x_coords, dmd.reconstructed_data[:, mid_t], 'r--', label='DMD Reconstruction')
    plt.title(f'Data vs DMD Reconstruction (t={mid_t})')
    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 比较重构效果
    # 比较重构效果
    reconstructed_dmd = dmd.reconstructed_data
    error_dmd = np.linalg.norm(data - reconstructed_dmd) / np.linalg.norm(data)
    print(f"带噪声数据下, 标准DMD重构与'干净'数据的误差: {error_dmd:.4%}")
    
    if has_optdmd:
        try:
            reconstructed_optdmd = optdmd.reconstructed_data
            error_optdmd = np.linalg.norm(data - reconstructed_optdmd) / np.linalg.norm(data)
            print(f"带噪声数据下, 优化DMD重构与'干净'数据的误差: {error_optdmd:.4%}")
        except NotImplementedError:
            print("OptDMD 的 reconstructed_data 尚未实现，无法计算其重构误差。")
    
    print("\nPyDMD 极大简化了DMD的应用流程，并提供了更鲁棒的算法。")

if __name__ == "__main__":

    exercises()
