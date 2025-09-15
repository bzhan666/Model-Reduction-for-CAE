# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:09:28 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt

def perform_weighted_pod_demo():
    print("=== bzhan666 ===\n")
    
    # 1. 创建非均匀网格和权重矩阵
    # 假设一个1D问题，网格在中间密集，两边稀疏
    x_sparse1 = np.linspace(0, 0.4, 10)
    x_dense = np.linspace(0.41, 0.59, 40)
    x_sparse2 = np.linspace(0.6, 1.0, 10)
    x = np.concatenate([x_sparse1, x_dense, x_sparse2])
    n_space = len(x)
    
    # 计算每个点的"控制体积" (这里是长度)
    dx = np.diff(x)
    weights_diag = np.zeros(n_space)
    weights_diag[0] = dx[0]/2
    weights_diag[-1] = dx[-1]/2
    weights_diag[1:-1] = (dx[:-1] + dx[1:])/2
    W = np.diag(weights_diag)
    
    # 2. 生成模拟数据
    # 假设一个模态集中在网格密集区
    n_time = 50
    t = np.linspace(0, 2*np.pi, n_time)
    mode_spatial = np.exp(-((x - 0.5)**2) / 0.005) # 高斯包，集中在中间
    mode_temporal = np.sin(t)
    A = np.outer(mode_spatial, mode_temporal) + 0.05 * np.random.randn(n_space, n_time)
    
    # 3. 执行标准POD (不加权)
    U_unweighted, _, _ = np.linalg.svd(A, full_matrices=False)
    
    # 4. 执行加权POD
    # L = sqrt(W) 因为W是对角阵
    L = np.sqrt(W)
    L_inv = np.diag(1.0 / np.sqrt(weights_diag))
    A_tilde = L @ A
    U_tilde, _, _ = np.linalg.svd(A_tilde, full_matrices=False)
    # 将模态转换回物理空间
    U_weighted = L_inv @ U_tilde
    
    # 5. 可视化对比第一模态
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(x, A[:, 0], 'b-')
    axes[0].set_title('First data snapshot')
    axes[0].grid(True)
    
    # 用离散点表示网格密度
    axes[1].plot(x, U_unweighted[:, 0], 'r-', label='Standard POD')
    axes[1].plot(x, np.zeros_like(x), 'k|', markersize=10) # 显示网格点
    axes[1].set_title('Standard POD first mode')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(x, U_weighted[:, 0], 'g-', label='Weighted POD')
    axes[2].plot(x, np.zeros_like(x), 'k|', markersize=10)
    axes[2].set_title('Weighted POD first mode')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.suptitle("Comparison of Standard POD vs Weighted POD on Non-Uniform Grids")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    print("观察: 标准POD模态的幅值在网格稀疏处被不自然地放大了。")
    print("加权POD正确地反映了物理结构，不受网格密度影响。")

if __name__ == "__main__":
    perform_weighted_pod_demo()