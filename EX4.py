# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:06:06 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt

class PODAnalyzer:
    def __init__(self, data_matrix):
        """
        初始化POD分析器
        :param data_matrix: 数据矩阵 A (n_space, n_time)
        """
        self.A = data_matrix
        self.n_space, self.n_time = self.A.shape
        self.U = None
        self.s = None
        self.Vt = None
    
    def decompose(self):
        """执行SVD分解计算POD模态"""
        print("执行SVD分解...")
        # 为了效率，如果空间点远多于时间点，SVD计算`full_matrices=False`很重要
        self.U, self.s, self.Vt = np.linalg.svd(self.A, full_matrices=False)
        print("分解完成.")
    
    def analyze_energy(self):
        """分析并可视化模态能量"""
        if self.s is None:
            raise ValueError("请先执行 decompose() 方法")
            
        # 计算能量
        energies = self.s**2
        total_energy = np.sum(energies)
        cumulative_energy = np.cumsum(energies) / total_energy
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 能量谱
        axes[0].semilogy(energies / total_energy, 'bo-', label='Energy proportion of a single mode')
        axes[0].set_title('Modal Energy Spectrum')
        axes[0].set_xlabel('Modal number')
        axes[0].set_ylabel('Energy proportion')
        axes[0].grid(True)
        axes[0].legend()
        
        # 累积能量
        axes[1].plot(cumulative_energy, 'ro-', label='accumulated energy')
        axes[1].set_title('accumulated energy')
        axes[1].set_xlabel('Modal number')
        axes[1].set_ylabel('Accumulated energy ratio')
        axes[1].grid(True)
        axes[1].axhline(y=0.99, color='g', linestyle='--', label='99% energy')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return cumulative_energy
        
    def reconstruct(self, r):
        """使用前 r 个模态重构数据"""
        if self.U is None:
            raise ValueError("请先执行 decompose() 方法")
        
        print(f"使用前 {r} 个模态进行重构...")
        A_reconstructed = self.U[:, :r] @ np.diag(self.s[:r]) @ self.Vt[:r, :]
        
        # 计算重构误差
        error = np.linalg.norm(self.A - A_reconstructed, 'fro') / np.linalg.norm(self.A, 'fro')
        print(f"重构相对误差: {error:.4%}")
        
        return A_reconstructed

# 主执行函数
def exercises():
    print("=== bzhan666 ===\n")
    
    # 使用上周的CFD数据生成函数
    # 模拟涡脱落现象
    nx, ny, nt = 50, 30, 100
    x = np.linspace(-2, 4, nx)
    y = np.linspace(-2, 2, ny)
    t = np.linspace(0, 10, nt)
    X, Y = np.meshgrid(x, y)
    flow_data = np.zeros((ny, nx, nt))
    for i, time in enumerate(t):
        vortex1 = 0.5 * np.exp(-((X-1)**2 + (Y-0.3)**2)/0.1) * np.sin(2*time)
        vortex2 = -0.5 * np.exp(-((X-1.5)**2 + (Y+0.3)**2)/0.1) * np.sin(2*time + np.pi)
        flow_data[:, :, i] = 1.0 + vortex1 + vortex2
    data_matrix = flow_data.reshape(-1, nt)
    
    # 1. 均值减法 (POD通常分析脉动)
    mean_field = np.mean(data_matrix, axis=1, keepdims=True)
    fluctuation_matrix = data_matrix - mean_field
    
    # 2. POD分析
    pod_analyzer = PODAnalyzer(fluctuation_matrix)
    pod_analyzer.decompose()
    cumulative_energy = pod_analyzer.analyze_energy()
    
    # 3. 确定截断秩
    r = np.argmax(cumulative_energy >= 0.99) + 1
    print(f"达到99%能量需要 {r} 个模态。")
    
    # 4. 重构并对比
    reconstructed_fluctuations = pod_analyzer.reconstruct(r)
    reconstructed_field_t50 = (reconstructed_fluctuations + mean_field).reshape(ny, nx, nt)[:, :, 50]
    original_field_t50 = data_matrix.reshape(ny, nx, nt)[:, :, 50]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    v_min = original_field_t50.min()
    v_max = original_field_t50.max()
    
    im1 = axes[0].contourf(X, Y, original_field_t50, levels=20, cmap='viridis', vmin=v_min, vmax=v_max)
    axes[0].set_title('original flow field (t=5.0)')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].contourf(X, Y, reconstructed_field_t50, levels=20, cmap='viridis', vmin=v_min, vmax=v_max)
    axes[1].set_title(f'POD reconstructs flow field (r={r})')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].contourf(X, Y, original_field_t50 - reconstructed_field_t50, levels=20, cmap='RdBu_r')
    axes[2].set_title('reconstruction error')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    exercises()