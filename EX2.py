# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:36:25 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, eig


class MatrixDecompositionVisualizer:
    def __init__(self):
        self.fig = None
        self.axes = None
    
    def demonstrate_evd_geometry(self):
        """演示特征值分解的几何意义"""
        # 创建一个2x2对称矩阵（代表某种物理过程）
        A = np.array([[3, 1], [1, 2]])
        
        # 计算特征值分解
        eigenvals, eigenvecs = eig(A)
        
        # 创建单位圆上的点
        theta = np.linspace(0, 2*np.pi, 100)
        unit_circle = np.array([np.cos(theta), np.sin(theta)])
        
        # 应用矩阵变换
        transformed = A @ unit_circle
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 原始单位圆
        ax1.plot(unit_circle[0], unit_circle[1], 'b-', linewidth=2, label='unit circle')
        ax1.quiver(0, 0, eigenvecs[0, 0], eigenvecs[1, 0], 
                  angles='xy', scale_units='xy', scale=1, color='r', width=0.005)
        ax1.quiver(0, 0, eigenvecs[0, 1], eigenvecs[1, 1], 
                  angles='xy', scale_units='xy', scale=1, color='g', width=0.005)
        ax1.set_title('Original space + eigenvector')
        ax1.grid(True)
        ax1.axis('equal')
        ax1.legend()
        
        # 变换后的椭圆
        ax2.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='After transformation')
        ax2.quiver(0, 0, eigenvals[0]*eigenvecs[0, 0], eigenvals[0]*eigenvecs[1, 0], 
                  angles='xy', scale_units='xy', scale=1, color='r', width=0.005)
        ax2.quiver(0, 0, eigenvals[1]*eigenvecs[0, 1], eigenvals[1]*eigenvecs[1, 1], 
                  angles='xy', scale_units='xy', scale=1, color='g', width=0.005)
        ax2.set_title('space after transformation')
        ax2.grid(True)
        ax2.axis('equal')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"特征值: {eigenvals}")
        print(f"特征向量:\n{eigenvecs}")
    
    def demonstrate_svd_step_by_step(self):
        """逐步演示SVD分解过程"""
        # 创建一个简单的"流场"数据矩阵
        # 行代表空间位置，列代表时间
        t = np.linspace(0, 4*np.pi, 50)
        x = np.linspace(0, 2*np.pi, 30)
        T, X = np.meshgrid(t, x)
        
        # 模拟两个主要模态的叠加
        mode1 = np.sin(X) * np.cos(0.5*T)  # 低频模态
        mode2 = 0.3 * np.sin(2*X) * np.cos(2*T)  # 高频模态
        noise = 0.1 * np.random.randn(*mode1.shape)  # 噪声
        
        A = mode1 + mode2 + noise
        
        # 执行SVD
        U, s, Vt = svd(A, full_matrices=False)
        
        # 可视化结果
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 原始数据
        im0 = axes[0, 0].imshow(A, aspect='auto', cmap='RdBu_r')
        axes[0, 0].set_title('raw data A')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 奇异值
        axes[0, 1].semilogy(s[:10], 'bo-')
        axes[0, 1].set_title('Singular Value Spectrum')
        axes[0, 1].set_xlabel('Modal number')
        axes[0, 1].set_ylabel('singular value')
        axes[0, 1].grid(True)
        
        # 前几个空间模态 (U的列)
        for i in range(2):
            axes[0, 2+i].plot(x, U[:, i])
            axes[0, 2+i].set_title(f'spatial mode {i+1}')
            axes[0, 2+i].grid(True)
        
        # 前几个时间模态 (V的行)
        for i in range(2):
            axes[1, 2+i].plot(t, Vt[i, :])
            axes[1, 2+i].set_title(f'Temporal modality {i+1}')
            axes[1, 2+i].grid(True)
        
        # 重构对比
        # 使用前k个模态重构
        k_values = [1, 3, 5]
        for idx, k in enumerate(k_values):
            A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            
            if idx == 0:
                im = axes[1, 0].imshow(A_k, aspect='auto', cmap='RdBu_r')
                axes[1, 0].set_title(f'Refactor (k={k})')
                plt.colorbar(im, ax=axes[1, 0])
            
            # 计算重构误差
            error = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
            print(f"使用前{k}个模态，相对误差: {error:.4f}")
        
        # 累积能量
        cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
        axes[1, 1].plot(cumulative_energy[:15], 'ro-')
        axes[1, 1].set_title('accumulated energy')
        axes[1, 1].set_xlabel('Number of modes')
        axes[1, 1].set_ylabel('Accumulated energy ratio')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return U, s, Vt, A
    
    def implement_power_iteration_svd(self, A, max_iter=100, tol=1e-6):
        """手动实现幂迭代求主奇异值"""
        m, n = A.shape
        
        # 初始化随机向量
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        print("幂迭代求解过程:")
        for i in range(max_iter):
            # v → A^T A v (右奇异向量方向)
            u = A @ v
            u = u / np.linalg.norm(u)
            
            # u → A A^T u (左奇异向量方向)  
            v_new = A.T @ u
            sigma = np.linalg.norm(v_new)
            v_new = v_new / sigma
            
            # 检查收敛
            if np.linalg.norm(v_new - v) < tol:
                print(f"收敛于第 {i+1} 次迭代")
                break
            
            v = v_new
            
            if i % 10 == 0:
                print(f"迭代 {i:3d}: σ = {sigma:.6f}")
        
        # 验证结果
        U_true, s_true, Vt_true = svd(A)
        print(f"\n验证结果:")
        print(f"手动实现的主奇异值: {sigma:.6f}")
        print(f"NumPy SVD的主奇异值: {s_true[0]:.6f}")
        print(f"误差: {abs(sigma - s_true[0]):.8f}")
        
        return sigma, u, v

# 主执行函数
def exercises():

    print("=== bzhan666===\n")
    
    visualizer = MatrixDecompositionVisualizer()
    
    print("1. 特征值分解几何演示:")
    visualizer.demonstrate_evd_geometry()
    
    print("\n2. SVD逐步分解演示:")
    U, s, Vt, A = visualizer.demonstrate_svd_step_by_step()
    
    print("\n3. 幂迭代SVD实现:")
    visualizer.implement_power_iteration_svd(A)

if __name__ == "__main__":
    exercises()