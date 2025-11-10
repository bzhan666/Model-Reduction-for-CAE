# -*- coding: utf-8 -*-
"""
Created on Mon Nov 9 21:32:58 2025

@author: bzhan666
"""

# (EX17代码的迭代)

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# 假设你的第六周代码(PODROM)保存在 EX16.py
try:
    from EX16 import PODROM
except ImportError:
    print("错误: 无法从 EX16.py 导入 PODROM 类。")
    class PODROM: pass # 桩代码

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class POD_DMD_Model:
    """
    基于POD和DMD的动态降阶模型 (v2.0 - 封装版)
    
    这个模型使用POD进行空间降维，然后在低维空间中使用DMD学习时间动态。
    """
    
    def __init__(self, n_pod_modes: int):
        """
        初始化模型
        
        Args:
            n_pod_modes: 用于空间降维的POD模态数量 (r)
        """
        if n_pod_modes <= 0:
            raise ValueError("n_pod_modes must be a positive integer.")
        self.r = n_pod_modes
        
        # 模型组件
        self.pod_rom = PODROM(n_modes=n_pod_modes)
        self.A_tilde = None        # 低维演化算子 (DMD operator)
        self.eigenvalues = None    # 低维空间的DMD特征值
        self.low_dim_modes = None  # 低维空间的DMD模态
        
        self.is_fitted = False

    def fit(self, snapshot_matrix: np.ndarray, dt: float = 1.0):
        """
        训练POD-DMD模型
        
        Args:
            snapshot_matrix: 原始快照矩阵 A (n_space, n_time)
            dt: 时间步长
        """
        print("="*60)
        print("开始训练 POD-DMD 模型...")
        print("="*60)
        
        self.dt = dt
        
        # 1. 训练POD-ROM以获得空间基，并投影到低维空间
        print("  步骤1: 执行POD降维...")
        low_dim_series = self.pod_rom.fit_transform(snapshot_matrix)
        
        # 2. 在低维空间应用DMD
        print("  步骤2: 在低维空间学习动力学 (DMD)...")
        X1 = low_dim_series[:, :-1]
        X2 = low_dim_series[:, 1:]
        
        # 使用SVD来稳定地计算伪逆和演化算子 (更鲁棒的方法)
        U, s, Vt = np.linalg.svd(X1, full_matrices=False)
        self.A_tilde = X2 @ Vt.T @ np.diag(1. / s) @ U.T
        
        # 3. 计算并存储低维空间的DMD特征值和模态
        self.eigenvalues, self.low_dim_modes = np.linalg.eig(self.A_tilde)
        
        self.is_fitted = True
        print("\n模型训练完成!")
        print(f"  POD模态数 (低维空间维度): {self.r}")
        print(f"  低维动力学算子 A_tilde 的维度: {self.A_tilde.shape}")
        return self

    def predict(self, initial_snapshot: np.ndarray, num_steps: int) -> np.ndarray:
        """
        从一个初始快照开始，向前预测多个时间步
        
        Args:
            initial_snapshot: 初始高维流场向量 x_0
            num_steps: 预测的时间步数
            
        Returns:
            predicted_snapshots: 预测出的高维快照矩阵 (n_space, num_steps)
        """
        self._check_fitted()
        
        # 1. 将初始快照投影到低维空间
        a0 = self.pod_rom.transform(initial_snapshot)
        
        # 2. 在低维空间进行时间演化
        predicted_low_dim_series = np.zeros((self.r, num_steps))
        current_a = a0
        
        for i in range(num_steps):
            predicted_low_dim_series[:, i] = current_a
            # 使用简单的矩阵乘法进行演化
            current_a = self.A_tilde @ current_a
            
        # 3. 将预测的低维序列重构回高维空间
        predicted_snapshots = self.pod_rom.inverse_transform(predicted_low_dim_series)
        
        return predicted_snapshots.real

    def plot_eigs(self, save_fig=False):
        """可视化低维空间DMD的特征值"""
        self._check_fitted()
        plt.figure(figsize=(8, 8))
        plt.scatter(self.eigenvalues.real, self.eigenvalues.imag, c='red', label='DMD特征值')
        
        unit_circle = plt.Circle((0., 0.), 1., color='blue', fill=False, linestyle='--', label='单位圆')
        plt.gca().add_artist(unit_circle)
        
        plt.title('低维空间DMD特征值分布', fontsize=16)
        plt.xlabel('实部', fontsize=12)
        plt.ylabel('虚部', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        plt.legend()
        if save_fig:
            plt.savefig("pod_dmd_eigenvalues.png", dpi=200)
        plt.show()

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")