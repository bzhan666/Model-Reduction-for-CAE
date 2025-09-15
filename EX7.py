# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:10:27 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt

class StandardDMD:
    def __init__(self, svd_rank=0):
        """
        初始化标准DMD
        :param svd_rank: SVD截断秩. 0表示不截断, -1表示自动选择.
        """
        self.svd_rank = svd_rank
        self.modes = None      # DMD Modes
        self.eigenvalues = None # DMD Eigenvalues
        self.amplitudes = None # Mode amplitudes
        self.dt = None         # Time step
        self.omega = None      # Continuous-time eigenvalues (frequencies)
    
    def fit(self, data, dt=1.0):
        """
        根据数据计算DMD
        :param data: 数据矩阵 (n_space, n_time)
        :param dt: 时间步长
        """
        print("开始执行标准DMD...")
        self.dt = dt
        X = data[:, :-1]
        X_prime = data[:, 1:]
        
        # 1. SVD of X
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        # 2. 确定截断秩
        r = self.svd_rank if self.svd_rank > 0 else len(s)
        print(f"SVD截断秩 r = {r}")
        U_r = U[:, :r]
        s_r = s[:r]
        Vt_r = Vt[:r, :]
        
        # 3. 计算低维算子 A_tilde
        A_tilde = U_r.T @ X_prime @ Vt_r.T @ np.diag(1.0 / s_r)
        
        # 4. 特征分解 A_tilde
        self.eigenvalues, W = np.linalg.eig(A_tilde)
        
        # 5. 计算DMD模态
        self.modes = U_r @ W
        
        # 6. 计算模态振幅 (初始时刻的投影)
        x0 = data[:, 0]
        self.amplitudes = np.linalg.pinv(self.modes) @ x0
        
        # 7. 计算连续时间特征值 (频率和增长率)
        self.omega = np.log(self.eigenvalues) / self.dt
        
        print("DMD计算完成.")
        return self

    def plot_eigs(self):
        """可视化DMD特征值"""
        plt.figure(figsize=(6, 6))
        plt.scatter(self.eigenvalues.real, self.eigenvalues.imag, c='red')
        # 绘制单位圆
        unit_circle = plt.Circle((0., 0.), 1., color='blue', fill=False, linestyle='--')
        plt.gca().add_artist(unit_circle)
        plt.title('DMD Eigenvalues on Complex Plane')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

# 主执行函数
def exercises():
    print("=== bzhan666 ===\n")
    
    # 创建一个合成的动态数据
    # 两个空间模态，两个不同的频率
    n_space = 128
    n_time = 100
    x_coords = np.linspace(-10, 10, n_space)
    t_coords = np.linspace(0, 8*np.pi, n_time)
    dt = t_coords[1] - t_coords[0]

    mode1_spatial = np.sin(x_coords)
    mode2_spatial = np.cos(3 * x_coords)
    
    freq1 = 1.0 # 稳定
    freq2 = 0.5 # 衰减
    
    dynamics1 = np.exp(1j * freq1 * t_coords) 
    dynamics2 = np.exp(1j * freq2 * t_coords) * np.exp(-0.2 * t_coords)
    
    data = (np.outer(mode1_spatial, dynamics1) + np.outer(mode2_spatial, dynamics2)).real
    
    # 执行DMD
    dmd = StandardDMD(svd_rank=5)
    dmd.fit(data, dt=dt)
    
    # 可视化特征值
    dmd.plot_eigs()
    
    # 打印计算出的频率
    calculated_freqs = np.abs(dmd.omega.imag / (2 * np.pi))
    print(f"真实频率 (Hz): {freq1/(2*np.pi):.4f}, {freq2/(2*np.pi):.4f}")
    print(f"DMD计算出的主频率 (Hz):\n{np.sort(calculated_freqs)[-2:]}")

if __name__ == "__main__":
    exercises()