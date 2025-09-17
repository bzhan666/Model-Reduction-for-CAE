# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:11:22 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt
from EX7 import StandardDMD # 假设你把上面的类保存在了这个文件

class DMDReconstructor(StandardDMD): # 继承上面的类
    def reconstruct(self, t_coords):
        """
        使用DMD结果重构时间序列
        :param t_coords: 需要重构的时间点坐标
        """
        if self.modes is None:
            raise ValueError("请先调用 fit() 方法")
        
        # time_dynamics 矩阵: (n_modes, n_times)
        time_dynamics = np.exp(np.outer(self.omega, t_coords))
        
        # 重构: (n_space, n_modes) @ (n_modes, n_modes) @ (n_modes, n_times)
        reconstructed_data = self.modes @ np.diag(self.amplitudes) @ time_dynamics
        
        return reconstructed_data.real

# 主执行函数
def exercises():
    print("=== bzhan666 ===\n")
    
    # 复用之前的数据
    n_space = 128
    n_time = 100
    x_coords = np.linspace(-10, 10, n_space)
    t_coords = np.linspace(0, 8*np.pi, n_time)
    dt = t_coords[1] - t_coords[0]
    
    mode1_spatial = np.sin(x_coords)
    mode2_spatial = np.cos(3 * x_coords)
    dynamics1 = np.exp(1j * 1.0 * t_coords) 
    dynamics2 = np.exp(1j * 0.5 * t_coords) * np.exp(-0.2 * t_coords)
    data = (np.outer(mode1_spatial, dynamics1) + np.outer(mode2_spatial, dynamics2)).real
    
    # 实例化重构器
    dmd_recon = DMDReconstructor(svd_rank=5)
    dmd_recon.fit(data, dt=dt)
    
    # 重构原始时间序列
    reconstructed_data = dmd_recon.reconstruct(t_coords)
    
    # 对比 t=50 时刻的快照
    snapshot_index = 50
    plt.figure(figsize=(12, 5))
    plt.plot(x_coords, data[:, snapshot_index], 'b-', label='Raw data')
    plt.plot(x_coords, reconstructed_data[:, snapshot_index], 'r--', label='DMD Reconstruction')
    plt.title(f'Snapshot at t = {t_coords[snapshot_index]:.2f}')
    plt.xlabel('Space coordinates x')
    plt.ylabel('value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 检查全局重构误差
    error = np.linalg.norm(data - reconstructed_data) / np.linalg.norm(data)
    print(f"全局重构相对误差: {error:.4%}")

if __name__ == "__main__":
    exercises()