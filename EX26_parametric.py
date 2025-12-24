# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 11:13:34 2025

@author: bzhan666

该脚本实现了一个基于全局POD（正交分解）和径向基函数（RBF）插值的参数化降阶模型（ROM）。
主要内容包括：
1. 为不同雷诺数下的流场生成参数化数据。
2. 使用POD对流场数据进行降维，训练降阶模型。
3. 使用RBF插值器预测新参数下的降维系数。
4. 可视化模型预测结果并进行误差分析。

这是一种常见的处理高维流体动力学仿真数据的模型降阶与参数化方法。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from typing import Dict, List, Tuple


try:
    from EX26 import FlexibleCoupledPOD # 或者你之前的 PODROM 类
except ImportError:
    # 简单的 SVD 桩代码
    class FlexibleCoupledPOD:
        def __init__(self, n_modes): self.n_modes = n_modes
        def fit(self, data_dict):
            # 这里的fit需要适应参数化数据的逻辑，下文会重写逻辑，
            # 所以这里只需要它是存在的即可。
            pass

class ParametricDataGenerator:
    """
    生成不同雷诺数下的流场数据
    参数 mu: 控制流速/雷诺数
    """
    def __init__(self, nx=80, ny=40):
        self.nx, self.ny = nx, ny
        self.x = np.linspace(0, 8, nx)
        self.y = np.linspace(-2, 2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def generate_snapshot(self, mu: float, t: float) -> np.ndarray:
        """
        根据参数 mu 和时间 t 生成一个快照
        mu 越大，涡街频率越高，尾迹越长
        """
        # 频率随 mu 增加
        omega = 2.0 * np.pi * (0.2 + 0.1 * mu) 
        
        # 涡街结构
        # 涡的衰减长度随 mu 变化
        decay = 0.5 / (1.0 + mu) 
        
        vortex = np.sin(omega * t - 2.0 * self.X) * np.exp(-decay * self.X) * np.cos(np.pi * self.Y / 2)
        
        # 主流速度随 mu 变化
        u_mean = 1.0 + 0.5 * mu
        
        field = u_mean + vortex
        return field

    def generate_dataset(self, params: List[float], n_snapshots=50) -> Dict[float, np.ndarray]:
        """生成参数集对应的训练数据"""
        dataset = {}
        print(f"生成参数化数据: 参数列表 {params}")
        for p in params:
            snapshots = []
            for i in range(n_snapshots):
                t = i * 0.1
                snap = self.generate_snapshot(mu=p, t=t)
                snapshots.append(snap.ravel())
            dataset[p] = np.array(snapshots).T # (N_space, N_time)
        return dataset

class ParametricROM:
    """参数化降阶模型 (Global POD + RBF Interpolation)"""
    
    def __init__(self, n_modes=10):
        self.n_modes = n_modes
        self.U = None     # 全局基
        self.mean = None  # 全局平均场
        self.rbf = None   # 插值器
        self.train_params = None # 训练参数
        self.projections = None  # 训练数据的投影系数
        
    def fit(self, dataset: Dict[float, np.ndarray]):
        """
        训练流程:
        1. 收集所有工况数据 -> 全局快照矩阵
        2. 全局 SVD -> 全局基 U
        3. 将每个工况投影到 U -> 得到系数 a(mu)
        4. 训练插值器 mu -> a
        """
        print("\n" + "="*60)
        print("训练参数化 ROM (Global POD + RBF)")
        print("="*60)
        
        # 1. 构建全局快照矩阵 (Snapshot Stacking)
        self.train_params = sorted(list(dataset.keys()))
        all_snapshots = []
        
        # 记录每个参数对应的数据列范围，以便后续拆分
        param_indices = {} 
        start_col = 0
        
        for p in self.train_params:
            data = dataset[p]
            all_snapshots.append(data)
            n_cols = data.shape[1]
            param_indices[p] = (start_col, start_col + n_cols)
            start_col += n_cols
            
        global_matrix = np.hstack(all_snapshots)
        print(f"全局快照矩阵形状: {global_matrix.shape}")
        
        # 2. 全局 POD
        self.mean = np.mean(global_matrix, axis=1, keepdims=True)
        X_prime = global_matrix - self.mean
        
        print("执行全局 SVD...")
        U, s, _ = np.linalg.svd(X_prime, full_matrices=False)
        self.U = U[:, :self.n_modes]
        
        energy = np.sum(s[:self.n_modes]**2) / np.sum(s**2)
        print(f"保留 {self.n_modes} 个模态，能量占比: {energy:.2%}")
        
        # 3. 投影并构建插值数据集
        # 我们需要构建 (mu, t) -> coefficients 的映射
        # 或者更简单：针对每个模态，构建 mu -> coeff_amplitude 的映射
        # 这里我们采用 "Parameter-only" 插值，假设所有工况的时间步是对齐的
        
        # 训练数据: X = [mu1, mu2, ...], Y = [Coeffs_mu1, Coeffs_mu2, ...]
        # Y 的形状是 (N_params, N_modes * N_time) - 我们把整个时间历程展平作为输出
        
        X_train = []
        Y_train = []
        
        for p in self.train_params:
            # 获取该参数下的原始数据
            data = dataset[p]
            # 投影: a = U^T * (x - mean)
            coeffs = self.U.T @ (data - self.mean) # (n_modes, n_time)
            
            X_train.append([p]) # 输入必须是 2D (n_samples, n_features)
            Y_train.append(coeffs.flatten()) # 将时间历程展平
            
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        
        # 4. 训练 RBF 插值器
        print("训练 RBF 插值器...")
        # kernel 可选: 'linear', 'thin_plate_spline', 'cubic', 'gaussian'
        self.rbf = RBFInterpolator(X_train, Y_train, kernel='thin_plate_spline')
        
        print("模型训练完成")
        
    def predict(self, mu: float, n_time_steps: int) -> np.ndarray:
        """
        预测新参数 mu 下的流场
        """
        # 1. 插值预测系数
        # 输入形状 (1, 1) -> [[mu]]
        # 输出形状 (1, n_modes * n_time)
        pred_coeffs_flat = self.rbf(np.array([[mu]]))
        
        # 重塑回 (n_modes, n_time)
        coeffs = pred_coeffs_flat.reshape(self.n_modes, n_time_steps)
        
        # 2. 重构流场: x = mean + U * a
        recon = self.mean + self.U @ coeffs
        
        return recon

def main():
    # 1. 准备数据
    # 假设我们有 mu = 1.0, 2.0, 3.0, 4.0 的数据
    # 我们想预测 mu = 2.5
    gen = ParametricDataGenerator()
    train_params = [1.0, 2.0, 3.0, 4.0]
    test_param = 2.5
    
    dataset = gen.generate_dataset(train_params, n_snapshots=100)
    
    # 2. 训练参数化 ROM
    prom = ParametricROM(n_modes=8)
    prom.fit(dataset)
    
    # 3. 预测未知工况
    print(f"\n预测新工况: mu = {test_param}")
    pred_field = prom.predict(mu=test_param, n_time_steps=100)
    
    # 4. 生成真实数据用于验证 (God Truth)
    truth_field = gen.generate_dataset([test_param], n_snapshots=100)[test_param]
    
    # 5. 误差分析与可视化
    error = np.linalg.norm(truth_field - pred_field) / np.linalg.norm(truth_field)
    print(f"相对预测误差: {error:.4%}")
    
    # 绘图
    t_idx = 50
    nx, ny = gen.nx, gen.ny
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im1 = axes[0].contourf(gen.X, gen.Y, truth_field[:, t_idx].reshape(ny, nx), levels=20, cmap='viridis')
    axes[0].set_title(f"True Field (mu={test_param})")
    
    im2 = axes[1].contourf(gen.X, gen.Y, pred_field[:, t_idx].reshape(ny, nx), levels=20, cmap='viridis')
    axes[1].set_title(f"Predicted Field (RBF)")
    
    err_map = truth_field[:, t_idx] - pred_field[:, t_idx]
    vmax = np.abs(err_map).max()
    im3 = axes[2].contourf(gen.X, gen.Y, err_map.reshape(ny, nx), levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"Error")
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()