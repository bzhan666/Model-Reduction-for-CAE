# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:03:22 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Part 1: 数据模拟与读取 ---

def generate_fluent_like_data(path='./fluent_data', nx=80, ny=40, nt=100):
    """
    模拟Fluent导出的ASCII数据文件。
    生成一个经典的2D圆柱绕流卡门涡街算例。
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    print(f"正在生成模拟数据到 '{path}' 目录...")
    x = np.linspace(-2, 14, nx)
    y = np.linspace(-4, 4, ny)
    X, Y = np.meshgrid(x, y)
    
    # 模拟一个简化的涡脱落
    for i in range(nt):
        t = i * 0.1
        # 主流 + 涡脱落的u速度分量
        u = 1 - np.exp(-((X - 5*t % 16 + 2)**2 + (Y - 0.5*np.sin(t))**2)) \
              + np.exp(-((X - 5*t % 16 + 2)**2 + (Y + 0.5*np.sin(t+np.pi)))**2)
        
        filename = os.path.join(path, f'flow_t_{i:04d}.dat')
        
        # 修复：使用手动写入方式，确保头部格式正确
        with open(filename, 'w') as f:
            f.write("VARIABLES = X, Y, U\n")
            f.write(f"ZONE T=time_{t:.2f}\n")
            # 写入数据
            data_to_save = np.vstack([X.ravel(), Y.ravel(), u.ravel()]).T
            for row in data_to_save:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
                
    print("数据生成完毕。")
    return nx, ny, nt

def read_and_assemble_snapshots(path, nt, field_index=2):
    """
    读取模拟的Fluent数据文件，并组装成快照矩阵。
    :param path: 数据文件所在目录
    :param nt: 快照总数
    :param field_index: 要提取的变量所在的列 (0:X, 1:Y, 2:U)
    """
    print("开始读取数据并组装快照矩阵...")
    # 从第一个文件读取网格信息和空间点数
    first_file = os.path.join(path, 'flow_t_0000.dat')
    
    # 修复：跳过两行头部信息
    grid_data = np.loadtxt(first_file, skiprows=2)
    n_space = grid_data.shape[0]
    coords = grid_data[:, :2] # 提取X, Y坐标
    
    # 初始化快照矩阵
    snapshot_matrix = np.zeros((n_space, nt))
    
    for i in range(nt):
        filename = os.path.join(path, f'flow_t_{i:04d}.dat')
        if i % 20 == 0:
            print(f"  正在读取: {filename}")
        # 修复：跳过两行头部信息
        data = np.loadtxt(filename, skiprows=2)
        snapshot_matrix[:, i] = data[:, field_index]
        
    print(f"快照矩阵组装完毕。尺寸: {snapshot_matrix.shape}")
    return snapshot_matrix, coords

def preprocess_data(snapshot_matrix):
    """
    对数据进行预处理，主要是均值减法。
    """
    # 计算时间平均场
    mean_field = np.mean(snapshot_matrix, axis=1)
    
    # 计算脉动场
    fluctuation_matrix = snapshot_matrix - mean_field[:, np.newaxis]
    
    print("数据预处理完成 (均值减法)。")
    return fluctuation_matrix, mean_field



# 主执行函数 (Day 29-31)
def exercises():
    DATA_PATH = './fluent_data'
    
    # 1. 生成数据
    nx, ny, nt = generate_fluent_like_data(path=DATA_PATH)
    
    # 2. 读取并组装
    snapshot_matrix_raw, coords = read_and_assemble_snapshots(path=DATA_PATH, nt=nt)
    
    # 3. 预处理
    fluctuation_matrix, mean_field = preprocess_data(snapshot_matrix_raw)
    
    # 4. 可视化验证
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # Reshape a vector back to 2D grid for plotting
    def reshape_field(field_vector):
        return field_vector.reshape(ny, nx)

    # 绘制平均流场
    im1 = axes[0].contourf(reshape_field(coords[:, 0]), reshape_field(coords[:, 1]), reshape_field(mean_field), cmap='viridis', levels=50)
    axes[0].set_title('Mean Flow Field')
    plt.colorbar(im1, ax=axes[0])
    
    # 绘制一个瞬时流场
    im2 = axes[1].contourf(reshape_field(coords[:, 0]), reshape_field(coords[:, 1]), reshape_field(snapshot_matrix_raw[:, -1]), cmap='viridis', levels=50)
    axes[1].set_title('Instantaneous Flow Field (last snapshot)')
    plt.colorbar(im2, ax=axes[1])
    
    # 绘制一个瞬时脉动流场
    im3 = axes[2].contourf(reshape_field(coords[:, 0]), reshape_field(coords[:, 1]), reshape_field(fluctuation_matrix[:, -1]), cmap='RdBu_r', levels=50)
    axes[2].set_title('Instantaneous Fluctuation Field')
    plt.colorbar(im3, ax=axes[2])
    
    for ax in axes:
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    
    return fluctuation_matrix, mean_field, coords, nx, ny

if __name__ == "__main__":
    fluctuation_matrix, mean_field, coords, nx, ny = exercises()
    


    
