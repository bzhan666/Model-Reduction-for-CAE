# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:39:37 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 向量空间基本操作验证
def verify_vector_space_properties():
    """验证向量空间的基本性质"""
    # 生成两个"流场"向量（简化为1D）
    u = np.random.randn(100)  # 模拟速度场
    v = np.random.randn(100)  # 模拟另一个速度场

    # 验证向量空间性质
    print("验证向量空间性质：")
    print(f"交换律: ||u + v - (v + u)||₂ = {np.linalg.norm(u + v - (v + u))}")
    print(f"结合律验证...")

    return u, v

# 2. 内积计算与可视化
def compute_inner_products(u, v):
    """计算和理解不同类型的内积"""
    # 标准内积
    standard_inner = np.dot(u, v)

    # 加权内积（模拟CFD中的加权积分）
    weights = np.linspace(0.5, 2.0, len(u))  # 模拟网格权重
    weighted_inner = np.sum(weights * u * v)

    print(f"标准内积: {standard_inner:.4f}")
    print(f"加权内积: {weighted_inner:.4f}")

    return standard_inner, weighted_inner

# 3. 范数的几何可视化
def visualize_norms():
    """可视化不同范数的几何意义"""
    # 创建单位圆/球面上的点
    theta = np.linspace(0, 2*np.pi, 100)

    # L1, L2, L∞ 范数的单位球
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # L1 范数 (菱形)
    x1 = np.cos(theta) / (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
    y1 = np.sin(theta) / (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
    axes[0].plot(x1, y1, 'b-', linewidth=2)
    axes[0].set_title('L₁ Norm Unit Ball')
    axes[0].grid(True)
    axes[0].axis('equal')

    # L2 范数 (圆形)
    x2 = np.cos(theta)
    y2 = np.sin(theta)
    axes[1].plot(x2, y2, 'r-', linewidth=2)
    axes[1].set_title('L₂ Norm Unit Ball')
    axes[1].grid(True)
    axes[1].axis('equal')

    # L∞ 范数 (正方形)
    square_x = [-1, 1, 1, -1, -1]
    square_y = [-1, -1, 1, 1, -1]
    axes[2].plot(square_x, square_y, 'g-', linewidth=2)
    axes[2].set_title('L∞ Norm Unit Ball')
    axes[2].grid(True)
    axes[2].axis('equal')

    plt.tight_layout()
    plt.show()

# 4. CFD数据的内积计算示例
def cfd_inner_product_example():
    """模拟CFD数据的内积计算"""
    # 模拟2D网格上的速度场
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # 模拟两个不同的速度场模态
    u1 = np.sin(np.pi * X) * np.cos(np.pi * Y)  # 第一个模态
    u2 = np.cos(2 * np.pi * X) * np.sin(np.pi * Y)  # 第二个模态

    # 计算内积 (离散形式的积分)
    dx, dy = 1/(nx-1), 1/(ny-1)
    inner_product = np.sum(u1 * u2) * dx * dy

    # 计算各自的L2范数
    norm_u1 = np.sqrt(np.sum(u1**2) * dx * dy)
    norm_u2 = np.sqrt(np.sum(u2**2) * dx * dy)

    print(f"模态1的L2范数: {norm_u1:.4f}")
    print(f"模态2的L2范数: {norm_u2:.4f}")
    print(f"两模态的内积: {inner_product:.4f}")
    print(f"正交性检验 (内积/范数乘积): {inner_product/(norm_u1*norm_u2):.6f}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im1 = axes[0].contourf(X, Y, u1, levels=20, cmap='RdBu_r')
    axes[0].set_title('model 1')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(X, Y, u2, levels=20, cmap='RdBu_r')
    axes[1].set_title('model 2')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].contourf(X, Y, u1 + u2, levels=20, cmap='RdBu_r')
    axes[2].set_title('model superposition')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()

    return u1, u2, inner_product

# 主执行函数
if __name__ == "__main__":
    print("=== BZHAN666===\n")

    # 执行所有练习
    u, v = verify_vector_space_properties()
    compute_inner_products(u, v)
    visualize_norms()
    cfd_inner_product_example()