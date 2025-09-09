# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:38:32 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt

class ProjectionAndSubspaceDemo:
    def __init__(self):
        pass
    
    def demonstrate_orthogonal_projection(self):
        """演示正交投影的几何意义"""
        # 创建3D空间中的一个2D子空间
        # 子空间由两个正交向量张成
        v1 = np.array([1, 1, 0]) / np.sqrt(2)
        v2 = np.array([-1, 1, 2]) / np.sqrt(6)
        
        # 构造投影矩阵
        V = np.column_stack([v1, v2])
        P = V @ V.T
        
        # 创建一些测试向量
        test_points = np.array([
            [2, 1, 3],
            [0, 0, 4],
            [1, -1, 1],
            [-2, 2, -1]
        ]).T
        
        # 计算投影
        projected_points = P @ test_points
        
        # 可视化
        fig = plt.figure(figsize=(12, 5))
        
        # 3D可视化
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 绘制子空间（用网格表示）
        u = np.linspace(-3, 3, 10)
        v = np.linspace(-3, 3, 10)
        U, V_mesh = np.meshgrid(u, v)
        X = U * v1[0] + V_mesh * v2[0]
        Y = U * v1[1] + V_mesh * v2[1]
        Z = U * v1[2] + V_mesh * v2[2]
        ax1.plot_surface(X, Y, Z, alpha=0.3, color='blue')
        
        # 绘制原始点和投影点
        ax1.scatter(test_points[0], test_points[1], test_points[2], 
                   c='red', s=100, label='original point')
        ax1.scatter(projected_points[0], projected_points[1], projected_points[2], 
                   c='green', s=100, label='Projection point')
        
        # 绘制投影线
        for i in range(test_points.shape[1]):
            ax1.plot([test_points[0,i], projected_points[0,i]],
                    [test_points[1,i], projected_points[1,i]],
                    [test_points[2,i], projected_points[2,i]], 'k--', alpha=0.5)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.set_title('3D Orthographic projection')
        
        # 投影距离分析
        ax2 = fig.add_subplot(122)
        distances = np.linalg.norm(test_points - projected_points, axis=0)
        ax2.bar(range(len(distances)), distances)
        ax2.set_title('Projection distance')
        ax2.set_xlabel('point number')
        ax2.set_ylabel('distance')
        
        plt.tight_layout()
        plt.show()
        
        # 验证投影性质
        print("投影矩阵性质验证:")
        print(f"P² = P: {np.allclose(P @ P, P)}")
        print(f"P是对称的: {np.allclose(P, P.T)}")
        
    def pca_vs_pod_comparison(self):
        """对比PCA和POD在CFD数据上的应用"""
        # 生成模拟的CFD时间序列数据
        nt, nx = 100, 50  # 100个时间步，50个空间点
        x = np.linspace(0, 2*np.pi, nx)
        t = np.linspace(0, 10, nt)
        
        # 创建两个主要模态
        mode1_spatial = np.sin(x)
        mode1_temporal = np.cos(0.5 * t)
        
        mode2_spatial = np.sin(2*x)
        mode2_temporal = np.sin(t) * np.exp(-0.1*t)
        
        # 组合数据矩阵 (空间 × 时间)
        data_matrix = (np.outer(mode1_spatial, mode1_temporal) + 
                      0.3 * np.outer(mode2_spatial, mode2_temporal) +
                      0.05 * np.random.randn(nx, nt))
        
        # 方法1: PCA视角（对时间进行主成分分析）
        # 中心化数据
        data_centered = data_matrix - np.mean(data_matrix, axis=1, keepdims=True)
        
        # 计算协方差矩阵的特征值分解
        cov_matrix = np.cov(data_centered)
        eigenvals_pca, eigenvecs_pca = np.linalg.eigh(cov_matrix)
        
        # 排序（降序）
        idx = np.argsort(eigenvals_pca)[::-1]
        eigenvals_pca = eigenvals_pca[idx]
        eigenvecs_pca = eigenvecs_pca[:, idx]
        
        # 方法2: POD视角（SVD分解）
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        
        # 可视化对比
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 原始数据
        im1 = axes[0, 0].imshow(data_matrix, aspect='auto', cmap='RdBu_r')
        axes[0, 0].set_title('raw data')
        axes[0, 0].set_xlabel('time')
        axes[0, 0].set_ylabel('space')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # PCA特征值
        axes[0, 1].semilogy(eigenvals_pca[:10], 'bo-', label='PCA eigenvalues')
        axes[0, 1].set_title('PCA eigenvalues')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # PCA主成分
        for i in range(2):
            axes[0, 2+i].plot(x, eigenvecs_pca[:, i])
            axes[0, 2+i].set_title(f'PCA principal components {i+1}')
            axes[0, 2+i].grid(True)
        
        # POD奇异值
        axes[1, 1].semilogy(s[:10], 'ro-', label='POD singular values')
        axes[1, 1].set_title('POD singular values')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        # POD空间模态
        for i in range(2):
            axes[1, 2+i].plot(x, U[:, i])
            axes[1, 2+i].set_title(f'POD spatial mode {i+1}')
            axes[1, 2+i].grid(True)
        
        # 重构比较
        k = 3
        # PCA重构
        pca_coeffs = eigenvecs_pca[:, :k].T @ data_centered
        pca_reconstructed = eigenvecs_pca[:, :k] @ pca_coeffs + np.mean(data_matrix, axis=1, keepdims=True)
        
        # POD重构
        pod_reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        im2 = axes[1, 0].imshow(pod_reconstructed, aspect='auto', cmap='RdBu_r')
        axes[1, 0].set_title(f'POD Refactor (k={k})')
        plt.colorbar(im2, ax=axes[1, 0])
        
        plt.tight_layout()
        plt.show()
        
        # 数值验证两种方法的等价性
        print("PCA vs POD 数值验证:")
        print(f"特征值 vs 奇异值²: {np.allclose(eigenvals_pca[:5], s[:5]**2)}")
        print(f"前5个特征值: {eigenvals_pca[:5]}")
        print(f"前5个奇异值²: {s[:5]**2}")
        
    def optimal_approximation_demo(self):
        """演示SVD截断的最优性（Eckart-Young定理）"""
        # 创建一个低秩矩阵加噪声
        m, n = 40, 30
        true_rank = 3
        
        # 构造真实的低秩矩阵
        A_true = np.random.randn(m, true_rank) @ np.random.randn(true_rank, n)
        
        # 添加噪声
        noise_level = 0.1
        A_noisy = A_true + noise_level * np.random.randn(m, n)
        
        # SVD分解
        U, s, Vt = np.linalg.svd(A_noisy, full_matrices=False)
        
        # 测试不同的截断秩
        ranks = range(1, min(m, n))
        svd_errors = []
        random_errors = []
        
        for k in ranks:
            # SVD最优截断
            A_svd_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            svd_error = np.linalg.norm(A_noisy - A_svd_k, 'fro')
            svd_errors.append(svd_error)
            
            # 随机低秩逼近（作为对比）
            Q, _ = np.linalg.qr(np.random.randn(m, k))
            R, _ = np.linalg.qr(np.random.randn(n, k))
            A_random_k = Q @ Q.T @ A_noisy @ R @ R.T
            random_error = np.linalg.norm(A_noisy - A_random_k, 'fro')
            random_errors.append(random_error)
        
        # 可视化最优性
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.semilogy(ranks, svd_errors, 'b-o', label='SVD Truncation (optimal)')
        plt.semilogy(ranks, random_errors, 'r--s', label='random projection')
        plt.axvline(x=true_rank, color='g', linestyle=':', label=f'true rank={true_rank}')
        plt.xlabel('truncated rank k')
        plt.ylabel('Frobenius error')
        plt.title('Optimality verification')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(s, 'bo-')
        plt.axvline(x=true_rank, color='g', linestyle=':', label=f'true rank={true_rank}')
        plt.xlabel('Singular value number')
        plt.ylabel('Singular value size')
        plt.title('Singular Value Spectrum')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"真实秩: {true_rank}")
        print(f"在真实秩处的SVD误差: {svd_errors[true_rank-1]:.4f}")
        print(f"在真实秩处的随机投影误差: {random_errors[true_rank-1]:.4f}")
        print(f"SVD优势: {random_errors[true_rank-1]/svd_errors[true_rank-1]:.2f}x")

    def cfd_projection_example(self):
        """CFD中的投影应用示例"""
        # 模拟翼型周围的流场数据
        # 创建网格
        nx, ny, nt = 50, 30, 100
        x = np.linspace(-2, 4, nx)
        y = np.linspace(-2, 2, ny)
        t = np.linspace(0, 10, nt)
        
        X, Y = np.meshgrid(x, y)
        
        # 模拟涡脱落现象（简化模型）
        flow_data = np.zeros((ny, nx, nt))
        
        for i, time in enumerate(t):
            # 主流
            u_mean = np.ones_like(X)
            
            # 涡脱落（简化）
            vortex1 = 0.5 * np.exp(-((X-1)**2 + (Y-0.3)**2)/0.1) * np.sin(2*time)
            vortex2 = -0.5 * np.exp(-((X-1.5)**2 + (Y+0.3)**2)/0.1) * np.sin(2*time + np.pi)
            
            flow_data[:, :, i] = u_mean + vortex1 + vortex2
        
        # 重排数据矩阵：(空间点 × 时间)
        data_matrix = flow_data.reshape(-1, nt)
        
        # POD分析
        U, s, Vt = np.linalg.svd(data_matrix, full_matrices=False)
        
        # 计算模态能量
        energy = s**2 / np.sum(s**2)
        cumulative_energy = np.cumsum(energy)
        
        # 选择保留90%能量的模态数
        n_modes = np.argmax(cumulative_energy >= 0.9) + 1
        
        print(f"保留90%能量需要 {n_modes} 个模态")
        print(f"压缩比: {n_modes/nt:.3f}")
        
        # 可视化结果
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始流场快照
        im1 = axes[0, 0].contourf(X, Y, flow_data[:, :, 25], levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('original flow field (t=2.5)')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 能量谱
        axes[0, 1].semilogy(energy[:20], 'bo-')
        axes[0, 1].set_title('Modal Energy Spectrum')
        axes[0, 1].set_xlabel('Modal number')
        axes[0, 1].set_ylabel('relative energy')
        axes[0, 1].grid(True)
        
        # 累积能量
        axes[0, 2].plot(cumulative_energy[:20], 'ro-')
        axes[0, 2].axhline(y=0.9, color='g', linestyle='--', label='90%')
        axes[0, 2].axvline(x=n_modes, color='g', linestyle='--')
        axes[0, 2].set_title('accumulated energy')
        axes[0, 2].set_xlabel('Modal number')
        axes[0, 2].set_ylabel('Accumulated energy ratio')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 前三个POD模态
        for i in range(3):
            mode_2d = U[:, i].reshape(ny, nx)
            im = axes[1, i].contourf(X, Y, mode_2d, levels=20, cmap='RdBu_r')
            axes[1, i].set_title(f'POD modal {i+1} (E={energy[i]:.3f})')
            axes[1, i].set_aspect('equal')
            plt.colorbar(im, ax=axes[1, i])
        
        plt.tight_layout()
        plt.show()
        
        # 重构验证
        reconstructed = U[:, :n_modes] @ np.diag(s[:n_modes]) @ Vt[:n_modes, :]
        reconstruction_error = np.linalg.norm(data_matrix - reconstructed, 'fro') / np.linalg.norm(data_matrix, 'fro')
        
        print(f"使用{n_modes}个模态的重构相对误差: {reconstruction_error:.4f}")
        
        return flow_data, U, s, Vt

# 主执行函数
def exercises():

    print("===bzhan666===\n")
    
    demo = ProjectionAndSubspaceDemo()
    
    print("1. 正交投影几何演示:")
    demo.demonstrate_orthogonal_projection()
    
    print("\n2. PCA vs POD 对比分析:")
    demo.pca_vs_pod_comparison()
    
    print("\n3. SVD最优性验证:")
    demo.optimal_approximation_demo()
    
    print("\n4. CFD投影应用示例:")
    flow_data, U, s, Vt = demo.cfd_projection_example()
    
    return flow_data, U, s, Vt

if __name__ == "__main__":
    exercises()