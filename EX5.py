# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:07:05 2025

@author: 25045
"""

import numpy as np
import time


def compare_pod_methods(n_space, n_time):
    """
    比较直接POD(SVD)和快照POD(Eig)的性能和结果
    """
    print(f"\n--- 测试 N_space={n_space}, N_time={n_time} ---")
    # 生成随机数据
    A = np.random.randn(n_space, n_time)

    # 方法1: 直接POD (基于SVD)
    start_time = time.time()
    U_svd, s_svd, _ = np.linalg.svd(A, full_matrices=False)
    end_time = time.time()
    print(f"直接POD (SVD) 耗时: {end_time - start_time:.6f} 秒")

    # 方法2: 快照POD (基于特征值分解)
    start_time = time.time()
    C_t = A.T @ A                                   # 时间协方差矩阵
    eigvals, V = np.linalg.eigh(C_t)                # 特征值分解
    idx = np.argsort(eigvals)[::-1]                 # 降序排序
    eigvals = eigvals[idx]
    V = V[:, idx]
    s_eig = np.sqrt(np.maximum(eigvals, 0.))        # 奇异值
    U_eig = A @ V @ np.diag(1.0 / (s_eig + 1e-12))  # 空间模态
    end_time = time.time()
    print(f"快照POD (Eig) 耗时: {end_time - start_time:.6f} 秒")

    # 结果验证
    k = min(n_space, n_time)                        # 可比较的最大阶数
    print("验证结果是否一致:")
    print(f"奇异值是否接近: {np.allclose(s_svd[:k], s_eig[:k])}")

    modes_match = True
    for i in range(k):
        dot_product = np.abs(U_svd[:, i] @ U_eig[:, i])
        if not np.isclose(dot_product, 1.0, atol=1e-4):
            modes_match = False
            print(f"模态 {i} 不匹配! 内积绝对值: {dot_product}")
            break
    print(f"空间模态是否一致: {modes_match}")
    return U_svd, U_eig


def exercises():
    print("=== bzhan666 ===\n")

    # 案例1: 时间步 > 空间点 (SVD应有优势)
    compare_pod_methods(n_space=100, n_time=200)

    # 案例2: 空间点 > 时间步 (快照法应有优势)
    compare_pod_methods(n_space=2000, n_time=100)

    # 案例3: 模拟真实CFD场景 (空间点 >> 时间步)
    compare_pod_methods(n_space=10000, n_time=150)
if __name__ == "__main__":
    exercises()

