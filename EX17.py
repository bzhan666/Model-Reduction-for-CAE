# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:29:52 2025

本程序实现了基于POD和DMD的动态降阶模型(Reduced Order Model, ROM)，用于流场
数据的时空预测。该模型结合了POD的空间降维能力和DMD的时间动力学学习能力。

主要功能:
----------
1. POD-DMD混合降阶模型
   - 使用POD进行空间模态分解和降维
   - 在低维空间中使用DMD学习时间演化规律
   - 实现从初始条件出发的多步预测

2. 模型训练与预测
   - 训练集/测试集划分
   - 低维动力学算子学习
   - 动态预测误差分析

3. 可视化与评估
   - 预测误差随时间演化曲线
   - 真实场与预测场的对比
   - 关键时间步的流场快照
   
@author: 25045
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import contextmanager
import time
from typing import Tuple, Optional, Union, List, Dict

# 假设你的第六章代码保存在 EX16.py
# 如果文件名不同，请修改这里的导入
try:
    from EX16 import PODROM
except ImportError:
    print("错误: 无法从 EX16.py 导入 PODROM 类。请确保文件存在且在Python路径中。")
    # 如果导入失败，提供一个最小化的桩代码以便后续代码能运行
    class PODROM: pass

# 假设你的第五章代码保存在 EX14_2.py
try:
    from EX14_2 import FluentDataGenerator, FluentDataReader
except ImportError:
    print("错误: 无法从 EX14_2.py 导入数据处理类。")
    # 桩代码
    class FluentDataGenerator: pass
    class FluentDataReader: pass

@contextmanager
def timer(description: str):
    """计时器上下文管理器"""
    start = time.time()
    yield
    end = time.time()
    print(f"{description}: {end - start:.2f}秒")

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class POD_DMD_Model:
    """
    基于POD和DMD的动态降阶模型
    
    这个模型使用POD进行空间降维，然后在低维空间中使用DMD学习时间动态。
    """
    
    def __init__(self, n_pod_modes: int):
        """
        初始化模型
        
        Args:
            n_pod_modes: 用于空间降维的POD模态数量
        """
        self.r = n_pod_modes
        self.pod_rom = PODROM(n_modes=n_pod_modes)
        self.A_tilde = None  # 低维演化算子 (DMD operator)
        self.is_fitted = False

    def fit(self, snapshot_matrix: np.ndarray, verbose: bool = True):
        """
        训练POD-DMD模型
        
        Args:
            snapshot_matrix: 原始快照矩阵 A (n_space, n_time)
            verbose: 是否打印详细信息
        """
        if verbose:
            print("\n" + "="*60)
            print("开始训练 POD-DMD 模型...")
            print("="*60)

        # 1. 训练POD-ROM以获得空间基
        with timer("  步骤1: 训练POD空间基"):
            self.pod_rom.fit(snapshot_matrix, verbose=False)
        
        # 2. 将所有训练快照投影到低维空间
        with timer("  步骤2: 投影到低维空间"):
            low_dim_series = self.pod_rom.transform(snapshot_matrix)
        
        # 3. 在低维空间应用DMD
        with timer("  步骤3: 在低维空间学习动力学 (DMD)"):
            X1 = low_dim_series[:, :-1]
            X2 = low_dim_series[:, 1:]
            
            # 使用SVD来稳定地计算伪逆和演化算子 (更鲁棒的方法)
            U, s, Vt = np.linalg.svd(X1, full_matrices=False)
            self.A_tilde = X2 @ Vt.T @ np.diag(1. / s) @ U.T
            
        self.is_fitted = True
        if verbose:
            print("\n模型训练完成!")
            print(f"  POD模态数: {self.r}")
            print(f"  低维动力学算子 A_tilde 的维度: {self.A_tilde.shape}")

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
            current_a = self.A_tilde @ current_a
            
        # 3. 将预测的低维序列重构回高维空间
        predicted_snapshots = self.pod_rom.inverse_transform(predicted_low_dim_series)
        
        return predicted_snapshots

    def _check_fitted(self):
        """检查模型是否已训练"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

# 主执行函数
def exercises(snapshot_matrix_raw: np.ndarray) -> Tuple[POD_DMD_Model, np.ndarray, np.ndarray]:
    """
    运行POD-DMD模型的训练流程
    """
    # 1. 划分训练集和测试集 (80% 训练, 20% 测试)
    split_idx = int(0.8 * snapshot_matrix_raw.shape[1])
    train_data = snapshot_matrix_raw[:, :split_idx]
    test_data = snapshot_matrix_raw[:, split_idx:]
    
    print(f"数据划分: {train_data.shape[1]} 步用于训练, {test_data.shape[1]} 步用于测试。")
    
    # 2. 实例化并训练模型
    # 选择一个合适的POD模态数，例如20
    pod_dmd_model = POD_DMD_Model(n_pod_modes=20)
    pod_dmd_model.fit(train_data)
    
    return pod_dmd_model, train_data, test_data

# (接续上面的代码)
def evaluate_dynamic_prediction(model: POD_DMD_Model, 
                                train_data: np.ndarray, 
                                test_data: np.ndarray,
                                save_fig: bool = True):
    """
    评估并可视化POD-DMD模型的动态预测性能
    
    Args:
        model: 训练好的POD_DMD_Model实例
        train_data: 训练数据
        test_data: 测试数据 (真实值)
        save_fig: 是否保存图片
    """
    print("\n" + "="*60)
    print("评估动态预测性能...")
    print("="*60)

    # 1. 进行预测
    # 使用训练集的最后一个快照作为预测的初始条件
    initial_condition = train_data[:, -1]
    num_predictions = test_data.shape[1]
    
    with timer("执行预测"):
        predicted_snapshots = model.predict(initial_condition, num_predictions)

    # 2. 计算逐点误差演化
    errors = []
    for i in range(num_predictions):
        true_snapshot = test_data[:, i]
        pred_snapshot = predicted_snapshots[:, i]
        error = np.linalg.norm(true_snapshot - pred_snapshot) / np.linalg.norm(true_snapshot)
        errors.append(error)
    
    print(f"预测 {num_predictions} 步后的最终相对误差: {errors[-1]:.2%}")
        
    # 3. 可视化
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    # a) 误差演化曲线
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(range(num_predictions), np.array(errors) * 100, 'r-o', markersize=4)
    ax1.set_title('预测误差随时间演化', fontsize=14, fontweight='bold')
    ax1.set_xlabel('预测时间步', fontsize=12)
    ax1.set_ylabel('相对误差 (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # b) 关键时间步的流场对比
    steps_to_show = [0, num_predictions // 2, num_predictions - 1]
    
    # 假设有网格信息（实际应用中需要传入）
    # 这里我们只做占位符，因为没有直接传入coords, nx, ny
    def plot_field(ax, field_data, title):
        # 简单的imshow可视化
        ny, nx = 40, 80 # 硬编码，应从外部传入
        im = ax.imshow(field_data.reshape(ny, nx), cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    for i, step in enumerate(steps_to_show):
        ax = fig.add_subplot(gs[1, i])
        
        # 拼接真实场和预测场
        true_field = test_data[:, step]
        pred_field = predicted_snapshots[:, step]
        
        # 为了并排对比，我们将它们reshape并拼接
        ny, nx = 40, 80
        combined_field = np.vstack([
            true_field.reshape(ny, nx),
            np.full((5, nx), np.nan), # 分割线
            pred_field.reshape(ny, nx)
        ])
        
        im = ax.imshow(combined_field, cmap='viridis', aspect='auto')
        ax.set_title(f'时间步: T_start + {step+1}', fontsize=12, fontweight='bold')
        ax.set_yticks([ny/2, ny + 5 + ny/2])
        ax.set_yticklabels(['真实场', '预测场'])
        ax.set_xticks([])
    
    plt.suptitle('POD-DMD 动态预测结果评估', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_fig:
        plt.savefig("pod_dmd_prediction_analysis.png", dpi=200)
        print("评估图已保存: pod_dmd_prediction_analysis.png")
        
    plt.show()

def main():
    """主函数，执行完整的POD-DMD流程"""
    print("=== POD-DMD 动态预测模型流水线 ===\n")
    
    # 1. 准备数据
    DATA_PATH = Path('./fluent_data')
    NX, NY, NT = 80, 40, 100
    
    if not DATA_PATH.exists():
        print("数据不存在，正在生成...")
        generator = FluentDataGenerator(nx=NX, ny=NY, nt=NT)
        generator.generate_karman_vortex_street(path=DATA_PATH)
    
    reader = FluentDataReader(path=DATA_PATH)
    snapshot_matrix_raw, coords = reader.read_snapshots(nt=NT)
    
    # 2. 训练模型
    pod_dmd_model, train_data, test_data = exercises(snapshot_matrix_raw)
    
    # 3. 评估预测
    # 注意：为了更好的可视化，实际应传入 coords, nx, ny
    evaluate_dynamic_prediction(pod_dmd_model, train_data, test_data)

if __name__ == "__main__":
    main()
