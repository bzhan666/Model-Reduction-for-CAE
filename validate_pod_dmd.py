# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 21:35:33 2025

@author: bzhan666
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pydmd import DMD # 导入基准模型

# 导入我们的模型和数据工具
try:
    from pod_dmd_model_v2 import POD_DMD_Model
    from EX14_2 import FluentDataGenerator, FluentDataReader, FlowVisualizer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 pod_dmd_model_v2.py 和 EX14_2.py 在同一目录下。")
    # 提供桩代码以便静态分析
    class POD_DMD_Model: pass
    class FluentDataGenerator: pass
    class FluentDataReader: pass
    class FlowVisualizer: pass

def predict_with_pydmd(dmd_instance, num_steps: int) -> np.ndarray:
    """
    手动使用PyDMD的组件进行标准的自回归预测。
    
    Args:
        dmd_instance: 已经fit好的PyDMD对象
        num_steps: 要预测的时间步数
        
    Returns:
        预测出的高维快照矩阵
    """
    # 1. 获取DMD的核心组件
    modes = dmd_instance.modes           # DMD模态 (Φ)
    eigs = dmd_instance.eigs             # DMD特征值 (λ)
    amplitudes = dmd_instance.amplitudes # 初始振幅 (b)
    
    # 2. 构建时间演化矩阵
    # vander的increasing=True会生成 [1, λ, λ^2, ...], 我们需要 [λ^0, λ^1, ...]
    # 注意，PyDMD的amplitudes是基于第一个快照的，所以我们的预测是从t=0开始的
    # 我们需要的是从训练结束(t_end)开始的预测
    # 首先，找到训练结束时(t_end-1)的振幅
    last_train_step_amps = amplitudes * (eigs ** (dmd_instance.dmd_time['t0'] - 1))
    
    # 现在从这个振幅开始演化
    time_dynamics = np.zeros((len(eigs), num_steps), dtype=np.complex128)
    for i in range(num_steps):
        time_dynamics[:, i] = last_train_step_amps * (eigs ** i)
        
    # 3. 重构高维快照
    predicted_snapshots = modes @ time_dynamics
    
    return predicted_snapshots.real


def main():
    print("="*70)
    print("POD-DMD 模型 与 纯DMD 模型对比验证 ")
    print("="*70)
    
    # 1. 准备数据
    DATA_PATH = Path('./fluent_data')
    NX, NY, NT = 80, 40, 100
    if not DATA_PATH.exists():
        FluentDataGenerator(NX, NY, NT).generate_karman_vortex_street(DATA_PATH)
    reader = FluentDataReader(DATA_PATH)
    snapshot_matrix, coords = reader.read_snapshots(nt=NT)
    
    # 2. 划分训练/测试数据
    split_idx = int(0.8 * NT)
    train_data = snapshot_matrix[:, :split_idx]
    test_data = snapshot_matrix[:, split_idx:]
    
    # 3. 定义模型参数
    SVD_RANK = 20
    num_predictions = test_data.shape[1]
    initial_condition = train_data[:, -1]
    
    # --- 模型1: 我们的 POD-DMD ---
    print("\n--- 训练并预测: POD-DMD 模型 ---")
    pod_dmd = POD_DMD_Model(n_pod_modes=SVD_RANK)
    pod_dmd.fit(train_data)
    pod_dmd_predictions = pod_dmd.predict(initial_condition, num_predictions)
    pod_dmd.plot_eigs()
    
    # --- 模型2: 纯DMD (使用PyDMD作为基准) ---
    print("\n--- 训练并预测: 纯DMD (PyDMD) 模型 ---")
    dmd_pure = DMD(svd_rank=SVD_RANK)
    dmd_pure.fit(train_data)
    
    # 使用我们修正后的预测函数
    dmd_pure_predictions = predict_with_pydmd(dmd_pure, num_predictions)
    
    # 4. 对比验证
    print("\n" + "="*70)
    print("对比验证结果")
    print("="*70)
    
    # 检查维度是否一致
    print(f"test_data shape: {test_data.shape}")
    print(f"pod_dmd_predictions shape: {pod_dmd_predictions.shape}")
    print(f"dmd_pure_predictions shape: {dmd_pure_predictions.shape}")
    
    # 确保维度一致后再计算误差
    if test_data.shape == dmd_pure_predictions.shape:
        error_pod_dmd = np.linalg.norm(test_data - pod_dmd_predictions, 'fro') / np.linalg.norm(test_data, 'fro')
        error_dmd_pure = np.linalg.norm(test_data - dmd_pure_predictions, 'fro') / np.linalg.norm(test_data, 'fro')
        
        print(f"\nPOD-DMD 预测总相对误差: {error_pod_dmd:.4%}")
        print(f"纯DMD   预测总相对误差: {error_dmd_pure:.4%}")
        
        if error_pod_dmd < error_dmd_pure:
            print("\n  POD-DMD 预测精度更高或相当。POD的去噪/压缩起到了积极作用。")
        else:
            print("\n  纯DMD 预测精度更高。可能POD截断损失了部分重要的动力学信息。")
            
        # 可视化对比
        visualizer = FlowVisualizer()
        fields_to_plot = {
            '真实场 (最后一步)': test_data[:, -1],
            'POD-DMD 预测': pod_dmd_predictions[:, -1],
            '纯DMD (PyDMD) 预测': dmd_pure_predictions[:, -1]
        }
        visualizer.plot_flow_comparison(fields_to_plot, coords, NX, NY)
    else:
        print("\n 错误: 预测结果与真实数据的维度不匹配，无法进行比较。")

if __name__ == "__main__":
    main()