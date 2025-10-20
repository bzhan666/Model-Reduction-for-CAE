# -*- coding: utf-8 -*-
"""
POD重构分析优化版 
基于 EX14-2.py 的优化数据管道

主要功能：
1. 多模态数重构对比
2. 重构误差分析
3. 能量-误差权衡分析
4. 模态贡献度分析
5. 时间演化重构

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import time
from contextlib import contextmanager

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@contextmanager
def timer(description: str):
    """计时器上下文管理器"""
    start = time.time()
    yield
    end = time.time()
    print(f"{description}: {end - start:.2f}秒")

class PODReconstructor:
    """POD重构器 - 扩展PODAnalyzer的重构功能"""
    
    def __init__(self, pod_analyzer):
        """
        初始化重构器
        
        Args:
            pod_analyzer: EX14-2.py中的PODAnalyzer实例
        """
        self.analyzer = pod_analyzer
        self.data = pod_analyzer.data
        self.U = pod_analyzer.U
        self.s = pod_analyzer.s
        self.Vt = pod_analyzer.Vt
        
        if self.U is None:
            raise RuntimeError("PODAnalyzer必须先执行compute_pod()方法")
    
    def reconstruct_full_matrix(self, num_modes: int) -> np.ndarray:
        """
        使用前r个模态重构整个数据矩阵
        
        Args:
            num_modes: 使用的模态数
            
        Returns:
            重构的数据矩阵
        """
        if num_modes > len(self.s):
            raise ValueError(f"请求的模态数 {num_modes} 超过可用模态数 {len(self.s)}")
        
        print(f"  使用前 {num_modes} 个模态重构全矩阵...")
        # A'_r = U_r @ Σ_r @ V_r^T
        with timer(f"  重构计算"):
            reconstructed = self.U[:, :num_modes] @ np.diag(self.s[:num_modes]) @ self.Vt[:num_modes, :]
        
        return reconstructed
    
    def reconstruct_snapshot(self, num_modes: int, time_index: int) -> np.ndarray:
        """
        重构单个时间快照
        
        Args:
            num_modes: 使用的模态数
            time_index: 时间步索引
            
        Returns:
            重构的快照
        """
        if num_modes > len(self.s):
            raise ValueError(f"请求的模态数 {num_modes} 超过可用模态数 {len(self.s)}")
        
        reconstructed = np.zeros(self.U.shape[0])
        for i in range(num_modes):
            reconstructed += self.s[i] * self.U[:, i] * self.Vt[i, time_index]
        
        return reconstructed
    
    def compute_reconstruction_error(self, num_modes: int, 
                                     metric: str = 'frobenius') -> float:
        """
        计算重构误差
        
        Args:
            num_modes: 使用的模态数
            metric: 误差度量方式 ('frobenius', 'relative', 'max')
            
        Returns:
            误差值（百分比）
        """
        reconstructed = self.reconstruct_full_matrix(num_modes)
        
        if metric == 'frobenius':
            error = np.linalg.norm(self.data - reconstructed, 'fro') / np.linalg.norm(self.data, 'fro')
        elif metric == 'relative':
            error = np.mean(np.abs(self.data - reconstructed) / (np.abs(self.data) + 1e-10))
        elif metric == 'max':
            error = np.max(np.abs(self.data - reconstructed)) / np.max(np.abs(self.data))
        else:
            raise ValueError(f"不支持的误差度量: {metric}")
        
        return error * 100
    
    def compute_energy_ratio(self, num_modes: int) -> float:
        """计算前r个模态的能量占比"""
        return np.sum(self.analyzer.energy_content[:num_modes]) * 100
    
    def analyze_mode_contribution(self, num_modes: int, time_index: int) -> Dict:
        """
        分析各模态对重构的贡献
        
        Args:
            num_modes: 分析的模态数
            time_index: 时间步索引
            
        Returns:
            包含各模态贡献信息的字典
        """
        contributions = {}
        base_reconstruction = np.zeros(self.U.shape[0])
        
        for i in range(num_modes):
            # 计算单个模态的贡献
            mode_contribution = self.s[i] * self.U[:, i] * self.Vt[i, time_index]
            base_reconstruction += mode_contribution
            
            contributions[f'mode_{i+1}'] = {
                'field': mode_contribution,
                'amplitude': np.linalg.norm(mode_contribution),
                'temporal_coeff': self.Vt[i, time_index],
                'spatial_amplitude': np.linalg.norm(self.U[:, i]),
                'singular_value': self.s[i]
            }
        
        return contributions

class ReconstructionVisualizer:
    """重构可视化器"""
    
    @staticmethod
    def plot_reconstruction_comparison(original_snapshot: np.ndarray,
                                      reconstructed_snapshots: Dict[int, np.ndarray],
                                      coords: np.ndarray, nx: int, ny: int,
                                      time_index: int, energy_ratios: Dict[int, float],
                                      errors: Dict[int, float],
                                      output_dir: str = 'output'):
        """
        对比可视化不同模态数的重构结果
        
        Args:
            original_snapshot: 原始快照
            reconstructed_snapshots: {num_modes: reconstructed_field}
            coords: 坐标
            nx, ny: 网格尺寸
            time_index: 时间步索引
            energy_ratios: 能量占比字典
            errors: 误差字典
            output_dir: 输出目录
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        num_plots = len(reconstructed_snapshots) + 1
        n_cols = min(2, num_plots)
        n_rows = (num_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if num_plots == 1:
            axes = np.array([axes])
        axes = axes.ravel()
        
        # 准备网格
        grid_x = coords[:, 0].reshape(ny, nx)
        grid_y = coords[:, 1].reshape(ny, nx)
        
        # 统一颜色范围
        v_min = original_snapshot.min()
        v_max = original_snapshot.max()
        
        # 绘制原始场
        im = axes[0].contourf(grid_x, grid_y, original_snapshot.reshape(ny, nx),
                             levels=40, cmap='viridis', vmin=v_min, vmax=v_max)
        axes[0].set_title(f'原始脉动场\n(时间步 {time_index})', 
                         fontsize=13, fontweight='bold')
        axes[0].set_aspect('equal')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 绘制重构结果
        for idx, (num_modes, reconstructed) in enumerate(sorted(reconstructed_snapshots.items())):
            ax = axes[idx + 1]
            
            im = ax.contourf(grid_x, grid_y, reconstructed.reshape(ny, nx),
                           levels=40, cmap='viridis', vmin=v_min, vmax=v_max)
            
            title = f'重构场: {num_modes} 模态\n'
            title += f'能量: {energy_ratios[num_modes]:.2f}% | 误差: {errors[num_modes]:.3f}%'
            
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隐藏多余子图
        for i in range(num_plots, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'reconstruction_comparison.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  对比图已保存: {output_path}")
        plt.show()
    
    @staticmethod
    def plot_error_analysis(mode_range: range,
                          errors: List[float],
                          energy_ratios: List[float],
                          output_dir: str = 'output'):
        """
        绘制误差-模态数分析图
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 重构误差 vs 模态数
        axes[0, 0].semilogy(mode_range, errors, 'b-o', markersize=4, linewidth=2)
        axes[0, 0].set_xlabel('模态数', fontsize=12)
        axes[0, 0].set_ylabel('重构误差 (%)', fontsize=12)
        axes[0, 0].set_title('重构误差 vs 模态数', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 能量捕获 vs 模态数
        axes[0, 1].plot(mode_range, energy_ratios, 'r-o', markersize=4, linewidth=2)
        axes[0, 1].axhline(y=90, color='orange', linestyle='--', label='90%阈值')
        axes[0, 1].axhline(y=99, color='green', linestyle='--', label='99%阈值')
        axes[0, 1].set_xlabel('模态数', fontsize=12)
        axes[0, 1].set_ylabel('累积能量 (%)', fontsize=12)
        axes[0, 1].set_title('能量捕获 vs 模态数', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 能量-误差权衡
        axes[1, 0].plot(energy_ratios, errors, 'g-o', markersize=5, linewidth=2)
        axes[1, 0].set_xlabel('累积能量 (%)', fontsize=12)
        axes[1, 0].set_ylabel('重构误差 (%)', fontsize=12)
        axes[1, 0].set_title('能量-误差权衡曲线', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].invert_yaxis()  # 误差越小越好
        
        # 4. 误差下降率
        error_reduction = [100]  # 从100%开始
        for i in range(1, len(errors)):
            reduction = (errors[i-1] - errors[i]) / errors[i-1] * 100
            error_reduction.append(reduction)
        
        axes[1, 1].plot(mode_range, error_reduction, 'm-o', markersize=4, linewidth=2)
        axes[1, 1].set_xlabel('模态数', fontsize=12)
        axes[1, 1].set_ylabel('误差下降率 (%)', fontsize=12)
        axes[1, 1].set_title('增加模态的边际收益', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'error_analysis.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  误差分析图已保存: {output_path}")
        plt.show()
    
    @staticmethod
    def plot_mode_contributions(contributions: Dict,
                              coords: np.ndarray, nx: int, ny: int,
                              output_dir: str = 'output'):
        """可视化各模态的贡献"""
        Path(output_dir).mkdir(exist_ok=True)
        
        num_modes = len(contributions)
        n_cols = min(3, num_modes)
        n_rows = (num_modes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if num_modes == 1:
            axes = np.array([axes])
        axes = axes.ravel()
        
        grid_x = coords[:, 0].reshape(ny, nx)
        grid_y = coords[:, 1].reshape(ny, nx)
        
        for idx, (mode_name, data) in enumerate(contributions.items()):
            field = data['field'].reshape(ny, nx)
            
            vmax = np.abs(field).max()
            im = axes[idx].contourf(grid_x, grid_y, field,
                                   levels=30, cmap='RdBu_r',
                                   vmin=-vmax, vmax=vmax)
            
            title = f"{mode_name}\n"
            title += f"σ={data['singular_value']:.3f} | "
            title += f"幅值={data['amplitude']:.3f}"
            
            axes[idx].set_title(title, fontsize=11)
            axes[idx].set_aspect('equal')
            axes[idx].set_xlabel('X')
            axes[idx].set_ylabel('Y')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        for i in range(num_modes, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('各模态对重构的贡献', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = Path(output_dir) / 'mode_contributions.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  模态贡献图已保存: {output_path}")
        plt.show()

def run_reconstruction_analysis(pod_analyzer, fluctuation_matrix, mean_field,
                               coords, nx, ny, output_dir='output'):
    """
    主执行函数  重构分析
    
    Args:
        pod_analyzer: EX14-2.py的PODAnalyzer实例
        fluctuation_matrix: 脉动矩阵
        mean_field: 平均场
        coords: 坐标
        nx, ny: 网格尺寸
        output_dir: 输出目录
    """
    print("\n" + "="*70)
    print("POD重构分析")
    print("="*70)
    
    # 初始化重构器
    reconstructor = PODReconstructor(pod_analyzer)
    
    # === 任务1: 不同模态数的重构对比 ===
    print("\n任务1: 不同模态数的重构对比")
    print("-" * 70)
    
    mode_numbers = [3, 5, 10, 20]
    time_index = min(75, fluctuation_matrix.shape[1] - 1)
    
    # 过滤有效的模态数
    max_modes = len(pod_analyzer.s)
    mode_numbers = [m for m in mode_numbers if m <= max_modes]
    print(f"分析模态数: {mode_numbers}")
    print(f"检查时间步: {time_index}")
    
    original_snapshot = fluctuation_matrix[:, time_index]
    reconstructed_snapshots = {}
    energy_ratios = {}
    errors = {}
    
    for num_modes in mode_numbers:
        print(f"\n处理 {num_modes} 模态重构...")
        reconstructed = reconstructor.reconstruct_snapshot(num_modes, time_index)
        reconstructed_snapshots[num_modes] = reconstructed
        energy_ratios[num_modes] = reconstructor.compute_energy_ratio(num_modes)
        errors[num_modes] = reconstructor.compute_reconstruction_error(num_modes)
        print(f"  能量占比: {energy_ratios[num_modes]:.2f}%")
        print(f"  重构误差: {errors[num_modes]:.3f}%")
    
    # 可视化对比
    ReconstructionVisualizer.plot_reconstruction_comparison(
        original_snapshot, reconstructed_snapshots, coords, nx, ny,
        time_index, energy_ratios, errors, output_dir
    )
    
    # === 任务2: 误差-模态数系统分析 ===
    print("\n任务2: 重构误差系统分析")
    print("-" * 70)
    
    max_analyze = min(50, len(pod_analyzer.s))
    mode_range = range(1, max_analyze + 1)
    
    all_errors = []
    all_energy_ratios = []
    
    print(f"分析1到{max_analyze}个模态的重构效果...")
    with timer("误差计算"):
        for r in mode_range:
            if r % 10 == 0:
                print(f"  处理模态 {r}/{max_analyze}")
            all_errors.append(reconstructor.compute_reconstruction_error(r))
            all_energy_ratios.append(reconstructor.compute_energy_ratio(r))
    
    # 可视化误差分析
    ReconstructionVisualizer.plot_error_analysis(
        mode_range, all_errors, all_energy_ratios, output_dir
    )
    
    # 找到关键阈值
    idx_90 = next((i for i, e in enumerate(all_energy_ratios) if e >= 90), None)
    idx_99 = next((i for i, e in enumerate(all_energy_ratios) if e >= 99), None)
    
    print(f"\n关键阈值分析:")
    if idx_90:
        print(f"  90%能量: {idx_90 + 1}模态, 误差={all_errors[idx_90]:.3f}%")
    if idx_99:
        print(f"  99%能量: {idx_99 + 1}模态, 误差={all_errors[idx_99]:.3f}%")
    
    # === 任务3: 模态贡献分析 ===
    print("\n任务3: 模态贡献度分析")
    print("-" * 70)
    
    num_analyze = min(6, len(pod_analyzer.s))
    contributions = reconstructor.analyze_mode_contribution(num_analyze, time_index)
    
    print(f"分析前{num_analyze}个模态的贡献:")
    for mode_name, data in contributions.items():
        print(f"  {mode_name}: σ={data['singular_value']:.4f}, "
              f"时间系数={data['temporal_coeff']:.4f}, "
              f"幅值={data['amplitude']:.4f}")
    
    ReconstructionVisualizer.plot_mode_contributions(
        contributions, coords, nx, ny, output_dir
    )
    
    # === 生成报告 ===
    print("\n" + "="*70)
    print("分析完成!")
    print(f"所有结果已保存到 '{output_dir}' 目录")
    print("="*70)
    
    return {
        'reconstructor': reconstructor,
        'mode_numbers': mode_numbers,
        'errors': errors,
        'energy_ratios': energy_ratios,
        'all_errors': all_errors,
        'all_energy_ratios': all_energy_ratios
    }

def main():
    """主函数 - 完整流水线"""
    print("=== POD重构分析程序 ===\n")
    
    try:
        # 导入并运行EX14-2
        print("导入EX14-2模块...")
        from EX14_2 import main as ex14_main
        
        print("\n运行EX14-2数据生成和基础POD分析...")
        ex14_results = ex14_main()
        
        # 提取结果
        pod_analyzer = ex14_results['pod_analyzer']
        fluctuation_matrix = ex14_results['fluctuation_matrix']
        mean_field = ex14_results['mean_field']
        coords = ex14_results['coords']
        nx = ex14_results['nx']
        ny = ex14_results['ny']
        
        # 运行重构分析
        print("\n" + "="*70)
        reconstruction_results = run_reconstruction_analysis(
            pod_analyzer, fluctuation_matrix, mean_field,
            coords, nx, ny, output_dir='output'
        )
        
        return {**ex14_results, **reconstruction_results}
        
    except ImportError as e:
        print(f"错误: 无法导入EX14-2模块")
        print(f"详细信息: {e}")
        print("\n请确保:")
        print("1. EX14-2.py 在同一目录下")
        print("2. 文件名正确 (EX14_2.py 或 EX14-2.py)")
        raise
    except Exception as e:
        print(f"程序执行出错: {e}")
        raise

if __name__ == "__main__":
    results = main()