# -*- coding: utf-8 -*-

"""
完整的POD降阶模型(ROM)工具 
sklearn风格的fit/transform/inverse_transform接口

主要功能：
1. 符合scikit-learn风格的API设计
2. 完整的编码-解码流程
3. 多种误差评估方法
4. 训练集/测试集验证
5. 在线预测和实时重构

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict
import time
from contextlib import contextmanager
import warnings

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


class PODROM:
    """
    POD降阶模型(Reduced Order Model)
    
    符合scikit-learn风格的API设计，提供完整的降阶建模能力
    
    Attributes:
        n_modes: 保留的模态数量
        mean_field: 时间平均场
        modes: POD空间模态 (U_r)
        singular_values: 奇异值
        temporal_modes: 时间模态 (V_r)
        is_fitted: 是否已训练标志
    """
    
    def __init__(self, n_modes: int, svd_method: str = 'standard'):
        """
        初始化POD-ROM
        
        Args:
            n_modes: 要保留的模态数量 (r)
            svd_method: SVD计算方法 ('standard', 'randomized')
        """
        if n_modes <= 0:
            raise ValueError("n_modes must be a positive integer.")
        
        self.n_modes = n_modes
        self.svd_method = svd_method
        
        # 模型参数
        self.mean_field = None
        self.modes = None  # U_r
        self.singular_values = None
        self.temporal_modes = None  # V_r^T
        
        # 训练数据信息
        self.n_space = None
        self.n_time_train = None
        self.fluctuation_energy = None
        
        # 状态标志
        self.is_fitted = False
    
    def fit(self, snapshot_matrix: np.ndarray, verbose: bool = True) -> 'PODROM':
        """
        根据训练数据计算平均场和POD模态
        
        Args:
            snapshot_matrix: 原始快照矩阵 A (n_space, n_time)
            verbose: 是否打印详细信息
            
        Returns:
            self: 返回自身以支持链式调用
        """
        if verbose:
            print("="*60)
            print("Fitting POD-ROM...")
            print("="*60)
        
        # 验证输入
        if snapshot_matrix.ndim != 2:
            raise ValueError("snapshot_matrix must be 2D array")
        
        self.n_space, self.n_time_train = snapshot_matrix.shape
        
        if self.n_modes > min(self.n_space, self.n_time_train):
            warnings.warn(f"n_modes ({self.n_modes}) exceeds min dimension "
                        f"({min(self.n_space, self.n_time_train)}), will be truncated")
            self.n_modes = min(self.n_modes, self.n_time_train)
        
        # 1. 计算并存储平均场
        with timer("  计算时间平均场"):
            self.mean_field = np.mean(snapshot_matrix, axis=1)
        
        # 2. 计算脉动场
        with timer("  计算脉动场"):
            fluctuation_matrix = snapshot_matrix - self.mean_field[:, np.newaxis]
            self.fluctuation_energy = np.linalg.norm(fluctuation_matrix, 'fro')**2
        
        # 3. SVD分解
        with timer(f"  SVD分解 (方法: {self.svd_method})"):
            if self.svd_method == 'randomized' and self.n_space > 5000:
                from sklearn.utils.extmath import randomized_svd
                U, s, Vt = randomized_svd(fluctuation_matrix, 
                                         n_components=self.n_modes,
                                         random_state=42)
            else:
                U, s, Vt = np.linalg.svd(fluctuation_matrix, full_matrices=False)
                U = U[:, :self.n_modes]
                s = s[:self.n_modes]
                Vt = Vt[:self.n_modes, :]
        
        # 4. 存储结果
        self.modes = U
        self.singular_values = s
        self.temporal_modes = Vt
        self.is_fitted = True
        
        # 5. 计算并显示能量信息
        captured_energy = np.sum(s**2)
        energy_ratio = captured_energy / self.fluctuation_energy * 100
        
        if verbose:
            print(f"\n训练完成!")
            print(f"  空间维度: {self.n_space}")
            print(f"  时间步数: {self.n_time_train}")
            print(f"  保留模态: {self.n_modes}")
            print(f"  能量占比: {energy_ratio:.2f}%")
            print(f"  压缩比: {self.n_space * self.n_time_train / (self.n_space * self.n_modes + self.n_modes + self.n_modes * self.n_time_train):.1f}x")
        
        return self
    
    def transform(self, snapshot: np.ndarray) -> np.ndarray:
        """
        将高维快照投影到低维空间 (编码)
        
        Args:
            snapshot: 高维流场向量或矩阵
                     - 1D: (n_space,) 单个快照
                     - 2D: (n_space, n_snapshots) 多个快照
                     
        Returns:
            modal_coeffs: 低维模态系数
                         - 1D: (n_modes,) 单个快照的系数
                         - 2D: (n_modes, n_snapshots) 多个快照的系数
        """
        self._check_fitted()
        
        # 处理输入维度
        is_single = snapshot.ndim == 1
        if is_single:
            snapshot = snapshot[:, np.newaxis]
        
        if snapshot.shape[0] != self.n_space:
            raise ValueError(f"Input dimension mismatch: expected {self.n_space}, got {snapshot.shape[0]}")
        
        # 投影: a = U_r^T @ (x - x_mean)
        fluctuation = snapshot - self.mean_field[:, np.newaxis]
        modal_coeffs = self.modes.T @ fluctuation
        
        return modal_coeffs.ravel() if is_single else modal_coeffs
    
    def inverse_transform(self, modal_coeffs: np.ndarray) -> np.ndarray:
        """
        从低维系数重构高维快照 (解码)
        
        Args:
            modal_coeffs: 低维模态系数
                         - 1D: (n_modes,) 单个快照的系数
                         - 2D: (n_modes, n_snapshots) 多个快照的系数
                         
        Returns:
            reconstructed: 重构的高维流场
                          - 1D: (n_space,) 单个快照
                          - 2D: (n_space, n_snapshots) 多个快照
        """
        self._check_fitted()
        
        # 处理输入维度
        is_single = modal_coeffs.ndim == 1
        if is_single:
            modal_coeffs = modal_coeffs[:, np.newaxis]
        
        if modal_coeffs.shape[0] != self.n_modes:
            raise ValueError(f"Coefficients dimension mismatch: expected {self.n_modes}, got {modal_coeffs.shape[0]}")
        
        # 重构: x_hat = x_mean + U_r @ a
        reconstructed = self.mean_field[:, np.newaxis] + self.modes @ modal_coeffs
        
        return reconstructed.ravel() if is_single else reconstructed
    
    def fit_transform(self, snapshot_matrix: np.ndarray) -> np.ndarray:
        """训练并转换 (sklearn风格)"""
        self.fit(snapshot_matrix)
        return self.transform(snapshot_matrix)
    
    def reconstruct(self, snapshot_matrix: np.ndarray) -> np.ndarray:
        """
        完整的重构流程: 编码 -> 解码
        
        Args:
            snapshot_matrix: 输入快照矩阵
            
        Returns:
            重构后的快照矩阵
        """
        modal_coeffs = self.transform(snapshot_matrix)
        return self.inverse_transform(modal_coeffs)
    
    def compute_reconstruction_error(self, snapshot_matrix: np.ndarray,
                                    metric: str = 'relative') -> float:
        """
        计算重构误差
        
        Args:
            snapshot_matrix: 原始快照矩阵
            metric: 误差度量 ('relative', 'absolute', 'max')
            
        Returns:
            误差值
        """
        self._check_fitted()
        
        reconstructed = self.reconstruct(snapshot_matrix)
        
        if metric == 'relative':
            error = np.linalg.norm(snapshot_matrix - reconstructed, 'fro') / \
                   np.linalg.norm(snapshot_matrix, 'fro')
        elif metric == 'absolute':
            error = np.linalg.norm(snapshot_matrix - reconstructed, 'fro')
        elif metric == 'max':
            error = np.max(np.abs(snapshot_matrix - reconstructed))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return error
    
    def evaluate_reconstruction_error_vs_modes(self, 
                                              max_modes: Optional[int] = None,
                                              save_fig: bool = False):
        """
        评估不同模态数下的理论重构误差
        
        Args:
            max_modes: 最大模态数，None表示使用全部
            save_fig: 是否保存图片
        """
        self._check_fitted()
        
        if max_modes is None:
            max_modes = len(self.singular_values)
        else:
            max_modes = min(max_modes, len(self.singular_values))
        
        # 计算能量和误差
        energies = self.singular_values**2
        total_energy = np.sum(energies)
        cumulative_energy = np.cumsum(energies) / total_energy
        
        # 相对误差: sqrt(1 - E_r/E_total)
        reconstruction_error = np.sqrt(1 - cumulative_energy)
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1: 重构误差
        modes_range = range(1, max_modes + 1)
        axes[0].semilogy(modes_range, reconstruction_error[:max_modes], 'b-o', markersize=4)
        axes[0].set_title('相对重构误差 vs 模态数', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('模态数 (r)', fontsize=11)
        axes[0].set_ylabel('相对误差 (Frobenius范数)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 添加关键阈值线
        error_thresholds = [0.01, 0.05, 0.1]
        colors = ['g', 'orange', 'r']
        for thresh, color in zip(error_thresholds, colors):
            idx = np.argmax(reconstruction_error < thresh)
            if idx > 0:
                axes[0].axhline(y=thresh, color=color, linestyle='--', 
                              label=f'{thresh*100:.0f}%误差: {idx+1}模态', alpha=0.7)
        axes[0].legend()
        
        # 子图2: 累积能量
        axes[1].plot(modes_range, cumulative_energy[:max_modes] * 100, 'r-o', markersize=4)
        axes[1].axhline(y=90, color='orange', linestyle='--', label='90%阈值', alpha=0.7)
        axes[1].axhline(y=99, color='green', linestyle='--', label='99%阈值', alpha=0.7)
        axes[1].set_title('累积能量 vs 模态数', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('模态数 (r)', fontsize=11)
        axes[1].set_ylabel('累积能量 (%)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('rom_error_analysis.png', dpi=200, bbox_inches='tight')
            print("图片已保存: rom_error_analysis.png")
        
        plt.show()
        
        # 打印关键信息
        print("\n关键阈值分析:")
        for energy_pct in [90, 95, 99]:
            idx = np.argmax(cumulative_energy >= energy_pct/100)
            if idx > 0:
                print(f"  {energy_pct}%能量: {idx+1}模态, 误差={reconstruction_error[idx]:.4f}")
    
    def validate_on_test_set(self, test_matrix: np.ndarray, 
                           visualize: bool = True,
                           coords: Optional[np.ndarray] = None,
                           nx: Optional[int] = None,
                           ny: Optional[int] = None):
        """
        在测试集上验证ROM性能
        
        Args:
            test_matrix: 测试快照矩阵
            visualize: 是否可视化
            coords, nx, ny: 可视化所需的网格信息
        """
        self._check_fitted()
        
        print("\n" + "="*60)
        print("测试集验证")
        print("="*60)
        
        n_test = test_matrix.shape[1]
        
        # 计算整体误差
        with timer("计算测试集重构误差"):
            error_relative = self.compute_reconstruction_error(test_matrix, 'relative')
            error_max = self.compute_reconstruction_error(test_matrix, 'max')
        
        # 计算每个快照的误差
        snapshot_errors = []
        for i in range(n_test):
            snap = test_matrix[:, i]
            recon = self.reconstruct(snap)
            err = np.linalg.norm(snap - recon) / np.linalg.norm(snap)
            snapshot_errors.append(err)
        
        print(f"\n测试结果:")
        print(f"  测试快照数: {n_test}")
        print(f"  平均相对误差: {error_relative:.4%}")
        print(f"  最大绝对误差: {error_max:.6f}")
        print(f"  单快照误差范围: [{min(snapshot_errors):.4%}, {max(snapshot_errors):.4%}]")
        
        # 可视化
        if visualize:
            self._visualize_test_results(test_matrix, snapshot_errors, 
                                        coords, nx, ny)
    
    def _visualize_test_results(self, test_matrix: np.ndarray,
                               snapshot_errors: List[float],
                               coords: Optional[np.ndarray],
                               nx: Optional[int],
                               ny: Optional[int]):
        """可视化测试结果"""
        fig = plt.figure(figsize=(16, 10))
        
        # 布局: 2行3列
        # 第一行: 原始场、重构场、误差场
        # 第二行: 误差分布直方图、时间演化、压缩示意图
        
        # 选择一个中间快照进行可视化
        idx = test_matrix.shape[1] // 2
        original = test_matrix[:, idx]
        reconstructed = self.reconstruct(original)
        error_field = original - reconstructed
        
        # 如果有网格信息,绘制2D场
        if coords is not None and nx is not None and ny is not None:
            grid_x = coords[:, 0].reshape(ny, nx)
            grid_y = coords[:, 1].reshape(ny, nx)
            
            # 原始场
            ax1 = plt.subplot(2, 3, 1)
            im1 = ax1.contourf(grid_x, grid_y, original.reshape(ny, nx),
                              levels=40, cmap='viridis')
            ax1.set_title(f'原始场 (快照 {idx})', fontsize=12, fontweight='bold')
            ax1.set_aspect('equal')
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            # 重构场
            ax2 = plt.subplot(2, 3, 2)
            im2 = ax2.contourf(grid_x, grid_y, reconstructed.reshape(ny, nx),
                              levels=40, cmap='viridis')
            ax2.set_title(f'重构场 ({self.n_modes}模态)', fontsize=12, fontweight='bold')
            ax2.set_aspect('equal')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            # 误差场
            ax3 = plt.subplot(2, 3, 3)
            vmax = np.abs(error_field).max()
            im3 = ax3.contourf(grid_x, grid_y, error_field.reshape(ny, nx),
                              levels=40, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax3.set_title('误差场', fontsize=12, fontweight='bold')
            ax3.set_aspect('equal')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 误差统计直方图
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(snapshot_errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(snapshot_errors), color='r', linestyle='--', 
                   label=f'平均: {np.mean(snapshot_errors):.4f}')
        ax4.set_xlabel('相对误差', fontsize=11)
        ax4.set_ylabel('频数', fontsize=11)
        ax4.set_title('测试集误差分布', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 误差时间演化
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(snapshot_errors, 'b-o', markersize=3)
        ax5.set_xlabel('快照索引', fontsize=11)
        ax5.set_ylabel('相对误差', fontsize=11)
        ax5.set_title('误差时间演化', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 压缩示意图
        ax6 = plt.subplot(2, 3, 6)
        categories = ['原始数据', 'ROM表示']
        original_size = self.n_space * test_matrix.shape[1]
        rom_size = self.n_space * self.n_modes + self.n_modes * test_matrix.shape[1]
        sizes = [original_size, rom_size]
        compression_ratio = original_size / rom_size
        
        bars = ax6.bar(categories, sizes, color=['coral', 'lightgreen'], alpha=0.7)
        ax6.set_ylabel('存储元素数', fontsize=11)
        ax6.set_title(f'数据压缩 (压缩比: {compression_ratio:.1f}x)', 
                     fontsize=12, fontweight='bold')
        
        # 在柱子上添加数值
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size/1e6:.2f}M' if size > 1e6 else f'{size/1e3:.1f}K',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('rom_test_validation.png', dpi=200, bbox_inches='tight')
        print("\n验证结果已保存: rom_test_validation.png")
        plt.show()
    
    def get_compression_ratio(self, n_snapshots: int) -> float:
        """
        计算数据压缩比
        
        Args:
            n_snapshots: 快照数量
            
        Returns:
            压缩比
        """
        self._check_fitted()
        
        original_size = self.n_space * n_snapshots
        rom_size = (self.n_space * self.n_modes +  # 空间模态
                   self.n_modes * n_snapshots +     # 时间系数
                   self.n_space)                     # 平均场
        
        return original_size / rom_size
    
    def _check_fitted(self):
        """检查模型是否已训练"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
    
    def __repr__(self):
        if self.is_fitted:
            return (f"PODROM(n_modes={self.n_modes}, "
                   f"fitted=True, "
                   f"n_space={self.n_space}, "
                   f"n_time_train={self.n_time_train})")
        else:
            return f"PODROM(n_modes={self.n_modes}, fitted=False)"


def run_rom_exercises(snapshot_matrix_raw: np.ndarray,
                     coords: Optional[np.ndarray] = None,
                     nx: Optional[int] = None,
                     ny: Optional[int] = None):
    """
    主执行函数 ROM练习
    
    Args:
        snapshot_matrix_raw: 原始快照矩阵
        coords, nx, ny: 网格信息(用于可视化)
    """
    print("\n" + "="*70)
    print(" POD降阶模型(ROM)完整流程")
    print("="*70)
    
    # === 任务1: 训练ROM ===
    print("\n任务1: 训练POD-ROM模型")
    print("-" * 70)
    
    n_modes = 20
    rom = PODROM(n_modes=n_modes, svd_method='standard')
    rom.fit(snapshot_matrix_raw, verbose=True)
    
    # === 任务2: 评估理论误差 ===
    print("\n任务2: 评估重构误差 vs 模态数")
    print("-" * 70)
    
    rom.evaluate_reconstruction_error_vs_modes(max_modes=50, save_fig=True)
    
    # === 任务3: 测试编码-解码流程 ===
    print("\n任务3: 编码-解码流程测试")
    print("-" * 70)
    
    # 选择一个测试快照
    test_idx = 90
    test_snapshot = snapshot_matrix_raw[:, test_idx]
    
    print(f"测试快照: 第{test_idx}个时间步")
    print(f"  原始维度: {test_snapshot.shape}")
    
    # 编码
    with timer("编码(投影到低维空间)"):
        low_dim_coeffs = rom.transform(test_snapshot)
    
    print(f"  低维系数维度: {low_dim_coeffs.shape}")
    print(f"  降维比: {test_snapshot.shape[0] / low_dim_coeffs.shape[0]:.1f}x")
    print(f"  前5个系数: {low_dim_coeffs[:5].round(3)}")
    
    # 解码
    with timer("解码(重构到高维空间)"):
        reconstructed_snapshot = rom.inverse_transform(low_dim_coeffs)
    
    print(f"  重构维度: {reconstructed_snapshot.shape}")
    
    # 验证误差
    error_relative = np.linalg.norm(test_snapshot - reconstructed_snapshot) / \
                    np.linalg.norm(test_snapshot)
    error_max = np.max(np.abs(test_snapshot - reconstructed_snapshot))
    
    print(f"\n单快照重构性能:")
    print(f"  相对误差 (L2): {error_relative:.6f} ({error_relative*100:.4f}%)")
    print(f"  最大绝对误差: {error_max:.6e}")
    
    # === 任务4: 训练集验证 ===
    print("\n任务4: 训练集整体重构验证")
    print("-" * 70)
    
    train_error = rom.compute_reconstruction_error(snapshot_matrix_raw, 'relative')
    print(f"训练集平均相对误差: {train_error:.6f} ({train_error*100:.4f}%)")
    
    # === 任务5: 测试集验证(如果有的话) ===
    print("\n任务5: 模拟测试集验证")
    print("-" * 70)
    
    # 用后10%的数据作为"测试集"
    split_idx = int(0.9 * snapshot_matrix_raw.shape[1])
    test_matrix = snapshot_matrix_raw[:, split_idx:]
    
    rom.validate_on_test_set(test_matrix, visualize=True, 
                            coords=coords, nx=nx, ny=ny)
    
    # === 任务6: 压缩性能分析 ===
    print("\n任务6: 数据压缩性能")
    print("-" * 70)
    
    compression_ratio = rom.get_compression_ratio(snapshot_matrix_raw.shape[1])
    print(f"压缩比: {compression_ratio:.2f}x")
    print(f"存储节省: {(1 - 1/compression_ratio)*100:.1f}%")
    
    print("\n" + "="*70)
    print("ROM分析完成!")
    print("="*70)
    
    return rom


def main():
    """主函数"""
    print("=== POD降阶模型(ROM)工具 ===\n")
    
    try:
        # 尝试从EX14-2导入数据
        print("方法1: 从EX14-2导入数据...")
        try:
            from EX14_2 import main as ex14_main
            
            print("运行EX14-2...")
            ex14_results = ex14_main()
            
            # 组合脉动矩阵和平均场得到原始数据
            fluctuation_matrix = ex14_results['fluctuation_matrix']
            mean_field = ex14_results['mean_field']
            snapshot_matrix_raw = fluctuation_matrix + mean_field[:, np.newaxis]
            
            coords = ex14_results['coords']
            nx = ex14_results['nx']
            ny = ex14_results['ny']
            
        except ImportError:
            print("无法导入EX14-2,尝试方法2...")
            
            # 尝试从EX14导入
            from EX14 import exercises_0
            
            print("运行EX14...")
            fluctuation_matrix, mean_field, coords, nx, ny = exercises_0()
            snapshot_matrix_raw = fluctuation_matrix + mean_field[:, np.newaxis]
        
        # 运行ROM练习
        rom = run_rom_exercises(snapshot_matrix_raw, coords, nx, ny)
        
        return rom
        
    except ImportError as e:
        print(f"\n错误: 无法导入所需模块")
        print(f"详细信息: {e}")
        print("\n请确保以下文件之一在同一目录:")
        print("  - EX14-2.py (或 EX14_2.py)")
        print("  - EX14.py")
        raise
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        raise


if __name__ == "__main__":
    rom_model = main()