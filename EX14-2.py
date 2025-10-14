# -*- coding: utf-8 -*-
"""
优化的流体POD分析代码
Created on Thu Sep 25 11:03:22 2025

主要优化：
1. 代码结构优化和模块化
2. 性能优化（内存和计算效率）
3. 错误处理和参数验证
4. 更丰富的分析功能
5. 更好的可视化效果

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
from typing import Tuple, Optional, Union
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

class FluentDataGenerator:
    """流体数据生成器类"""
    
    def __init__(self, nx: int = 80, ny: int = 40, nt: int = 100):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        
    def generate_karman_vortex_street(self, path: Union[str, Path] = './fluent_data') -> Tuple[int, int, int]:
        """
        生成卡门涡街流场数据
        
        Args:
            path: 数据保存路径
            
        Returns:
            (nx, ny, nt): 网格尺寸和时间步数
        """
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        print(f"正在生成模拟数据到 '{path}' 目录...")
        
        # 创建网格
        x = np.linspace(-2, 14, self.nx)
        y = np.linspace(-4, 4, self.ny)
        X, Y = np.meshgrid(x, y)
        
        # 添加圆柱体几何
        cylinder_center = (2, 0)
        cylinder_radius = 0.5
        cylinder_mask = (X - cylinder_center[0])**2 + (Y - cylinder_center[1])**2 <= cylinder_radius**2
        
        with timer("数据生成"):
            for i in range(self.nt):
                t = i * 0.1
                
                # 改进的涡街模拟（更真实的物理行为）
                # 主流速度
                u_base = np.ones_like(X)
                
                # 涡脱落频率和强度
                vortex_freq = 0.2
                vortex_strength = 0.3
                downstream_decay = 0.1
                
                # 上下涡的位置和强度
                vortex_y_offset = 1.0
                phase_shift = np.pi
                
                # 涡脱落模拟
                for n in range(1, 4):  # 多个涡对
                    x_vortex = 4 + n * 3
                    
                    # 上涡
                    vortex_upper = vortex_strength * np.exp(
                        -(((X - x_vortex)**2 + (Y - vortex_y_offset * np.sin(vortex_freq * t + n * phase_shift))**2) / (1 + downstream_decay * (X - 2)))
                    )
                    
                    # 下涡（相位相反）
                    vortex_lower = -vortex_strength * np.exp(
                        -(((X - x_vortex)**2 + (Y + vortex_y_offset * np.sin(vortex_freq * t + n * phase_shift + np.pi))**2) / (1 + downstream_decay * (X - 2)))
                    )
                    
                    u_base += vortex_upper + vortex_lower
                
                # 在圆柱体内部设置边界条件
                u_base[cylinder_mask] = 0
                
                # 添加一些随机扰动来模拟湍流
                if i > 20:  # 让前面几步稳定
                    noise = 0.02 * np.random.normal(size=u_base.shape)
                    u_base += noise
                
                # 保存数据
                filename = path / f'flow_t_{i:04d}.dat'
                self._save_data_file(filename, X, Y, u_base, t)
                
                if i % 20 == 0:
                    print(f"  已生成 {i+1}/{self.nt} 个时间步")
        
        print("数据生成完毕。")
        return self.nx, self.ny, self.nt
    
    @staticmethod
    def _save_data_file(filename: Path, X: np.ndarray, Y: np.ndarray, U: np.ndarray, t: float):
        """保存单个数据文件"""
        with open(filename, 'w') as f:
            f.write("VARIABLES = X, Y, U\n")
            f.write(f"ZONE T=time_{t:.2f}\n")
            
            # 向量化写入，提高效率
            data = np.column_stack([X.ravel(), Y.ravel(), U.ravel()])
            np.savetxt(f, data, fmt='%.6f')

class FluentDataReader:
    """流体数据读取器类"""
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.path}")
    
    def read_snapshots(self, nt: int, field_index: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取并组装快照矩阵
        
        Args:
            nt: 快照总数
            field_index: 变量列索引 (0:X, 1:Y, 2:U)
            
        Returns:
            (snapshot_matrix, coords): 快照矩阵和坐标
        """
        print("开始读取数据并组装快照矩阵...")
        
        # 从第一个文件获取网格信息
        first_file = self.path / 'flow_t_0000.dat'
        if not first_file.exists():
            raise FileNotFoundError(f"第一个数据文件不存在: {first_file}")
        
        with timer("读取网格信息"):
            grid_data = np.loadtxt(first_file, skiprows=2)
            n_space = grid_data.shape[0]
            coords = grid_data[:, :2]
        
        # 预分配内存
        snapshot_matrix = np.zeros((n_space, nt), dtype=np.float32)  # 使用float32节省内存
        
        with timer("读取所有时间步数据"):
            for i in range(nt):
                filename = self.path / f'flow_t_{i:04d}.dat'
                if not filename.exists():
                    warnings.warn(f"文件不存在，跳过: {filename}")
                    continue
                    
                if i % 20 == 0:
                    print(f"  正在读取: {filename.name} ({i+1}/{nt})")
                
                data = np.loadtxt(filename, skiprows=2)
                snapshot_matrix[:, i] = data[:, field_index].astype(np.float32)
        
        print(f"快照矩阵组装完毕。尺寸: {snapshot_matrix.shape}")
        return snapshot_matrix, coords

class DataPreprocessor:
    """数据预处理器类"""
    
    @staticmethod
    def mean_subtraction(snapshot_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """均值减法预处理"""
        print("执行均值减法预处理...")
        
        with timer("计算时间平均场"):
            mean_field = np.mean(snapshot_matrix, axis=1, keepdims=True)
        
        with timer("计算脉动场"):
            fluctuation_matrix = snapshot_matrix - mean_field
        
        print("数据预处理完成。")
        return fluctuation_matrix, mean_field.ravel()
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
        """
        数据标准化
        
        Args:
            data: 输入数据
            method: 标准化方法 ('standard', 'minmax', 'robust')
        """
        if method == 'standard':
            mean_val = np.mean(data, axis=1, keepdims=True)
            std_val = np.std(data, axis=1, keepdims=True)
            normalized = (data - mean_val) / (std_val + 1e-8)
            params = {'mean': mean_val, 'std': std_val}
        elif method == 'minmax':
            min_val = np.min(data, axis=1, keepdims=True)
            max_val = np.max(data, axis=1, keepdims=True)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            params = {'min': min_val, 'max': max_val}
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        return normalized, params

class PODAnalyzer:
    """POD分析器类"""
    
    def __init__(self, data_matrix: np.ndarray):
        self.data = data_matrix.astype(np.float64)  # 确保精度
        self.U = None
        self.s = None
        self.Vt = None
        self.energy_content = None
        self.cumulative_energy = None
        
    def compute_pod(self, rank: Optional[int] = None, energy_threshold: float = 0.99):
        """
        计算POD模态
        
        Args:
            rank: 截断秩，None表示自动确定
            energy_threshold: 能量阈值，用于自动确定截断秩
        """
        print("\n开始执行POD分析...")
        
        with timer("SVD分解"):
            # 对于大矩阵，使用随机SVD加速
            if self.data.shape[0] > 5000 and self.data.shape[1] > 100:
                from sklearn.decomposition import TruncatedSVD
                n_components = min(50, self.data.shape[1] - 1)
                svd = TruncatedSVD(n_components=n_components)
                self.U = svd.fit_transform(self.data)
                self.s = svd.singular_values_
                self.Vt = svd.components_
            else:
                self.U, self.s, self.Vt = np.linalg.svd(self.data, full_matrices=False)
        
        # 计算能量分布
        self.energy_content = self.s**2
        total_energy = np.sum(self.energy_content)
        self.energy_content = self.energy_content / total_energy
        self.cumulative_energy = np.cumsum(self.energy_content)
        
        # 自动确定截断秩
        if rank is None:
            rank = np.argmax(self.cumulative_energy >= energy_threshold) + 1
            print(f"自动选择前 {rank} 个模态（能量占比 {energy_threshold*100:.1f}%）")
        
        # 截断
        if rank > 0:
            self.U = self.U[:, :rank]
            self.s = self.s[:rank]
            self.Vt = self.Vt[:rank, :]
            
        print(f"POD分析完成。保留 {len(self.s)} 个模态。")
        
    def plot_energy_analysis(self, save_fig: bool = False):
        """绘制能量分析图"""
        if self.s is None:
            raise RuntimeError("请先调用 compute_pod() 方法")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 奇异值
        axes[0, 0].semilogy(self.s, 'bo-', markersize=4)
        axes[0, 0].set_title('奇异值衰减')
        axes[0, 0].set_xlabel('模态序号')
        axes[0, 0].set_ylabel('奇异值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 归一化能量
        axes[0, 1].semilogy(self.energy_content, 'ro-', markersize=4)
        axes[0, 1].set_title('归一化模态能量')
        axes[0, 1].set_xlabel('模态序号')
        axes[0, 1].set_ylabel('能量占比')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 累积能量
        axes[1, 0].plot(self.cumulative_energy, 'go-', markersize=4)
        axes[1, 0].axhline(y=0.9, color='r', linestyle='--', label='90%')
        axes[1, 0].axhline(y=0.99, color='orange', linestyle='--', label='99%')
        axes[1, 0].set_title('累积能量占比')
        axes[1, 0].set_xlabel('模态数量')
        axes[1, 0].set_ylabel('累积能量占比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 前几个模态的时间系数
        if self.Vt is not None:
            num_plot = min(4, self.Vt.shape[0])
            time_indices = np.arange(self.Vt.shape[1])
            for i in range(num_plot):
                axes[1, 1].plot(time_indices, self.Vt[i, :], label=f'模态 {i+1}')
            axes[1, 1].set_title('前几个模态的时间系数')
            axes[1, 1].set_xlabel('时间步')
            axes[1, 1].set_ylabel('时间系数')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('pod_energy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_modes(self, coords: np.ndarray, nx: int, ny: int, 
                       num_modes: int = 6, save_fig: bool = False):
        """可视化POD模态"""
        if self.U is None:
            raise RuntimeError("请先调用 compute_pod() 方法")
            
        num_modes = min(num_modes, self.U.shape[1])
        print(f"可视化前 {num_modes} 个POD空间模态...")
        
        # 创建子图
        rows = 2
        cols = (num_modes + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 8))
        axes = axes.flatten() if num_modes > 1 else [axes]
        
        # 重构网格
        grid_x = coords[:, 0].reshape(ny, nx)
        grid_y = coords[:, 1].reshape(ny, nx)
        
        for i in range(num_modes):
            mode_field = self.U[:, i].reshape(ny, nx)
            
            # 自适应颜色范围
            vmax = np.abs(mode_field).max()
            vmin = -vmax
            
            im = axes[i].contourf(grid_x, grid_y, mode_field, 
                                levels=30, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[i].set_title(f'模态 {i+1}\n能量占比: {self.energy_content[i]:.1%}', 
                            fontsize=12)
            axes[i].set_aspect('equal')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[i], orientation='horizontal', 
                        pad=0.1, shrink=0.8)
        
        # 隐藏多余的子图
        for i in range(num_modes, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('主导POD模态', fontsize=16)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('pod_modes.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def reconstruct_field(self, mode_indices: Optional[list] = None, 
                         time_index: int = -1) -> np.ndarray:
        """
        重构流场
        
        Args:
            mode_indices: 使用的模态索引，None表示使用所有模态
            time_index: 时间步索引
            
        Returns:
            重构的流场
        """
        if self.U is None or self.Vt is None:
            raise RuntimeError("请先调用 compute_pod() 方法")
        
        if mode_indices is None:
            mode_indices = list(range(len(self.s)))
        
        # 重构
        reconstructed = np.zeros(self.U.shape[0])
        for i in mode_indices:
            reconstructed += self.s[i] * self.U[:, i] * self.Vt[i, time_index]
        
        return reconstructed

class FlowVisualizer:
    """流场可视化器类"""
    
    @staticmethod
    def plot_flow_comparison(fields_dict: dict, coords: np.ndarray, 
                           nx: int, ny: int, save_fig: bool = False):
        """对比显示多个流场"""
        num_fields = len(fields_dict)
        fig, axes = plt.subplots(1, num_fields, figsize=(6*num_fields, 5))
        if num_fields == 1:
            axes = [axes]
        
        grid_x = coords[:, 0].reshape(ny, nx)
        grid_y = coords[:, 1].reshape(ny, nx)
        
        for idx, (title, field) in enumerate(fields_dict.items()):
            field_2d = field.reshape(ny, nx)
            
            im = axes[idx].contourf(grid_x, grid_y, field_2d, 
                                  levels=50, cmap='viridis')
            axes[idx].set_title(title)
            axes[idx].set_aspect('equal')
            axes[idx].set_xlabel('X')
            axes[idx].set_ylabel('Y')
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('flow_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数 - 完整的POD分析流水线"""
    
    print("=== 流体POD分析程序 ===\n")
    
    # 参数设置
    DATA_PATH = './fluent_data'
    NX, NY, NT = 80, 40, 100
    
    try:
        # 1. 数据生成
        print("步骤1: 数据生成")
        generator = FluentDataGenerator(nx=NX, ny=NY, nt=NT)
        nx, ny, nt = generator.generate_karman_vortex_street(path=DATA_PATH)
        
        # 2. 数据读取
        print("\n步骤2: 数据读取")
        reader = FluentDataReader(path=DATA_PATH)
        snapshot_matrix, coords = reader.read_snapshots(nt=nt)
        
        # 3. 数据预处理
        print("\n步骤3: 数据预处理")
        fluctuation_matrix, mean_field = DataPreprocessor.mean_subtraction(snapshot_matrix)
        
        # 4. 初始可视化
        print("\n步骤4: 初始流场可视化")
        visualizer = FlowVisualizer()
        fields_to_plot = {
            '时间平均场': mean_field,
            '瞬时流场': snapshot_matrix[:, -1],
            '脉动流场': fluctuation_matrix[:, -1]
        }
        visualizer.plot_flow_comparison(fields_to_plot, coords, nx, ny)
        
        # 5. POD分析
        print("\n步骤5: POD分析")
        pod_analyzer = PODAnalyzer(fluctuation_matrix)
        pod_analyzer.compute_pod(energy_threshold=0.95)
        
        # 6. 结果分析和可视化
        print("\n步骤6: 结果分析")
        pod_analyzer.plot_energy_analysis()
        pod_analyzer.visualize_modes(coords, nx, ny, num_modes=6)
        
        # 7. 流场重构演示
        print("\n步骤7: 流场重构演示")
        # 使用前3个模态重构
        reconstructed = pod_analyzer.reconstruct_field(mode_indices=[0, 1, 2], time_index=-1)
        reconstruction_fields = {
            '原始脉动场': fluctuation_matrix[:, -1],
            '3模态重构': reconstructed,
            '重构误差': fluctuation_matrix[:, -1] - reconstructed
        }
        visualizer.plot_flow_comparison(reconstruction_fields, coords, nx, ny)
        
        print("\n=== POD分析完成 ===")
        
        return {
            'pod_analyzer': pod_analyzer,
            'fluctuation_matrix': fluctuation_matrix,
            'mean_field': mean_field,
            'coords': coords,
            'nx': nx,
            'ny': ny
        }
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        raise

if __name__ == "__main__":
    results = main()
