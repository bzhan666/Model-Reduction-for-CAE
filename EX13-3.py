# -*- coding: utf-8 -*-
"""
优化的流体数据处理代码 (EX13)
Created on Thu Sep 25 11:03:22 2025

主要优化：
1. 面向对象设计，提高代码可维护性
2. 错误处理和参数验证
3. 性能优化和内存管理
4. 更好的可视化效果
5. 日志和进度显示

@author: bzhan666
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from pathlib import Path
import warnings
from typing import Tuple, Optional, Union
import time
from contextlib import contextmanager

# ===== 强制中文字体设置 =====
def setup_chinese_font_force():
    """强制设置中文字体"""
    # 方法1：清除matplotlib缓存
    try:
        matplotlib.font_manager._rebuild()
    except:
        pass
    
    # 方法2：多重字体设置
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 方法3：强制设置字体属性
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['SimHei', 'Microsoft YaHei'],
        'axes.unicode_minus': False,
        'figure.autolayout': True
    })
    
    print("✅ 中文字体设置完成")

# 执行字体设置
setup_chinese_font_force()

# 设置图表样式
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

@contextmanager
def timer(description: str):
    """计时器上下文管理器"""
    start = time.time()
    print(f"开始 {description}...")
    yield
    end = time.time()
    print(f"完成 {description} - 耗时: {end - start:.2f}秒")

class FluentDataSimulator:
    """Fluent数据模拟器类"""
    
    def __init__(self, nx: int = 80, ny: int = 40, nt: int = 100):
        """
        初始化模拟器
        
        Args:
            nx: x方向网格点数
            ny: y方向网格点数  
            nt: 时间步数
        """
        self.nx = max(10, nx)  # 参数验证
        self.ny = max(10, ny)
        self.nt = max(1, nt)
        
        # 创建计算网格
        self.x = np.linspace(-2, 14, self.nx)
        self.y = np.linspace(-4, 4, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        print(f"初始化模拟器 - 网格尺寸: {self.nx}×{self.ny}, 时间步: {self.nt}")

    def generate_karman_vortex_data(self, output_path: Union[str, Path] = './fluent_data') -> Tuple[int, int, int]:
        """
        生成卡门涡街流场数据
        
        Args:
            output_path: 输出路径
            
        Returns:
            (nx, ny, nt): 网格和时间参数
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"正在生成卡门涡街数据到: {output_path}")
        
        with timer("卡门涡街数据生成"):
            # 预计算一些常量以提高性能
            dt = 0.1
            vortex_strength = 0.5
            decay_rate = 0.05
            
            for i in range(self.nt):
                t = i * dt
                
                # 改进的涡街物理模型
                u_velocity = self._compute_velocity_field(t, vortex_strength, decay_rate)
                
                # 保存数据文件
                filename = output_path / f'flow_t_{i:04d}.dat'
                self._save_fluent_format(filename, self.X, self.Y, u_velocity, t)
                
                # 进度显示
                if i % max(1, self.nt // 10) == 0:
                    progress = (i + 1) / self.nt * 100
                    print(f"  进度: {progress:.1f}% ({i+1}/{self.nt})")
        
        print("数据生成完成!")
        return self.nx, self.ny, self.nt

    def _compute_velocity_field(self, t: float, strength: float, decay: float) -> np.ndarray:
        """
        计算更真实的速度场
        
        Args:
            t: 当前时间
            strength: 涡强度
            decay: 衰减率
        """
        # 基础流速
        u_base = np.ones_like(self.X)
        
        # 圆柱体几何 (可选)
        cylinder_x, cylinder_y = 2.0, 0.0
        cylinder_radius = 0.5
        
        # 涡脱落参数
        vortex_frequency = 0.2
        shedding_amplitude = 1.2
        downstream_positions = [4, 7, 10, 13]  # 多个涡的下游位置
        
        # 生成交替涡
        for i, x_pos in enumerate(downstream_positions):
            phase = vortex_frequency * t + i * np.pi  # 交替相位
            
            # 上涡
            y_upper = shedding_amplitude * np.sin(phase)
            vortex_upper = strength * np.exp(
                -(((self.X - x_pos)**2 + (self.Y - y_upper)**2) * 
                  (1 + decay * (self.X - cylinder_x)))
            )
            
            # 下涡 (相位差π)
            y_lower = -shedding_amplitude * np.sin(phase + np.pi)
            vortex_lower = -strength * np.exp(
                -(((self.X - x_pos)**2 + (self.Y - y_lower)**2) * 
                  (1 + decay * (self.X - cylinder_x)))
            )
            
            u_base += vortex_upper + vortex_lower
        
        # 添加边界条件和噪声
        mask = (self.X - cylinder_x)**2 + (self.Y - cylinder_y)**2 <= cylinder_radius**2
        u_base[mask] = 0  # 圆柱内部无滑移边界条件
        
        # 添加小扰动模拟湍流
        if t > 2.0:  # 初始阶段保持稳定
            noise = 0.01 * np.random.normal(size=u_base.shape)
            u_base += noise
            
        return u_base

    @staticmethod
    def _save_fluent_format(filename: Path, X: np.ndarray, Y: np.ndarray, 
                           U: np.ndarray, time: float):
        """保存Fluent格式数据文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("VARIABLES = X, Y, U\n")
                f.write(f"ZONE T=time_{time:.2f}\n")
                
                # 向量化数据写入
                data_array = np.column_stack([X.ravel(), Y.ravel(), U.ravel()])
                np.savetxt(f, data_array, fmt='%.6f')
        except IOError as e:
            raise IOError(f"无法保存文件 {filename}: {e}")

class FluentDataLoader:
    """Fluent数据加载器类"""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        初始化加载器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = Path(data_path)
        self._validate_path()
        
    def _validate_path(self):
        """验证数据路径"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")
        if not self.data_path.is_dir():
            raise NotADirectoryError(f"路径不是目录: {self.data_path}")

    def load_snapshot_matrix(self, num_snapshots: int, 
                           field_index: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载并组装快照矩阵
        
        Args:
            num_snapshots: 快照数量
            field_index: 字段索引 (0:X, 1:Y, 2:U)
            
        Returns:
            (snapshot_matrix, coordinates): 快照矩阵和坐标信息
        """
        if not isinstance(num_snapshots, int) or num_snapshots <= 0:
            raise ValueError("快照数量必须是正整数")
        if not 0 <= field_index <= 2:
            raise ValueError("字段索引必须在0-2之间")
            
        print(f"开始加载 {num_snapshots} 个快照...")
        
        # 从第一个文件获取网格信息
        first_file = self.data_path / 'flow_t_0000.dat'
        if not first_file.exists():
            raise FileNotFoundError(f"第一个数据文件不存在: {first_file}")
        
        with timer("网格信息读取"):
            grid_info = self._load_single_file(first_file)
            spatial_points = grid_info.shape[0]
            coordinates = grid_info[:, :2].copy()
        
        # 预分配内存 - 使用float32节省内存
        snapshot_matrix = np.zeros((spatial_points, num_snapshots), dtype=np.float32)
        
        with timer("批量数据加载"):
            successful_loads = 0
            for i in range(num_snapshots):
                filename = self.data_path / f'flow_t_{i:04d}.dat'
                
                try:
                    data = self._load_single_file(filename)
                    snapshot_matrix[:, i] = data[:, field_index].astype(np.float32)
                    successful_loads += 1
                except (FileNotFoundError, ValueError) as e:
                    warnings.warn(f"跳过文件 {filename}: {e}")
                    continue
                
                # 进度显示
                if i % max(1, num_snapshots // 10) == 0:
                    progress = (i + 1) / num_snapshots * 100
                    print(f"  加载进度: {progress:.1f}% ({i+1}/{num_snapshots})")
        
        if successful_loads == 0:
            raise RuntimeError("没有成功加载任何数据文件")
        
        print(f"成功加载 {successful_loads}/{num_snapshots} 个快照")
        print(f"快照矩阵尺寸: {snapshot_matrix.shape}")
        
        return snapshot_matrix, coordinates

    @staticmethod
    def _load_single_file(filename: Path) -> np.ndarray:
        """加载单个数据文件"""
        try:
            return np.loadtxt(filename, skiprows=2)
        except (OSError, ValueError) as e:
            raise ValueError(f"文件格式错误: {filename}") from e

class DataProcessor:
    """数据处理器类"""
    
    @staticmethod
    def compute_mean_subtraction(snapshot_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算均值减法预处理
        
        Args:
            snapshot_matrix: 原始快照矩阵
            
        Returns:
            (fluctuation_matrix, mean_field): 脉动矩阵和时间平均场
        """
        if snapshot_matrix.size == 0:
            raise ValueError("输入矩阵为空")
            
        print("执行数据预处理...")
        
        with timer("时间平均计算"):
            # 计算时间平均场
            mean_field = np.mean(snapshot_matrix, axis=1)
            
        with timer("脉动场计算"):
            # 计算脉动场 (原场减去平均场)
            fluctuation_matrix = snapshot_matrix - mean_field[:, np.newaxis]
        
        # 计算统计信息
        rms_fluctuation = np.sqrt(np.mean(fluctuation_matrix**2))
        print(f"脉动场RMS: {rms_fluctuation:.4f}")
        
        return fluctuation_matrix, mean_field

    @staticmethod
    def compute_statistics(data: np.ndarray) -> dict:
        """计算数据统计信息"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'rms': np.sqrt(np.mean(data**2))
        }

class FlowFieldVisualizer:
    """流场可视化器类"""
    
    def __init__(self, figsize: Tuple[int, int] = (18, 5)):
        self.figsize = figsize
        
    def plot_flow_fields(self, fields_data: dict, coordinates: np.ndarray, 
                        nx: int, ny: int, save_path: Optional[str] = None):
        """
        绘制多个流场对比图
        
        Args:
            fields_data: 包含不同流场数据的字典
            coordinates: 网格坐标
            nx, ny: 网格尺寸
            save_path: 保存路径（可选）
        """
        # 再次确保字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        num_fields = len(fields_data)
        if num_fields == 0:
            raise ValueError("没有提供流场数据")
            
        fig, axes = plt.subplots(1, num_fields, figsize=(6*num_fields, 5))
        if num_fields == 1:
            axes = [axes]
        
        # 重构网格
        grid_x = coordinates[:, 0].reshape(ny, nx)
        grid_y = coordinates[:, 1].reshape(ny, nx)
        
        for idx, (title, field_data) in enumerate(fields_data.items()):
            field_2d = field_data.reshape(ny, nx)
            
            # 创建等高线图
            if '脉动' in title or 'fluctuation' in title.lower():
                # 脉动场使用对称颜色映射
                vmax = np.abs(field_2d).max()
                levels = np.linspace(-vmax, vmax, 50)
                contour = axes[idx].contourf(grid_x, grid_y, field_2d, 
                                           levels=levels, cmap='RdBu_r')
            else:
                # 其他场使用常规颜色映射
                contour = axes[idx].contourf(grid_x, grid_y, field_2d, 
                                           levels=50, cmap='viridis')
            
            # 设置标题和标签 - 显式设置字体
            axes[idx].set_title(title, fontsize=14, fontweight='bold', 
                              fontproperties='SimHei')
            axes[idx].set_xlabel('X坐标', fontsize=12, fontproperties='SimHei')
            axes[idx].set_ylabel('Y坐标', fontsize=12, fontproperties='SimHei')
            axes[idx].set_aspect('equal')
            
            # 添加颜色条
            cbar = plt.colorbar(contour, ax=axes[idx])
            cbar.ax.tick_params(labelsize=10)
            
            # 添加网格
            axes[idx].grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle('流场对比分析', fontsize=16, fontweight='bold',
                    fontproperties='SimHei')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()

    def plot_statistics_summary(self, data_dict: dict):
        """绘制数据统计摘要"""
        # 再次确保字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        field_names = list(data_dict.keys())
        statistics = [DataProcessor.compute_statistics(data) for data in data_dict.values()]
        
        # 均值对比
        means = [stat['mean'] for stat in statistics]
        bars1 = axes[0, 0].bar(field_names, means)
        axes[0, 0].set_title('时间平均值对比', fontproperties='SimHei')
        axes[0, 0].set_ylabel('平均值', fontproperties='SimHei')
        axes[0, 0].tick_params(axis='x', rotation=45)
        # 设置x轴标签字体
        for label in axes[0, 0].get_xticklabels():
            label.set_fontproperties('SimHei')
        
        # 标准差对比
        stds = [stat['std'] for stat in statistics]
        bars2 = axes[0, 1].bar(field_names, stds, color='orange')
        axes[0, 1].set_title('标准差对比', fontproperties='SimHei')
        axes[0, 1].set_ylabel('标准差', fontproperties='SimHei')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for label in axes[0, 1].get_xticklabels():
            label.set_fontproperties('SimHei')
        
        # 极值对比
        mins = [stat['min'] for stat in statistics]
        maxs = [stat['max'] for stat in statistics]
        x_pos = np.arange(len(field_names))
        bars3 = axes[1, 0].bar(x_pos, maxs, label='最大值', alpha=0.7)
        bars4 = axes[1, 0].bar(x_pos, mins, label='最小值', alpha=0.7)
        axes[1, 0].set_title('极值对比', fontproperties='SimHei')
        axes[1, 0].set_ylabel('数值', fontproperties='SimHei')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(field_names, rotation=45)
        for label in axes[1, 0].get_xticklabels():
            label.set_fontproperties('SimHei')
        # 设置图例字体
        legend = axes[1, 0].legend(prop='SimHei')
        
        # RMS对比
        rms_values = [stat['rms'] for stat in statistics]
        bars5 = axes[1, 1].bar(field_names, rms_values, color='green')
        axes[1, 1].set_title('RMS值对比', fontproperties='SimHei')
        axes[1, 1].set_ylabel('RMS', fontproperties='SimHei')
        axes[1, 1].tick_params(axis='x', rotation=45)
        for label in axes[1, 1].get_xticklabels():
            label.set_fontproperties('SimHei')
        
        plt.tight_layout()
        plt.show()

def run_complete_analysis(data_path: str = './fluent_data', 
                         nx: int = 80, ny: int = 40, nt: int = 100,
                         regenerate_data: bool = True) -> dict:
    """
    运行完整的流体数据分析流水线
    
    Args:
        data_path: 数据路径
        nx, ny, nt: 网格和时间参数
        regenerate_data: 是否重新生成数据
        
    Returns:
        包含所有分析结果的字典
    """
    print("=" * 50)
    print("流体数据处理与可视化分析程序")
    print("=" * 50)
    
    results = {}
    
    try:
        # 步骤1: 数据生成
        if regenerate_data:
            print("\n📊 步骤1: 数据生成")
            simulator = FluentDataSimulator(nx=nx, ny=ny, nt=nt)
            nx, ny, nt = simulator.generate_karman_vortex_data(output_path=data_path)
            results.update({'nx': nx, 'ny': ny, 'nt': nt})
        
        # 步骤2: 数据加载
        print("\n📁 步骤2: 数据加载")
        loader = FluentDataLoader(data_path)
        snapshot_matrix, coordinates = loader.load_snapshot_matrix(num_snapshots=nt)
        results.update({
            'snapshot_matrix': snapshot_matrix,
            'coordinates': coordinates
        })
        
        # 步骤3: 数据预处理
        print("\n⚙️ 步骤3: 数据预处理")
        processor = DataProcessor()
        fluctuation_matrix, mean_field = processor.compute_mean_subtraction(snapshot_matrix)
        results.update({
            'fluctuation_matrix': fluctuation_matrix,
            'mean_field': mean_field
        })
        
        # 步骤4: 可视化分析
        print("\n📈 步骤4: 可视化分析")
        visualizer = FlowFieldVisualizer()
        
        # 流场对比
        flow_fields = {
            '时间平均流场': mean_field,
            '瞬时流场': snapshot_matrix[:, -1],
            '瞬时脉动流场': fluctuation_matrix[:, -1]
        }
        visualizer.plot_flow_fields(flow_fields, coordinates, nx, ny)
        
        # 统计分析
        visualizer.plot_statistics_summary(flow_fields)
        
        print("\n✅ 分析完成!")
        print(f"📊 数据矩阵尺寸: {snapshot_matrix.shape}")
        print(f"📍 空间点数: {coordinates.shape[0]}")
        print(f"⏱️ 时间步数: {nt}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 程序执行错误: {e}")
        raise

if __name__ == "__main__":
    # 运行完整分析
    analysis_results = run_complete_analysis(
        data_path='./fluent_data',
        nx=80, 
        ny=40, 
        nt=100,
        regenerate_data=True
    )
    
    print("\n" + "=" * 50)
    print("分析结果已保存在 analysis_results 变量中")
    print("=" * 50)