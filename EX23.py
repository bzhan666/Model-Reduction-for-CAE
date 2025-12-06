# -*- coding: utf-8 -*-
"""
多物理场耦合POD

1. 正确处理平均场
2. 完整的POD流程(去均值→SVD→重构+均值)
3. 添加详细误差分析

@author: bzhan666
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MultiPhysicsDataGenerator:
    """生成流热耦合的模拟数据"""
    def __init__(self, nx=80, ny=40, nt=100):
        self.nx, self.ny, self.nt = nx, ny, nt
        
    def generate_data(self):
        """生成速度场和温度场"""
        print("生成流热耦合数据...")
        x = np.linspace(0, 8, self.nx)
        y = np.linspace(-2, 2, self.ny)
        X, Y = np.meshgrid(x, y)
        
        u_field = np.zeros((self.nx * self.ny, self.nt))
        T_field = np.zeros((self.nx * self.ny, self.nt))
        
        for i in range(self.nt):
            t = i * 0.2
            # 速度场 (涡脱落)
            vortex_1 = np.exp(-((X - 2 - t*0.5)**2 + (Y - 0.5*np.sin(t))**2)*2)
            vortex_2 = -np.exp(-((X - 3 - t*0.5)**2 + (Y + 0.5*np.sin(t))**2)*2)
            u_snap = 1.0 + (vortex_1 + vortex_2) * 2.0
            
            # 温度场
            T_base = np.exp(-((X - 2 - t*0.5)**2 + Y**2)*0.5) * 300.0
            
            u_field[:, i] = u_snap.ravel()
            T_field[:, i] = T_base.ravel() + 273.15
            
        return u_field, T_field, X, Y


class CoupledPOD:
    """正确实现的多物理场耦合POD"""
    def __init__(self, n_modes=10):
        self.n_modes = n_modes
        self.scalers = {}
        self.field_dims = {}
        self.var_names = []
        
        # POD核心参数
        self.U = None
        self.s = None
        self.Vt = None
        self.mean_combined = None  # 关键：保存拼接后的平均场
        
    def fit(self, data_dict: Dict[str, np.ndarray]):
        """
        训练耦合POD模型
        
        Args:
            data_dict: {'velocity': (N_u, nt), 'temperature': (N_T, nt)}
        """
        print("\n" + "="*60)
        print("多物理场耦合POD训练")
        print("="*60)
        
        # === 步骤1: 归一化每个场 ===
        processed_data_list = []
        self.var_names = list(data_dict.keys())
        
        print("\1.数据预处理:")
        for name, data in data_dict.items():
            self.field_dims[name] = data.shape[0]
            
            # 计算统计量
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            std[std < 1e-10] = 1.0
            
            self.scalers[name] = {'mean': mean, 'std': std}
            
            # 标准化: z = (x - μ) / σ
            norm_data = (data - mean) / std
            processed_data_list.append(norm_data)
            
            print(f"  {name:12s}: [{data.min():8.2f}, {data.max():8.2f}] "
                  f"→ [{norm_data.min():6.2f}, {norm_data.max():6.2f}]")
        
        # === 步骤2: 拼接多个场 ===
        combined_matrix = np.vstack(processed_data_list)
        print(f"\2.拼接矩阵: {combined_matrix.shape}")
        
        # === 步骤3: 计算时间平均场并去均值 ⭐ ===
        print("\3.POD标准流程:")
        self.mean_combined = np.mean(combined_matrix, axis=1, keepdims=True)
        fluctuation_matrix = combined_matrix - self.mean_combined
        print(f"  时间平均场: {self.mean_combined.shape}")
        print(f"  波动场: {fluctuation_matrix.shape}")
        
        # === 步骤4: SVD分解 ===
        U, s, Vt = np.linalg.svd(fluctuation_matrix, full_matrices=False)
        self.U = U[:, :self.n_modes]
        self.s = s[:self.n_modes]
        self.Vt = Vt[:self.n_modes, :]
        
        # 能量占比
        total_energy = np.sum(s**2)
        captured_energy = np.sum(self.s**2)
        energy_ratio = captured_energy / total_energy * 100
        
        print(f"  保留模态: {self.n_modes}")
        print(f"  能量占比: {energy_ratio:.2f}%")
        print("\n 训练完成!")
        
    def reconstruct(self, r=None):
        """
        重构并反归一化
        
        Args:
            r: 使用的模态数,None表示使用全部训练的模态
        """
        if r is None:
            r = self.n_modes
        if r > self.n_modes:
            raise ValueError(f"r={r} 超过训练的模态数 {self.n_modes}")
            
        # === POD标准重构: X_recon = mean + U_r @ Σ_r @ V_r^T ===
        # 注意: 这是在归一化空间的重构
        fluctuation_recon = self.U[:, :r] @ np.diag(self.s[:r]) @ self.Vt[:r, :]
        recon_combined = self.mean_combined + fluctuation_recon  # ⭐ 加回平均场
        
        # === 拆分并反归一化 ===
        results = {}
        start_row = 0
        
        for name in self.var_names:
            n_rows = self.field_dims[name]
            end_row = start_row + n_rows
            
            # 提取归一化的重构数据
            norm_recon = recon_combined[start_row:end_row, :]
            
            # 反归一化: x = z * σ + μ
            scaler = self.scalers[name]
            orig_recon = norm_recon * scaler['std'] + scaler['mean']
            
            results[name] = orig_recon
            start_row = end_row
            
        return results
    
    def compute_errors(self, data_dict: Dict[str, np.ndarray], r=None):
        """计算重构误差"""
        recon_results = self.reconstruct(r)
        errors = {}
        
        for name, orig_data in data_dict.items():
            recon_data = recon_results[name]
            
            # 相对L2误差
            rel_error = np.linalg.norm(orig_data - recon_data, 'fro') / \
                       np.linalg.norm(orig_data, 'fro')
            
            # 最大绝对误差
            max_error = np.max(np.abs(orig_data - recon_data))
            
            errors[name] = {
                'relative_L2': rel_error,
                'max_absolute': max_error
            }
            
        return errors


def visualize_comparison(X, Y, data_dict, recon_results, snap_idx, nx, ny):
    """可视化对比"""
    n_vars = len(data_dict)
    fig, axes = plt.subplots(n_vars, 3, figsize=(15, 5*n_vars))
    
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    for i, (name, orig_data) in enumerate(data_dict.items()):
        recon_data = recon_results[name]
        
        # 提取快照
        orig_snap = orig_data[:, snap_idx].reshape(ny, nx)
        recon_snap = recon_data[:, snap_idx].reshape(ny, nx)
        error_snap = orig_snap - recon_snap
        
        # 颜色映射
        cmap = 'jet' if 'vel' in name.lower() else 'inferno'
        
        # 原始场
        im1 = axes[i, 0].contourf(X, Y, orig_snap, levels=40, cmap=cmap)
        axes[i, 0].set_title(f'原始 {name}', fontsize=12, fontweight='bold')
        axes[i, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
        
        # 重构场
        im2 = axes[i, 1].contourf(X, Y, recon_snap, levels=40, cmap=cmap)
        axes[i, 1].set_title(f'重构 {name}', fontsize=12, fontweight='bold')
        axes[i, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
        
        # 误差场
        vmax = np.abs(error_snap).max()
        im3 = axes[i, 2].contourf(X, Y, error_snap, levels=40, 
                                  cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i, 2].set_title(f'误差 {name}', fontsize=12, fontweight='bold')
        axes[i, 2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('coupled_pod_comparison.png', dpi=200, bbox_inches='tight')
    print("\n可视化结果已保存: coupled_pod_comparison.png")
    plt.show()


def analyze_mode_convergence(model, data_dict, X, Y, nx, ny):
    """分析模态收敛性"""
    max_modes = min(model.n_modes, 20)
    mode_range = range(1, max_modes + 1)
    
    errors_by_var = {name: [] for name in data_dict.keys()}
    
    print("\n" + "="*60)
    print("模态收敛性分析")
    print("="*60)
    
    for r in mode_range:
        errors = model.compute_errors(data_dict, r)
        for name, err_dict in errors.items():
            errors_by_var[name].append(err_dict['relative_L2'])
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, errors in errors_by_var.items():
        ax.semilogy(mode_range, errors, 'o-', label=name, linewidth=2, markersize=6)
    
    ax.set_xlabel('模态数 r', fontsize=12)
    ax.set_ylabel('相对L2误差', fontsize=12)
    ax.set_title('重构误差 vs 模态数', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mode_convergence.png', dpi=200, bbox_inches='tight')
    print("收敛性分析已保存: mode_convergence.png")
    plt.show()
    
    # 打印关键信息
    print("\n关键误差信息:")
    for name, errors in errors_by_var.items():
        print(f"  {name}:")
        print(f"  1模态: {errors[0]:.4%}")
        print(f"  5模态: {errors[4]:.4%}" if len(errors) > 4 else "")
        print(f"  {max_modes}模态: {errors[-1]:.4%}")


def main():
    # 1. 生成数据
    gen = MultiPhysicsDataGenerator()
    u_data, T_data, X, Y = gen.generate_data()
    
    data_input = {
        'Velocity': u_data,
        'Temperature': T_data
    }
    
    # 2. 训练模型
    model = CoupledPOD(n_modes=15)
    model.fit(data_input)
    
    # 3. 重构
    recon_results = model.reconstruct()
    
    # 4. 计算误差
    print("\n" + "="*60)
    print("重构误差评估")
    print("="*60)
    errors = model.compute_errors(data_input)
    for name, err_dict in errors.items():
        print(f"\n{name}:")
        print(f"相对L2误差: {err_dict['relative_L2']:.6f} ({err_dict['relative_L2']*100:.4f}%)")
        print(f"最大绝对误差: {err_dict['max_absolute']:.6e}")
    
    # 5. 可视化对比
    snap_idx = 50
    visualize_comparison(X, Y, data_input, recon_results, snap_idx, gen.nx, gen.ny)
    
    # 6. 模态收敛性分析
    analyze_mode_convergence(model, data_input, X, Y, gen.nx, gen.ny)
    
    return model, recon_results


if __name__ == "__main__":
    model, results = main()