# -*- coding: utf-8 -*-
"""
圆柱绕流真实数据 + 多物理场耦合POD
复用 EX23.py 的 CoupledPOD 类

数据: CYLINDER_ALL.mat (449x199, 151 snapshots)
      from J. Nathan Kutz & Steven L. Brunton
依赖: EX23.py (CoupledPOD类)
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat
from typing import Dict, Tuple

# 直接导入现成的类
from EX23 import CoupledPOD

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RealDataBridge:
    """数据加载 + VTK导出"""
    
    def __init__(self, output_dir="./cylinder_coupled"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.mesh = None
        self.grid_shape = None

    def load_all_fields(self, mat_file="CYLINDER_ALL.mat"):
        """
        加载圆柱绕流的全部三个物理场
        
        Returns:
            data_dict: {'u_velocity': (n_space, nt), ...}
            grid_shape: (nx, ny)
        """
        print("="*70)
        print(f"加载: {mat_file}")
        print("="*70)
        
        mat_data = loadmat(mat_file)
        
        # 先检查数据形状
        print("\n文件内容:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"  {key}: {mat_data[key].shape}")
        
        # 提取数据
        u_all = mat_data['UALL']
        v_all = mat_data['VALL']
        vort_all = mat_data['VORTALL']
        
        print(f"\n原始数据形状: {u_all.shape}")
        
        # 判断数据格式
        if u_all.ndim == 3:
            # 格式1: (ny, nx, nt) 或 (nx, ny, nt)
            ny, nx, nt = u_all.shape
            print(f"检测到3D数组: ({ny}, {nx}, {nt})")
            # 转为快照矩阵
            u_snapshots = u_all.reshape(-1, nt, order='F')
            v_snapshots = v_all.reshape(-1, nt, order='F')
            vort_snapshots = vort_all.reshape(-1, nt, order='F')
            
        elif u_all.ndim == 2:
            # 格式2: (n_space, nt) 已经是快照矩阵
            print(f"检测到2D数组: {u_all.shape}")
            u_snapshots = u_all
            v_snapshots = v_all
            vort_snapshots = vort_all
            
            # 需要推断网格大小
            n_space, nt = u_all.shape
            # 假设是接近正方形的网格
            nx = int(np.sqrt(n_space * 449 / 199))  # 根据描述 449x199
            ny = n_space // nx
            print(f"推断网格: {nx} x {ny}")
            
        else:
            raise ValueError(f"不支持的数据维度: {u_all.ndim}")
        
        self.grid_shape = (nx, ny)
        
        print(f"\n网格: {nx} x {ny} = {nx*ny:,} 点")
        print(f"时间步: {nt}")
        
        # 构建数据字典
        data_dict = {
            'u_velocity': u_snapshots,
            'v_velocity': v_snapshots,
            'vorticity': vort_snapshots
        }
        
        print("\n物理量范围:")
        for name, data in data_dict.items():
            print(f"  {name:12s}: [{data.min():8.2f}, {data.max():8.2f}]")
        
        # 创建VTK网格
        x = np.linspace(0, 10, nx)
        y = np.linspace(-2, 2, ny)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        zz = np.zeros_like(xx)
        self.mesh = pv.StructuredGrid(xx, yy, zz)
        
        print("\n加载完成\n")
        return data_dict, (nx, ny)

    def save_to_vtk(self, data_dict: Dict[str, np.ndarray], 
                    time_idx: int, suffix: str = "orig"):
        """保存多场到单个VTK文件"""
        grid = self.mesh.copy()
        
        for name, vector in data_dict.items():
            grid.point_data[name] = vector
        
        filename = f"{suffix}_{time_idx:04d}.vtk"
        grid.save(self.output_dir / filename)


def visualize_coupled_modes(model: CoupledPOD, grid_shape: Tuple[int, int],
                            n_show: int = 4, save_path=None):
    """可视化耦合模态"""
    nx, ny = grid_shape
    
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 4*n_show))
    
    for mode_idx in range(n_show):
        mode_vector = model.U[:, mode_idx]
        
        start = 0
        for field_idx, name in enumerate(model.var_names):
            n_rows = model.field_dims[name]
            end = start + n_rows
            
            field_mode = mode_vector[start:end].reshape(ny, nx)
            vmax = np.abs(field_mode).max()
            
            ax = axes[mode_idx, field_idx]
            im = ax.imshow(field_mode, cmap='RdBu_r', aspect='auto',
                          vmin=-vmax, vmax=vmax)
            
            if mode_idx == 0:
                ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            if field_idx == 0:
                ax.set_ylabel(f'Mode {mode_idx+1}\nσ={model.s[mode_idx]:.2f}',
                             fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            start = end
    
    plt.suptitle('耦合POD模态', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.show()


def visualize_reconstruction(orig_dict: Dict, recon_dict: Dict,
                            grid_shape: Tuple, time_idx: int = 0, 
                            save_path=None):
    """可视化重构对比"""
    nx, ny = grid_shape
    n_vars = len(orig_dict)
    
    fig, axes = plt.subplots(n_vars, 3, figsize=(15, 5*n_vars))
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    for i, name in enumerate(orig_dict.keys()):
        orig = orig_dict[name][:, time_idx].reshape(ny, nx)
        recon = recon_dict[name][:, time_idx].reshape(ny, nx)
        error = orig - recon
        
        cmap = 'RdBu_r' if 'vort' in name.lower() else 'viridis'
        
        # 原始
        im1 = axes[i, 0].imshow(orig, cmap=cmap, aspect='auto')
        axes[i, 0].set_title(f'{name} - 原始', fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
        
        # 重构
        im2 = axes[i, 1].imshow(recon, cmap=cmap, aspect='auto')
        rel_err = np.linalg.norm(orig - recon) / np.linalg.norm(orig)
        axes[i, 1].set_title(f'{name} - 重构 ({rel_err:.4%})',
                            fontsize=11, fontweight='bold')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
        
        # 误差
        vmax = np.abs(error).max()
        im3 = axes[i, 2].imshow(error, cmap='seismic', aspect='auto',
                               vmin=-vmax, vmax=vmax)
        axes[i, 2].set_title(f'{name} - 误差', fontsize=11, fontweight='bold')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f" 已保存: {save_path}")
    plt.show()


def visualize_energy(model: CoupledPOD, save_path=None):
    """能量谱"""
    s = model.s
    energy = s**2
    cumulative = np.cumsum(energy) / np.sum(energy) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 奇异值
    ax1.semilogy(range(1, len(s)+1), s, 'bo-', markersize=6)
    ax1.set_xlabel('模态数', fontsize=12)
    ax1.set_ylabel('奇异值', fontsize=12)
    ax1.set_title('奇异值衰减', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 累积能量
    ax2.plot(range(1, len(s)+1), cumulative, 'r^-', markersize=6)
    ax2.axhline(90, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax2.axhline(99, color='green', linestyle='--', alpha=0.5, label='99%')
    ax2.set_xlabel('模态数', fontsize=12)
    ax2.set_ylabel('累积能量 (%)', fontsize=12)
    ax2.set_title('能量收敛', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.show()


def main():
    """主函数"""
    print("="*70)
    print(" 圆柱绕流多物理场耦合 POD")
    print("="*70)
    print()
    
    # === 配置 ===
    MAT_FILE = "CYLINDER_ALL.mat"
    N_MODES = 20
    N_TRAIN = 150
    N_EXPORT = 30
    
    # === 1. 加载数据 ===
    bridge = RealDataBridge()
    data_dict, grid_shape = bridge.load_all_fields(MAT_FILE)
    
    # 分割训练/测试
    train_dict = {k: v[:, :N_TRAIN] for k, v in data_dict.items()}
    test_dict = {k: v[:, N_TRAIN:] for k, v in data_dict.items()}
    
    # === 2. 训练 (直接用EX23的类!) ===
    model = CoupledPOD(n_modes=N_MODES)
    model.fit(train_dict)
    
    # === 3. 评估 ===
    print("\n" + "="*70)
    print("性能评估")
    print("="*70)
    
    train_errors = model.compute_errors(train_dict)
    print("\n训练集:")
    for name, err in train_errors.items():
        print(f"  {name:12s}: {err['relative_L2']:.6f} ({err['relative_L2']*100:.4f}%)")
    
    if test_dict[list(test_dict.keys())[0]].shape[1] > 0:
        test_errors = model.compute_errors(test_dict)
        print("\n测试集:")
        for name, err in test_errors.items():
            print(f"  {name:12s}: {err['relative_L2']:.6f} ({err['relative_L2']*100:.4f}%)")
    
    # === 4. 可视化 ===
    print("\n" + "="*70)
    print("生成可视化")
    print("="*70)
    
    visualize_coupled_modes(model, grid_shape, n_show=4,
                           save_path=bridge.output_dir/'coupled_modes.png')
    
    recon_dict = model.reconstruct()
    visualize_reconstruction(train_dict, recon_dict, grid_shape, time_idx=75,
                            save_path=bridge.output_dir/'reconstruction.png')
    
    visualize_energy(model, save_path=bridge.output_dir/'energy.png')
    
    # === 5. 导出VTK ===
    print("\n" + "="*70)
    print("导出 VTK")
    print("="*70)
    
    n_export = min(N_EXPORT, N_TRAIN)
    print(f"导出 {n_export} 个时间步...\n")
    
    recon_dict_full = model.reconstruct()
    
    for i in range(n_export):
        # 原始
        orig_snap = {k: v[:, i] for k, v in train_dict.items()}
        bridge.save_to_vtk(orig_snap, i, "orig")
        
        # 重构
        recon_snap = {k: v[:, i] for k, v in recon_dict_full.items()}
        bridge.save_to_vtk(recon_snap, i, "recon")
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{n_export}")
    
    print(f"\nVTK已保存: {bridge.output_dir.absolute()}")
    
    # === 总结 ===
    print("\n" + "="*70)
    print("分析完成")
    print("="*70)
    print(f"输出: {bridge.output_dir}")



if __name__ == "__main__":
    main()