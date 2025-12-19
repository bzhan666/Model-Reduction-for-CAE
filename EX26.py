# -*- coding: utf-8 -*-
"""
圆柱绕流多物理场耦合 POD - 增强版

新功能:
1. 灵活的输入输出选择 (部分场重构)
2. 模型保存/加载 (pickle)
3. 在线预测能力

数据: CYLINDER_ALL.mat
依赖: EX23.py (CoupledPOD基类)
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse.linalg import svds
from typing import Dict, Tuple, List, Optional
import pickle
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FlexibleCoupledPOD:
    """
    增强版耦合POD - 支持灵活输入输出
    
    核心思想:
    - 训练时: 用全部变量 (u, v, vort)
    - 预测时: 输入部分变量 → 重构全部或指定变量
    """
    
    def __init__(self, n_modes=20):
        self.n_modes = n_modes
        self.scalers = {}
        self.field_dims = {}
        self.var_names = []
        
        # POD核心
        self.U = None
        self.s = None
        self.Vt = None
        self.mean_combined = None
        self.is_fitted = False
        
        # 元数据
        self.metadata = {
            'version': '1.0',
            'n_modes': n_modes,
            'trained_on': None,
            'input_vars': [],
            'output_vars': []
        }
    
    def fit(self, data_dict: Dict[str, np.ndarray], verbose=True):
        """训练模型"""
        if verbose:
            print("\n" + "="*70)
            print("多物理场耦合 POD 训练")
            print("="*70)
        
        # 归一化并拼接
        processed_data_list = []
        self.var_names = list(data_dict.keys())
        
        if verbose:
            print("\n1️⃣ 数据预处理:")
        
        for name, data in data_dict.items():
            self.field_dims[name] = data.shape[0]
            
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            std[std < 1e-10] = 1.0
            
            self.scalers[name] = {'mean': mean, 'std': std}
            # 使用 float32 节省内存
            norm_data = ((data - mean) / std).astype(np.float32)
            
            # 确保是二维数组 (n_space, n_time)
            if norm_data.ndim == 1:
                norm_data = norm_data.reshape(-1, 1)
            
            processed_data_list.append(norm_data)
            
            if verbose:
                print(f"  {name:12s}: [{data.min():8.2f}, {data.max():8.2f}] "
                      f"→ [{norm_data.min():6.2f}, {norm_data.max():6.2f}]")
                print(f"    Shape: {norm_data.shape}")
        
        # 拼接
        try:
            combined_matrix = np.vstack(processed_data_list)
        except ValueError as e:
            print(f"\n 拼接失败: {e}")
            for i, arr in enumerate(processed_data_list):
                print(f"  Array {i}: shape={arr.shape}, dtype={arr.dtype}")
            raise
        
        # 释放内存
        del processed_data_list
        import gc
        gc.collect()
        
        if verbose:
            print(f"\n2️⃣ 拼接矩阵: {combined_matrix.shape}")
            print(f"\n3️⃣ SVD分解...")
        
        # POD标准流程
        self.mean_combined = np.mean(combined_matrix, axis=1, keepdims=True)
        # 原地操作减少内存占用
        combined_matrix -= self.mean_combined
        # fluctuation_matrix = combined_matrix # 别名
        
        # 使用 svds 计算部分 SVD (显著节省内存)
        # 注意: svds 返回的奇异值是升序的,需要翻转
        U, s, Vt = svds(combined_matrix, k=self.n_modes)
        
        # 翻转顺序 (从小到大 -> 从大到小)
        U = U[:, ::-1]
        s = s[::-1]
        Vt = Vt[::-1, :]
        
        self.U = U
        self.s = s
        self.Vt = Vt
        
        # 能量分析 (近似)
        total_energy = np.sum(np.var(combined_matrix, axis=1)) * (combined_matrix.shape[1] - 1)
        captured_energy = np.sum(self.s**2)
        energy_ratio = captured_energy / total_energy * 100
        
        if verbose:
            print(f"  保留模态: {self.n_modes}")
            print(f"  能量占比: {energy_ratio:.2f}%")
            print("\n 训练完成!")
        
        self.is_fitted = True
        
        # 更新元数据
        self.metadata['trained_on'] = self.var_names
        self.metadata['input_vars'] = self.var_names
        self.metadata['output_vars'] = self.var_names
        
        return self
    
    def reconstruct(self, r=None, output_vars: Optional[List[str]] = None):
        """
        重构 (可指定输出变量)
        
        Args:
            r: 使用的模态数
            output_vars: 要输出的变量列表 (None=全部)
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练")
        
        if r is None:
            r = self.n_modes
        
        # 完整重构
        fluctuation_recon = self.U[:, :r] @ np.diag(self.s[:r]) @ self.Vt[:r, :]
        recon_combined = self.mean_combined + fluctuation_recon
        
        # 拆分
        results = {}
        start_row = 0
        
        for name in self.var_names:
            n_rows = self.field_dims[name]
            end_row = start_row + n_rows
            
            # 如果指定了输出变量且当前变量不在其中,跳过
            if output_vars is not None and name not in output_vars:
                start_row = end_row
                continue
            
            norm_recon = recon_combined[start_row:end_row, :]
            scaler = self.scalers[name]
            orig_recon = norm_recon * scaler['std'] + scaler['mean']
            
            results[name] = orig_recon
            start_row = end_row
        
        return results
    
    def predict(self, 
                input_dict: Dict[str, np.ndarray],
                output_vars: Optional[List[str]] = None,
                r: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        在线预测: 给定部分场,预测全部场
        
        Args:
            input_dict: 输入场 {'u': (n_space, n_time), 'v': ...}
            output_vars: 要输出的变量 (None=全部)
            r: 使用的模态数
            
        Returns:
            predicted_dict: 预测的场
            
        Example:
            # 用u和v预测vort
            pred = model.predict({'u': u_new, 'v': v_new}, 
                                output_vars=['vorticity'])
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练")
        
        if r is None:
            r = self.n_modes
        
        # 检查输入变量
        for name in input_dict.keys():
            if name not in self.var_names:
                raise ValueError(f"未知变量: {name}")
        
        # 构建部分观测的拼接向量
        combined_input = []
        input_mask = []  # 记录哪些行有输入
        
        start_row = 0
        for name in self.var_names:
            n_rows = self.field_dims[name]
            
            if name in input_dict:
                # 有输入: 归一化
                data = input_dict[name]
                if data.ndim == 1:
                    data = data[:, np.newaxis] # 确保是 (N, 1)
                elif data.ndim == 2 and data.shape[1] > data.shape[0] and data.shape[0] < 5000:
                    # 如果看起来是 (Time, Space), 转置
                    # 注意: 这里假设 n_space > n_time, 且 n_time 通常较小
                    # 更稳健的方法是直接reshape成 (-1, n_time) 如果我们知道 n_time
                     pass 

                # 确保时间维度匹配
                n_time = data.shape[1]
                
                scaler = self.scalers[name]
                norm_data = (data - scaler['mean']) / scaler['std']
                combined_input.append(norm_data)
                input_mask.extend([True] * n_rows)
            else:
                # 无输入: 用零填充
                # 需要知道时间步数. 从 input_dict 中任意一个已知的获取
                # 如果是第一个变量且缺失, 需要先找到后面的变量确定 n_time
                if 'n_time' not in locals():
                     # 寻找第一个有效输入的 n_time
                     for k, v in input_dict.items():
                         if v.ndim == 1: n_time = 1
                         else: n_time = v.shape[1]
                         break
                
                combined_input.append(np.zeros((n_rows, n_time)))
                input_mask.extend([False] * n_rows)
            
            start_row += n_rows
        
        combined_input = np.vstack(combined_input)
        input_mask = np.array(input_mask)
        
        # 投影到POD空间 (仅使用有输入的行)
        fluctuation_input = combined_input - self.mean_combined
        
        # 最小二乘投影: min ||U_obs @ a - x_obs||
        # 解: a = (U_obs^T @ U_obs)^-1 @ U_obs^T @ x_obs
        U_obs = self.U[input_mask, :r]
        x_obs = fluctuation_input[input_mask, :]
        
        # 模态系数
        a = np.linalg.lstsq(U_obs, x_obs, rcond=None)[0]
        
        # 重构全场
        fluctuation_recon = self.U[:, :r] @ a
        recon_combined = self.mean_combined + fluctuation_recon
        
        # 拆分并反归一化
        results = {}
        start_row = 0
        
        for name in self.var_names:
            n_rows = self.field_dims[name]
            end_row = start_row + n_rows
            
            if output_vars is not None and name not in output_vars:
                start_row = end_row
                continue
            
            norm_recon = recon_combined[start_row:end_row, :]
            scaler = self.scalers[name]
            orig_recon = norm_recon * scaler['std'] + scaler['mean']
            
            results[name] = orig_recon.squeeze() if orig_recon.shape[1] == 1 else orig_recon
            start_row = end_row
        
        return results
    
    def save_model(self, filepath: str):
        """保存模型 (pickle格式)"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练,无法保存")
        
        model_data = {
            'n_modes': self.n_modes,
            'scalers': self.scalers,
            'field_dims': self.field_dims,
            'var_names': self.var_names,
            'U': self.U,
            's': self.s,
            'Vt': self.Vt,
            'mean_combined': self.mean_combined,
            'metadata': self.metadata
        }
        
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 同时保存JSON元数据
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'n_modes': self.n_modes,
                'var_names': self.var_names,
                'field_dims': {k: int(v) for k, v in self.field_dims.items()},
                'energy_ratio': float(np.sum(self.s**2) / np.sum(self.Vt**2)),
                'metadata': self.metadata
            }, f, indent=2)
        
        print(f"  模型已保存:")
        print(f"   - {filepath}")
        print(f"   - {meta_path}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(n_modes=model_data['n_modes'])
        model.scalers = model_data['scalers']
        model.field_dims = model_data['field_dims']
        model.var_names = model_data['var_names']
        model.U = model_data['U']
        model.s = model_data['s']
        model.Vt = model_data['Vt']
        model.mean_combined = model_data['mean_combined']
        model.metadata = model_data['metadata']
        model.is_fitted = True
        
        print(f"   模型已加载: {filepath}")
        print(f"   变量: {model.var_names}")
        print(f"   模态数: {model.n_modes}")
        
        return model
    
    def compute_errors(self, data_dict: Dict[str, np.ndarray], r=None):
        """计算重构误差"""
        recon_results = self.reconstruct(r)
        errors = {}
        
        for name, orig_data in data_dict.items():
            recon_data = recon_results[name]
            
            rel_error = np.linalg.norm(orig_data - recon_data, 'fro') / \
                       np.linalg.norm(orig_data, 'fro')
            max_error = np.max(np.abs(orig_data - recon_data))
            
            errors[name] = {
                'relative_L2': rel_error,
                'max_absolute': max_error
            }
        
        return errors


# === 以下复用之前的可视化和数据加载代码 ===

class RealDataBridge:
    """数据加载"""
    def __init__(self, output_dir="./cylinder_flexible"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.mesh = None
        self.grid_shape = None

    def load_all_fields(self, mat_file="CYLINDER_ALL.mat"):
        print("="*70)
        print(f"加载: {mat_file}")
        print("="*70)
        
        # 加载时尽可能节省内存
        try:
            mat_data = loadmat(mat_file)
        except MemoryError:
            print(" 内存不足无法加载MAT文件")
            raise
        
        print("\n文件内容:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"  {key}: {mat_data[key].shape}")
        
        # 使用 float32 减少内存占用
        # 统一数据形状处理
        raw_data = {}
        
        for key, target_name in [('UALL', 'u_velocity'), ('VALL', 'v_velocity'), ('VORTALL', 'vorticity')]:
            if key not in mat_data:
                print(f" Warning: {key} not found in MAT file")
                continue
                
            arr = mat_data[key]
            print(f"  {key} raw shape: {arr.shape}")
            
            # 1. 处理 3D -> 2D
            if arr.ndim == 3:
                d1, d2, nt = arr.shape
                # 假设较小的维度是时间? 不一定. 通常最后一维是时间
                # 这里保持原有逻辑: reshape(-1, nt)
                arr = arr.reshape(-1, nt, order='F')
                print(f"    -> Reshaped 3D to: {arr.shape}")
            
            # 2. 智能转置: 假设空间维度 >> 时间维度
            if arr.ndim == 2:
                rows, cols = arr.shape
                # 启发式规则: 如果行数远小于列数,且行数看起来像时间步(例如 < 2000), 则转置
                if rows < cols and rows < 5000: 
                    print(f"    -> Transposing {arr.shape} to (Space, Time)")
                    arr = arr.T
            
            raw_data[target_name] = arr.astype(np.float32)
            
        # 3. 检查时间步一致性
        shapes = {k: v.shape for k, v in raw_data.items()}
        for k, s in shapes.items():
            print(f"  {k} final shape: {s}")
            
        if not shapes:
            raise ValueError("No valid data loaded!")
            
        min_time = min(s[1] for s in shapes.values())
        print(f"  Min time steps: {min_time}")
        
        data_dict = {}
        for name, arr in raw_data.items():
            if arr.shape[1] > min_time:
                print(f" {name} has {arr.shape[1]} time steps, truncating to {min_time}")
                arr = arr[:, :min_time]
            
            # 确保至少是 2D
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
                
            data_dict[name] = arr
            
        # 释放原始字典
        del mat_data
        import gc
        gc.collect()
        
        # 验证最终形状
        sample_shape = data_dict['u_velocity'].shape
        print(f"\n统一后 u_velocity 形状: {sample_shape}")
        
        nx = 449
        ny = sample_shape[0] // nx
        self.grid_shape = (nx, ny)
        
        print(f"\n网格: {nx} x {ny} = {nx*ny:,} 点")
        print(f"时间步: {sample_shape[1]}")
        
        print("\n物理量范围:")
        for name, data in data_dict.items():
            print(f"  {name:12s}: [{data.min():8.2f}, {data.max():8.2f}]")
        
        print("\n 加载完成\n")
        return data_dict, (nx, ny)


def demo_flexible_io(model, train_dict, grid_shape):
    """演示灵活输入输出功能"""
    print("\n" + "="*70)
    print("演示: 灵活输入输出")
    print("="*70)
    
    # 场景1: 用u和v预测vort
    print("\n场景1: 输入u+v → 预测vort")
    print("-" * 70)
    
    time_idx = 75
    u_test = train_dict['u_velocity'][:, time_idx]
    v_test = train_dict['v_velocity'][:, time_idx]
    vort_true = train_dict['vorticity'][:, time_idx]
    
    # 预测
    pred = model.predict(
        {'u_velocity': u_test, 'v_velocity': v_test},
        output_vars=['vorticity']
    )
    
    vort_pred = pred['vorticity']
    error = np.linalg.norm(vort_true - vort_pred) / np.linalg.norm(vort_true)
    
    print(f"  涡量预测误差: {error:.6f} ({error*100:.4f}%)")
    
    # 场景2: 只输入u,重构全部场
    print("\n场景2: 输入u → 重构全部场")
    print("-" * 70)
    
    pred_all = model.predict({'u_velocity': u_test})
    
    for name in ['v_velocity', 'vorticity']:
        true_val = train_dict[name][:, time_idx]
        pred_val = pred_all[name]
        err = np.linalg.norm(true_val - pred_val) / np.linalg.norm(true_val)
        print(f"  {name:12s} 误差: {err:.6f} ({err*100:.4f}%)")


def main():
    print("="*70)
    print(" 圆柱绕流 - 增强版耦合POD")
    print("="*70)
    print()
    
    MAT_FILE = "CYLINDER_ALL.mat"
    N_MODES = 20
    N_TRAIN = 150
    
    # 加载数据
    bridge = RealDataBridge()
    data_dict, grid_shape = bridge.load_all_fields(MAT_FILE)
    
    train_dict = {k: v[:, :N_TRAIN] for k, v in data_dict.items()}
    test_dict = {k: v[:, N_TRAIN:] for k, v in data_dict.items()}
    
    # 训练模型
    model = FlexibleCoupledPOD(n_modes=N_MODES)
    model.fit(train_dict)
    
    # 评估
    print("\n" + "="*70)
    print("性能评估")
    print("="*70)
    
    train_errors = model.compute_errors(train_dict)
    print("\n训练集:")
    for name, err in train_errors.items():
        print(f"  {name:12s}: {err['relative_L2']:.6f} ({err['relative_L2']*100:.4f}%)")
    
    # 演示灵活输入输出
    demo_flexible_io(model, train_dict, grid_shape)
    
    # 保存模型
    print("\n" + "="*70)
    print("模型保存")
    print("="*70)
    model.save_model(bridge.output_dir / "coupled_pod_model.pkl")
    
    print("\n" + "="*70)
    print("完成")
    print("="*70)
    print(f" 输出: {bridge.output_dir}")
    print(" 模型可通过以下方式复用:")
    print(" 1. Python: model = FlexibleCoupledPOD.load_model('...')")
    
    return model


if __name__ == "__main__":
    model = main()