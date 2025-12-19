# -*- coding: utf-8 -*-
"""
POD模型 → NVIDIA Omniverse 快速导出工具

输出:
    omniverse_export/
    ├── pod_model.onnx              # GPU推理模型
    ├── flow_field.npy              # 点云数据
    ├── metadata.json               # 元数据
    └── import_to_omniverse.py      # Omniverse导入脚本
"""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime


class OmniverseQuickExport:
    """快速导出到Omniverse"""
    
    def __init__(self, model_pkl_path: str):
        with open(model_pkl_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.var_names = self.model_data['var_names']
        self.n_modes = self.model_data['n_modes']
        self.field_dims = self.model_data['field_dims']
        
        print(f"POD模型已加载: {self.var_names}, {self.n_modes}模态")
    
    def export_for_omniverse(self, 
                            snapshot_data: dict,
                            grid_shape: tuple,
                            output_dir: str = "./omniverse_export"):
        """导出Omniverse所需文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("导出Omniverse资源包")
        print("="*70)
        
        # 1. ONNX模型
        print("\n导出ONNX模型...")
        self._export_onnx(output_dir / "pod_model.onnx")
        
        # 2. 点云数据
        print("\n生成点云数据...")
        self._export_point_cloud(snapshot_data, grid_shape, 
                                output_dir / "flow_field.npy")
        
        # 3. 元数据
        print("\n生成元数据...")
        self._export_metadata(output_dir / "metadata.json")
        
        # 4. Omniverse导入脚本
        print("\n生成导入脚本...")
        self._generate_omniverse_script(output_dir / "import_to_omniverse.py")
        
        print( "="*70)
        print("导出完成!")
        print("="*70)
        print(f"\输出目录: {output_dir.absolute()}")
        print("\n下一步:")
        print("1. 打开 NVIDIA Omniverse (Create/Isaac Sim)")
        print("2. Window → Script Editor")
        print("3. 加载并运行 import_to_omniverse.py")
        print("\n或手动:")
        print("1. File → Import")
        print("2. 选择 flow_field.npy (作为点云)")
    
    def _export_onnx(self, output_path: Path):
        """导出ONNX模型"""
        try:
            import torch
            import torch.nn as nn
            
            class PODModel(nn.Module):
                def __init__(self, U, s, mean, scalers, field_dims, var_names):
                    super().__init__()
                    self.register_buffer('U', torch.from_numpy(U).float())
                    self.register_buffer('s', torch.from_numpy(s).float())
                    self.register_buffer('mean', torch.from_numpy(mean).float())
                    
                    # 拼接归一化参数
                    std_list = []
                    mean_list = []
                    for name in var_names:
                        std_list.append(torch.from_numpy(scalers[name]['std']).float())
                        mean_list.append(torch.from_numpy(scalers[name]['mean']).float())
                    
                    self.register_buffer('std_combined', torch.cat(std_list))
                    self.register_buffer('mean_data', torch.cat(mean_list))
                
                def forward(self, x):
                    # 归一化
                    x_norm = (x - self.mean_data.T) / self.std_combined.T
                    # POD投影和重构
                    x_fluc = x_norm - self.mean.T
                    a = torch.matmul(x_fluc, self.U)
                    x_recon_norm = self.mean.T + torch.matmul(a, self.U.T)
                    # 反归一化
                    x_recon = x_recon_norm * self.std_combined.T + self.mean_data.T
                    return x_recon
            
            model = PODModel(
                self.model_data['U'],
                self.model_data['s'],
                self.model_data['mean_combined'],
                self.model_data['scalers'],
                self.field_dims,
                self.var_names
            )
            model.eval()
            
            total_dim = sum(self.field_dims.values())
            dummy_input = torch.randn(1, total_dim)
            
            torch.onnx.export(
                model, dummy_input, str(output_path),
                opset_version=17,
                input_names=['combined_input'],
                output_names=['combined_output'],
                dynamic_axes={
                    'combined_input': {0: 'batch'},
                    'combined_output': {0: 'batch'}
                }
            )
            
            print(f"    {output_path.name}")
            print(f"      输入维度: {total_dim}")
            print(f"      模态数: {self.n_modes}")
            
        except ImportError:
            print("     跳过ONNX导出 (需要PyTorch)")
            print("      安装: pip install torch")
        except Exception as e:
            print(f"   ONNX导出失败: {e}")
    
    def _export_point_cloud(self, snapshot_data: dict, 
                           grid_shape: tuple, output_path: Path):
        """导出点云数据"""
        nx, ny = grid_shape
        
        # 生成网格
        x = np.linspace(0, 10, nx)
        y = np.linspace(-2, 2, ny)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        
        # 选择第一个变量
        var_name = list(snapshot_data.keys())[0]
        data = snapshot_data[var_name]
        
        # 取第一个时间步
        if data.ndim == 2:
            field = data[:, 0]
        else:
            field = data
        
        field_2d = field.reshape(ny, nx)
        
        # 归一化到[0,1]用于颜色
        field_min, field_max = field_2d.min(), field_2d.max()
        if field_max > field_min:
            field_norm = (field_2d - field_min) / (field_max - field_min)
        else:
            field_norm = np.zeros_like(field_2d)
        
        # 颜色映射 (蓝→绿→红)
        colors_r = np.clip(2 * field_norm - 0.5, 0, 1)
        colors_g = 1 - 2 * np.abs(field_norm - 0.5)
        colors_b = np.clip(1 - 2 * field_norm, 0, 1)
        
        # 构建点云: [x, y, z, r, g, b]
        point_cloud = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.zeros(nx * ny),
            colors_r.ravel(),
            colors_g.ravel(),
            colors_b.ravel()
        ])
        
        np.save(output_path, point_cloud)
        
        print(f"      {output_path.name}")
        print(f"      点数: {len(point_cloud):,}")
        print(f"      变量: {var_name}")
        print(f"      范围: [{field_min:.2f}, {field_max:.2f}]")
    
    def _export_metadata(self, output_path: Path):
        """导出元数据"""
        metadata = {
            'model_info': {
                'variables': self.var_names,
                'n_modes': self.n_modes,
                'field_dims': {k: int(v) for k, v in self.field_dims.items()}
            },
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'format': 'Omniverse-compatible'
            },
            'omniverse': {
                'recommended_app': 'Omniverse Create 2023.2+',
                'gpu_required': True,
                'real_time_capable': True
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   {output_path.name}")
    
    def _generate_omniverse_script(self, output_path: Path):
        """生成Omniverse导入脚本"""
        script = '''#!/usr/bin/env python3
"""
Omniverse导入脚本
在 Omniverse Create/Isaac Sim 的 Script Editor 中运行

使用方法:
1. 打开 Omniverse (Create 或 Isaac Sim)
2. Window → Script Editor
3. 加载此脚本
4. 点击运行
"""

import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf

def import_pod_flow_field(npy_path="flow_field.npy"):
    """导入POD流场到Omniverse场景"""
    
    print("="*70)
    print("导入POD流场")
    print("="*70)
    
    # 1. 加载点云数据
    print("\\n1️⃣ 加载点云数据...")
    points = np.load(npy_path)
    print(f"   加载 {len(points):,} 个点")
    
    # 2. 获取当前Stage
    stage = omni.usd.get_context().get_stage()
    
    if stage is None:
        print("无法获取Stage,请确保场景已打开")
        return
    
    # 3. 创建Points几何
    print("\\n2️⃣ 创建点云几何...")
    points_path = "/World/POD_FlowField"
    
    # 删除已存在的
    if stage.GetPrimAtPath(points_path):
        stage.RemovePrim(points_path)
    
    points_prim = UsdGeom.Points.Define(stage, points_path)
    
    # 4. 设置点位置
    print("3️⃣ 设置点位置和颜色...")
    positions = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points]
    points_prim.GetPointsAttr().Set(positions)
    
    # 5. 设置点颜色
    colors = [Gf.Vec3f(float(p[3]), float(p[4]), float(p[5])) for p in points]
    points_prim.GetDisplayColorAttr().Set(colors)
    
    # 6. 设置点大小
    widths = [0.02] * len(points)  # 可调整点大小
    points_prim.GetWidthsAttr().Set(widths)
    
    # 7. 设置渲染模式
    points_prim.CreateDisplayOpacityAttr().Set([1.0])
    
    print("\\n 流场已导入到场景!")
    print(f"   路径: {points_path}")
    print(f"   点数: {len(points):,}")
    print("\\n提示:")
    print("- 在Viewport中可以旋转查看")
    print("- 调整点大小: widths参数")
    print("- 调整颜色: displayColor属性")
    
    return points_path

# 运行导入
if __name__ == "__main__":
    try:
        result = import_pod_flow_field()
        print(f"\\n导入成功: {result}")
    except Exception as e:
        print(f"\\n导入失败: {e}")
        import traceback
        traceback.print_exc()
'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script)
        
        print(f"   {output_path.name}")


def main():
    """主函数"""
    print("="*70)
    print(" POD → NVIDIA Omniverse 导出工具")
    print("="*70)
    
    # 配置
    MODEL_PKL = "cylinder_flexible/coupled_pod_model.pkl"
    GRID_SHAPE = (449, 199)
    
    # 检查模型文件
    if not Path(MODEL_PKL).exists():
        print(f"\n找不到模型: {MODEL_PKL}")
        print("\n请先运行 EX25.py 训练并保存模型")
        return
    
    # 加载模型获取测试数据
    with open(MODEL_PKL, 'rb') as f:
        model_data = pickle.load(f)
    
    # 从模型中提取一个测试快照
    # 这里我们用平均场作为示例
    print("\n准备数据...")
    
    # 拆分平均场
    test_data = {}
    start = 0
    for name in model_data['var_names']:
        dim = model_data['field_dims'][name]
        end = start + dim
        test_data[name] = model_data['mean_combined'][start:end, 0]
        start = end
    
    print(f"  使用变量: {list(test_data.keys())}")
    
    # 导出
    exporter = OmniverseQuickExport(MODEL_PKL)
    exporter.export_for_omniverse(test_data, GRID_SHAPE)


if __name__ == "__main__":
    main()