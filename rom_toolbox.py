# rom_toolbox.py (全能版)

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Union

# --- 导入所有阶段的模型 ---
try:
    # 第5周数据工具
    from EX14_2 import FluentDataReader, FluentDataGenerator
    # 第6周 POD
    from EX16 import PODROM
    # 第9周(基于第7周) POD-DMD
    from pod_dmd_model_v2 import POD_DMD_Model
    # 第8周 神经网络 (新增!)
    from EX18 import AutoEncoderReductor, AE_LSTM_ROM 
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 EX14_2, EX16, EX18, pod_dmd_model_v2 在同一目录下")

class ROMToolbox:
    """
    全能降阶建模工具箱后端
    支持：POD, POD-DMD, AE, AE-LSTM
    """
    
    def __init__(self):
        self.data_matrix: Optional[np.ndarray] = None
        self.coords: Optional[np.ndarray] = None
        # 支持所有模型类型
        self.model: Union[PODROM, POD_DMD_Model, AutoEncoderReductor, AE_LSTM_ROM, None] = None
        self.model_type: str = ""
        self.train_data: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None
        
        # 存储额外的训练参数
        self.train_params = {} 

    def load_data(self, data_path: str, nx: int, ny: int, nt: int) -> str:
        """加载数据 (保持不变)"""
        path = Path(data_path)
        if not path.exists():
            if "fluent_data" in str(path):
                print("生成示例数据...")
                FluentDataGenerator(nx, ny, nt).generate_karman_vortex_street(path)
            else:
                return f"错误: 路径 {path} 不存在"

        try:
            reader = FluentDataReader(path)
            self.data_matrix, self.coords = reader.read_snapshots(nt=nt)
            return f"成功加载数据: {self.data_matrix.shape}"
        except Exception as e:
            return f"加载失败: {str(e)}"

    def init_model(self, model_type: str, **kwargs):
        """初始化模型 (增加了神经网络的支持)"""
        self.model_type = model_type
        
        # 获取通用参数
        rank = int(kwargs.get('rank', 10))
        
        # 存储特定于训练的参数 (Epochs等)
        self.train_params = {
            'epochs': int(kwargs.get('epochs', 50)),
            'batch_size': 32
        }

        try:
            if model_type == "Linear: Static POD":
                self.model = PODROM(n_modes=rank)
                
            elif model_type == "Linear: Dynamic POD-DMD":
                self.model = POD_DMD_Model(n_pod_modes=rank)
                
            elif model_type == "Non-linear: Static AE":
                # AE使用 latent_dim 作为 rank
                self.model = AutoEncoderReductor(n_latent_dim=rank)
                
            elif model_type == "Non-linear: Dynamic AE-LSTM":
                seq_len = int(kwargs.get('seq_len', 10))
                self.model = AE_LSTM_ROM(n_latent_dim=rank, sequence_length=seq_len)
                
            else:
                raise ValueError(f"未知类型: {model_type}")
                
            return f"模型已初始化: {model_type} (Rank/Latent={rank})"
        except Exception as e:
            return f"初始化失败: {str(e)}"

    def train_model(self, split_ratio: float = 0.8) -> str:
        """训练模型 (适配不同的 fit 接口)"""
        if self.data_matrix is None or self.model is None:
            return "错误: 数据未加载或模型未初始化"

        # 划分数据
        split_idx = int(self.data_matrix.shape[1] * split_ratio)
        self.train_data = self.data_matrix[:, :split_idx]
        self.test_data = self.data_matrix[:, split_idx:]

        try:
            # 根据模型类型调用不同的 fit 方法
            if "Non-linear" in self.model_type:
                # 神经网络需要 epochs
                epochs = self.train_params.get('epochs', 50)
                
                if isinstance(self.model, AutoEncoderReductor):
                    self.model.fit(self.train_data, epochs=epochs)
                elif isinstance(self.model, AE_LSTM_ROM):
                    # AE-LSTM 需要两个 epoch 参数，这里简化处理
                    self.model.fit(self.train_data, ae_epochs=epochs, lstm_epochs=int(epochs*0.6))
            else:
                # 线性模型通常只需要数据
                self.model.fit(self.train_data)
                
            return "训练完成！"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"训练出错: {str(e)}"

    def run_task(self) -> Dict:
        """执行预测或重构任务"""
        if self.model is None or not self.model.is_fitted:
            raise RuntimeError("模型未训练")

        results = {}
        
        # --- 静态重构任务 (POD, AE) ---
        if "Static" in self.model_type:
            # 取测试集最后一张快照进行重构对比
            target_snapshot = self.test_data[:, -1]
            
            if isinstance(self.model, PODROM):
                reconstructed = self.model.reconstruct(target_snapshot)
            else: # AEROM
                # AE transform 返回的是 (latent,)，需要 reshape 或直接处理
                latent = self.model.transform(target_snapshot)
                reconstructed = self.model.inverse_transform(latent)
                
            results['truth'] = target_snapshot
            results['prediction'] = reconstructed.ravel() # 确保是一维
            results['title'] = f"{self.model_type} 重构对比"

        # --- 动态预测任务 (POD-DMD, AE-LSTM) ---
        elif "Dynamic" in self.model_type:
            num_steps = self.test_data.shape[1]
            
            if isinstance(self.model, POD_DMD_Model):
                initial_cond = self.train_data[:, -1]
                prediction = self.model.predict(initial_cond, num_steps)
                
            else: # AE_LSTM_Model
                # LSTM 需要一段历史序列作为输入
                seq_len = self.model.seq_len
                # 取训练集最后 seq_len 个步长
                initial_seq = self.train_data[:, -seq_len:]
                prediction = self.model.predict(initial_seq, num_steps)

            # 取最后一步进行展示
            results['truth'] = self.test_data[:, -1]
            results['prediction'] = prediction[:, -1]
            results['title'] = f"{self.model_type} 预测对比 (+{num_steps}步)"
            
        return results

    def save_model(self, filepath: str):
        """
        保存模型
        注意：对于Keras模型 (AE/LSTM)，pickle可能无法直接保存整个对象。
        这里我们尝试保存，如果报错，提示用户（实际工程中需要分离保存权重）。
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            return f"模型已保存至 {filepath}"
        except Exception as e:
            return f"保存失败 (Keras模型可能需要特殊处理): {str(e)}"

    def load_model(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            # 推断类型用于UI显示
            if hasattr(self.model, 'lstm_model'):
                self.model_type = "Non-linear: Dynamic AE-LSTM"
            elif hasattr(self.model, 'autoencoder'):
                self.model_type = "Non-linear: Static AE"
            elif hasattr(self.model, 'A_tilde'):
                self.model_type = "Linear: Dynamic POD-DMD"
            else:
                self.model_type = "Linear: Static POD"
            return f"模型已加载: {self.model_type}"
        except Exception as e:
            return f"加载失败: {str(e)}"