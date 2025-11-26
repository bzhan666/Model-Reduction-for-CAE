# -*- coding: utf-8 -*-
"""
AE-LSTM 降阶模型
结合自编码器(空间降维)和LSTM(时间动态建模)

Created on Tue Nov 11 2025
@author: bzhan666
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict
import time
from contextlib import contextmanager

# TensorFlow 导入
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("="*60)
    print("错误: TensorFlow 未安装。")
    print("请运行: pip install tensorflow")
    print("="*60)
    raise

# 导入数据处理工具
try:
    from EX14_2 import FluentDataGenerator, FluentDataReader
except ImportError:
    print("警告: 无法从 EX14_2.py 导入数据处理类。")
    class FluentDataGenerator: pass
    class FluentDataReader: pass

# ============================================================================
# 工具函数
# ============================================================================

@contextmanager
def timer(description: str):
    """计时器上下文管理器"""
    start = time.time()
    yield
    end = time.time()
    print(f"{description}: {end - start:.2f}秒")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 第一部分: 非线性空间降维器 (AutoEncoder)
# ============================================================================

class AutoEncoderReductor:
    """
    自编码器空间降维器
    
    功能:
    - 将高维流场 (n_space,) 压缩到低维隐空间 (r,)
    - 提供编码(encode)和解码(decode)功能
    """
    
    def __init__(self, n_latent_dim: int, hidden_layers: list = [256, 128]):
        """
        初始化自编码器
        
        Args:
            n_latent_dim: 隐空间维度 (r)
            hidden_layers: 编码器隐藏层神经元数量列表
        """
        self.r = n_latent_dim
        self.hidden_layers = hidden_layers
        
        # 模型组件
        self.autoencoder: Optional[Model] = None
        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None
        
        # 数据归一化参数
        self.data_mean = None
        self.data_std = None
        self.n_space = None
        
        self.is_fitted = False
    
    def build_model(self, n_space: int):
        """构建对称的自编码器网络"""
        self.n_space = n_space
        
        # ==================== 编码器 ====================
        encoder_input = Input(shape=(n_space,), name='encoder_input')
        x = encoder_input
        
        # 逐层压缩
        for i, units in enumerate(self.hidden_layers):
            x = Dense(units, activation='relu', name=f'encoder_dense_{i}')(x)
            x = BatchNormalization(name=f'encoder_bn_{i}')(x)
            x = Dropout(0.1, name=f'encoder_dropout_{i}')(x)
        
        # 瓶颈层 (隐空间)
        latent = Dense(self.r, activation='linear', name='latent_space')(x)
        self.encoder = Model(encoder_input, latent, name='encoder')
        
        # ==================== 解码器 ====================
        decoder_input = Input(shape=(self.r,), name='decoder_input')
        x = decoder_input
        
        # 对称地逐层解压缩
        for i, units in enumerate(reversed(self.hidden_layers)):
            x = Dense(units, activation='relu', name=f'decoder_dense_{i}')(x)
            x = BatchNormalization(name=f'decoder_bn_{i}')(x)
            x = Dropout(0.1, name=f'decoder_dropout_{i}')(x)
        
        # 输出层
        decoder_output = Dense(n_space, activation='linear', name='decoder_output')(x)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        
        # ==================== 端到端自编码器 ====================
        autoencoder_output = self.decoder(self.encoder(encoder_input))
        self.autoencoder = Model(encoder_input, autoencoder_output, name='autoencoder')
        
        # 编译模型
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("\n" + "="*60)
        print("自编码器模型结构:")
        print("="*60)
        print(f"输入维度: {n_space}")
        print(f"隐空间维度: {self.r}")
        print(f"压缩比: {n_space/self.r:.1f}x")
        self.autoencoder.summary()
    
    def fit(self, snapshot_matrix: np.ndarray, 
            epochs: int = 100, 
            batch_size: int = 32,
            validation_split: float = 0.2) -> Dict:
        """
        训练自编码器
        
        Args:
            snapshot_matrix: 快照矩阵 (n_space, n_time)
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
            
        Returns:
            训练历史字典
        """
        print("\n" + "="*60)
        print("第一步: 训练空间降维器 (AutoEncoder)")
        print("="*60)
        
        # 数据验证
        if snapshot_matrix.ndim != 2:
            raise ValueError(f"快照矩阵必须是2维,当前: {snapshot_matrix.ndim}维")
        
        n_space, n_time = snapshot_matrix.shape
        print(f"训练数据: {n_space} 空间点 × {n_time} 时间步")
        
        # 数据标准化 (使用Z-score标准化,更稳定)
        self.data_mean = np.mean(snapshot_matrix, axis=1, keepdims=True)
        self.data_std = np.std(snapshot_matrix, axis=1, keepdims=True) + 1e-8
        normalized_data = (snapshot_matrix - self.data_mean) / self.data_std
        
        # 转换为 Keras 格式: (n_samples, n_features)
        train_data = normalized_data.T  # (n_time, n_space)
        
        # 构建模型
        self.build_model(n_space)
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # 训练
        with timer("  训练耗时"):
            history = self.autoencoder.fit(
                train_data, train_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
        
        self.is_fitted = True
        
        # 训练结果
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"\n训练完成!")
        print(f"  最终训练损失: {final_train_loss:.6f}")
        print(f"  最终验证损失: {final_val_loss:.6f}")
        
        return history.history
    
    def encode(self, snapshots: np.ndarray) -> np.ndarray:
        """
        编码: 高维 -> 低维
        
        Args:
            snapshots: (n_space,) 或 (n_space, n_time)
            
        Returns:
            latent_codes: (r,) 或 (r, n_time)
        """
        self._check_fitted()
        
        is_single = snapshots.ndim == 1
        if is_single:
            snapshots = snapshots[:, np.newaxis]
        
        # 标准化
        normalized = (snapshots - self.data_mean) / self.data_std
        
        # 编码
        latent_codes = self.encoder.predict(normalized.T, verbose=0).T
        
        return latent_codes.ravel() if is_single else latent_codes
    
    def decode(self, latent_codes: np.ndarray) -> np.ndarray:
        """
        解码: 低维 -> 高维
        
        Args:
            latent_codes: (r,) 或 (r, n_time)
            
        Returns:
            reconstructed: (n_space,) 或 (n_space, n_time)
        """
        self._check_fitted()
        
        is_single = latent_codes.ndim == 1
        if is_single:
            latent_codes = latent_codes[np.newaxis, :]
        else:
            latent_codes = latent_codes.T
        
        # 解码
        normalized = self.decoder.predict(latent_codes, verbose=0).T
        
        # 反标准化
        reconstructed = normalized * self.data_std + self.data_mean
        
        return reconstructed.ravel() if is_single else reconstructed
    
    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("模型未训练。请先调用 fit() 方法。")


# ============================================================================
# 第二部分: 非线性时间动态模型 (LSTM)
# ============================================================================

class LSTMDynamicsModel:
    """
    LSTM 时间动态模型
    
    功能:
    - 在低维隐空间中建模时间演化规律
    - 基于历史序列预测未来状态
    """
    
    def __init__(self, n_latent_dim: int, sequence_length: int, 
                 lstm_units: list = [64, 32]):
        """
        初始化 LSTM 模型
        
        Args:
            n_latent_dim: 隐空间维度 (r)
            sequence_length: 输入序列长度
            lstm_units: LSTM 层的单元数列表
        """
        self.r = n_latent_dim
        self.seq_len = sequence_length
        self.lstm_units = lstm_units
        
        self.model: Optional[Model] = None
        self.is_fitted = False
    
    def build_model(self):
        """构建 LSTM 网络"""
        model = Sequential(name='LSTM_Dynamics')
        
        # 第一个 LSTM 层
        model.add(LSTM(
            self.lstm_units[0],
            activation='tanh',
            return_sequences=len(self.lstm_units) > 1,
            input_shape=(self.seq_len, self.r),
            name='lstm_1'
        ))
        model.add(Dropout(0.2))
        
        # 后续 LSTM 层
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)
            model.add(LSTM(
                units,
                activation='tanh',
                return_sequences=return_seq,
                name=f'lstm_{i}'
            ))
            model.add(Dropout(0.2))
        
        # 输出层
        model.add(Dense(self.r, activation='linear', name='output'))
        
        # 编译
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        print("\n" + "="*60)
        print("LSTM 动态模型结构:")
        print("="*60)
        print(f"输入序列长度: {self.seq_len}")
        print(f"隐空间维度: {self.r}")
        self.model.summary()
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建 LSTM 训练序列
        
        Args:
            data: 时间序列数据 (n_time, r)
            
        Returns:
            X: 输入序列 (n_samples, seq_len, r)
            y: 目标值 (n_samples, r)
        """
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])
            y.append(data[i + self.seq_len])
        return np.array(X), np.array(y)
    
    def fit(self, latent_series: np.ndarray, 
            epochs: int = 50, 
            batch_size: int = 16,
            validation_split: float = 0.2) -> Dict:
        """
        训练 LSTM 模型
        
        Args:
            latent_series: 隐空间时间序列 (r, n_time)
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
            
        Returns:
            训练历史字典
        """
        print("\n" + "="*60)
        print("第二步: 训练时间动态模型 (LSTM)")
        print("="*60)
        
        # 转换为 (n_time, r) 格式
        latent_series = latent_series.T
        print(f"隐空间数据: {latent_series.shape[0]} 时间步 × {latent_series.shape[1]} 维")
        
        # 创建序列
        X_train, y_train = self._create_sequences(latent_series)
        print(f"生成序列数: {len(X_train)}")
        print(f"  输入形状: {X_train.shape}")
        print(f"  输出形状: {y_train.shape}")
        
        # 构建模型
        self.build_model()
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # 训练
        with timer("  训练耗时"):
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
        
        self.is_fitted = True
        
        # 训练结果
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"\n训练完成!")
        print(f"  最终训练损失: {final_train_loss:.6f}")
        print(f"  最终验证损失: {final_val_loss:.6f}")
        
        return history.history
    
    def predict(self, initial_sequence: np.ndarray, num_steps: int) -> np.ndarray:
        """
        自回归预测
        
        Args:
            initial_sequence: 初始序列 (r, seq_len)
            num_steps: 预测步数
            
        Returns:
            predictions: 预测结果 (r, num_steps)
        """
        self._check_fitted()
        
        # 转换为 (seq_len, r)
        history = initial_sequence.T
        
        if history.shape[0] != self.seq_len:
            raise ValueError(f"初始序列长度必须为 {self.seq_len}")
        
        predictions = []
        
        # 自回归预测
        for step in range(num_steps):
            # 准备输入: (1, seq_len, r)
            input_seq = history[-self.seq_len:].reshape(1, self.seq_len, self.r)
            
            # 预测下一步
            next_state = self.model.predict(input_seq, verbose=0)[0]  # (r,)
            predictions.append(next_state)
            
            # 更新历史
            history = np.vstack([history, next_state])
        
        # 转换为 (r, num_steps)
        return np.array(predictions).T
    
    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("模型未训练。请先调用 fit() 方法。")


# ============================================================================
# 集成模型: AE-LSTM ROM
# ============================================================================

class AE_LSTM_ROM:
    """
    AE-LSTM 降阶模型
    
    集成空间降维 (AutoEncoder) 和时间动态 (LSTM)
    """
    
    def __init__(self, n_latent_dim: int, sequence_length: int,
                 ae_hidden_layers: list = [256, 128],
                 lstm_units: list = [64, 32]):
        """
        初始化 AE-LSTM ROM
        
        Args:
            n_latent_dim: 隐空间维度
            sequence_length: LSTM 输入序列长度
            ae_hidden_layers: AutoEncoder 隐藏层配置
            lstm_units: LSTM 层配置
        """
        self.r = n_latent_dim
        self.seq_len = sequence_length
        
        # 初始化两个组件
        self.ae = AutoEncoderReductor(n_latent_dim, ae_hidden_layers)
        self.lstm = LSTMDynamicsModel(n_latent_dim, sequence_length, lstm_units)
        
        self.is_fitted = False

    def save(self, path: Union[str, Path]):
        """
        保存整个 AE-LSTM 模型
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，无法保存")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 分别保存 AE 和 LSTM 组件
        self.ae.save(path / "autoencoder")
        self.lstm.save(path / "lstm")
        
        # 保存配置信息
        config = {
            'n_latent_dim': self.r,
            'sequence_length': self.seq_len
        }
        with open(path / "config.json", 'w') as f:
            import json
            json.dump(config, f)
            
        print(f"AE-LSTM ROM 已保存至 {path}")

    @classmethod
    def load(cls, path: Union[str, Path]):
        """
        加载整个 AE-LSTM 模型
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"路径 {path} 不存在")
        
        # 加载配置
        with open(path / "config.json", 'r') as f:
            import json
            config = json.load(f)
            
        # 加载组件
        ae = AutoEncoderReductor.load(path / "autoencoder")
        lstm = LSTMDynamicsModel.load(path / "lstm", 
                                      n_latent_dim=config['n_latent_dim'],
                                      sequence_length=config['sequence_length'])
        
        # 创建实例
        instance = cls(n_latent_dim=config['n_latent_dim'], 
                       sequence_length=config['sequence_length'])
        instance.ae = ae
        instance.lstm = lstm
        instance.is_fitted = True
        
        print(f"AE-LSTM ROM 已从 {path} 加载")
        return instance

    def fit(self, snapshot_matrix: np.ndarray,
            ae_epochs: int = 100,
            lstm_epochs: int = 50,
            batch_size: int = 32):
        """
        训练完整模型
        
        Args:
            snapshot_matrix: 快照矩阵 (n_space, n_time)
            ae_epochs: AutoEncoder 训练轮数
            lstm_epochs: LSTM 训练轮数
            batch_size: 批次大小
        """
        print("\n" + "="*70)
        print("开始训练 AE-LSTM 降阶模型")
        print("="*70)
        
        # 步骤1: 训练 AutoEncoder
        ae_history = self.ae.fit(
            snapshot_matrix,
            epochs=ae_epochs,
            batch_size=batch_size
        )
        
        # 步骤2: 将数据投影到隐空间
        print("\n投影训练数据到隐空间...")
        latent_series = self.ae.encode(snapshot_matrix)
        print(f"隐空间数据形状: {latent_series.shape}")
        
        # 步骤3: 训练 LSTM
        lstm_history = self.lstm.fit(
            latent_series,
            epochs=lstm_epochs,
            batch_size=batch_size
        )
        
        self.is_fitted = True
        
        print("\n" + "="*70)
        print("AE-LSTM ROM 训练完成!")
        print("="*70)
        
        return {
            'ae_history': ae_history,
            'lstm_history': lstm_history
        }
    
    def predict(self, initial_snapshots: np.ndarray, num_steps: int) -> np.ndarray:
        """
        预测未来流场
        
        Args:
            initial_snapshots: 初始快照序列 (n_space, seq_len)
            num_steps: 预测步数
            
        Returns:
            predicted_snapshots: 预测的快照 (n_space, num_steps)
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练。请先调用 fit() 方法。")
        
        # 1. 编码到隐空间
        initial_latent = self.ae.encode(initial_snapshots)  # (r, seq_len)
        
        # 2. LSTM 预测
        predicted_latent = self.lstm.predict(initial_latent, num_steps)  # (r, num_steps)
        
        # 3. 解码回物理空间
        predicted_snapshots = self.ae.decode(predicted_latent)  # (n_space, num_steps)
        
        return predicted_snapshots
    
    def evaluate_reconstruction(self, test_snapshots: np.ndarray) -> Dict:
        """评估重构误差"""
        reconstructed = self.ae.decode(self.ae.encode(test_snapshots))
        
        mse = np.mean((test_snapshots - reconstructed)**2)
        mae = np.mean(np.abs(test_snapshots - reconstructed))
        relative_error = np.linalg.norm(test_snapshots - reconstructed) / np.linalg.norm(test_snapshots)
        
        return {
            'mse': mse,
            'mae': mae,
            'relative_error': relative_error
        }


# ============================================================================
# 可视化函数
# ============================================================================

def visualize_reconstruction(original: np.ndarray, 
                            reconstructed: np.ndarray,
                            nx: int = 80, ny: int = 40,
                            time_idx: int = 0):
    """可视化重构效果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    def plot_field(ax, data, title, cmap='viridis'):
        im = ax.imshow(data.reshape(ny, nx), cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
        return im
    
    # 原始场
    plot_field(axes[0], original[:, time_idx], "原始流场")
    
    # 重构场
    plot_field(axes[1], reconstructed[:, time_idx], "AE 重构")
    
    # 误差场
    error = original[:, time_idx] - reconstructed[:, time_idx]
    vmax = np.abs(error).max()
    plot_field(axes[2], error, "重构误差", cmap='RdBu_r')
    axes[2].images[0].set_clim(-vmax, vmax)
    
    plt.tight_layout()
    plt.show()


def visualize_prediction(true: np.ndarray,
                        predicted: np.ndarray,
                        nx: int = 80, ny: int = 40,
                        time_steps: list = [0, 5, 10]):
    """可视化预测效果"""
    n_steps = len(time_steps)
    fig, axes = plt.subplots(3, n_steps, figsize=(6*n_steps, 15))
    
    for i, t in enumerate(time_steps):
        # 真实值
        im1 = axes[0, i].imshow(true[:, t].reshape(ny, nx), cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'真实值 (t={t})', fontsize=12)
        plt.colorbar(im1, ax=axes[0, i])
        
        # 预测值
        im2 = axes[1, i].imshow(predicted[:, t].reshape(ny, nx), cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'预测值 (t={t})', fontsize=12)
        plt.colorbar(im2, ax=axes[1, i])
        
        # 误差
        error = true[:, t] - predicted[:, t]
        vmax = np.abs(error).max()
        im3 = axes[2, i].imshow(error.reshape(ny, nx), cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[2, i].set_title(f'预测误差 (t={t})', fontsize=12)
        plt.colorbar(im3, ax=axes[2, i])
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    print("="*70)
    print("AE-LSTM 降阶模型 - 卡门涡街预测")
    print("="*70)
    
    # ========== 1. 数据准备 ==========
    DATA_PATH = Path('./fluent_data')
    NX, NY, NT = 80, 40, 100
    
    if not DATA_PATH.exists():
        print("\n生成卡门涡街数据...")
        FluentDataGenerator(NX, NY, NT).generate_karman_vortex_street(DATA_PATH)
    
    reader = FluentDataReader(DATA_PATH)
    snapshot_matrix, coords = reader.read_snapshots(nt=NT)
    
    print(f"\n数据形状: {snapshot_matrix.shape}")
    print(f"  空间点数: {snapshot_matrix.shape[0]}")
    print(f"  时间步数: {snapshot_matrix.shape[1]}")
    
    # ========== 2. 数据划分 ==========
    split_ratio = 0.8
    split_idx = int(split_ratio * NT)
    
    train_data = snapshot_matrix[:, :split_idx]
    test_data = snapshot_matrix[:, split_idx:]
    
    print(f"\n训练集: {train_data.shape[1]} 步")
    print(f"测试集: {test_data.shape[1]} 步")
    
    # ========== 3. 训练模型 ==========
    model = AE_LSTM_ROM(
        n_latent_dim=8,          # 隐空间维度
        sequence_length=10,       # LSTM 序列长度
        ae_hidden_layers=[256, 128],
        lstm_units=[64, 32]
    )
    
    history = model.fit(
        train_data,
        ae_epochs=50,
        lstm_epochs=30,
        batch_size=16
    )
    
    # ========== 4. 评估重构性能 ==========
    print("\n" + "="*70)
    print("评估 AutoEncoder 重构性能")
    print("="*70)
    
    reconstruction_metrics = model.evaluate_reconstruction(test_data)
    print(f"测试集重构误差:")
    print(f"  MSE: {reconstruction_metrics['mse']:.6f}")
    print(f"  MAE: {reconstruction_metrics['mae']:.6f}")
    print(f"  相对误差: {reconstruction_metrics['relative_error']:.2%}")
    
    # 可视化重构
    reconstructed_test = model.ae.decode(model.ae.encode(test_data))
    visualize_reconstruction(test_data, reconstructed_test, NX, NY, time_idx=10)
    
    # ========== 5. 预测未来流场 ==========
    print("\n" + "="*70)
    print("预测未来流场")
    print("="*70)
    
    # 使用训练集末尾作为初始条件
    initial_sequence = train_data[:, -model.seq_len:]
    num_predictions = test_data.shape[1]
    
    print(f"初始序列: {initial_sequence.shape}")
    print(f"预测步数: {num_predictions}")
    
    predicted_snapshots = model.predict(initial_sequence, num_predictions)
    
    # 计算预测误差
    pred_mse = np.mean((test_data - predicted_snapshots)**2, axis=0)
    pred_mae = np.mean(np.abs(test_data - predicted_snapshots), axis=0)
    
    print(f"\n预测误差:")
    print(f"  平均 MSE: {np.mean(pred_mse):.6f}")
    print(f"  平均 MAE: {np.mean(pred_mae):.6f}")
    
    # 可视化预测结果
    visualize_prediction(test_data, predicted_snapshots, NX, NY, [0, 5, 10, 15])
    
    # ========== 6. 误差随时间变化 ==========
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(pred_mse, 'b-', linewidth=2)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('均方误差随时间变化', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(pred_mae, 'r-', linewidth=2)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('平均绝对误差随时间变化', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ========== 7. 对比某一空间点的时间演化 ==========
    point_idx = snapshot_matrix.shape[0] // 2  # 中心点
    
    plt.figure(figsize=(14, 5))
    
    # 训练阶段
    plt.subplot(1, 2, 1)
    plt.plot(range(split_idx), train_data[point_idx, :], 'b-', linewidth=2, label='训练数据')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('流场值', fontsize=12)
    plt.title(f'空间点 #{point_idx} - 训练阶段', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 预测阶段
    plt.subplot(1, 2, 2)
    time_range = range(split_idx, NT)
    plt.plot(time_range, test_data[point_idx, :], 'b-', linewidth=2, label='真实值')
    plt.plot(time_range, predicted_snapshots[point_idx, :], 'r--', linewidth=2, label='AE-LSTM预测')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('流场值', fontsize=12)
    plt.title(f'空间点 #{point_idx} - 预测阶段', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)


def demo_autoencoder_only():
    """演示单独的 AutoEncoder 功能"""
    print("="*70)
    print("演示: AutoEncoder 空间降维")
    print("="*70)
    
    # 准备数据
    DATA_PATH = Path('./fluent_data')
    NX, NY, NT = 80, 40, 100
    
    if not DATA_PATH.exists():
        FluentDataGenerator(NX, NY, NT).generate_karman_vortex_street(DATA_PATH)
    
    reader = FluentDataReader(DATA_PATH)
    snapshot_matrix, coords = reader.read_snapshots(nt=NT)
    
    # 训练 AutoEncoder
    ae = AutoEncoderReductor(n_latent_dim=8, hidden_layers=[256, 128])
    ae.fit(snapshot_matrix, epochs=50, batch_size=32)
    
    # 测试重构
    test_idx = NT // 2
    original_snapshot = snapshot_matrix[:, test_idx]
    
    # 编码-解码
    latent_code = ae.encode(original_snapshot)
    reconstructed_snapshot = ae.decode(latent_code)
    
    print(f"\n原始维度: {original_snapshot.shape[0]}")
    print(f"隐空间维度: {latent_code.shape[0]}")
    print(f"压缩比: {original_snapshot.shape[0] / latent_code.shape[0]:.1f}x")
    
    # 误差分析
    error = np.abs(original_snapshot - reconstructed_snapshot)
    print(f"\n重构误差:")
    print(f"  最大误差: {error.max():.6f}")
    print(f"  平均误差: {error.mean():.6f}")
    print(f"  相对误差: {np.linalg.norm(error) / np.linalg.norm(original_snapshot):.2%}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    def plot_field(ax, data, title):
        im = ax.imshow(data.reshape(NY, NX), cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    
    plot_field(axes[0], original_snapshot, "原始流场")
    plot_field(axes[1], reconstructed_snapshot, "AutoEncoder 重构")
    
    vmax = error.max()
    im = axes[2].imshow(error.reshape(NY, NX), cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    axes[2].set_title("重构误差 (绝对值)", fontsize=14)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


def demo_lstm_only():
    """演示单独的 LSTM 功能"""
    print("="*70)
    print("演示: LSTM 时间动态建模")
    print("="*70)
    
    # 生成简单的测试数据 (正弦波)
    t = np.linspace(0, 4*np.pi, 200)
    r = 5  # 隐空间维度
    
    # 创建多个相位不同的正弦波
    latent_data = np.zeros((r, len(t)))
    for i in range(r):
        latent_data[i, :] = np.sin(t + i * 0.5) + 0.1 * np.random.randn(len(t))
    
    print(f"测试数据: {latent_data.shape}")
    
    # 划分数据
    split_idx = 160
    train_latent = latent_data[:, :split_idx]
    test_latent = latent_data[:, split_idx:]
    
    # 训练 LSTM
    lstm = LSTMDynamicsModel(n_latent_dim=r, sequence_length=10, lstm_units=[32, 16])
    lstm.fit(train_latent, epochs=30, batch_size=16)
    
    # 预测
    initial_seq = train_latent[:, -10:]
    num_predictions = test_latent.shape[1]
    predictions = lstm.predict(initial_seq, num_predictions)
    
    print(f"\n预测形状: {predictions.shape}")
    
    # 可视化
    fig, axes = plt.subplots(r, 1, figsize=(14, 2*r))
    
    for i in range(r):
        ax = axes[i] if r > 1 else axes
        
        # 训练数据
        ax.plot(range(split_idx), train_latent[i, :], 'b-', linewidth=1.5, label='训练数据', alpha=0.7)
        
        # 测试数据
        test_range = range(split_idx, split_idx + num_predictions)
        ax.plot(test_range, test_latent[i, :], 'g-', linewidth=2, label='真实值')
        ax.plot(test_range, predictions[i, :], 'r--', linewidth=2, label='LSTM预测')
        
        ax.axvline(x=split_idx, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel(f'维度 {i+1}', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=10, loc='upper right')
        if i == r-1:
            ax.set_xlabel('时间步', fontsize=12)
    
    plt.suptitle('LSTM 时间序列预测', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 误差分析
    mse = np.mean((test_latent - predictions)**2)
    mae = np.mean(np.abs(test_latent - predictions))
    print(f"\n预测误差:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "ae":
            demo_autoencoder_only()
        elif mode == "lstm":
            demo_lstm_only()
        else:
            print("使用方法:")
            print("  python script.py       - 运行完整 AE-LSTM 模型")
            print("  python script.py ae    - 仅演示 AutoEncoder")
            print("  python script.py lstm  - 仅演示 LSTM")
    else:
        main()