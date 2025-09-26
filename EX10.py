# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:40:56 2025

@author: 25045
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


# 强制设置中文字体
def setup_chinese_font_force():
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



# 1. 生成模拟的非线性CFD数据
def generate_nonlinear_data(n_space=128, n_time=200):
    """生成一个移动的激波/锋面作为非线性特征"""
    x = np.linspace(-1, 1, n_space)
    t = np.linspace(0, 2, n_time)
    data = np.zeros((n_space, n_time))
    for i in range(n_time):
        center = np.sin(t[i] * np.pi) # 移动的中心
        width = 0.1
        data[:, i] = 0.5 * (1 + np.tanh((x - center) / width))
    return data

# 2. 构建自编码器模型
def build_autoencoder(input_dim, latent_dim):
    # 编码器
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(latent_dim, activation='relu', name='bottleneck')(encoded)
    
    # 解码器
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded) # 用sigmoid因为数据在0-1之间
    
    # 完整模型
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# 主执行函数
def exercises():
    print("=== bzhan666 ===\n")
    
    # 生成数据
    data = generate_nonlinear_data()
    # Keras需要 (n_samples, n_features) 格式
    data_train = data.T 
    
    input_dim = data_train.shape[1]
    latent_dim = 3 # 尝试用3个维度来压缩
    
    # 构建并训练模型
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    print("开始训练自编码器...")
    history = autoencoder.fit(data_train, data_train, 
                              epochs=100, batch_size=16, 
                              shuffle=True, verbose=0)
    print("训练完成.")
    
    # 可视化训练过程
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'])
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.show()
    
    # 重构并对比
    reconstructed_data = autoencoder.predict(data_train).T
    
    snapshot_idx = 100
    plt.figure(figsize=(12, 5))
    plt.plot(data[:, snapshot_idx], 'b-', label='原始数据')
    plt.plot(reconstructed_data[:, snapshot_idx], 'r--', label=f'AE重构 (latent_dim={latent_dim})')
    plt.title(f'Snapshot Comparison at t={snapshot_idx}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    exercises()