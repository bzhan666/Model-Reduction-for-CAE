# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:56:05 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
from EX10 import generate_nonlinear_data, build_autoencoder # 假设你把之前的函数保存在这

# 1. 准备LSTM的训练数据
def create_lstm_sequences(data, sequence_length):
    """将时间序列数据转换为 (n_samples, sequence_length, n_features) 格式"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# 主执行函数
def exercises():
    print("=== bzhan666 ===\n")
    
    # a. 先获取训练好的AE和Encoder
    data = generate_nonlinear_data()
    data_train_ae = data.T
    input_dim = data_train_ae.shape[1]
    latent_dim = 3
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.fit(data_train_ae, data_train_ae, epochs=100, batch_size=16, verbose=0)
    
    # b. 用Encoder将高维数据降维
    latent_space_data = encoder.predict(data_train_ae)
    
    # c. 准备LSTM训练序列
    sequence_length = 10
    X_lstm, y_lstm = create_lstm_sequences(latent_space_data, sequence_length)
    
    # d. 构建并训练LSTM模型
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, latent_dim)),
        Dense(latent_dim)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    print("开始训练LSTM模型...")
    lstm_model.fit(X_lstm, y_lstm, epochs=50, verbose=0)
    print("训练完成.")
    
    # e. 进行一步预测并验证
    test_idx = 150
    input_sequence = latent_space_data[test_idx-sequence_length:test_idx].reshape(1, sequence_length, latent_dim)
    
    predicted_latent_vector = lstm_model.predict(input_sequence)
    
    # f. 使用AE的解码器重构回高维流场
    decoder_input = Input(shape=(latent_dim,))
    deco = autoencoder.layers[-3](decoder_input)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(decoder_input, deco)
    
    predicted_snapshot = decoder.predict(predicted_latent_vector)
    
    # g. 可视化对比
    plt.figure(figsize=(12, 5))
    plt.plot(data[:, test_idx], 'b-', label=f'真实快照 @ t={test_idx}')
    plt.plot(predicted_snapshot.T, 'g--', label=f'AE-LSTM 预测快照 @ t={test_idx}')
    plt.title('One-step Prediction using AE-LSTM')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    exercises()