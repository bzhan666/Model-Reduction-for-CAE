# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:56:47 2025

@author: 25045
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Burgers' equation: u_t + u * u_x - nu * u_xx = 0
NU = 0.01 / np.pi

# 1. 定义PINN模型
def build_pinn_model(layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    for i in range(1, len(layers) - 1):
        model.add(tf.keras.layers.Dense(layers[i], activation='tanh'))
    model.add(tf.keras.layers.Dense(layers[i+1]))
    return model

# 2. 定义物理残差损失
def physics_loss(model, t, x):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        tape.watch(x)
        u = model(tf.concat([t, x], axis=1))
        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    del tape
    
    residual = u_t + u * u_x - NU * u_xx
    return tf.reduce_mean(tf.square(residual))

# 主执行函数
def exercises():
    print("=== bzhan666 ===\n")
    
    # a. 准备数据点
    # 初始条件 (t=0)
    x_ic = np.linspace(-1, 1, 100).reshape(-1, 1)
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(np.pi * x_ic)
    # 边界条件 (x=-1, x=1)
    t_bc = np.linspace(0, 1, 100).reshape(-1, 1)
    x_bc1 = -np.ones_like(t_bc)
    x_bc2 = np.ones_like(t_bc)
    u_bc = np.zeros_like(t_bc)
    
    # b. 准备配置点 (用于计算物理损失)
    N_collocation = 10000
    t_col = np.random.rand(N_collocation, 1)
    x_col = np.random.uniform(-1, 1, (N_collocation, 1))

    # c. 构建模型和优化器
    model = build_pinn_model([2, 20, 20, 20, 1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # d. 训练循环
    print("开始训练PINN...")
    for epoch in range(2000):
        with tf.GradientTape() as tape:
            # 数据损失
            u_pred_ic = model(tf.concat([t_ic, x_ic], 1))
            u_pred_bc1 = model(tf.concat([t_bc, x_bc1], 1))
            u_pred_bc2 = model(tf.concat([t_bc, x_bc2], 1))
            loss_d = tf.reduce_mean(tf.square(u_ic - u_pred_ic)) + \
                     tf.reduce_mean(tf.square(u_bc - u_pred_bc1)) + \
                     tf.reduce_mean(tf.square(u_bc - u_pred_bc2))
            # 物理损失
            loss_p = physics_loss(model, tf.constant(t_col, dtype=tf.float32), 
                                  tf.constant(x_col, dtype=tf.float32))
            
            total_loss = loss_d + loss_p
        
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss={total_loss.numpy():.4f}, Data Loss={loss_d.numpy():.4f}, Physics Loss={loss_p.numpy():.4f}")
    
    print("训练完成.")
    
    # e. 可视化结果
    t_test, x_test = np.meshgrid(np.linspace(0, 1, 101), np.linspace(-1, 1, 101))
    tx_test = np.hstack((t_test.flatten()[:,None], x_test.flatten()[:,None]))
    u_pred = model.predict(tx_test).reshape(t_test.shape)
    
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(t_test, x_test, u_pred, cmap='rainbow')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('PINN Solution for 1D Burgers Equation')
    plt.show()

if __name__ == "__main__":
    exercises()