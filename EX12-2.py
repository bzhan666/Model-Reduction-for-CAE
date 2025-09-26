# -*- coding: utf-8 -*-
"""
优化版PINN求解1D Burgers方程 

@author: bzhan666
优化要点：
1. 修复数值稳定性问题
2. 改进训练监控
3. 优化可视化效果
4. 保持代码简洁易懂
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Burgers' equation: u_t + u * u_x - nu * u_xx = 0
NU = 0.01 / np.pi

def build_pinn_model(layers):
    """构建PINN模型 - 添加了更好的初始化"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    
    for i in range(1, len(layers) - 1):
        model.add(tf.keras.layers.Dense(
            layers[i], 
            activation='tanh',
            kernel_initializer='glorot_normal'  # 改进初始化
        ))
    
    model.add(tf.keras.layers.Dense(
        layers[-1],
        kernel_initializer='glorot_normal'
    ))
    return model

def physics_loss(model, t, x):
    """计算物理残差损失 - 改进了数值稳定性"""
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        tape.watch(x)
        u = model(tf.concat([t, x], axis=1))
        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
    
    # 检查梯度是否为None
    if u_t is None or u_x is None:
        return tf.constant(0.0, dtype=tf.float32)
    
    u_xx = tape.gradient(u_x, x)
    del tape
    
    if u_xx is None:
        return tf.constant(0.0, dtype=tf.float32)
    
    # Burgers方程残差
    residual = u_t + u * u_x - NU * u_xx
    return tf.reduce_mean(tf.square(residual))

def exercises():
    print("=== bzhan666 ===\n")
    
    # a. 准备数据点
    print("准备训练数据...")
    
    # 初始条件 (t=0)
    x_ic = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(np.pi * x_ic).astype(np.float32)
    
    # 边界条件 (x=-1, x=1)
    t_bc = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)
    x_bc1 = -np.ones_like(t_bc)
    x_bc2 = np.ones_like(t_bc)
    u_bc = np.zeros_like(t_bc)
    
    # b. 准备配置点 (用于计算物理损失)
    N_collocation = 10000
    t_col = np.random.rand(N_collocation, 1).astype(np.float32)
    x_col = np.random.uniform(-1, 1, (N_collocation, 1)).astype(np.float32)
    
    # c. 构建模型和优化器
    print("构建模型...")
    layers = [2, 20, 20, 20, 1]
    model = build_pinn_model(layers)
    
    # 使用学习率衰减
    initial_lr = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=1000,
        decay_rate=0.95
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    print(f"网络结构: {layers}")
    print(f"参数总数: {model.count_params()}")
    
    # d. 训练循环
    print("\n开始训练PINN...")
    epochs = 3000
    loss_history = {'total': [], 'data': [], 'physics': []}
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # 数据损失
            u_pred_ic = model(tf.concat([t_ic, x_ic], 1))
            u_pred_bc1 = model(tf.concat([t_bc, x_bc1], 1))
            u_pred_bc2 = model(tf.concat([t_bc, x_bc2], 1))
            
            loss_ic = tf.reduce_mean(tf.square(u_ic - u_pred_ic))
            loss_bc = tf.reduce_mean(tf.square(u_bc - u_pred_bc1)) + \
                     tf.reduce_mean(tf.square(u_bc - u_pred_bc2))
            loss_data = loss_ic + loss_bc
            
            # 物理损失 - 转换为TensorFlow张量
            loss_phys = physics_loss(model, 
                                   tf.constant(t_col, dtype=tf.float32), 
                                   tf.constant(x_col, dtype=tf.float32))
            
            # 总损失 - 添加权重平衡
            alpha = 1.0  # 数据损失权重
            beta = 1.0   # 物理损失权重
            total_loss = alpha * loss_data + beta * loss_phys
        
        # 计算梯度并更新
        grads = tape.gradient(total_loss, model.trainable_variables)
        
        # 梯度裁剪（防止梯度爆炸）
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # 记录损失
        loss_history['total'].append(float(total_loss))
        loss_history['data'].append(float(loss_data))
        loss_history['physics'].append(float(loss_phys))
        
        # 打印进度
        if epoch % 500 == 0:
            current_lr = optimizer.learning_rate.numpy()
            print(f"Epoch {epoch:4d}: 总损失={total_loss.numpy():.6f}, "
                  f"数据损失={loss_data.numpy():.6f}, "
                  f"物理损失={loss_phys.numpy():.6f}, "
                  f"学习率={current_lr:.2e}")
    
    print(f"训练完成! 最终损失: {total_loss.numpy():.6f}")
    
    # e. 可视化结果
    print("\n生成预测结果...")
    t_test, x_test = np.meshgrid(np.linspace(0, 1, 101), np.linspace(-1, 1, 101))
    tx_test = np.hstack((t_test.flatten()[:,None], x_test.flatten()[:,None])).astype(np.float32)
    u_pred = model.predict(tx_test, verbose=0).reshape(t_test.shape)
    
    # 创建更好的可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('PINN求解1D Burgers方程结果', fontsize=14)
    
    # 1. 解的热力图
    im1 = ax1.pcolormesh(t_test, x_test, u_pred, cmap='RdYlBu_r', shading='auto')
    ax1.set_xlabel('时间 t')
    ax1.set_ylabel('空间 x')
    ax1.set_title('PINN预测解')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 不同时刻的解
    times = [0.0, 0.25, 0.5, 1.0]
    colors = ['blue', 'green', 'orange', 'red']
    for i, t_val in enumerate(times):
        idx = np.argmin(np.abs(np.linspace(0, 1, 101) - t_val))
        ax2.plot(np.linspace(-1, 1, 101), u_pred[:, idx], 
                color=colors[i], linewidth=2, label=f't = {t_val}')
    ax2.set_xlabel('空间 x')
    ax2.set_ylabel('u(t, x)')
    ax2.set_title('不同时刻的解')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练损失曲线
    epochs_list = range(len(loss_history['total']))
    ax3.semilogy(epochs_list, loss_history['total'], 'b-', linewidth=2, label='总损失')
    ax3.semilogy(epochs_list, loss_history['data'], 'r--', linewidth=2, label='数据损失')
    ax3.semilogy(epochs_list, loss_history['physics'], 'g:', linewidth=2, label='物理损失')
    ax3.set_xlabel('训练轮数')
    ax3.set_ylabel('损失值')
    ax3.set_title('训练损失曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 初始条件对比
    x_line = np.linspace(-1, 1, 101)
    u_initial_true = -np.sin(np.pi * x_line)
    u_initial_pred = model.predict(np.column_stack([np.zeros(101), x_line]).astype(np.float32), verbose=0).flatten()
    
    ax4.plot(x_line, u_initial_true, 'b-', linewidth=3, label='真实初始条件')
    ax4.plot(x_line, u_initial_pred, 'r--', linewidth=2, label='PINN预测')
    ax4.set_xlabel('空间 x')
    ax4.set_ylabel('u(0, x)')
    ax4.set_title('初始条件拟合效果')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 性能总结
    print("\n=== 训练总结 ===")
    print(f"网络参数: {model.count_params()}")
    print(f"训练轮数: {epochs}")
    print(f"最终总损失: {loss_history['total'][-1]:.6f}")
    print(f"最终数据损失: {loss_history['data'][-1]:.6f}")
    print(f"最终物理损失: {loss_history['physics'][-1]:.6f}")

if __name__ == "__main__":
    # 设置随机种子（可选）
    tf.random.set_seed(42)
    np.random.seed(42)
    
    exercises()