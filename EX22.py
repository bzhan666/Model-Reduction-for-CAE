# -*- coding: utf-8 -*-
"""
对比 Numpy 和 Dask 在处理大数组时的性能和内存行为。
Created on Thu Dec  4 15:16:48 2025

@author: bzhan666
"""

import numpy as np
import dask.array as da
import time
import os

def create_large_array(shape, filename):
    """创建一个大的磁盘数组用于测试"""
    if not os.path.exists(filename):
        print(f"生成测试数据 {filename}...")
        x = np.memmap(filename, dtype='float64', mode='w+', shape=shape)
        # 填充随机数
        chunk_size = 1000
        for i in range(0, shape[0], chunk_size):
            end = min(i + chunk_size, shape[0])
            x[i:end] = np.random.random((end-i, shape[1]))
        x.flush()
    return filename

def numpy_processing(filename, shape):
    print("\n--- Numpy (Memmap) 处理 ---")
    start = time.time()
    
    # 加载
    data = np.memmap(filename, dtype='float64', mode='r', shape=shape)
    
    # 计算平均场 (这是一个I/O密集型 + 计算密集型操作)
    mean_field = np.mean(data, axis=0)
    
    # 计算标准差
    std_field = np.std(data, axis=0)
    
    # 强制计算
    print(f"结果形状: {mean_field.shape}")
    end = time.time()
    print(f"Numpy 耗时: {end - start:.2f} 秒")

def dask_processing(filename, shape, chunks):
    print("\n--- Dask 处理 ---")
    start = time.time()
    
    # 1. 定义 Dask 数组 (不会读取数据)
    # chunks定义了每个小块的大小，例如 (1000, 10000)
    data_dask = da.from_array(np.memmap(filename, dtype='float64', mode='r', shape=shape), chunks=chunks)
    
    # 2. 定义计算图 (不会执行计算)
    mean_task = da.mean(data_dask, axis=0)
    std_task = da.std(data_dask, axis=0)
    
    # 3. 执行计算 (利用多核)
    # dask会自动安排多个线程同时读取不同的文件块并计算
    mean_field, std_field = da.compute(mean_task, std_task)
    
    print(f"结果形状: {mean_field.shape}")
    end = time.time()
    print(f"Dask 耗时: {end - start:.2f} 秒")

def main():
    # 模拟 2GB+ 数据
    # 10000 时间步 x 25000 空间点
    N_TIME = 10000
    N_SPACE = 25000
    SHAPE = (N_TIME, N_SPACE)
    FILE = 'dask_test.dat'
    
    create_large_array(SHAPE, FILE)
    
    # 1. Numpy 单核运行
    numpy_processing(FILE, SHAPE)
    
    # 2. Dask 多核运行
    # 将数据切分为更小的块，例如每块包含 2000 个时间步
    # chunk size的选择对性能影响很大，通常要在 100MB 左右
    CHUNKS = (2000, N_SPACE) 
    dask_processing(FILE, SHAPE, CHUNKS)
    
    # 清理
    # os.remove(FILE)

if __name__ == "__main__":
    main()