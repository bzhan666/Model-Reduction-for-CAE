# -*- coding: utf-8 -*-
"""
1. 生成“伪”大数据
2. 实现增量POD流程

Created on Tue Dec  2 22:33:41 2025

@author: bzhan666
"""

import numpy as np
import os
from sklearn.decomposition import IncrementalPCA
import time

class BigDataPOD:
    def __init__(self, filename, n_space, n_time, n_components=20):
        self.filename = filename
        self.n_space = n_space
        self.n_time = n_time
        self.n_components = n_components
        self.ipca = IncrementalPCA(n_components=n_components)
        
        # 内存映射数组 (在磁盘上，不占内存)
        self.data_mmap = None

    def generate_fake_big_data(self):
        """
        生成一个大的二进制文件，模拟大型CFD数据。
        使用 memmap 直接写入磁盘。
        """
        print(f"正在生成大数据文件: {self.filename} ...")
        print(f"维度: {self.n_space} x {self.n_time}")
        expected_size_gb = (self.n_space * self.n_time * 8) / (1024**3) # float64 = 8 bytes
        print(f"预计大小: {expected_size_gb:.2f} GB")

        # 创建内存映射文件 (写入模式 'w+')
        fp = np.memmap(self.filename, dtype='float64', mode='w+', shape=(self.n_time, self.n_space))
        
        # 分块写入，防止内存溢出
        chunk_size = 100
        for i in range(0, self.n_time, chunk_size):
            end = min(i + chunk_size, self.n_time)
            # 生成一些随机波动数据模拟流场
            # 注意：scikit-learn的PCA默认是 (n_samples, n_features) 即 (时间, 空间)
            # 这与我们之前的 (空间, 时间) 不同，要注意转置关系
            fp[i:end] = np.random.randn(end-i, self.n_space) + np.sin(np.linspace(i, end, end-i))[:, None]
            if i % 1000 == 0:
                print(f"  已生成 {i} / {self.n_time} 步")
        
        # 刷新到磁盘
        fp.flush()
        print("数据生成完毕。")

    def run_incremental_pod(self, batch_size=500):
        """
        使用增量算法计算POD
        """
        print("\n开始执行增量POD (Incremental POD)...")
        start_time = time.time()
        
        # 以只读模式打开映射
        X_mmap = np.memmap(self.filename, dtype='float64', mode='r', shape=(self.n_time, self.n_space))
        
        # 分批次训练 (Partial Fit)
        # 这种方式永远不会把整个数据读入内存
        for i in range(0, self.n_time, batch_size):
            end = min(i + batch_size, self.n_time)
            batch = X_mmap[i:end]
            self.ipca.partial_fit(batch)
            print(f"  已处理批次: {i}-{end}")
            
        end_time = time.time()
        print(f"增量POD完成。耗时: {end_time - start_time:.2f} 秒")
        
        # 获取结果
        # components_ 是模态 (n_components, n_features) -> (r, n_space)
        self.modes = self.ipca.components_ 
        self.singular_values = self.ipca.singular_values_
        print(f"获得模态形状: {self.modes.shape}")

    def transform_big_data(self, batch_size=500):
        """
        将大数据投影到低维空间 (也是分块进行)
        """
        print("\n开始投影大数据...")
        X_mmap = np.memmap(self.filename, dtype='float64', mode='r', shape=(self.n_time, self.n_space))
        
        # 结果存入内存 (因为它很小: n_time x r)
        reduced_data = np.zeros((self.n_time, self.n_components))
        
        for i in range(0, self.n_time, batch_size):
            end = min(i + batch_size, self.n_time)
            batch = X_mmap[i:end]
            reduced_data[i:end] = self.ipca.transform(batch)
            
        print(f"投影完成。低维数据形状: {reduced_data.shape}")
        return reduced_data

def main():
    # 模拟一个较大的场景
    # 空间点: 100,000 (比如 100x100x10 的网格)
    # 时间步: 2,000
    # 数据量约 1.6 GB (你可以根据自己电脑内存调整，试试增加到 5000 步)
    N_SPACE = 100000 
    N_TIME = 2000
    FILE_NAME = 'large_cfd_data.dat'
    
    worker = BigDataPOD(FILE_NAME, N_SPACE, N_TIME, n_components=20)
    
    # 1. 生成数据
    if not os.path.exists(FILE_NAME):
        worker.generate_fake_big_data()
    else:
        print("数据文件已存在，直接使用。")
        
    # 2. 增量训练
    worker.run_incremental_pod(batch_size=200)
    
    # 3. 投影
    coeffs = worker.transform_big_data()
    
    
    # 4. 清理 (可选)
    #os.remove(FILE_NAME)

if __name__ == "__main__":
    main()