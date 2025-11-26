# -*- coding: utf-8 -*-
"""
降阶模型(ROM)工具箱的图形用户界面(GUI)。

功能:
- 提供一个可视化的操作界面来使用 `rom_toolbox.py` 中的功能。
- 用户可以通过界面加载数据、选择模型、设置参数、执行训练和预测。
- 将模型预测结果与真实值进行可视化对比。

@author: bzhan666
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from rom_toolbox import ROMToolbox

class ROMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("全能 CAE 降阶建模工具箱 ")
        self.root.geometry("1280x800")
        
        self.toolbox = ROMToolbox()
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self._init_layout()
        
    def _init_layout(self):
        # 主布局：左侧控制栏，右侧绘图区
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(main_paned, padding="10", width=300)
        self.plot_frame = ttk.Frame(main_paned, padding="10")
        
        main_paned.add(control_frame, weight=1)
        main_paned.add(self.plot_frame, weight=4)
        
        # === 1. 数据加载 ===
        group1 = ttk.LabelFrame(control_frame, text="1. 数据源", padding="10")
        group1.pack(fill=tk.X, pady=5)
        
        self.entry_path = ttk.Entry(group1)
        self.entry_path.insert(0, "./fluent_data")
        self.entry_path.pack(fill=tk.X, pady=2)
        
        ttk.Button(group1, text="加载/生成数据", command=self.on_load_data).pack(fill=tk.X, pady=2)
        self.lbl_status = ttk.Label(group1, text="就绪", foreground="gray", font=("Arial", 8))
        self.lbl_status.pack(anchor=tk.W)

        # === 2. 模型选择与参数 ===
        group2 = ttk.LabelFrame(control_frame, text="2. 模型配置", padding="10")
        group2.pack(fill=tk.X, pady=5)
        
        ttk.Label(group2, text="选择模型架构:").pack(anchor=tk.W)
        self.combo_model = ttk.Combobox(group2, values=[
            "Linear: Static POD",
            "Linear: Dynamic POD-DMD",
            "Non-linear: Static AE",
            "Non-linear: Dynamic AE-LSTM"
        ], state="readonly")
        self.combo_model.current(0)
        self.combo_model.pack(fill=tk.X, pady=5)
        self.combo_model.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # 参数网格
        param_frame = ttk.Frame(group2)
        param_frame.pack(fill=tk.X, pady=5)
        
        # Rank / Latent Dim
        ttk.Label(param_frame, text="Rank/Latent:").grid(row=0, column=0, sticky=tk.W)
        self.entry_rank = ttk.Entry(param_frame, width=8)
        self.entry_rank.insert(0, "10")
        self.entry_rank.grid(row=0, column=1, padx=5, pady=2)
        
        # Epochs (NN only)
        ttk.Label(param_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W)
        self.entry_epochs = ttk.Entry(param_frame, width=8)
        self.entry_epochs.insert(0, "50")
        self.entry_epochs.grid(row=1, column=1, padx=5, pady=2)
        
        # Seq Len (LSTM only)
        ttk.Label(param_frame, text="Seq Len:").grid(row=2, column=0, sticky=tk.W)
        self.entry_seq = ttk.Entry(param_frame, width=8)
        self.entry_seq.insert(0, "10")
        self.entry_seq.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Button(group2, text="初始化模型", command=self.on_init_model).pack(fill=tk.X, pady=10)

        # === 3. 执行 ===
        group3 = ttk.LabelFrame(control_frame, text="3. 执行", padding="10")
        group3.pack(fill=tk.X, pady=5)
        
        ttk.Button(group3, text="开始训练", command=self.on_train).pack(fill=tk.X, pady=2)
        ttk.Button(group3, text="运行预测/重构", command=self.on_run).pack(fill=tk.X, pady=2)
        
        # === 4. 持久化 ===
        group4 = ttk.LabelFrame(control_frame, text="4. 保存/加载", padding="10")
        group4.pack(fill=tk.X, pady=5)
        
        frame_io = ttk.Frame(group4)
        frame_io.pack(fill=tk.X)
        ttk.Button(frame_io, text="保存", width=8, command=self.on_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_io, text="加载", width=8, command=self.on_load).pack(side=tk.LEFT, padx=2)

        # === 日志 ===
        ttk.Label(control_frame, text="系统日志:").pack(anchor=tk.W, pady=(10,0))
        self.txt_log = tk.Text(control_frame, height=8, width=30, font=("Consolas", 8))
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # === 绘图区 ===
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化状态
        self.on_model_change(None)

    def log(self, msg):
        self.txt_log.insert(tk.END, "> " + msg + "\n")
        self.txt_log.see(tk.END)

    def on_model_change(self, event):
        """根据选择的模型启用/禁用参数框"""
        model = self.combo_model.get()
        
        # 默认所有启用
        self.entry_epochs.state(['!disabled'])
        self.entry_seq.state(['!disabled'])
        
        if "Linear" in model:
            self.entry_epochs.state(['disabled']) # 线性模型不需要 Epochs
            if "Static" in model:
                self.entry_seq.state(['disabled']) # 静态模型不需要 Seq Len
        elif "Non-linear" in model:
            if "Static" in model:
                self.entry_seq.state(['disabled']) # AE 不需要 Seq Len
        
    def on_load_data(self):
        path = self.entry_path.get()
        self.log(f"加载数据: {path}")
        self.root.update()
        msg = self.toolbox.load_data(path, 80, 40, 100)
        self.log(msg)
        if "成功" in msg:
            self.lbl_status.config(text="数据已加载", foreground="green")

    def on_init_model(self):
        m_type = self.combo_model.get()
        try:
            rank = int(self.entry_rank.get())
            epochs = int(self.entry_epochs.get())
            seq = int(self.entry_seq.get())
            
            msg = self.toolbox.init_model(m_type, rank=rank, epochs=epochs, seq_len=seq)
            self.log(msg)
        except ValueError:
            messagebox.showerror("错误", "参数必须为整数")

    def on_train(self):
        self.log("训练中，请稍候（神经网络可能较慢）...")
        self.root.update()
        msg = self.toolbox.train_model()
        self.log(msg)
        if "完成" in msg:
            messagebox.showinfo("完成", "模型训练完毕")

    def on_run(self):
        try:
            results = self.toolbox.run_task()
            self.update_plot(results)
            self.log("任务执行成功")
        except Exception as e:
            self.log(f"执行失败: {e}")
            messagebox.showerror("错误", str(e))

    def on_save(self):
        f = filedialog.asksaveasfilename(defaultextension=".pkl")
        if f: self.log(self.toolbox.save_model(f))

    def on_load(self):
        f = filedialog.askopenfilename()
        if f: 
            self.log(self.toolbox.load_model(f))
            # 更新下拉框显示
            self.combo_model.set(self.toolbox.model_type)
            self.on_model_change(None)

    def update_plot(self, results):
        self.ax[0].clear()
        self.ax[1].clear()
        
        nx, ny = 80, 40 # 示例固定值，实际应从toolbox.data_shape获取
        
        truth = results['truth'].reshape(ny, nx)
        pred = results['prediction'].reshape(ny, nx)
        
        vmin = min(truth.min(), pred.min())
        vmax = max(truth.max(), pred.max())
        
        im1 = self.ax[0].imshow(truth, cmap='jet', vmin=vmin, vmax=vmax)
        self.ax[0].set_title("Ground Truth")
        self.ax[0].axis('off')
        
        im2 = self.ax[1].imshow(pred, cmap='jet', vmin=vmin, vmax=vmax)
        self.ax[1].set_title(results.get('title', "Prediction"))
        self.ax[1].axis('off')
        
        self.fig.suptitle(results.get('title', "Result Analysis"))
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ROMApp(root)
    root.mainloop()
