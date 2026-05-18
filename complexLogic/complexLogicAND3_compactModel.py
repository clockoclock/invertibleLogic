import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Information
# Created by: WHH
# Date: 2026-04-20
# Description: 级联堆叠的 3 输入 AND 逻辑门模拟退火统计分析
# 该代码实现了一个基于 Ising 模型的级联堆叠 3 输入 AND 逻辑门模拟器，使用了模拟退火算法来观察系统状态的演化轨迹，并通过统计分析展示了系统在不同阶段的状态分布情况。
# 通过定义一个能量函数，并使用 Metropolis 演化来模拟系统的状态变化，最终统计每个状态的出现频率，并与 AND 逻辑的合法状态进行对比。
# 用户可以调整 num_samples、steps 和降温计划来观察不同条件下系统的行为和合法状态的捕获率，从而更深入地理解 Ising 模型在逻辑门实现中的应用。
# 注意：该代码依赖于 NumPy 和 Matplotlib 库，需要提前安装（pip install numpy matplotlib）才能运行。
# Result: 功能正常，能够正确模拟级联堆叠的 3 输入 AND 逻辑门的行为，并且在统计分析中显示合法状态的捕获率，同时通过柱状图展示了系统状态的分布情况。

# 测试了在AND1和AND2设置了不同的耦合强度之间的区别，实测发现，当AND2的耦合强度设置为与AND1相同（1.0）时，合法状态的捕获率更高，且分布更集中；而当AND2的耦合强度设置为AND1的1.5倍（1.5）时，合法状态的捕获率明显降低，且分布更分散。这表明在级联堆叠的逻辑门设计中，保持各级之间的耦合强度一致有助于提高系统稳定性和合法状态的捕获率。   

# 修订05-18
# Created by: WHH
# 原代码温度恒定，未能执行退火过程，现添加初始和终止温度参数，并在每次蒙特卡洛步骤中逐渐降低温度，以更真实地模拟退火过程。
# 通过调整 T_start 和 T_end 参数，用户可以观察不同退火速率对系统状态分布的影响，从而更深入地理解退火过程在 Ising 模型中的作用。

def run_cascaded_statistical_analysis(num_samples=1000, T_start=5.0, T_end=0.1, steps=1000):
    # --- 级联堆叠参数 (带能量梯度补偿) ---
    # 节点索引: 0:In1, 1:In2, 2:In3, 3:Out, 4:Aux
    # 权重逻辑: 后级(Gate B)强度是前级(Gate A)的 1.5 倍
    h = [-1.0, -1.0, -1.0, 2.0, 1.0]  # h4 = 1.0(A_out) + -1.5(B_in) AND2和AND1同等耦合强度
    # h = [-1.0, -1.0, -1.5, 3.0, 0.5]  # h4 = 1.0(A_out) + -1.5(B_in) AND2和AND1差1.5倍耦合强度
    j = {
        (0, 1): 1.0, (0, 4): -2.0, (1, 4): -2.0,  # Gate A: m0,m1 -> m4
        # (4, 2): 1.5, (4, 3): -3.0, (2, 3): -3.0   # Gate B: m4,m2 -> m3
        (4, 2): 1.0, (4, 3): -2.0, (2, 3): -2.0   # Gate B: m4,m2 -> m3
    }
    
    # 16个状态的描述 (用于图表展示)
    state_labels = [f"{i:04b}" for i in range(16)]
    # 合法状态 (In1, In2, In3, Out)
    valid_states = [0, 2, 4, 6, 8, 10, 12, 15]

    def get_energy(s):
        e = sum(h[i] * s[i] for i in range(5))
        for (u, v), val in j.items():
            e += val * s[u] * s[v]
        return e

    final_results = []

    print(f"Starting {num_samples} annealing runs...")
    for n in range(num_samples):
        # 初始随机状态
        s = np.random.choice([-1, 1], size=5)
        
        # 快速退火计划
        for step in range(steps):
            # 线性降温计划
            T = T_start + (T_end - T_start) * (step / steps)
            
            i = np.random.randint(0, 5)
            s_flip = s.copy()
            s_flip[i] *= -1
            dE = get_energy(s_flip) - get_energy(s)
            
            if dE < 0 or (T > 1e-11 and np.random.rand() < np.exp(-dE/T)):
                s = s_flip
        
        # 记录最后一步的逻辑位 (前4位)
        state_int = int("".join(map(str, ((s[:4] + 1) // 2).astype(int))), 2)
        final_results.append(state_int)

    # 2. 统计频次
    counts = Counter(final_results)
    total_counts = [counts.get(i, 0) for i in range(16)]
    
    # 3. 绘图展示
    plt.figure(figsize=(12, 7), dpi=100)
    colors = ['#2ecc71' if i in valid_states else '#e74c3c' for i in range(16)]
    
    bars = plt.bar(range(16), total_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.xticks(range(16), state_labels, rotation=45, family='monospace')
    plt.title(f"Statistical Distribution of 1000 Annealing Runs\n(Cascaded 3-Input AND Model)", fontsize=14)
    plt.ylabel("Frequency (Times Observed)", fontsize=12)
    plt.xlabel("System Logic State (In1 In2 In3 Out)", fontsize=12)
    
    # 绘制辅助线说明
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#2ecc71', lw=4, label='Legal States (Energy Minima)'),
                       Line2D([0], [0], color='#e74c3c', lw=4, label='Illegal States (High Energy)')]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

run_cascaded_statistical_analysis()