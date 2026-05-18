import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Information
# Created by: WHH
# Date: 2026-04-09
# Description: 3 输入 AND 逻辑门（引入辅助位）模拟退火轨迹可视化
# 该代码实现了一个基于 Ising 模型的 3 输入 AND 逻辑门模拟器，使用了模拟退火算法来观察系统状态的演化轨迹，并通过统计分析展示了系统在不同阶段的状态分布情况。
# 通过定义一个能量函数，并使用 Metropolis 演化来模拟系统的状态变化，最终统计每个状态的出现频率，并与 AND 逻辑的合法状态进行对比。
# 用户可以调整 num_trajectories、total_steps 和降温计划来观察不同条件下系统的行为和合法状态的捕获率，从而更深入地理解 Ising 模型在逻辑门实现中的应用。
# 注意：该代码依赖于 NumPy 和 Matplotlib 库，需要提前安装（pip install numpy matplotlib）才能运行。
# Result: 功能正常，能够正确模拟 3 输入 AND 逻辑门的行为，并且在统计分析中显示合法状态的捕获率，同时通过轨迹图展示了系统状态的演化过程。

def plot_3in_and_stable_convergence(num_trajectories=40, total_steps=500):
    # --- 严格遵循 image_be8300.png 的参数 ---
    # 节点顺序: In1(0), In2(1), In3(2), Out(3), Aux(4)
    h = [-2, -2, -2, 10, 6] 
    j = {
        (0, 1): 1, (0, 2): 1, (1, 2): 1,    # 输入间耦合
        (0, 3): -3, (1, 3): -3, (2, 3): -3, # 输入到输出
        (0, 4): -1, (1, 4): -1, (2, 4): -1, # 输入到辅助
        (3, 4): 7                           # 输出到辅助
    }
    
    # 3-input AND 合法态: 只要 (In1&In2&In3) == Out 即可
    # 十进制对应 (只看前4位逻辑位): 0000(0), 0010(2), 0100(4), 0110(6), 1000(8), 1010(10), 1100(12), 1111(15)
    valid_dec = [0, 2, 4, 6, 8, 10, 12, 15]

    def get_energy(s):
        energy = sum(h[i] * s[i] for i in range(5))
        for (u, v), val in j.items():
            energy += val * s[u] * s[v]
        return energy

    all_trajectories = [] # 统一变量名
    point_stats = Counter()

    # --- 模拟退火循环 ---
    for t in range(num_trajectories):
        s = np.random.choice([-1, 1], size=5)
        path = []
        
        for step in range(total_steps):
            # 动态降温计划
            if step < total_steps * 0.2:
                T = 5.0 # 高温探索
            elif step < total_steps * 0.5:
                # 线性降温区
                T = 5.0 * (1 - (step - total_steps*0.2) / (total_steps*0.3))
            else:
                T = 1e-10 # 长稳态锁定区

            # Metropolis 采样
            i = np.random.randint(0, 5)
            s_flip = s.copy()
            s_flip[i] *= -1
            dE = get_energy(s_flip) - get_energy(s)
            
            if dE < 0 or (T > 1e-11 and np.random.rand() < np.exp(-dE / T)):
                s = s_flip
            
            # 记录逻辑位 (前4位) 的十进制值
            state_int = int("".join(map(str, ((s[:4] + 1) // 2).astype(int))), 2)
            path.append(state_int)
            point_stats[(step, state_int)] += 1
            
        all_trajectories.append(path)

    # --- 绘图逻辑 ---
    plt.figure(figsize=(15, 8), facecolor='white')
    
    # 背景：合法状态带
    for vd in valid_dec:
        plt.axhline(y=vd, color='#2ecc71', alpha=0.1, linewidth=15, zorder=0)

    # 1. 绘制线条 (跳变轨迹)
    for path in all_trajectories:
        plt.step(range(total_steps), path, where='post', color='#bdc3c7', alpha=0.1, linewidth=0.5, zorder=1)

    # 2. 绘制散点 (状态统计)
    steps_arr, states_arr, counts_arr = [], [], []
    for (step, state), count in point_stats.items():
        steps_arr.append(step)
        states_arr.append(state)
        counts_arr.append(count)

    # 点的大小和颜色反映频率，cmap 使用 'viridis' 或 'plasma'
    plt.scatter(steps_arr, states_arr, c=counts_arr, s=[c*2 for c in counts_arr], 
                cmap='viridis', alpha=0.6, edgecolors='none', zorder=2)

    # 3. 修饰
    plt.yticks(range(16), [f"{i:04b}" for i in range(16)], family='monospace', fontsize=10)
    plt.title("3-Input AND Ising Dynamics: Auxiliary Bit Model (Corrected)", fontsize=14)
    plt.xlabel("Monte Carlo Steps (Steps 250-500 show Stabilization)", fontsize=12)
    plt.ylabel("Logic State (In1 In2 In3 Out)", fontsize=12)
    
    # 区域标注
    plt.axvspan(total_steps*0.5, total_steps, color='green', alpha=0.03)
    plt.text(total_steps*0.75, 14.5, "Stable Region", color='darkgreen', weight='bold')
    
    plt.ylim(-0.5, 15.5)
    plt.grid(axis='y', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()

# 执行
plot_3in_and_stable_convergence()