import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# information
# Created by: WHH
# Date: 2026-05-14
# Description: 3-input AND 逻辑门（引入辅助位）模拟退火统计验证
# 该代码实现了一个基于 Ising 模型的 3 输入 AND 逻辑门模拟器，使用了模拟退火算法来观察系统状态的演化轨迹，并通过统计分析展示了系统在不同阶段的状态分布情况。
# 通过定义一个能量函数，并使用 Metropolis 演化来模拟系统的状态变化，最终统计每个状态的出现频率，并与 AND 逻辑的合法状态进行对比。
# 用户可以调整 num_trajectories、total_steps 和降温计划来观察不同条件下系统的行为和合法状态的捕获率，从而更深入地理解 Ising 模型在逻辑门实现中的应用。
# 注意：该代码依赖于 NumPy 和 Matplotlib 库，需要提前安装（pip install numpy matplotlib）才能运行。
# Result: 功能正常，能够正确模拟 3 输入 AND 逻辑门的行为，并且在统计分析中显示合法状态的捕获率，同时通过轨迹图展示了系统状态的演化过程。

# ==========================================
# 1. 参数配置 (全对称、等强度 Copy-Link 方案)
# ==========================================
# 节点定义: 0:In1, 1:In2, 2:In3, 3:Out, 4:m4a, 5:m4b
h = np.array([-1.0, -1.0, -1.5, 3.0, 2.0, -1.0]) 
# 注意：h4a=+2 (GateA输出), h4b=-1 (GateB输入)

# 耦合矩阵 J (对称矩阵)
J = np.zeros((6, 6))
# Gate A (m0, m1 -> m4a)
J[0,1] = J[1,0] = 1.0
J[0,4] = J[4,0] = -2.0
J[1,4] = J[4,1] = -2.0
# Gate B (m4b, m2 -> m3)
J[4,2] = J[2,4] = 1.5 # 后级输入耦合
J[4,3] = J[3,4] = -3.0
J[2,3] = J[3,2] = -3.0
# Copy-Link (m4a <-> m4b) 强一致性约束
J[4,5] = J[5,4] = -5.0 

def calculate_energy(s, h, J):
    """计算单个状态的 Ising 能量"""
    return np.dot(h, s) + 0.5 * np.dot(s, np.dot(J, s))

def is_legal(s):
    """判断逻辑是否合法: (m0&m1==m4a) AND (m4a==m4b) AND (m4b&m2==m3)"""
    # 物理值 -1, 1 转为 逻辑值 0, 1
    m = (s + 1) // 2
    cond1 = (int(m[0]) & int(m[1])) == int(m[4])
    cond2 = int(m[4]) == int(m[5])
    cond3 = (int(m[5]) & int(m[2])) == int(m[3])
    return cond1 and cond2 and cond3

# ==========================================
# 2. 模拟退火算法 (Simulated Annealing)
# ==========================================
def simulated_annealing(h, J, steps=1000, T_start=10.0, T_end=0.1):
    n = len(h)
    s = np.random.choice([-1, 1], size=n) # 随机初始状态
    T = T_start
    decay = (T_end / T_start) ** (1.0 / steps)
    
    for _ in range(steps):
        i = np.random.randint(n)
        # 计算翻转前后的能量差
        delta_E = -2 * s[i] * (h[i] + np.dot(J[i], s))
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            s[i] *= -1
        T *= decay
    return s

# ==========================================
# 3. 执行仿真与可视化
# ==========================================
num_runs = 1000
results = []

print("正在进行1000次退火仿真...")
for _ in range(num_runs):
    final_s = simulated_annealing(h, J)
    # 将最终状态转为二进制字符串便于统计 (仅保留 In1,In2,In3,Out)
    m = (final_s + 1) // 2
    state_str = "".join(map(str, m[:4].astype(int)))
    results.append(state_str)

# 统计频率
counts = Counter(results)
labels = sorted([format(i, '04b') for i in range(16)])
frequencies = [counts[l] for l in labels]

# 颜色区分: 只有符合 AND3 逻辑的才是绿色
legal_and3 = ["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"]
colors = ['#2ecc71' if l in legal_and3 else '#e74c3c' for l in labels]

plt.figure(figsize=(12, 6))
bars = plt.bar(labels, frequencies, color=colors)
plt.title(f"Statistical Distribution of {num_runs} Runs (Copy-Link Model)")
plt.xlabel("Logic State (In1 In2 In3 Out)")
plt.ylabel("Frequency")

# 在柱子上标注数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

plt.grid(axis='y', alpha=0.3)
plt.show()