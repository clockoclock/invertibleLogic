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

# 修订05-18
# Created by: WHH
# 添加对于系统状态分布的统计分析，记录每个状态的出现频率，并通过柱状图展示系统在不同阶段的状态分布情况。
# 通过调整 num_samples、steps 和降温计划，用户可以观察不同条件下系统的行为和合法状态的捕获率，从而更深入地理解 Ising 模型在逻辑门实现中的应用。

# ==========================================
# 1. 参数配置 (完全保持原全对称、等强度 Copy-Link 方案不变)
# ==========================================
# 节点定义: 0:In1, 1:In2, 2:In3, 3:Out, 4:m4a, 5:m4b
h = np.array([-1.0, -1.0, -1.0, 2.0, 2.0, -1.0])

# 耦合矩阵 J (对称矩阵)
J = np.zeros((6, 6))
# Gate A (m0, m1 -> m4a)
J[0,1] = J[1,0] = 1.0
J[0,4] = J[4,0] = -2.0
J[1,4] = J[4,1] = -2.0
# Gate B (m4b, m2 -> m3)
J[4,2] = J[2,4] = 1.0
J[4,3] = J[3,4] = -2.0
J[2,3] = J[3,2] = -2.0
# Copy-Link (m4a <-> m4b) 强一致性约束
J[4,5] = J[5,4] = -2.0

# ==========================================
# 新增：定义全系统总能量计算函数 (依据经典哈密顿量公式)
# ==========================================
def get_system_energy(s):
    """计算当前全自旋系统下的总物理能量 (Hamiltonian)"""
    # 局部场能量项: sum(h_i * s_i)
    energy_h = np.dot(h, s)
    # 耦合能量项: 0.5 * s^T * J * s (由于 J 是全对称的，为了消除双向重复计算必须乘 0.5)
    energy_j = 0.5 * np.dot(s, np.dot(J, s))
    return energy_h + energy_j

def is_legal(s):
    """判断逻辑是否合法: (m0&m1==m4a) AND (m4a==m4b) AND (m4b&m2==m3)"""
    m = (s + 1) // 2
    cond1 = (int(m[0]) & int(m[1])) == int(m[4])
    cond2 = int(m[4]) == int(m[5])
    cond3 = (int(m[5]) & int(m[2])) == int(m[3])
    return cond1 and cond2 and cond3

# ==========================================
# 2. 修改点：替换为由全系统总能量差驱动的模拟退火算法
# ==========================================
def simulated_annealing_global_energy(h, J, steps=1000, T_start=10.0, T_end=0.1):
    n = len(h)
    s = np.random.choice([-1, 1], size=n)  # 随机初始状态
    T = T_start
    decay = (T_end / T_start) ** (1.0 / steps)
    
    for _ in range(steps):
        i = np.random.randint(n)
        
        # 状态拷贝与单自旋翻转构建候选态
        s_flip = s.copy()
        s_flip[i] *= -1
        
        # 依据全系统总能量函数作差：dE = E_new - E_old
        delta_E = get_system_energy(s_flip) - get_system_energy(s)
        
        # Metropolis 接受准则更新状态
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            s = s_flip
        T *= decay
    return s

# ==========================================
# 3. 执行仿真、控制台报表生成与可视化统计分布
# ==========================================
num_runs = 1000
results = []

# 16个宏观逻辑状态定义与 3输入 AND 真值表合法态映射
labels = sorted([format(i, '04b') for i in range(16)])
legal_and3 = ["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"]

print("正在进行1000次由【全系统总能量差】驱动的退火仿真与统计分析...")
for _ in range(num_runs):
    final_s = simulated_annealing_global_energy(h, J)
    # 提取前4位宏观逻辑状态位 (In1, In2, In3, Out)
    m = (final_s + 1) // 2
    state_str = "".join(map(str, m[:4].astype(int)))
    results.append(state_str)

# 统计频次
counts = Counter(results)
frequencies = [counts[l] for l in labels]

# --- 新增：打印详细的控制台数据统计报表 ---
print("\n" + "="*65)
print("     3 输入 AND 门 (Copy-Link拓扑) 宏观逻辑状态收敛统计")
print("="*65)
print(f"{'逻辑状态(In123 Out)':<22} | {'收敛次数':<10} | {'收敛概率':<10} | {'状态性质'}")
print("-"*65)

total_valid_observed = 0
for l in labels:
    c = counts[l]
    prob = (c / num_runs) * 100
    is_valid = "✔ 合法基态" if l in legal_and3 else "✘ 非法高能态"
    if l in legal_and3:
        total_valid_observed += c
    print(f"      {l}          |    {c:<7} |   {prob:>5.1f}%   | {is_valid}")
    
print("-"*65)
print(f"全局逻辑准确率（真值表总捕获率）: {total_valid_observed / num_runs * 100:.2f}%")
print("="*65)

# --- 4. 绘图展示全面分布 (优化色彩与图例) ---
colors = ['#2ecc71' if l in legal_and3 else '#e74c3c' for l in labels]

plt.figure(figsize=(12, 6), dpi=100)
bars = plt.bar(labels, frequencies, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
plt.title(f"Statistical Distribution of {num_runs} Runs\n(Whole-System $\\Delta E$ Driven & Copy-Link Scheme)", fontsize=13)
plt.xlabel("Logic State (In1 In2 In3 Out)", fontsize=12)
plt.ylabel("Frequency (Times Observed)", fontsize=12)

# 在柱状图顶部动态标注捕获次数
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=10)

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='#2ecc71', lw=4, label='Legal States (True Table Match)'),
                   Line2D([0], [0], color='#e74c3c', lw=4, label='Illegal States (Logic Mismatch)')]
plt.legend(handles=legend_elements, loc='upper right')

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()