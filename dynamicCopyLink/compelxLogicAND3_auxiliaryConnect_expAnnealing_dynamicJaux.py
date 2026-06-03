import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.lines import Line2D

# information
# Created by: WHH
# Date: 2026-05-22
# Description: 3-input AND 逻辑门（引入辅助位）- 指数退火与可调三段线性分段耦合强度调度仿真验证

# ==========================================
# 1. 全局静态参数配置（子模块内部权重硬核固定）
# ==========================================
# 节点索引定义: 0:In1, 1:In2, 2:In3, 3:Out, 4:m4a, 5:m4b
h = np.array([-1.0, -1.0, -1.0, 2.0, 2.0, -1.0])

def get_system_energy(s, J_matrix):
    """计算当前全自旋系统下的总物理能量 (Hamiltonian)"""
    energy_h = np.dot(h, s)
    # 耦合能量项: 0.5 * s^T * J * s
    energy_j = 0.5 * np.dot(s, np.dot(J_matrix, s))
    return energy_h + energy_j

# ==========================================
# 2. 核心函数：支持指数退火与分段耦合调度的状态机模拟器
# ==========================================
def simulated_annealing_piecewise_dynamic(
    steps=200,             # 退火总轮次（温度衰减次数）
    iters_per_step=10,     # 每轮温度下的单次自旋翻转迭代次数（可调）
    T_start=50.0,          # 针对最大能垒科学量化的高温起点
    T_end=0.2,             # 铁腕锁死低温终点
    # 三段区间的比例（必须相加等于 1.0）
    boundaries=(0.3, 0.4, 0.3), 
    # 每一段对应的辅助节点（Copy-Link）耦合强度
    J_aux_stages=(0.2, 2.0, 5.0)
):
    n = len(h)
    s = np.random.choice([-1, 1], size=n)  # 随机初始状态
    
    # 初始化基础静态耦合矩阵（辅助节点暂设为0，后续动态注入）
    J_base = np.zeros((6, 6))
    # Gate A 内部约束 (m0, m1 -> m4a)
    J_base[0,1] = J_base[1,0] = 1.0
    J_base[0,4] = J_base[4,0] = -2.0
    J_base[1,4] = J_base[4,1] = -2.0
    # Gate B 内部约束 (m4b, m2 -> m3)
    J_base[5,2] = J_base[2,5] = 1.0
    J_base[5,3] = J_base[3,5] = -2.0
    J_base[2,3] = J_base[3,2] = -2.0

    # 计算分段线性时间轴的硬件开关切换点
    stage1_end = int(boundaries[0] * steps)
    stage2_end = stage1_end + int(boundaries[1] * steps)
    
    # 核心：计算指数退火衰减系数 alpha
    T = T_start
    alpha = (T_end / T_start) ** (1.0 / (steps - 1))
    
    # 开始退火轮次循环
    for step in range(steps):
        # 硬件分段状态机：根据当前时钟轮次分配对应的 J_aux 强度
        if step < stage1_end:
            J_aux = J_aux_stages[0]  # 第一阶段：高温自由探索
        elif step < stage2_end:
            J_aux = J_aux_stages[1]  # 第二阶段：黄金相变过滤
        else:
            J_aux = J_aux_stages[2]  # 第三阶段：铁腕锁死收割
        
        # 将当前的 J_aux 刚性注入到全局突触矩阵中（Copy-Link: m4a <-> m4b）
        # 注意：原本代码中互连为 -2.0，此处符号保持与您原代码物理一致（即 -J_aux）
        J_current = J_base.copy()
        J_current[4,5] = J_current[5,4] = -J_aux
        
        # 在当前温度和当前耦合地形下，进行内部蒙特卡洛迭代
        for _ in range(iters_per_step):
            i = np.random.randint(n)
            
            s_flip = s.copy()
            s_flip[i] *= -1
            
            # 依据包含动态 J_aux 的当前总能量做差
            delta_E = get_system_energy(s_flip, J_current) - get_system_energy(s, J_current)
            
            # Metropolis 接受准则
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                s = s_flip
                
        # 完成当前轮次，温度按照指数轨迹衰减
        T *= alpha
        
    return s

# ==========================================
# 3. 统计验证与多弹道仿真运行
# ==========================================
num_runs = 1000
results = []

labels = sorted([format(i, '04b') for i in range(16)])
legal_and3 = ["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"]

print("="*70)
print(" 正在执行【指数退火】+【动态三段线性分段耦合】级联加速芯片仿真...")
print("="*70)

# =========================================================================
# 您可以在这里高度自定义您的退火参数与控制策略：
# =========================================================================
SIM_CONFIG = {
    'steps': 200,                    # 退火总轮次
    'iters_per_step': 8,             # 单次温度下的自旋翻转尝试次数
    'T_start': 50.0,                 # 初始高温
    'T_end': 0.1,                   # 终止低温
    'boundaries': (0.3, 0.4, 0.3),   # 三段区间占比划分 (3:4:3)
    'J_aux_stages': (0.2, 0.5, 5.0)  # 第一段(J_min), 第二段(J_mid), 第三段(J_max)
}

for _ in range(num_runs):
    final_s = simulated_annealing_piecewise_dynamic(**SIM_CONFIG)
    # 提取前4位宏观逻辑状态位 (In1, In2, In3, Out)
    m = (final_s + 1) // 2
    state_str = "".join(map(str, m[:4].astype(int)))
    results.append(state_str)

# 统计频次
counts = Counter(results)
frequencies = [counts[l] for l in labels]

# 报表打印
print("\n" + "="*65)
print("     3 输入 AND 门 (可调三段式指数退火) 宏观状态收敛报表")
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
print(f"配置参数: T_start={SIM_CONFIG['T_start']}, T_end={SIM_CONFIG['T_end']}, 阶段划分={SIM_CONFIG['boundaries']}")
print(f"耦合调度: J_min={SIM_CONFIG['J_aux_stages'][0]} -> J_mid={SIM_CONFIG['J_aux_stages'][1]} -> J_max={SIM_CONFIG['J_aux_stages'][2]}")
print(f"全局逻辑准确率（真值表总捕获率）: {total_valid_observed / num_runs * 100:.2f}%")
print("="*65)

# ==========================================
# 4. 绘图展示
# ==========================================
colors = ['#2ecc71' if l in legal_and3 else '#e74c3c' for l in labels]

plt.figure(figsize=(12, 6), dpi=100)
bars = plt.bar(labels, frequencies, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
plt.title(f"Statistical Distribution under Piecewise $J_{{aux}}$ & Exponential Annealing\n"
          f"(Total Steps: {SIM_CONFIG['steps']}, Iters/Step: {SIM_CONFIG['iters_per_step']})", fontsize=13)
plt.xlabel("Logic State (In1 In2 In3 Out)", fontsize=12)
plt.ylabel("Frequency (Times Observed)", fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=10)

legend_elements = [Line2D([0], [0], color='#2ecc71', lw=4, label='Legal States (True Table Match)'),
                   Line2D([0], [0], color='#e74c3c', lw=4, label='Illegal States (Logic Mismatch)')]
plt.legend(handles=legend_elements, loc='upper right')

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()