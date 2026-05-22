import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.lines import Line2D

# information
# Created by: WHH
# Date: 2026-05-23
# Description: 3-input AND 逻辑门算法验证 - 指数退火与可调三段线性分段耦合强度调度（支持固定输出）

def get_system_energy(s, J_matrix):
    """纯粹的算法能量函数计算 (Hamiltonian)"""
    h = np.array([-1.0, -1.0, -1.0, 2.0, 2.0, -1.0])
    energy_h = np.dot(h, s)
    energy_j = 0.5 * np.dot(s, np.dot(J_matrix, s))
    return energy_h + energy_j

# ==========================================
# 核心函数：纯算法驱动的退火与分段耦合调度模拟器
# ==========================================
def simulated_annealing_pure_algorithm(
    target_out=None,       # 算法控制：None(自主寻优), 0(固定输出为0), 1(固定输出为1)
    steps=200,             # 退火总轮次（温度衰减次数）
    iters_per_step=10,     # 每轮温度下的单次自旋翻转迭代次数
    T_start=50.0,          # 初始高温
    T_end=0.1,            # 终止低温
    boundaries=(0, 1.0, 0),   # 三段区间占比划分
    J_aux_stages=(0.2, 50, 5.0)  # 每一段对应的辅助节点耦合强度
):
    n = 6
    s = np.random.choice([-1, 1], size=n)  # 随机初始状态
    
    # 算法级初始化：如果要固定输出，直接在初始状态里强制赋对应的值，且后续不再改变
    if target_out == 0:
        s[3] = -1  # 算法对应逻辑 0
    elif target_out == 1:
        s[3] = 1   # 算法对应逻辑 1

    # 初始化基础静态耦合矩阵（子模块内部硬核固定）
    J_base = np.zeros((6, 6))
    J_base[0,1] = J_base[1,0] = 1.0
    J_base[0,4] = J_base[4,0] = -2.0
    J_base[1,4] = J_base[4,1] = -2.0
    J_base[4,2] = J_base[2,4] = 1.0
    J_base[4,3] = J_base[3,4] = -2.0
    J_base[2,3] = J_base[3,2] = -2.0

    # 计算分段线性时间轴的切换点
    stage1_end = int(boundaries[0] * steps)
    stage2_end = stage1_end + int(boundaries[1] * steps)
    
    # 计算指数退火衰减系数 alpha
    T = T_start
    alpha = (T_end / T_start) ** (1.0 / (steps - 1))
    # 线性退火
    decay = (T_end / T_start) ** (1.0 / steps)
    
    # 开始退火轮次循环
    for step in range(steps):
        # 分段状态机：根据当前轮次分配对应的 J_aux 强度
        if step < stage1_end:
            J_aux = J_aux_stages[0]  # 第一阶段：高温自由探索
        elif step < stage2_end:
            J_aux = J_aux_stages[1]  # 第二阶段：黄金相变过滤
        else:
            J_aux = J_aux_stages[2]  # 第三阶段：铁腕锁死收割
        
        # 组装当前阶段的耦合矩阵
        J_current = J_base.copy()
        J_current[4,5] = J_current[5,4] = -J_aux
        
        # 单温度下的迭代
        for _ in range(iters_per_step):
            # 【核心修改】：如果你指定了固定输出，随机挑选节点时，把节点 3（输出位）排除在外
            if target_out is not None:
                # 只能在 [0, 1, 2, 4, 5] 中随机选择翻转节点
                available_nodes = [0, 1, 2, 4, 5]
                i = np.random.choice(available_nodes)
            else:
                # 自主寻优时，所有 6 个节点都可以参与翻转
                i = np.random.randint(n)
            
            s_flip = s.copy()
            s_flip[i] *= -1
            
            # 计算纯粹的算法能量差
            delta_E = get_system_energy(s_flip, J_current) - get_system_energy(s, J_current)
            
            # Metropolis 接受准则
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                s = s_flip
                
        # 温度按照指数轨迹衰减
        # T *= alpha
        # 线性退火
        T *= decay
        
    return s

# ==========================================
# 3. 统计验证与多弹道仿真运行
# ==========================================
num_runs = 1000
results = []
labels = sorted([format(i, '04b') for i in range(16)])

# =========================================================================
# 算法验证控制开关：None(自主寻优), 0(固定输出为0), 1(固定输出为1)
# =========================================================================
TARGET_OUTPUT_MODE = 1  # <--- 在这里修改你想验证的算法模式

SIM_CONFIG = {
    'target_out': TARGET_OUTPUT_MODE,
    'steps': 1000,
    'iters_per_step': 1,
    'T_start': 10.0,
    'T_end': 0.1,
    'boundaries': (0.0, 1.0, 0.0),
    'J_aux_stages': (0.2, 1.5, 5.0)
}

# 依据模式定义算法期待的合法状态集
if TARGET_OUTPUT_MODE == 0:
    legal_states = ["0000", "0010", "0100", "0110", "1000", "1010", "1100"] 
    mode_str = "固定输出 = 0"
elif TARGET_OUTPUT_MODE == 1:
    legal_states = ["1111"] 
    mode_str = "固定输出 = 1"
else:
    legal_states = ["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"]
    mode_str = "自主寻优"

print("="*70)
print(f" 正在执行算法验证: 【{mode_str}】 + 【指数退火】 + 【三段动态耦合强度】")
print("="*70)

for _ in range(num_runs):
    final_s = simulated_annealing_pure_algorithm(**SIM_CONFIG)
    m = (final_s + 1) // 2
    state_str = "".join(map(str, m[:4].astype(int)))
    results.append(state_str)

# 统计频次
counts = Counter(results)
frequencies = [counts[l] for l in labels]

# 报表打印
print("\n" + "="*65)
print(f"     3 输入 AND 门 ({mode_str}) 算法收敛统计")
print("="*65)
print(f"{'逻辑状态(In123 Out)':<22} | {'收敛次数':<10} | {'收敛概率':<10} | {'状态性质'}")
print("-"*65)

total_valid_observed = 0
for l in labels:
    c = counts[l]
    prob = (c / num_runs) * 100
    is_valid = "✔ 目标期望态" if l in legal_states else "✘ 冲突/非目标态"
    if l in legal_states:
        total_valid_observed += c
    print(f"      {l}          |    {c:<7} |   {prob:>5.1f}%   | {is_valid}")
    
print("-"*65)
print(f"算法目标状态总捕获率: {total_valid_observed / num_runs * 100:.2f}%")
print("="*65)

# ==========================================
# 4. 绘图展示
# ==========================================
colors = ['#2ecc71' if l in legal_states else '#e74c3c' for l in labels]

plt.figure(figsize=(12, 6), dpi=100)
bars = plt.bar(labels, frequencies, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
plt.title(f"Pure Algorithm Verification ({mode_str})\n"
          f"(J_stages: {SIM_CONFIG['J_aux_stages']}, Boundaries: {SIM_CONFIG['boundaries']})", fontsize=13)
plt.xlabel("Logic State (In1 In2 In3 Out)", fontsize=12)
plt.ylabel("Frequency (Times Observed)", fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=10)

legend_elements = [Line2D([0], [0], color='#2ecc71', lw=4, label='Target States (Matched)'),
                   Line2D([0], [0], color='#e74c3c', lw=4, label='Filtered States')]
plt.legend(handles=legend_elements, loc='upper right')

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()