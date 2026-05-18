import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# information
# Created by: WHH
# Date: 2026-05-14
# Description: 3-input AND 逻辑门（引入辅助位）单次模拟退火轨迹详细记录与分析
# Result：功能异常，待分析

# 绘图数据记录
history_steps = []
history_states = []
history_energies = []

def run_detailed_sa_and_export(h, J, max_steps=2000, T_init=10.0, T_min=0.01):
    num_nodes = len(h)
    # 初始状态随机或指定为全1/全0
    state = np.random.choice([-1, 1], size=num_nodes)
    
    # 指数冷却计划
    steps = np.arange(max_steps)
    T_schedule = T_init * (T_min / T_init) ** (steps / (max_steps - 1))
    
    debug_log = []

    def calculate_energy(s, h_vec, J_mat):
        # 严格执行: E = sum(h*s) + sum(J*s*s)
        # 假定 J 为上三角矩阵，不使用 0.5 系数以匹配 -11 理论值
        bias_energy = np.sum(h_vec * s)
        coupling_energy = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if J_mat[i, j] != 0:
                    coupling_energy += J_mat[i, j] * s[i] * s[j]
        return bias_energy + coupling_energy
        

    for step in range(max_steps):
        T = T_schedule[step]
        # 1. 核心: 1/6 概率随机选中一个节点
        node_to_flip = np.random.randint(0, num_nodes)
        
        # 2. 计算当前能量与翻转后的能量
        e_curr = calculate_energy(state, h, J)
        
        next_state = state.copy()
        next_state[node_to_flip] *= -1
        e_next = calculate_energy(next_state, h, J)
        
        # 3. 计算能量差 delta_E
        delta_e = e_next - e_curr
        
        # 4. 判断接受概率
        accept = False
        prob = 1.0
        
        if delta_e <= 0:
            # 包含等能跳转 (delta_e == 0)
            accept = True
        else:
            # 玻尔兹曼分布接受较差解
            prob = np.exp(-delta_e / T)
            if np.random.rand() < prob:
                accept = True
        
        # 5. 记录所有细节 (关键对账数据)
        debug_log.append({
            'Step': step,
            'Temp': round(T, 6),
            'Node_Attempted': node_to_flip,
            'Current_State': "".join(['1' if x==1 else '0' for x in state]),
            'Target_State': "".join(['1' if x==1 else '0' for x in next_state]),
            'Energy_Curr': round(e_curr, 4),
            'Energy_Next': round(e_next, 4),
            'Delta_E': round(delta_e, 8),
            'Accept_Prob': prob,
            'Accepted': accept
        })

        # 记录 4-bit 逻辑状态索引 (In1 In2 In3 Out)
        m = (state + 1) // 2
        logic_idx = int("".join(map(str, np.append(m[:2], m[-2:]).astype(int))), 2)

        # 记录绘图数据
        history_steps.append(step)
        history_states.append(logic_idx)
        history_energies.append(e_curr)
        
        if accept:
            state = next_state

    # 导出 Excel
    df = pd.DataFrame(debug_log)
    df.to_excel("sa_debug_trace_full_0421_10h36m.xlsx", index=False)
    print("退火完成，详细轨迹已保存至 sa_debug_trace_full_0421_10h36m.xlsx")
    return state

def export_full_energy_map(h, J):
    num_nodes = len(h)
    records = []
    
    # 遍历所有 64 个状态 (0 到 63)
    for i in range(2**num_nodes):
        # 转换为 6 位二进制
        bin_str = format(i, f'0{num_nodes}b')
        # 逻辑位 m (0, 1) -> 自旋变量 s (-1, 1)
        m = np.array([int(b) for b in bin_str])
        s = 2 * m - 1
        
        # 1. 计算偏置项能量
        bias_list = h * s
        total_bias = np.sum(bias_list)
        
        # 2. 计算耦合项能量 (严格按照上三角矩阵计算)
        total_coupling = 0
        coupling_details = []
        for row in range(num_nodes):
            for col in range(row + 1, num_nodes):
                if J[row, col] != 0:
                    val = J[row, col] * s[row] * s[col]
                    total_coupling += val
        
        total_energy = total_bias + total_coupling
        
        # 记录每一行的详细数据
        records.append({
            'Index': i,
            'Binary_State': bin_str,
            'Total_Energy': round(total_energy, 4),
            'Bias_Sum': round(total_bias, 4),
            'Coupling_Sum': round(total_coupling, 4),
            'm0': m[0], 'm1': m[1], 'm4': m[2], 'm5': m[3], 'm2': m[4], 'm3': m[5]
        })

    # 导出到 Excel
    # df = pd.DataFrame(records)
    # df.to_excel("Ising_Full_Energy_Map_0421_10h07m.xlsx", index=False)
    print("全状态能量图已成功导出至 Ising_Full_Energy_Map_0421_10h07m.xlsx")

# --- 你的配置参数 ---
# 索引顺序按照你 Excel 的: m0, m1, m4, m5, m2, m3 (s0-s5)
h_params = np.array([-1.0, -1.0, 2.0, -1.0, -1.0, 2.0]) 
J_params = np.zeros((6, 6))

# Gate A
J_params[0, 1] = 1.0
J_params[0, 2] = -2.0
J_params[1, 2] = -2.0
# Gate B
J_params[4, 5] = -2.0
J_params[3, 4] = 1.0
J_params[3, 5] = -2.0
# Copy-Link (m4-m5 强耦合)
J_params[2, 3] = -5.0

# 运行
final_state = run_detailed_sa_and_export(h_params, J_params, max_steps=2000)

export_full_energy_map(h_params, J_params)

# --- 绘图 ---
plt.figure(figsize=(15, 8))

# 绘制路径线和状态点
plt.plot(history_steps, history_states, color='gray', alpha=0.2, linewidth=1, zorder=1)
plt.scatter(history_steps, history_states, c=history_energies, cmap='viridis', s=20, edgecolors='white', linewidths=0.3, zorder=2)

# 标注纵坐标
logic_labels = [format(i, '04b') for i in range(16)]
plt.yticks(range(16), logic_labels)

# 标注 8 个合法状态 (AND3 真值表)
# 合法态：0000, 0010, 0100, 0110, 1000, 1010, 1100, 1111
legal_indices = [0, 2, 4, 6, 8, 10, 12, 15]
for idx in legal_indices:
    plt.axhline(y=idx, color='green', alpha=0.15, linewidth=12, zorder=0)

plt.colorbar(label='Ising Energy (E)')
plt.title("Forward Logic Dynamics: Single Annealing Path (All Bits Flip)", fontsize=14)
plt.xlabel("Monte Carlo Iteration Step", fontsize=12)
plt.ylabel("Logic State (In1 In2 In3 Out)", fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()