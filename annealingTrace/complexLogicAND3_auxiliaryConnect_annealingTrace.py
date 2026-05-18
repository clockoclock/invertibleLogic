import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Information
# Created by: WHH
# Date: 2026-05-18
# Description: 6节点 Copy-Link 方案的 3 输入 AND 逻辑门模拟退火轨迹与收敛均匀性分析

def plot_copylink_3in_and_dynamics(num_trajectories=100, total_steps=1000, T_start=5.0, T_end=0.1):
    # ==========================================
    # 1. 参数配置 (6节点 Copy-Link 方案)
    # ==========================================
    # 节点对应: 0:m0(In1), 1:m1(In2), 2:m4a, 3:m4b, 4:m2(In3), 5:m3(Out)
    h = np.array([-1.0, -1.0, 2.0, -1.0, -1.0, 2.0])

    J = np.zeros((6, 6))
    # Gate A (m0, m1 -> m4a)
    J[0, 1] = J[1, 0] = 1.0
    J[0, 2] = J[2, 0] = -2.0
    J[1, 2] = J[2, 1] = -2.0

    # Gate B (m4b, m2 -> m3)
    J[4, 5] = J[5, 4] = -2.0
    J[3, 4] = J[4, 3] = 1.0
    J[3, 5] = J[5, 3] = -2.0

    # Copy-Link (m4a <-> m4b 强耦合约束)
    J[2, 3] = J[3, 2] = -5.0

    # 自动找出 6 节点 64 种物理状态下的完备合法基态解 (用于图表绿线绘制和概率判定)
    valid_dec = []
    for b in range(64):
        m_check = np.array([(b >> i) & 1 for i in range(6)])
        cond1 = (m_check[0] & m_check[1]) == m_check[2]
        cond2 = m_check[2] == m_check[3]
        cond3 = (m_check[3] & m_check[4]) == m_check[5]
        if cond1 and cond2 and cond3:
            # 转换为 4 位宏观态的十进制索引: (In1, In2, In3, Out) -> (m0, m1, m4, m5)
            macro_idx = (m_check[0] << 3) | (m_check[1] << 2) | (m_check[4] << 1) | m_check[5]
            if macro_idx not in valid_dec:
                valid_dec.append(macro_idx)
    valid_dec.sort() # 对应 [0, 2, 4, 6, 8, 10, 12, 15]

    def get_energy(s):
        """计算 6 节点 Ising 能量"""
        return np.dot(h, s) + 0.5 * np.dot(s, np.dot(J, s))

    all_trajectories = []
    point_stats = Counter()
    final_state_counter = Counter()

    # ==========================================
    # 2. 多轨迹模拟退火循环
    # ==========================================
    print(f"正在进行 {num_trajectories} 次 6 节点 Copy-Link 模型退火仿真...")
    for t in range(num_trajectories):
        s = np.random.choice([-1, 1], size=6)
        path = []
        for step in range(total_steps):
            # 线性降温计划
            T = T_start + (T_end - T_start) * (step / total_steps)

            i = np.random.randint(0, 6)
            s_flip = s.copy()
            s_flip[i] *= -1
            
            # 计算能量差
            dE = get_energy(s_flip) - get_energy(s)
            if dE < 0 or (T > 1e-11 and np.random.rand() < np.exp(-dE / T)):
                s = s_flip
            
            # 提取当前的 0/1 状态值
            m = (s + 1) // 2
            # 映射为 4 位宏观态二进制: m0 m1 m4 m2 -> In1 In2 In3 Out
            macro_int = (int(m[0]) << 3) | (int(m[1]) << 2) | (int(m[4]) << 1) | int(m[5])
            
            path.append(macro_int)
            point_stats[(step, macro_int)] += 1
        
        all_trajectories.append(path)
        # 记录当前退火轨迹终点的宏观状态
        final_state_counter[path[-1]] += 1

    # =========================================================================
    # 3. 终点落点均匀性统计输出
    # =========================================================================
    print("\n" + "="*70)
    print("     6节点 Copy-Link 模型模拟退火最终收敛状态概率均匀性统计")
    print("="*70)
    print(f"{'逻辑状态 (In123 Out)':<22} | {'总收敛次数':<10} | {'实际概率':<10} | {'状态性质'}")
    print("-"*70)
    
    total_valid_runs = 0
    for state_idx in range(16):
        runs = final_state_counter[state_idx]
        probability = (runs / num_trajectories) * 100
        is_legal = "✔ 合法解 (基态)" if state_idx in valid_dec else "✘ 非法解 (亚稳态)"
        
        if state_idx in valid_dec:
            total_valid_runs += runs
            
        print(f"      {state_idx:04b}          |    {runs:<7} |   {probability:>5.1f}%   | {is_legal}")
        
    print("-"*70)
    print(f"总体合法解捕获率 (Ground State Capture Rate): {total_valid_runs / num_trajectories * 100:.2f}%")
    print(f"理想对称状态下，每个合法解的目标收敛概率应为: {100 / len(valid_dec):.2f}%")
    print("="*70 + "\n")

    # ==========================================
    # 4. 绘图逻辑 (动态散点图与灰度轨迹线)
    # ==========================================
    plt.figure(figsize=(15, 8))
    
    # 绘制合法基态的目标绿带提示
    for vd in valid_dec: 
        plt.axhline(y=vd, color='green', alpha=0.08, linewidth=15)
        
    # 绘制所有灰色退火路径演化轨迹
    for path in all_trajectories:
        plt.step(range(total_steps), path, where='post', color='gray', alpha=0.1, linewidth=0.5)
    
    # 提取时间步长密度散点
    steps_arr, states_arr, counts_arr = zip(*[(k[0], k[1], v) for k, v in point_stats.items()])
    plt.scatter(steps_arr, states_arr, c=counts_arr, s=[c*1.5 for c in counts_arr], cmap='plasma', alpha=0.6)
    
    plt.yticks(range(16), [f"{i:04b}" for i in range(16)], family='monospace')
    plt.xlabel("Annealing Steps")
    plt.ylabel("Macro Logic State (In1 In2 In3 Out)")
    plt.title("6-Node Copy-Link 3-Input AND Model: Trajectory & Phase Space Density Analysis")
    plt.colorbar(label='State Occupancy Count')
    plt.show()

if __name__ == "__main__":
    plot_copylink_3in_and_dynamics()