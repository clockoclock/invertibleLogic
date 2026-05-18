import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Information
# Created by: WHH
# Date: 2026-05-18
# Description: 级联堆叠的 3 输入 AND 逻辑门模拟退火轨迹分析（增加终点状态收敛概率均匀性分析）

# 修订05-18
# Created by: WHH
# 可以修改num_samples、T_start、T_end和steps参数来观察不同退火条件下系统状态分布的变化，同时新增了对每条退火轨迹终点状态的统计分析，以评估系统在合法解（基态）与非法解（亚稳态）之间的收敛概率均匀性，从而更深入地理解退火过程在 Ising 模型中的作用和系统的稳定性。

def plot_cascaded_3in_and_dynamics(num_trajectories=100, total_steps=1000, T_start=5.0, T_end=0.1):
    # --- 级联拓扑参数 ---
    # 节点: 0:In1, 1:In2, 2:In3, 3:Out, 4:Aux(Internal)
    h = [-1.0, -1.0, -1.0, 2.0, 1.0]  # h4 = 1.0(A_out) + 1.0(B_in) AND2和AND1同等耦合强度
    # 仅保留级联内的耦合，消除全局耦合
    j = {
        (0, 1): 1, (0, 4): -2, (1, 4): -2,  # 第一级: m0,m1 -> m4
        (4, 2): 1, (4, 3): -2, (2, 3): -2  # 第二级: m4,m2 -> m3
    }
    
    # 全局合法态 (In1, In2, In3, Out):
    valid_dec = [0, 2, 4, 6, 8, 10, 12, 15]

    def get_energy(s):
        energy = sum(h[i] * s[i] for i in range(5))
        for (u, v), val in j.items():
            energy += val * s[u] * s[v]
        return energy

    all_trajectories = []
    point_stats = Counter()
    
    # 新增：用于单独统计每条退火轨迹终点的计数器
    final_state_counter = Counter()

    for t in range(num_trajectories):
        s = np.random.choice([-1, 1], size=5)
        path = []
        for step in range(total_steps):
            # 线性降温计划
            T = T_start + (T_end - T_start) * (step / total_steps)

            i = np.random.randint(0, 5)
            s_flip = s.copy(); s_flip[i] *= -1
            dE = get_energy(s_flip) - get_energy(s)
            if dE < 0 or (T > 1e-11 and np.random.rand() < np.exp(-dE / T)):
                s = s_flip
            
            state_int = int("".join(map(str, ((s[:4] + 1) // 2).astype(int))), 2)
            path.append(state_int)
            point_stats[(step, state_int)] += 1
        
        all_trajectories.append(path)
        
        # 新增：在当前轨迹完成全部退火步数后，记录最终收敛的宏观状态 (终点)
        final_state_counter[path[-1]] += 1

    # =========================================================================
    # 新增功能：分析最终状态收敛频次与概率的统一性
    # =========================================================================
    print("\n" + "="*70)
    print("        Ising 模拟退火最终收敛状态（终点落点）概率均匀性统计")
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

    # --- 绘图逻辑 (完全保持原有功能不变) ---
    plt.figure(figsize=(15, 8))
    for vd in valid_dec: plt.axhline(y=vd, color='green', alpha=0.08, linewidth=15)
    for path in all_trajectories:
        plt.step(range(total_steps), path, where='post', color='gray', alpha=0.1, linewidth=0.5)
    
    steps_arr, states_arr, counts_arr = zip(*[(k[0], k[1], v) for k, v in point_stats.items()])
    plt.scatter(steps_arr, states_arr, c=counts_arr, s=[c*1.5 for c in counts_arr], cmap='plasma', alpha=0.6)
    
    plt.yticks(range(16), [f"{i:04b}" for i in range(16)], family='monospace')
    plt.title("Cascaded 3-Input AND: Energy Gradient Strategy (h0,h1 < h2,h3)")
    plt.show()

plot_cascaded_3in_and_dynamics()