import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.lines import Line2D

# Information
# Created by: WHH
# Date: 2026-05-18
# Description: 10节点 双门级联XOR拓扑 在固定输出端 (Clamped Output) 下的逆向退火统计分析

def run_clamped_xor3_analysis(fixed_out_val=1, num_samples=1000, T_start=10.0, T_end=0.1, steps=1200):
    """
    fixed_out_val: 固定最终输出端 m7 的值，0 或 1。
    - 如果固定为 1：理想情况下，输入 [m0, m1, m6] 有 4 种异或为 1 的可能（001, 010, 100, 111）
    - 如果固定为 0：理想情况下，输入 [m0, m1, m6] 有 4 种异或为 0 的可能（000, 011, 101, 110）
    """
    # =========================================================================
    # 1. 严格配置 10 节点物理参数 (h向量与对称化 J矩阵)
    # =========================================================================
    # 节点对应: 0:m0, 1:m1, 2:m2, 3:m3, 4:m4, 5:m5, 6:m6, 7:m7, 8:m8, 9:m9
    h = np.array([0, 0, 3, 3, -3, 0, 0, 3, 3, -3], dtype=float)

    J = np.zeros((10, 10))
    # --- Gate 1 耦合 ---
    J[0, 1] = 5.0; J[0, 3] = 5.0; J[1, 3] = 5.0; J[0, 4] = 5.0; J[1, 4] = 5.0
    J[2, 3] = 6.0; J[2, 4] = -6.0; J[3, 4] = -1.0
    
    # --- Gate 2 耦合 (索引全部平移 +5) ---
    J[0+5, 1+5] = 5.0; J[0+5, 3+5] = 5.0; J[1+5, 3+5] = 5.0; J[0+5, 4+5] = 5.0; J[1+5, 4+5] = 5.0
    J[2+5, 3+5] = 6.0; J[2+5, 4+5] = -6.0; J[3+5, 4+5] = -1.0
    
    # --- Copy-Link 约束 ---
    J[2, 5] = -1000.0
    
    # 矩阵对称化
    J = J + J.T

    # 转换逻辑固定值到物理自旋空间 (0 -> -1, 1 -> 1)
    fixed_out_spin = 1 if fixed_out_val == 1 else -1
    
    # =========================================================================
    # 2. 真值表全集与预期可解集合过滤
    # =========================================================================
    state_labels = [f"{i:04b}" for i in range(16)]
    
    # 3输入 XOR 全局合法基态（满足 (m0^m1^m6) == m7）
    global_valid_states = []
    for i in range(16):
        m0 = (i >> 3) & 1
        m1 = (i >> 2) & 1
        m6 = (i >> 1) & 1
        m7 = (i >> 0) & 1
        if (m0 ^ m1 ^ m6) == m7:
            global_valid_states.append(i)
            
    # 根据固定输出筛选当前钳位状态下的“目标预期解”
    current_expected_states = [i for i in global_valid_states if ((i & 1) == fixed_out_val)]

    def get_energy(s):
        """依据经典哈密顿量公式计算全系统物理总能量"""
        return np.dot(h, s) + 0.5 * np.dot(s, np.dot(J, s))

    final_results = []
    print(f"开始执行模拟退火演化... 约束条件: 最终输出端 m7(索引7) 强制固定为 = {fixed_out_val}")
    print(f"进行 {num_samples} 轮独立平行实验，每次退火迭代 {steps} 步.")

    # =========================================================================
    # 3. 核心退火仿真循环
    # =========================================================================
    for n in range(num_samples):
        # 初始化 10 维随机自旋
        s = np.random.choice([-1, 1], size=10)
        # 强制将钳位目标项锁定
        s[7] = fixed_out_spin
        
        for step in range(steps):
            # 动态线性温度衰减
            T = T_start + (T_end - T_start) * (step / steps)
            
            # 【关键修改】选择翻转节点时，将固定的输出端节点 7 严格排除在外
            i = np.random.choice([0, 1, 2, 3, 4, 5, 6, 8, 9])
            
            s_flip = s.copy()
            s_flip[i] *= -1
            
            dE = get_energy(s_flip) - get_energy(s)
            
            # Metropolis 接受判定
            if dE < 0 or (T > 1e-11 and np.random.rand() < np.exp(-dE / T)):
                s = s_flip
                
        # 退火冻结，提取核心逻辑位: [m0, m1, m6, m7]
        m = ((s + 1) // 2).astype(int)
        state_int = (m[0] << 3) | (m[1] << 2) | (m[6] << 1) | m[7]
        final_results.append(state_int)

    # 统计数据频次
    counts = Counter(final_results)
    total_counts = [counts.get(i, 0) for i in range(16)]

    # =========================================================================
    # 4. 规范化打印控制台精细统计数据报表
    # =========================================================================
    print("\n" + "="*80)
    print(f"      【10节点级联XOR模型 钳位输出 Out(m7) = {fixed_out_val}】 逆向求解均匀性分析")
    print("="*80)
    print(f"{'核心逻辑状态 (m0 m1 m6 m7)':<25} | {'收敛频次':<10} | {'实际概率':<10} | {'解的硬件性质'}")
    print("-"*80)
    
    total_expected_observed = 0
    for i in range(16):
        c = total_counts[i]
        prob = (c / num_samples) * 100
        
        if i in current_expected_states:
            status_str = "✔ 符合逆向预期合法解 (系统全局基态)"
            total_expected_observed += c
        elif i in global_valid_states:
            status_str = "⚠ 理论合法解，但被固定钳位条件排除"
        else:
            status_str = "✘ 逻辑错误态 (高能亚稳态/非法解)"
            
        print(f"        {i:04b} (Index:{i:<2})          |    {c:<7} |   {prob:>5.1f}%   | {status_str}")
        
    print("-"*80)
    print(f"当前输出约束下合法目标解的总捕获率 (Target Capture Rate): {total_expected_observed / num_samples * 100:.2f}%")
    print(f"理想均匀概率下，这 {len(current_expected_states)} 个有效解的平摊概率应各自接近: {100 / len(current_expected_states):.2f}%")
    print("="*80 + "\n")

    # =========================================================================
    # 5. 可视化柱状图绘制 (直观展现多解收敛的均匀分布)
    # =========================================================================
    plt.figure(figsize=(13, 7), dpi=100)
    
    colors = []
    for i in range(16):
        if i in current_expected_states:
            colors.append('#2ecc71')   # 绿色: 当前钳位目标解
        elif i in global_valid_states:
            colors.append('#bdc3c7')   # 灰色: 被锁死的其他逻辑态
        else:
            colors.append('#e74c3c')   # 红色: 逻辑完全不匹配的非法高能态
            
    bars = plt.bar(range(16), total_counts, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
    
    # 顶部标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + num_samples*0.005,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(range(16), state_labels, rotation=45, family='monospace', fontsize=11)
    plt.title(f"Clamped Output (m7 = {fixed_out_val}) Reverse Simulation\n({num_samples} Runs, 10-Node Twin-XOR Cascaded Circuit)", fontsize=13)
    plt.ylabel("Observed Frequency (Times)", fontsize=12)
    plt.xlabel("Core Logic Space State (m0 m1 m6 m7)", fontsize=12)
    
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', lw=4, label=f'Expected Target Solutions (Out={fixed_out_val})'),
        Line2D([0], [0], color='#bdc3c7', lw=4, label='Valid XOR States Blocked by Clamping'),
        Line2D([0], [0], color='#e74c3c', lw=4, label='Illegal Mismatch States')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 可直接通过修改此处的参数(0或1)，进行不同输出强约束下的反向均匀性探测
    run_clamped_xor3_analysis(fixed_out_val=0, num_samples=1000)