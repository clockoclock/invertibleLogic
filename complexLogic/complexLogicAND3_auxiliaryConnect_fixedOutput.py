import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.lines import Line2D

# Information
# Created by: WHH
# Date: 2026-05-18
# Description: 6节点 Copy-Link 拓扑 3输入 AND 门在固定输出端 (Clamped Output) 下的逆向退火统计分析

def run_clamped_output_analysis_copylink(fixed_out_val=0, num_samples=1000, T_start=5.0, T_end=0.1, steps=1000):
    """
    fixed_out_val: 固定输出的值，0 或 1。
    - 如果固定为 1：理想情况下，输入只有一种可能，即 111 (对应 1111)。
    - 如果固定为 0：理想情况下，输入有 7 种可能（0000, 0010, 0100, 0110, 1000, 1010, 1100）。
    """
    # ==========================================
    # 1. 参数配置 (6节点 Copy-Link 方案)
    # ==========================================
    # 节点对应关系: 0:m0(In1), 1:m1(In2), 2:m4a, 3:m4b, 4:m2(In3), 5:m3(Out)
    h = np.array([-1.0, -1.0, 2.0, -1.0, -1.0, 2.0])

    J = np.zeros((6, 6))
    # Gate A (m0, m1 -> m4a)
    J[0, 1] = J[1, 0] = 1.0
    J[0, 2] = J[2, 0] = -2.0
    J[1, 2] = J[2, 1] = -2.0

    # Gate B (m4b, m2 -> m3)
    J[3, 4] = J[4, 3] = 1.0
    J[3, 5] = J[5, 3] = -2.0
    J[4, 5] = J[5, 4] = -2.0

    # Copy-Link (m4a <-> m4b 强耦合约束)
    J[2, 3] = J[3, 2] = -2.0

    # 转换逻辑固定值到物理自旋空间 (0 -> -1, 1 -> 1)
    fixed_out_spin = 1 if fixed_out_val == 1 else -1
    
    # 全局 16 种宏观逻辑状态标签
    state_labels = [f"{i:04b}" for i in range(16)]
    
    # 全局合法基态列表 (对应真值表合法解)
    global_valid_states = [0, 2, 4, 6, 8, 10, 12, 15]
    
    # 根据固定输出筛选当前约束下的“合法预期解”
    # 状态格式为 4-bit 整数: (In1*8 + In2*4 + In3*2 + Out*1)
    if fixed_out_val == 1:
        current_expected_states = [15]  # 即 1111
    else:
        current_expected_states = [0, 2, 4, 6, 8, 10, 12]  # 后缀为0的那些合法态

    def get_energy(s):
        """计算 6 节点全局哈密顿量总能量"""
        return np.dot(h, s) + 0.5 * np.dot(s, np.dot(J, s))

    final_results = []

    print(f"开始模拟退火... 约束条件: 输出端 Out 强制固定为 = {fixed_out_val} (共进行 {num_samples} 轮独立实验)")
    
    for n in range(num_samples):
        # 1. 初始化状态 (6节点)
        s = np.random.choice([-1, 1], size=6)
        # 强行硬约束：初始化时就将输出位(索引5, 即m3)设为目标物理自旋值
        s[5] = fixed_out_spin
        
        # 2. 退火演化
        for step in range(steps):
            # 动态线性降温
            T = T_start + (T_end - T_start) * (step / steps)
            
            # 关键修改：随机选择翻转节点时，排除已被固定的输出端节点 5
            # 只在 0, 1, 2, 3, 4 (输入位与内部复制辅助位) 之间随机选点
            i = np.random.choice([0, 1, 2, 3, 4])
            
            s_flip = s.copy()
            s_flip[i] *= -1
            dE = get_energy(s_flip) - get_energy(s)
            
            if dE < 0 or (T > 1e-11 and np.random.rand() < np.exp(-dE/T)):
                s = s_flip
        
        # 3. 提取最终冻结时的宏观逻辑状态位 (剥离内部辅助位差异)
        m = ((s + 1) // 2).astype(int)
        # 映射为 4 位宏观态二进制: m0 m1 m4 m3 -> In1 In2 In3 Out
        state_int = (m[0] << 3) | (m[1] << 2) | (m[4] << 1) | m[5]
        final_results.append(state_int)

    # 4. 统计处理
    counts = Counter(final_results)
    total_counts = [counts.get(i, 0) for i in range(16)]
    
    # 5. 控制台精确数据报表
    print("\n" + "="*75)
    print(f"    【6节点 Copy-Link 模型 Out = {fixed_out_val}】 反向可逆求解概率与均匀性统计")
    print("="*75)
    print(f"{'宏观逻辑状态 (In123 Out)':<24} | {'收敛频次':<10} | {'实际概率':<10} | {'解的性质'}")
    print("-"*75)
    
    total_expected_observed = 0
    for i in range(16):
        c = total_counts[i]
        prob = (c / num_samples) * 100
        
        if i in current_expected_states:
            status_str = "✔ 符合预期合法解"
            total_expected_observed += c
        elif i in global_valid_states:
            status_str = "⚠ 合法解，但被固定条件排除"
        else:
            status_str = "✘ 非法高能态"
            
        print(f"       {i:04b}           |    {c:<7} |   {prob:>5.1f}%   | {status_str}")
        
    print("-"*75)
    print(f"当前固定约束下的解捕获率 (Target Capture Rate): {total_expected_observed / num_samples * 100:.2f}%")
    if len(current_expected_states) > 1:
        print(f"理想均匀状态下，各有效解的平摊概率应接近: {100 / len(current_expected_states):.2f}%")
    print("="*75)

    # 6. 可视化柱状图展示
    plt.figure(figsize=(12, 7), dpi=100)
    colors = []
    for i in range(16):
        if i in current_expected_states:
            colors.append('#2ecc71')   # 绿色 (目标解)
        elif i in global_valid_states:
            colors.append('#bdc3c7')   # 灰色 (受限解)
        else:
            colors.append('#e74c3c')   # 红色 (非法解)
            
    bars = plt.bar(range(16), total_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + num_samples*0.005,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.xticks(range(16), state_labels, rotation=45, family='monospace')
    plt.title(f"Clamped Output (Out = {fixed_out_val}) Energy Landscape Exploration\n({num_samples} Annealing Runs, 6-Node Copy-Link AND)", fontsize=13)
    plt.ylabel("Observed Frequency (Times)", fontsize=12)
    plt.xlabel("System Logic State (In1 In2 In3 Out)", fontsize=12)
    
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', lw=4, label=f'Expected Valid States under Out={fixed_out_val}'),
        Line2D([0], [0], color='#bdc3c7', lw=4, label='Valid States Blocked by Clamping'),
        Line2D([0], [0], color='#e74c3c', lw=4, label='Illegal High Energy States')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 可在此处修改固定输出的值 (0 或 1) 以进行不同的逆向验证
    run_clamped_output_analysis_copylink(fixed_out_val=0, num_samples=1000)