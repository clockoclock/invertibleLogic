import numpy as np
import collections
import itertools

# Information
# Created by: WHH
# Date: 2026-04-08
# Description: Ising 模型参数求解器 - 3 输入 AND 逻辑门（引入辅助位）逆向验证，固定输出为0  
# Result：功能正确

# 修订05-18
# Created by: WHH
# 原代码温度恒定，未能执行退火过程，现添加初始和终止温度参数，并在每次蒙特卡洛步骤中逐渐降低温度，以更真实地模拟退火过程。
# 通过调整 T_start 和 T_end 参数，用户可以观察不同退火速率对系统状态分布的影响，从而更深入地理解退火过程在 Ising 模型中的作用。

def verify_3in_and_inverse(num_samples=1000, T_start=1.0, T_end=0.1， steps=250):
    """
    num_samples: 总退火次数
    T_start: 初始温度
    T_end: 最终温度
    steps: 每次退火的蒙特卡洛步数
    """
    # 1. 载入 image_be8300.png 的参数配置
    h = [-2, -2, -2, 10, 6]  # [In1, In2, In3, Out, Aux]
    
    j = {}
    j[(0, 1)] = j[(0, 2)] = j[(1, 2)] = 1      # 输入间
    j[(0, 3)] = j[(1, 3)] = j[(2, 3)] = -3     # 输入到输出
    j[(0, 4)] = j[(1, 4)] = j[(2, 4)] = -1     # 输入到辅助
    j[(3, 4)] = 7                              # 输出到辅助

    # 固定输出位 m3 = 0 (逻辑0对应物理-1)
    clamp_node = 3
    clamp_val_phys = -1 

    def get_energy(s):
        energy = sum(h[i] * s[i] for i in range(5))
        for (u, v), val in j.items():
            energy += val * s[u] * s[v]
        return energy

    final_results = []

    # 2. 模拟退火循环
    for _ in range(num_samples):
        # 初始化：固定输出位为 -1，辅助位随机，输入位随机
        s = np.random.choice([-1, 1], size=5)
        s[clamp_node] = clamp_val_phys
        
        # 退火演化
        for _ in range(250):
            for i in range(5):
                # 跳过被钳位的输出位索引 3
                if i == clamp_node:
                    continue
                
                s_flip = s.copy()
                s_flip[i] *= -1
                dE = get_energy(s_flip) - get_energy(s)
                
                # 线性降温计划
                T = T_start + (T_end - T_start) * (_ / 250)

                # Metropolis 准则
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    s = s_flip
        
        # 记录逻辑状态 (A, B, C, Out)
        logical_state = tuple((s[:4] + 1) // 2)
        final_results.append(logical_state)

    # 3. 频数统计与分析
    counts = collections.Counter(final_results)
    
    print(f"{'='*65}")
    print(f"逆向验证：固定输出 Out = 0 (样本总数: {num_samples})")
    print(f"{'状态 (A B C Out)':<20} | {'次数':<8} | {'占比':<8} | {'备注'}")
    print("-" * 65)

    total_illegal = 0
    # 遍历所有输出为 0 的输入组合 (2^3 = 8种)
    for in_bits in itertools.product([0, 1], repeat=3):
        state = in_bits + (0,) # 组合成 (A, B, C, 0)
        count = counts[state]
        percentage = (count / num_samples) * 100
        
        # 逻辑判定：对于 AND 门，若 Out=0，输入不能全为 1
        is_legal = (sum(in_bits) < 3) 
        if not is_legal and count > 0:
            total_illegal += count
            note = "✘ (逻辑冲突!)"
        elif is_legal:
            note = "✔ (合法解)"
        else:
            note = ""

        print(f"{str(state):<20} | {count:<8} | {percentage:>6.1f}% | {note}")

    print("-" * 65)
    print(f"非法状态 (1,1,1,0) 出现总数: {total_illegal}")
    print(f"逆向计算成功率: {(num_samples - total_illegal) / num_samples * 100:.2f}%")
    print(f"{'='*65}")

verify_3in_and_inverse(num_samples=1000, T_start=1.0, T_end=0.1， steps=1000)