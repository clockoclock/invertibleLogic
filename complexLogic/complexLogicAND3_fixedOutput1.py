import numpy as np
import collections
import itertools

# Information
# Created by: WHH
# Date: 2026-04-08
# Description: Ising 模型参数求解器 - 3 输入 AND 逻辑门（引入辅助位）逆向验证，固定输出为1
# Result：功能正确

def verify_3in_and_inverse_fixed_1(num_samples=1000, T=0.5):
    # 1. 载入 image_be8300.png 的参数配置
    h = [-2, -2, -2, 10, 6]  # [In1, In2, In3, Out, Aux]
    
    j = {}
    j[(0, 1)] = j[(0, 2)] = j[(1, 2)] = 1      # 输入间耦合
    j[(0, 3)] = j[(1, 3)] = j[(2, 3)] = -3     # 输入到输出耦合
    j[(0, 4)] = j[(1, 4)] = j[(2, 4)] = -1     # 输入到辅助耦合
    j[(3, 4)] = 7                              # 输出到辅助耦合

    # 固定输出位 m3 = 1 (逻辑1对应物理+1)
    clamp_node = 3
    clamp_val_phys = 1 

    def get_energy(s):
        energy = sum(h[i] * s[i] for i in range(5))
        for (u, v), val in j.items():
            energy += val * s[u] * s[v]
        return energy

    final_results = []

    # 2. 模拟退火循环
    for _ in range(num_samples):
        # 初始化：固定输出位为 1，其余随机
        s = np.random.choice([-1, 1], size=5)
        s[clamp_node] = clamp_val_phys
        
        # 退火演化
        for _ in range(250):
            for i in range(5):
                if i == clamp_node: continue
                
                s_flip = s.copy()
                s_flip[i] *= -1
                dE = get_energy(s_flip) - get_energy(s)
                
                # Metropolis 准则
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    s = s_flip
        
        # 记录逻辑状态 (A, B, C, Out)
        logical_state = tuple((s[:4] + 1) // 2)
        final_results.append(logical_state)

    # 3. 统计输出
    counts = collections.Counter(final_results)
    
    print(f"{'='*65}")
    print(f"逆向验证：固定输出 Out = 1 (总样本: {num_samples})")
    print(f"{'状态 (A B C Out)':<20} | {'次数':<8} | {'占比':<8} | {'备注'}")
    print("-" * 65)

    # 遍历所有输出为 1 的输入组合 (2^3 = 8种)
    for in_bits in itertools.product([0, 1], repeat=3):
        state = in_bits + (1,) 
        count = counts[state]
        percentage = (count / num_samples) * 100
        
        # 逻辑判定：只有 (1,1,1,1) 是合法解
        is_correct_factor = (sum(in_bits) == 3)
        if is_correct_factor:
            note = "★ (正确反推解)"
        else:
            note = "✘ (逻辑错误)" if count > 0 else ""

        print(f"{str(state):<20} | {count:<8} | {percentage:>6.1f}% | {note}")

    print("-" * 65)
    print(f"逆向计算成功率 (命中 1,1,1): {counts[(1,1,1,1)] / num_samples * 100:.2f}%")
    print(f"{'='*65}")

verify_3in_and_inverse_fixed_1()