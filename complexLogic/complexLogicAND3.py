import numpy as np
import collections

# Information
# Date: 2026-04-10
# Created by WHH
# Description: 验证 3-input AND 逻辑门的 Ising 模型参数是否正确，通过统计模拟退火的结果来评估模型的准确性。 
# 参数来源于之前求解的结果，包含了输入节点、输出节点和辅助节点的偏置和耦合强度。
# 通过统计最终状态的分布，评估模型在满足 AND 逻辑关系的情况下的表现，并计算全局逻辑准确率。
# 注意：参数的选择和模拟退火的温度 T 可能需要根据实际情况进行调整，以确保模型能够正确地收敛到预期的逻辑状态。
# Results: 输出每种输入组合对应的输出状态的统计结果，并计算全局逻辑准确率。
# 功能正确，但合法状态之间的偏差稍大

def verify_3in_and_with_aux_counts(num_samples=10000, T=5):
    # 1. 填入你求解出的参数 (image_be8300.png)
    h = [-2, -2, -2, 10, 6]  # [h0, h1, h2, h3, h4]
    
    # 构建 J 字典
    j = {}
    # 输入间耦合 (j01, j02, j12)
    j[(0, 1)] = j[(0, 2)] = j[(1, 2)] = 1
    # 输入到输出 (j03, j13, j23)
    j[(0, 3)] = j[(1, 3)] = j[(2, 3)] = -3
    # 输入到辅助 (j04, j14, j24)
    j[(0, 4)] = j[(1, 4)] = j[(2, 4)] = -1
    # 输出到辅助 (j34)
    j[(3, 4)] = 7

    def get_energy(s):
        # 基础偏置能量
        energy = sum(h[i] * s[i] for i in range(5))
        # 耦合能量
        for (u, v), val in j.items():
            energy += val * s[u] * s[v]
        return energy

    final_states = []

    # 2. 进行 1000 次独立模拟退火
    for _ in range(num_samples):
        # 随机初始化 5 个节点状态为 {-1, 1}
        s = np.random.choice([-1, 1], size=5)
        
        # 演化步数，确保收敛
        for _ in range(200):
            for i in range(5):
                s_flip = s.copy()
                s_flip[i] *= -1
                dE = get_energy(s_flip) - get_energy(s)
                
                # Metropolis 接受概率
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    s = s_flip
        
        # 转换为逻辑 0/1 格式，只记录 [In1, In2, In3, Out]
        # 辅助位 (index 4) 不参与逻辑展示
        logical_state = tuple((s[:4] + 1) // 2)
        final_states.append(logical_state)

    # 3. 统计结果
    counts = collections.Counter(final_states)
    
    print(f"{'='*60}")
    print(f"3-Input AND 验证结果 (含辅助位, 总样本: {num_samples})")
    print(f"{'状态 (A B C Out)':<18} | {'次数':<8} | {'占比':<8} | {'逻辑性'}")
    print("-" * 60)

    # 遍历所有 16 种可能的逻辑组合
    total_valid = 0
    for bits in itertools.product([0, 1], repeat=4):
        s1, s2, s3, s4 = bits
        is_legal = ((s1 & s2 & s3) == s4)
        count = counts[bits]
        percentage = (count / num_samples) * 100
        
        if is_legal:
            total_valid += count
            tag = "✔ (合法)"
        else:
            tag = "✘ (冲突)" if count > 0 else ""
            
        print(f"{str(bits):<18} | {count:<8} | {percentage:>6.1f}% | {tag}")

    print("-" * 60)
    print(f"全局逻辑准确率 (Total Accuracy): {total_valid/num_samples*100:.2f}%")
    print(f"{'='*60}")

import itertools
verify_3in_and_with_aux_counts()