import numpy as np
import collections
import itertools

# Information
# Date: 2026-05-16
# Created by JCZ, WHH
# Description: 验证 3-input XOR 逻辑门的 Ising 模型参数是否正确，通过统计模拟退火的结果来评估模型的准确性。 
# 参数来源于之前求解的结果，包含了输入节点、输出节点和辅助节点的偏置和耦合强度。
# 通过统计最终状态的分布，评估模型在满足 XOR 逻辑关系的情况下的表现，并计算全局逻辑准确率。
# 注意：参数的选择和模拟退火的温度 T 可能需要根据实际情况进行调整，以确保模型能够正确地收敛到预期的逻辑状态。
# Results: 输出每种输入组合对应的输出状态的统计结果，并计算全局逻辑准确率。
# 功能正确，且激活了退火函数，使基态收敛更加稳定。

def verify_3in_xor_with_2aux_counts(num_samples=1000, T=5, steps=1000):
    # 1. 填入你求解出的参数 (image_be8300.png)
    h = [0, 0, 3, 3, -3, 0, 0, 3, 3, -3]  # [h0, h1, h2, h3, h4, h5, h6, h7, h8, h9]
    
    # 构建 J 字典
    j = {}
    #输入间耦合
    j[(0, 1)] = 5
    #输入到输出
    j[(0, 2)] = j[(1, 2)] = 0
    #输入到辅助
    j[(0, 3)] = j[(1, 3)] = 5
    j[(0, 4)] = j[(1, 4)] = 5
    #输出到辅助
    j[(2, 3)] = 6
    j[(2, 4)] = -6
    #辅助间耦合
    j[(3, 4)] = -1

    #输入间耦合
    j[(0+5, 1+5)] = 5
    #输入到输出
    j[(0+5, 2+5)] = j[(1+5, 2+5)] = 0
    #输入到辅助
    j[(0+5, 3+5)] = j[(1+5, 3+5)] = 5
    j[(0+5, 4+5)] = j[(1+5, 4+5)] = 5
    #输出到辅助
    j[(2+5, 3+5)] = 6
    j[(2+5, 4+5)] = -6
    #辅助间耦合
    j[(3+5, 4+5)] = -1

    # Copy-Link (m2 <-> m6) 强一致性约束
    j[(2, 5)] = j[(5, 2)] = -2.0

    # 上三角的耦合系数参与哈密顿量计算
    def get_energy(s):
        energy = sum(h[i] * s[i] for i in range(10))
        for (u, v), val in j.items():
            if u < v:  # 严格限制仅上三角参与计算
                energy += val * s[u] * s[v]
        return energy
    
    final_states = []
    
    # 2. 进行多次独立模拟演化
    for _ in range(num_samples):
        # 随机初始化 10 个节点状态为物理自旋 {-1, 1}
        s = np.random.choice([-1, 1], size=10)
        
        # ================== 【核心修改：激活模拟退火过程】 ==================
        for step in range(steps):
            t = T * (1 - step / steps) + 1e-5  # 从初始 T 线性降温至接近 0
            for i in range(10):
                s_flip = s.copy()
                s_flip[i] *= -1
                dE = get_energy(s_flip) - get_energy(s)
                
                # Metropolis 接受准则（使用当前步数的温度 t）
                if dE < 0 or np.random.rand() < np.exp(-dE / t):
                    s = s_flip
        # ===================================================================
        
        # 转换为数字逻辑 0/1 格式，只记录核心逻辑关注的 [In1, In2, In3, Out]
        logical_state = tuple((s[[0, 1, 6, 7]] + 1) // 2)
        final_states.append(logical_state)

    # 3. 统计并打印结果
    counts = collections.Counter(final_states)
    
    print(f"{'='*60}")
    print(f"3-Input XOR 正向功能验证结果 (总样本数: {num_samples})")
    print(f"{'状态 (In1 In2 In3 Out)':<22} | {'出现次数':<8} | {'占比':<8} | {'逻辑判定'}")
    print("-" * 60)

    total_valid = 0
    # 遍历所有 16 种可能的输入输出逻辑组合 (2^4)
    for bits in itertools.product([0, 1], repeat=4):
        s1, s2, s3, s4 = bits
        # XOR 逻辑判定：输入相异时输出为 1，相同时输出为 0
        is_legal = ((s1 ^ s2 ^ s3) == s4)
        count = counts[bits]
        percentage = (count / num_samples) * 100
        
        if is_legal:
            total_valid += count
            tag = "✔ (合法基态)"
        else:
            tag = "✘ (非法冲突)" if count > 0 else ""
            
        print(f"{str(bits):<18} | {count:<8} | {percentage:>6.1f}% | {tag}")

    print("-" * 60)
    print(f"全局逻辑准确率 (Total Accuracy): {total_valid / num_samples * 100:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    verify_3in_xor_with_2aux_counts()