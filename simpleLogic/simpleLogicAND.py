import numpy as np
import collections

# Information
# Created by: WHH 
# Date: 2026-04-07
# Description: AND 可逆逻辑实现
# 该代码实现了一个基于 Ising 模型的 AND 逻辑门模拟器，使用了模拟退火算法来验证求解得到的参数配置。
# 通过定义一个能量函数，并使用 Gibbs Sampling 或 Metropolis 演化来模拟系统的状态变化，最终统计每个状态的出现频率，并与 AND 逻辑的合法状态进行对比。
# 用户可以调整 num_samples、T 和 steps 参数来观察不同条件下系统的行为和合法状态的捕获率，从而更深入地理解 Ising 模型在逻辑门实现中的应用。
# 注意：该代码依赖于 NumPy 库，需要提前安装（pip install numpy）才能运行。
# Result: 功能正常，能够正确模拟 AND 逻辑门的行为，并且在统计分析中显示合法状态的捕获率。

def verify_ising_gate_with_counts(h, j_dict, num_samples=1000, T=1.0, steps=100):
    """
    h: 偏置列表 [h0, h1, h2]
    j_dict: 耦合字典 {(0,1): v, (0,2): v, (1,2): v}
    num_samples: 总退火次数
    T: 模拟温度
    steps: 每次退火的蒙特卡洛步数
    """
    results = []
    
    # 物理能量函数 (基于 Bipolar -1, 1 空间)
    def get_energy(s):
        e_h = h[0]*s[0] + h[1]*s[1] + h[2]*s[2]
        e_j = j_dict[(0,1)]*s[0]*s[1] + j_dict[(0,2)]*s[0]*s[2] + j_dict[(1,2)]*s[1]*s[2]
        return e_h + e_j

    # 模拟退火循环
    for _ in range(num_samples):
        # 1. 随机初始化状态
        s = np.random.choice([-1, 1], size=3)
        
        # 2. Gibbs Sampling / Metropolis 演化
        for _ in range(steps): 
            for i in range(3):
                s_flip = s.copy()
                s_flip[i] *= -1
                dE = get_energy(s_flip) - get_energy(s)
                
                # 接受准则
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    s = s_flip
        
        # 3. 记录结果 (转回逻辑 0/1 方便查看)
        logical_state = tuple((s + 1) // 2)
        results.append(logical_state)

    # 4. 统计与展示
    counts = collections.Counter(results)
    valid_states = [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]
    
    print(f"{'='*50}")
    print(f"统计分析结果 (总样-m pip --version本数: {num_samples}, 温度: {T})")
    print(f"{'状态 (A B C)':<15} | {'出现次数':<10} | {'占比':<8} | {'是否合法'}")
    print("-" * 50)
    
    # 遍历所有可能的 8 种状态
    total_valid_counts = 0
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                state = (a, b, c)
                count = counts[state]
                percentage = (count / num_samples) * 100
                is_valid = "✔ (合法)" if state in valid_states else "✘ (非法)"
                if state in valid_states: total_valid_counts += count
                
                print(f"{str(state):<15} | {count:<10} | {percentage:>6.1f}% | {is_valid}")
    
    print("-" * 50)
    print(f"总合法状态捕获率: {total_valid_counts / num_samples * 100:.2f}%")
    print(f"{'='*50}")

# 示例：使用你刚才求得的 Bipolar 参数
# 建议：如果发现非法状态多，尝试减小 T 或增大 energy_gap 重新求解 h 和 j
h_params = [-1, -1, 2] 
j_params = {(0, 1): 1, (0, 2): -2, (1, 2): -2}

verify_ising_gate_with_counts(h_params, j_params, num_samples=10000, T=5.0)