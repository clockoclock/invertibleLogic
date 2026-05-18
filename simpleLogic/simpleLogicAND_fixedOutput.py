import numpy as np
import collections

# Information
# Created by WHH
# Date: 2026-04-09
# Description: AND 可逆逻辑的逆验证
# 固定输出为0或者1，查看输入状态的分布情况，验证是否符合 AND 门的逻辑约束
# 修改变量为clamp_val，可以修改为0,1，分别验证输出为0和1的情况，对应自旋节点的物理值为-1和1
# Result：功能验证正确，输出符合预期的逻辑约束

# 修订05-18
# Created by: WHH
# 原代码温度恒定，未能执行退火过程，现添加初始和终止温度参数，并在每次蒙特卡洛步骤中逐渐降低温度，以更真实地模拟退火过程。
# 通过调整 T_start 和 T_end 参数，用户可以观察不同退火速率对系统状态分布的影响，从而更深入地理解退火过程在 Ising 模型中的作用。

def verify_inverse_ising(h, j_dict, clamp_node=2, clamp_val=1, num_samples=1000, T_start=1.0, T_end=0.1, steps=100):
    """
    h: 偏置 [h0, h1, h2]
    j_dict: 耦合 {(0,1): v, (0,2): v, (1,2): v}
    clamp_node: 要固定的节点索引 (2 代表输出位 m3)
    clamp_val: 固定的逻辑值 (0 或 1)
    """
    results = []
    # 将逻辑固定值转换为物理 Bipolar 值 (-1 或 1)
    v_clamp = 2 * clamp_val - 1

    def get_energy(s):
        # 计算当前 3-node 系统的总能量
        e_h = h[0]*s[0] + h[1]*s[1] + h[2]*s[2]
        e_j = j_dict[(0,1)]*s[0]*s[1] + j_dict[(0,2)]*s[0]*s[2] + j_dict[(1,2)]*s[1]*s[2]
        return e_h + e_j

    for _ in range(num_samples):
        # 1. 初始化：固定位设为 v_clamp，其余位随机
        s = np.random.choice([-1, 1], size=3)
        s[clamp_node] = v_clamp
        
        # 2. 迭代演化 (Gibbs Sampling)
        for step in range(steps):
            # 线性降温计划
            T = T_start + (T_end - T_start) * (step / steps)
            
            for i in range(3):
                # 跳过被固定的节点，它不翻转
                if i == clamp_node: continue
                
                s_flip = s.copy()
                s_flip[i] *= -1
                dE = get_energy(s_flip) - get_energy(s)
                
                # Metropolis 准则
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    s = s_flip
        
        results.append(tuple((s + 1) // 2))

    # 3. 统计结果
    counts = collections.Counter(results)
    print(f"{'='*55}")
    print(f"逆向验证：固定输出 m{clamp_node} = {clamp_val} (样本数: {num_samples})")
    print(f"{'状态 (m1 m2 m3)':<18} | {'出现次数':<10} | {'占比':<8} | {'备注'}")
    print("-" * 55)

    for a in [0, 1]:
        for b in [0, 1]:
            # 只显示符合固定条件的状态
            state = (a, b, clamp_val)
            count = counts[state]
            percentage = (count / num_samples) * 100
            
            # 逻辑检查：对于 AND 门，若 m3=0，则 (1,1,0) 是非法的
            is_illegal = (a & b != clamp_val)
            note = "✘ (非法/冲突)" if is_illegal else "✔ (合法解)"
            
            print(f"{str(state):<18} | {count:<10} | {percentage:>6.1f}% | {note}")
    
    print(f"{'='*55}")

# ---------------------------------------------------------
# 建议使用的标准 Bipolar 参数 (h=[-1, -1, 2], J=[1, -2, -2])
# 这些参数经过对称性优化，能更好地支持逆向计算
h_params = [-1, -1, 2] 
j_params = {(0, 1): 1, (0, 2): -2, (1, 2): -2}

# 执行逆向验证
verify_inverse_ising(h_params, j_params, clamp_node=2, clamp_val=1, T_start=1.0, T_end=0.1, steps=1000)