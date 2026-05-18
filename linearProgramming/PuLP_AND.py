import pulp

# Information
# Created by: WHH
# Date: 2026-04-07
# Description: Ising 模型参数求解器 - AND 逻辑门
# 本代码实现了一个基于整数线性规划的 Ising 模型参数求解器，针对 AND 逻辑门设计。
# 通过定义变量 h 和 j 来模拟 Ising 模型中的偏置项和耦合项，并使用约束确保合法状态和非法状态之间的能量差距。
# 该实现采用了 bipolar 映射（0/1 转换为 -1/+1），以更贴近物理 Ising 模型的表示方式。
# 用户可以通过调整 energy_gap 参数来控制合法状态和非法状态之间的能量差距，从而影响模型的鲁棒性和可行解的空间。
# 该代码还包括了对称性约束，确保输入 A 和 B 在逻辑中地位对等，简化了模型并减少了求解空间。
# 最后，代码输出了求解得到的参数值以及每个状态的能量分布，帮助用户理解模型的行为和验证结果的正确性。
# 注意：该代码依赖于 PuLP 库，需要提前安装（pip install pulp）才能运行。
# Result: 功能正常。根据输入的逻辑门类型，能够输出正确的参数配置

# 修订05-18
# Created by: WHH
# 可以通过修改energy_gap参数来调整合法状态和非法状态之间的能量差距，从而影响模型的鲁棒性和可行解的空间。
# 可以通过修改lowBound和upBound参数来调整h和j的取值范围，以适应不同的物理实现或优化需求。

def solve_ising_bipolar(logic_gate='AND', energy_gap=1):
    prob = pulp.LpProblem(f"Ising_{logic_gate}_Bipolar", pulp.LpMinimize)

    # 1. 定义变量：h 和 j 依然使用整数，模拟硬件的离散控制
    # 输入端 h[0], h[1]，输出端 h[2]
    h = pulp.LpVariable.dicts("h", range(3), lowBound=-5, upBound=5, cat='Integer')
    # 耦合项 j01 (输入间), j02 (A到C), j12 (B到C)
    j = pulp.LpVariable.dicts("j", [(0, 1), (0, 2), (1, 2)], lowBound=-2, upBound=2, cat='Integer')
    
    # 基态能量偏置
    e_g = pulp.LpVariable("E_ground", lowBound=-20, upBound=20, cat='Continuous')

    # 2. 增加对称性约束：输入 A 和 B 在逻辑中地位对等
    prob += (h[0] == h[1], "Symmetry_H")
    prob += (j[(0, 2)] == j[(1, 2)], "Symmetry_J")

    # 3. 遍历逻辑状态 (0/1 用于判断逻辑，-1/+1 用于计算能量)
    for s0 in [0, 1]:
        for s1 in [0, 1]:
            for s2 in [0, 1]:
                # 核心映射：将逻辑 [0, 1] 转换为物理 [-1, 1]
                v0, v1, v2 = (2*s0 - 1), (2*s1 - 1), (2*s2 - 1)
                
                # 计算物理能量公式: E = sum(h_i * v_i) + sum(j_ij * v_i * v_j)
                current_energy = -(h[0]*v0 + h[1]*v1 + h[2]*v2 + 
                                  j[(0, 1)]*v0*v1 + j[(0, 2)]*v0*v2 + j[(1, 2)]*v1*v2)
                
                # 定义 AND 逻辑真值表
                is_valid = (s0 & s1 == s2)
                
                if is_valid:
                    # 合法状态必须落在同一个势能面上
                    prob += (current_energy == e_g)
                else:
                    # 非法状态必须高于势能面一个能量间隙
                    prob += (current_energy >= e_g + energy_gap)

    # 4. 执行求解
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[status] == 'Optimal':
        print(f"--- {logic_gate} 门参数 (Bipolar 映射) ---")
        print(f"偏置项 h (A, B, C): {[int(pulp.value(h[i])) for i in range(3)]}")
        print(f"耦合项 j01 (输入间): {int(pulp.value(j[(0, 1)]))}")
        print(f"耦合项 j_in_out (A->C, B->C): {int(pulp.value(j[(0, 2)]))}")
        print(f"基态能量 E_g: {pulp.value(e_g)}")
        
        # 验证能量分布
        print("\n能量分布表 (s0 s1 s2 | Energy):")
        for s0 in [0, 1]:
            for s1 in [0, 1]:
                for s2 in [0, 1]:
                    v = [2*s0-1, 2*s1-1, 2*s2-1]
                    e = -(pulp.value(h[0])*v[0] + pulp.value(h[1])*v[1] + pulp.value(h[2])*v[2] +
                         pulp.value(j[(0,1)])*v[0]*v[1] + pulp.value(j[(0,2)])*v[0]*v[2] + 
                         pulp.value(j[(1,2)])*v[1]*v[2])
                    print(f"{s0} {s1} {s2} | {e:.1f} {'*' if s0&s1==s2 else ''}")
    else:
        print("未找到可行解。")

solve_ising_bipolar(logic_gate='AND', energy_gap=2)