import pulp
import itertools

# information
# Created by: WHH
# Date: 2026-04-08
# Description: Ising 模型参数求解器 - 3 输入 AND 逻辑门（引入辅助位）
# 本代码实现了一个基于整数线性规划的 Ising 模型参数求解器，针对 3 输入 AND 逻辑门设计，并引入了一个隐藏辅助位以增加系统的自由度。
# 通过定义变量 h 和 j 来模拟 Ising 模型中的偏置项和耦合项，并使用约束确保合法状态和非法状态之间的能量差距。
# 该实现采用了 bipolar 映射（0/1 转换为 -1/+1），以更贴近物理 Ising 模型的表示方式。
# 用户可以通过调整 energy_gap 参数来控制合法状态和非法状态之间的能量差距，从而影响模型的鲁棒性和可行解的空间。
# 该代码还包括了对称性约束，确保输入 A、B 和 C 在逻辑中地位对等，简化了模型并减少了求解空间。
# 最后，代码输出了求解得到的参数值以及每个状态的能量分布，帮助用户理解模型的行为和验证结果的正确性。
# 注意：该代码依赖于 PuLP 库，需要提前安装（pip install pulp）才能运行。
# Result: 功能正常。根据输入的逻辑门类型，能够输出正确的参数配置，并且引入辅助位后系统的可行解空间得到扩展，能够更好地满足能量约束。

# Version v1.0
# Date: 2026-05-14
# Comments: 该方案通过线性规划直接求解3输入AND的Ising参数，但需要引入额外的辅助位，如果后续需要设计更复杂的逻辑门，则需要更多的辅助位，极大的增加了系统的复杂度和求解难度
# 该方案仅用来初步验证可逆逻辑在稍微多节点下的可行性，后续会有级联方案，尝试可重构设计更复杂的可逆逻辑
# Result: 功能正常

def solve_3in_and_with_aux(energy_gap=1.5):
    # 1. 创建优化问题
    prob = pulp.LpProblem("Ising_3In_AND_with_Aux", pulp.LpMinimize)

    # 2. 定义节点：0,1,2(输入), 3(输出), 4(辅助位)
    # h 为偏置，j 为耦合强度
    h = pulp.LpVariable.dicts("h", range(5), lowBound=-10, upBound=10, cat='Integer')
    pairs = list(itertools.combinations(range(5), 2))
    j = pulp.LpVariable.dicts("j", pairs, lowBound=-10, upBound=10, cat='Integer')
    
    # E_ground 为所有合法逻辑状态对应的基态能量
    e_g = pulp.LpVariable("E_ground", lowBound=-50, upBound=50, cat='Continuous')

    # 3. 对称性约束：确保输入 m1, m2, m3 在硬件上是对等的
    # 偏置对称
    prob += (h[0] == h[1])
    prob += (h[1] == h[2])
    # 输入间耦合对称 j01=j02=j12
    prob += (j[(0, 1)] == j[(0, 2)])
    prob += (j[(0, 2)] == j[(1, 2)])
    # 输入到输出耦合对称 j03=j13=j23
    prob += (j[(0, 3)] == j[(1, 3)])
    prob += (j[(1, 3)] == j[(2, 3)])
    # 输入到辅助位耦合对称 j04=j14=j24
    prob += (j[(0, 4)] == j[(1, 4)])
    prob += (j[(1, 4)] == j[(2, 4)])

    # 4. 遍历 16 种逻辑状态 (2^4)
    for s in itertools.product([0, 1], repeat=4):
        s1, s2, s3, s4 = s  # s1,s2,s3为输入，s4为输出
        is_valid = ((s1 & s2 & s3) == s4)
        v = [2*x - 1 for x in s] # 转换为物理空间 {-1, 1}

        # 辅助位 a (index 4) 可能为 -1 或 1
        # 定义能量计算函数
        def get_energy_expr(v_list, v_aux):
            # 基础偏置能量
            energy = sum(h[i] * v_list[i] for i in range(4)) + h[4] * v_aux
            # 节点间耦合 (除辅助位外)
            energy += sum(j[pair] * v_list[pair[0]] * v_list[pair[1]] for pair in pairs if pair[1] < 4)
            # 节点与辅助位间的耦合
            energy += sum(j[(i, 4)] * v_list[i] * v_aux for i in range(4))
            return energy

        if is_valid:
            # 对于合法状态：至少存在一个辅助位状态 (1或-1)，使得能量等于 e_g
            # 且另一个辅助位状态的能量必须 >= e_g
            # 引入二进制变量 z 来选择辅助位的状态
            z = pulp.LpVariable(f"z_{s}", cat='Binary')
            e_pos = get_energy_expr(v, 1)
            e_neg = get_energy_expr(v, -1)
            
            # 使用 Big-M 法表示：如果 z=1, e_pos=e_g; 如果 z=0, e_neg=e_g
            M = 100
            prob += (e_pos - e_g <= M * (1 - z))
            prob += (e_pos - e_g >= -M * (1 - z))
            prob += (e_neg - e_g <= M * z)
            prob += (e_neg - e_g >= -M * z)
            
            # 辅助位的另一种状态不能低于基态
            prob += (e_pos >= e_g)
            prob += (e_neg >= e_g)
        else:
            # 对于非法状态：无论辅助位是 1 还是 -1，能量都必须显著高于 e_g
            prob += (get_energy_expr(v, 1) >= e_g + energy_gap)
            prob += (get_energy_expr(v, -1) >= e_g + energy_gap)

    # 5. 求解
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[status] == 'Optimal':
        print(f"找到可行解！Energy Gap = {energy_gap}")
        print(f"偏置 h (In1, In2, In3, Out, Aux): {[int(pulp.value(h[i])) for i in range(5)]}")
        print("-" * 30)
        print("耦合项 J 矩阵关键值:")
        print(f"输入间耦合 (j01, j02, j12): {int(pulp.value(j[(0,1)]))}")
        print(f"输入到输出 (j03, j13, j23): {int(pulp.value(j[(0,3)]))}")
        print(f"输入到辅助 (j04, j14, j24): {int(pulp.value(j[(0,4)]))}")
        print(f"输出到辅助 (j34): {int(pulp.value(j[(3,4)]))}")
        print(f"基态能量 E_g: {pulp.value(e_g)}")
    else:
        print("依然无解，请尝试调整约束或参数范围。")

solve_3in_and_with_aux(energy_gap=1.5)