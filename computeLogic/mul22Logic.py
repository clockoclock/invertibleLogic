import numpy as np
import itertools
import collections

# =============================================================================
# Information
# Created by: jcz
# Date: 2026-05-25
# Description:
# 本代码实现了一个基于 Ising 模型的 2×2 bit 乘法器可逆逻辑系统，
# 采用模拟退火 (Simulated Annealing) 对整体 Hamiltonian 进行全局能量优化，
# 验证逻辑正确性。
#
# 架构组成:
#   - 4 × AND Gate（部分积生成）
#   - 2 × Half Adder（结果累加）
#
# 特点:
#   1. 完全复用用户已有 AND / HA 的线性规划最优参数
#   2. 采用模块拼接构建全局 Ising Hamiltonian
#   3. 使用 Metropolis 模拟退火验证逻辑正确性
#
# 节点映射:
#   输入层:
#       0(a0), 1(a1), 2(b0), 3(b1)
#
#   AND层:
#       4(p00), 5(p01), 6(p10), 7(p11)
#
#   Half Adder 1:
#       8(z1), 9(c1), 10(anc1)
#
#   Half Adder 2:
#       11(z2), 12(z3), 13(anc2)
#
#   输出:
#       z0 = p00
#       z1 = 节点 8
#       z2 = 节点 11
#       z3 = 节点 12
#
# Result:
#   输出统计每个状态出现概率，并验证乘法逻辑正确性。
# =============================================================================

# =============================================================================# Information
# Created by: whh
# Date: 2026-06-02
# Description:
# 更新版本，现在不同模块的节点是直接拼合在一起的，以下修改为通过copy逻辑来实现参数的全局构建
# 后续可以通过调节copy逻辑的权重来实现不同模块间的补偿设计
# 根据级联方案的设计，需要在AND与HA之间引入3个辅助节点，在HA之间引入1个辅助节点
# 以下把这些辅助节点的参数也直接写入全局参数构建逻辑中，保持代码的整体性和清晰性
# 节点映射:
#   输入层:
#       0(a0), 1(a1), 2(b0), 3(b1)
#
#   AND层:
#       4(p00), 5(p01), 6(p10), 7(p11)
#
#   Half Adder 1:
#       8(in1), 9(in2), 10(sum) -> z1, 11(carry), 12(anc1)
#
#   Half Adder 2:
#       13(in1), 14(in2), 15(sum) -> z2, 16(carry) -> z3, 17(anc2)
#
#   Copy Logic:
#       5(p01) → 8(in1), 6(p10) → 9(in2), 7(p11) → 13(in1), 11(carry) → 14(in2)
#
#   输出:
#       z0 = p00
#       z1 = 节点 10
#       z2 = 节点 15
#       z3 = 节点 16
# =============================================================================

def verify_2x2_multiplier_ising_counts(num_samples=1000, T=10, T_start=10, T_end=0.00001, steps=1000):

    # 将这些节点之间完全分割，共需要4*2+5*2=18个节点来实现完整的2x2乘法器逻辑系统
    # 在AND门的输入之间不需要引入辅助节点，仅在输出之间引入辅助节点来连接到HA门；在HA门之间引入一个辅助节点来连接两个HA门
    num_nodes = 18

    # ========================== 【全局参数容器】 ==========================
    h = np.zeros(num_nodes)
    J = {}

    # 节点深度记录（用于后续可能的深度补偿权重设计）
    def add_J(i, j, val):
        key = tuple(sorted((i, j)))
        J[key] = J.get(key, 0) + val

    # =============================================================================
    # 【模块1：AND Gate 嵌入】
    # 参数来源: Logic_AND.py :contentReference[oaicite:0]{index=0}
    #
    # 逻辑: p = A · B
    # 节点: (A, B, P)
    # =============================================================================
    h_and = [-1, -1, 2]
    j_and = {(0, 1): 1, (0, 2): -2, (1, 2): -2}

    # AND Gate 嵌入函数
    def embed_and(a, b, p) -> None:
        h[a] += h_and[0]
        h[b] += h_and[1]
        h[p] += h_and[2]

        add_J(a, b, j_and[(0, 1)])
        add_J(a, p, j_and[(0, 2)])
        add_J(b, p, j_and[(1, 2)])

    # 四个部分积
    embed_and(0, 2, 4)  # p00
    embed_and(0, 3, 5)  # p01
    embed_and(1, 2, 6)  # p10
    embed_and(1, 3, 7)  # p11

    # =============================================================================
    # 【模块2：Half Adder 嵌入】
    # 参数来源: Logic_HA_1aux.py :contentReference[oaicite:1]{index=1}
    #
    # 逻辑:
    #   Sum = A ⊕ B
    #   Carry = A · B
    #
    # 节点: (A, B, Sum, Carry, Ancilla)
    # =============================================================================
    h_ha = [-1, -1, 1, 1, 1]

    j_ha = {
        (0, 1): 2,
        (0, 2): -2, (1, 2): -2,
        (0, 3): -2, (1, 3): -2,
        (0, 4): -2, (1, 4): -2,
        (2, 3): 2, (2, 4): 2,
        (3, 4): 0
    }

    # Half Adder 嵌入函数
    def embed_ha(a, b, s, c, anc):
        mapping = {0: a, 1: b, 2: s, 3: c, 4: anc}

        for i in range(5):
            h[mapping[i]] += h_ha[i]

        for (u, v), val in j_ha.items():
            add_J(mapping[u], mapping[v], val)

    # 此处需要考虑用于连接AND层和HA层的辅助节点设计，以及HA层之间的连接设计
    # 连接设计方案: 输入节点为8(p01), 9(p10)，输出节点为10(z1), 11(c1)，辅助节点为12(anc1)
    # 其中输入节点8,9分别通过copy逻辑连接到AND层的p01和p10节点（对应序号5,6），输出节点10,11分别对应HA1的Sum和Carry，辅助节点12为HA1的单辅助位
    # HA1: p01 + p10 → z1, c1
    embed_ha(8, 9, 10, 11, 12)  

    # 连接设计方案：输入节点为13(p11), 14(c1)，输出节点为15(z2), 16(z3)，辅助节点为17(anc2)
    # 其中输入节点13通过copy逻辑连接到AND层的p11节点（对应序号7），输入节点14通过copy逻辑连接到HA1的Carry节点（对应序号11），输出节点15,16分别对应HA2的Sum和Carry，辅助节点17为HA2的单辅助位
    # HA2: p11 + c1 → z2, z3
    embed_ha(13, 14, 15, 16, 17)  

    # =============================================================================
    # 辅助节点的copy逻辑设计
    # 连接设计方案：共计4条copy逻辑，分别为：
    #   1. p01 (节点5) → 输入节点8 
    #   2. p10 (节点6) → 输入节点9
    #   3. p11 (节点7) → 输入节点13
    #   4. c1 (节点11) → 输入节点14
    # 其中每条copy逻辑可以设计为一个简单的线性耦合，参数可以设置为 -2 来强制输入输出节点保持一致的状态
    # 由于这些节点的偏置已经被确定了，因此只需要重新设计这些节点之间的耦合参数即可实现copy逻辑的功能
    # =============================================================================
    j_copy = [-2.5]  # 4条copy逻辑的耦合参数，此处设置为相同值

    # 定义copy逻辑的连接关系
    def embed_copy(src, dst):
        add_J(src, dst, j_copy[0])  

    # 执行copy逻辑的嵌入
    embed_copy(5, 8)   # p01 → 输入节点8
    embed_copy(6, 9)   # p10 → 输入节点9
    embed_copy(7, 13)  # p11 → 输入节点13
    embed_copy(11, 14) # c1 → 输入节点14

    # =============================================================================
    # 【全局能量函数】
    # =============================================================================
    pairs = list(J.keys())

    def get_energy(s):
        energy = sum(h[i] * s[i] for i in range(num_nodes))
        for (u, v) in pairs:
            energy += J[(u, v)] * s[u] * s[v]
        return energy

    # =============================================================================
    # 【模拟退火核心引擎】
    # =============================================================================
    results = []

    print("正在执行 2×2 乘法器 Ising 系统的模拟退火演化...")

    for _ in range(num_samples):
        s = np.random.choice([1, -1], size=num_nodes)

        # steps = 1000
        for step in range(steps):
            t = T * (1 - step / steps) + 1e-5
            # t = T_start
            # decay = (T_end / T_start) ** (1.0 / steps)

            i = np.random.randint(0, num_nodes)
            s_flip = s.copy()
            s_flip[i] *= -1

            dE = get_energy(s_flip) - get_energy(s)

            if dE < 0 or np.random.rand() < np.exp(-dE / t):
                s = s_flip
            # t *= decay

        # ========================== 【结果提取】 ==========================
        a0 = (s[0] + 1) // 2
        a1 = (s[1] + 1) // 2
        b0 = (s[2] + 1) // 2
        b1 = (s[3] + 1) // 2

        z0 = (s[4] + 1) // 2
        z1 = (s[10] + 1) // 2
        z2 = (s[15] + 1) // 2
        z3 = (s[16] + 1) // 2

        results.append((a1, a0, b1, b0, z3, z2, z1, z0))

    # =============================================================================
    # 【统计分析】
    # =============================================================================
    counts = collections.Counter(results)

    print("=" * 90)
    print("2×2 乘法器 模拟退火验证结果")
    print(f"{'状态 (a1 a0 b1 b0 | z3 z2 z1 z0)':<35} | {'次数':<8} | 判定")
    print("-" * 90)

    total_valid = 0

    for bits in itertools.product([0, 1], repeat=8):
        a1, a0, b1, b0, z3, z2, z1, z0 = bits

        A = a1 * 2 + a0
        B = b1 * 2 + b0
        Z = z3 * 8 + z2 * 4 + z1 * 2 + z0

        count = counts[bits]
        valid = (A * B == Z)

        if valid:
            total_valid += count
            tag = "✔ (合法基态)"
        else:
            tag = "✘ (非法态)" if count > 0 else ""

        print(f"{str(bits):<35} | {count:<8} | {tag}")

    print("-" * 90)
    print(f"全局逻辑正确率: {total_valid / num_samples * 100:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    verify_2x2_multiplier_ising_counts(num_samples=1000, T=5, T_start=50, T_end=0.00001, steps = 10000)