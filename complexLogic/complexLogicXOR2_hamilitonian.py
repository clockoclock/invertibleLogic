import numpy as np
import pandas as pd

# information
# Created by: WHH
# Date: 2026-05-18
# Description: 5节点 双辅助位 XOR 逻辑门的哈密顿量空间投影计算 (依据全新耦合系数与偏置)

def main():
    # ==========================================
    # 1. 严格配置 5 节点 XOR 物理参数
    # ==========================================
    # 节点对应: 0:m0(In1), 1:m1(In2), 2:m2(Out), 3:m3(Aux1), 4:m4(Aux2)
    h = np.array([0, 0, 3, 3, -3], dtype=float)

    J = np.zeros((5, 5))
    # 输入间耦合
    J[0, 1] = J[1, 0] = 5.0
    # 输入到输出 (为 0，保持默认)
    J[0, 2] = J[2, 0] = 0.0
    J[1, 2] = J[2, 1] = 0.0
    # 输入到辅助
    J[0, 3] = J[3, 0] = 5.0
    J[1, 3] = J[3, 1] = 5.0
    J[0, 4] = J[4, 0] = 5.0
    J[1, 4] = J[4, 1] = 5.0
    # 输出到辅助
    J[2, 3] = J[3, 2] = 6.0
    J[2, 4] = J[4, 2] = -6.0
    # 辅助间耦合
    J[3, 4] = J[4, 3] = -1.0

    # 辅助位状态分类容器 (m3 m4 共有 4 种微观组合)
    aux_cases = ["00", "01", "10", "11"]
    subspace_data = {case: [] for case in aux_cases}

    # 存储 32 种微观状态的大总表
    big_table_data = []

    # ==========================================
    # 2. 遍历 32 种微观状态 (5位二进制生成)
    # ==========================================
    for b in range(32):
        m = np.zeros(5, dtype=int)
        m[0] = (b >> 0) & 1  # m0 (In1)
        m[1] = (b >> 1) & 1  # m1 (In2)
        m[2] = (b >> 2) & 1  # m2 (Out)
        m[3] = (b >> 3) & 1  # m3 (Aux1)
        m[4] = (b >> 4) & 1  # m4 (Aux2)

        # 转换为自旋状态 s 轴 (-1, +1)
        s = 2 * m - 1

        # 计算哈密顿能量值: E = h·s + 0.5 * s·J·s^T
        energy = np.dot(h, s) + 0.5 * np.dot(s, np.dot(J, s))

        # 构建展示标签
        bit_str = f"{m[0]} {m[1]} {m[2]} {m[3]} {m[4]}"
        macro_str = f"{m[0]}{m[1]}{m[2]}"  # 宏观逻辑态: In1 In2 Out
        aux_str = f"{m[3]}{m[4]}"          # 辅助位状态: m3 m4

        # 纯粹的 XOR 逻辑真值表合法性检验
        is_legal_xor = (m[0] ^ m[1]) == m[2]
        is_match = "✔ 基态解" if is_legal_xor else "✘ 非法态"

        # 载入大总表数据
        big_table_data.append({
            "Index": b,
            "Bit_String": bit_str,
            "Macro_State(In12_Out)": macro_str,
            "Energy": round(energy, 1),
            "Status": is_match
        })

        # 分发给对应内部状态子表
        subspace_data[aux_str].append({
            "MacroState_In12Out": macro_str,
            "Hamiltonian_Energy": round(energy, 1),
            "XOR_Status": is_match
        })

    # 设置 pandas 打印配置
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ==========================================
    # 3. 规范化打印输出
    # ==========================================
    print("\n" + "="*75)
    print("                    32 种全状态微观物理能量大总表")
    print("="*75)
    df_big = pd.DataFrame(big_table_data)
    print(df_big.to_string(index=False))

    # 循环提取 4 种辅助位子空间，并对宏观逻辑态做标准的二进制递增排序 (000 -> 111)
    for case_name in aux_cases:
        print("\n" + "="*65)
        print(f"  辅助位配置 (m3, m4) = {case_name} 的哈密顿量子分布空间（标准升序）")
        print("="*65)
        
        df_sub = pd.DataFrame(subspace_data[case_name])
        # 强制按宏观状态字符（In1 In2 Out）升序排列输出
        df_sub_sorted = df_sub.sort_values(by="MacroState_In12Out", ascending=True)
        
        print(df_sub_sorted.to_string(index=False))

if __name__ == "__main__":
    main()