import numpy as np
import pandas as pd

# information
# Created by: WHH
# Date: 2026-05-18
# Description: 5节点 紧凑型级联 3-input AND 逻辑门的哈密顿量在 m4 空间下的投影计算

def main():
    # ==========================================
    # 1. 配置 5 节点紧凑型级联物理参数
    # ==========================================
    # 节点索引对应: 0:m0(In1), 1:m1(In2), 2:m4(Aux), 3:m2(In3), 4:m3(Out)
    h = np.array([-1.0, -1.0, 1.0, -1.0, 2.0])

    J = np.array([
        [ 0,  1, -2,  0,  0],  # m0 (In1)
        [ 1,  0, -2,  0,  0],  # m1 (In2)
        [-2, -2,  0,  1, -2],  # m4 (Aux)
        [ 0,  0,  1,  0, -2],  # m2 (In3)
        [ 0,  0, -2, -2,  0]   # m3 (Out)
    ])

    # 按照辅助位 m4 的两种可能（0 和 1）划分的子空间容器
    aux_cases = ["0", "1"]
    subspace_data = {case: [] for case in aux_cases}

    # 存储 32 种微观状态的大总表
    big_table_data = []

    # ==========================================
    # 2. 遍历 32 种微观自旋状态 (5位二进制生成)
    # ==========================================
    for b in range(32):
        m = np.zeros(5, dtype=int)
        m[0] = (b >> 0) & 1  # m0 (In1)
        m[1] = (b >> 1) & 1  # m1 (In2)
        m[2] = (b >> 2) & 1  # m4 (Aux)
        m[3] = (b >> 3) & 1  # m2 (In3)
        m[4] = (b >> 4) & 1  # m3 (Out)

        # 转换为 Ising 系统的自旋轴状态 s (-1, +1)
        s = 2 * m - 1

        # 计算经典哈密顿量公式总能量: E = h·s + 0.5 * s·J·s^T
        energy = np.dot(h, s) + 0.5 * np.dot(s, np.dot(J, s))

        # 构建可读的字符串标签
        bit_str = f"{m[0]} {m[1]} {m[2]} {m[3]} {m[4]}"
        macro_str = f"{m[0]}{m[1]}{m[3]}{m[4]}"  # 提取映射到系统的 In1 In2 In3 Out
        aux_str = f"{m[2]}"                        # 提取 m4 辅助位状态

        # 级联逻辑完整解合法性校验
        cond1 = (m[0] & m[1]) == m[2]    # 第一级 AND：m0 & m1 == m4
        cond2 = (m[2] & m[3]) == m[4]    # 第二级 AND：m4 & m2 == m3
        is_match = "✔ 基态解" if (cond1 and cond2) else "✘ 非法态"

        # 载入 32 种物理大总表
        big_table_data.append({
            "Index": b,
            "Bit_String": bit_str,
            "Macro_State": macro_str,
            "Energy": round(energy, 1),
            "Status": is_match
        })

        # 按 m4 的具体状态分发给对应的能量分布子空间
        subspace_data[aux_str].append({
            "MacroState_In123Out": macro_str,
            "Hamiltonian_Energy": round(energy, 1),
            "Logic_Status": is_match
        })

    # 设置 pandas 打印配置，使其能完美对齐显示
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ==========================================
    # 3. 规范化报表打印输出
    # ==========================================
    print("\n" + "="*75)
    print("                    32 种全状态微观物理能量大总表")
    print("="*75)
    df_big = pd.DataFrame(big_table_data)
    print(df_big.to_string(index=False))

    # 循环提取子空间，并对宏观逻辑态做标准的文本递增排序 (0000 -> 1111)
    for case_name in aux_cases:
        print("\n" + "="*65)
        print(f"  辅助位配置 m4 = {case_name} 的哈密顿量子分布空间（标准升序）")
        print("="*65)
        
        # 转换为 DataFrame 并依据宏观态字符排序
        df_sub = pd.DataFrame(subspace_data[case_name])
        df_sub_sorted = df_sub.sort_values(by="MacroState_In123Out", ascending=True)
        
        print(df_sub_sorted.to_string(index=False))

if __name__ == "__main__":
    main()