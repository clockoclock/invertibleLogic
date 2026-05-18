import numpy as np
import pandas as pd

# information
# Created by: WHH
# Date: 2026-05-18
# Description: 3-input AND 逻辑门的哈密顿量计算 - 引入辅助节点的 Copy-Link 方案

def main():
    # ==========================================
    # 1. 严格配置物理参数 (根据 image_8937af.png)
    # ==========================================
    # 节点索引小端序对应: 0:m0(In1), 1:m1(In2), 2:m4a, 3:m4b, 4:m2(In3), 5:m3(Out)
    h = np.array([-1.0, -1.0, 2.0, -1.0, -1.0, 2.0])

    J = np.zeros((6, 6))
    # Gate A (m0, m1 -> m4a)
    J[0, 1] = J[1, 0] = 1.0
    J[0, 2] = J[2, 0] = -2.0
    J[1, 2] = J[2, 1] = -2.0

    # Copy-Link (m4a <-> m4b)
    J[2, 3] = J[3, 2] = -2.0

    # Gate B (m4b, m2 -> m3)
    J[3, 4] = J[4, 3] = 1.0
    J[3, 5] = J[5, 3] = -2.0
    J[4, 5] = J[5, 4] = -2.0

    # 辅助位状态分类容器
    aux_cases = ["00", "01", "10", "11"]
    subspace_data = {case: [] for case in aux_cases}

    # 存储大总表数据
    big_table_data = []

    # ==========================================
    # 2. 遍历 64 种微观状态 (二进制小端序生成)
    # ==========================================
    for b in range(64):
        m = np.zeros(6, dtype=int)
        m[0] = (b >> 0) & 1  # m0 (In1)
        m[1] = (b >> 1) & 1  # m1 (In2)
        m[2] = (b >> 2) & 1  # m4a
        m[3] = (b >> 3) & 1  # m4b
        m[4] = (b >> 4) & 1  # m2 (In3)
        m[5] = (b >> 5) & 1  # m3 (Out)

        # 转换为自旋状态 s 轴 (-1, +1)
        s = 2 * m - 1

        # 计算哈密顿能量值: E = h*s^T + 0.5 * s*J*s^T
        energy = np.dot(h, s) + 0.5 * np.dot(s, np.dot(J, s))

        # 构建可读的字符串标签
        bit_str = f"{m[0]} {m[1]} {m[2]} {m[3]} {m[4]} {m[5]}"
        macro_str = f"{m[0]}{m[1]}{m[4]}{m[5]}"  # In1 In2 In3 Out
        aux_str = f"{m[2]}{m[3]}"                # m4a m4b

        # 完备逻辑合法性校验
        cond1 = (m[0] & m[1]) == m[2]
        cond2 = m[2] == m[3]
        cond3 = (m[3] & m[4]) == m[5]
        is_match = "✔ 基态解" if (cond1 and cond2 and cond3) else "✘ 非法态"

        # 载入大总表数据
        big_table_data.append({
            "Index": b,
            "Bit_String": bit_str,
            "Macro_State": macro_str,
            "Energy": round(energy, 1),
            "Status": is_match
        })

        # 分发给对应内部状态子表
        subspace_data[aux_str].append({
            "MacroState_In123Out": macro_str,
            "Hamiltonian_Energy": round(energy, 1),
            "Logic_Status": is_match
        })

    # 设置 pandas 打印配置，使其能完美对齐显示
    pd.set_option('display.max_rows', 70)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ==========================================
    # 3. 规范化打印输出
    # ==========================================
    print("\n" + "="*75)
    print("                    64 种全状态微观物理能量大总表")
    print("="*75)
    df_big = pd.DataFrame(big_table_data)
    print(df_big.to_string(index=False))

    # 循环提取子空间，并对宏观逻辑态做标准的文本递增排序 (0000 -> 1111)
    for case_name in aux_cases:
        print("\n" + "="*65)
        print(f"  辅助位配置 (m4a, m4b) = {case_name} 的哈密顿量子分布空间（标准升序）")
        print("="*65)
        
        # 转换为 DataFrame
        df_sub = pd.DataFrame(subspace_data[case_name])
        
        # 【核心操作】强制按宏观状态字符升序排列，使 0001 紧跟 0000 稳定输出
        df_sub_sorted = df_sub.sort_values(by="MacroState_In123Out", ascending=True)
        
        print(df_sub_sorted.to_string(index=False))

if __name__ == "__main__":
    main()