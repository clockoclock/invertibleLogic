import numpy as np
import pandas as pd

def calculate_ising_energy_landscape():
    # 1. 定义节点索引映射 (按照矩阵顺序对齐)
    # 节点: m0, m1, m4, m2, m3
    # 物理含义假设: m0, m1, m2 为输入; m3 为最终输出; m4 为级联辅助位
    node_names = ['m0', 'm1', 'm4', 'm2', 'm3']
    
    # 2. 严格根据您上传的图片构建 J 矩阵 (对称矩阵)
    J = np.array([
        [ 0,  1, -2,  0,  0],  # m0
        [ 1,  0, -2,  0,  0],  # m1
        [-2, -2,  0,  1, -2],  # m4
        [ 0,  0,  1,  0, -2],  # m2
        [ 0,  0, -2, -2,  0]   # m3
    ])
    
    # 3. 构建 h 偏置向量
    h = np.array([-1.0, -1.0, 1.0, -1.0, 2.0])
    
    # 4. 生成所有 32 种自旋配置空间 (m_i 属于 {-1, 1})
    # 采用二进制网格生成，0 映射为 -1，1 映射为 1
    num_nodes = len(node_names)
    results = []
    
    for idx in range(2**num_nodes):
        # 提取当前状态的 5 位二进制表示
        # 为了方便阅读，按标准的二进制高位到低位或固定位序展开
        # 这里直接通过位运算提取状态
        s_bits = [(idx >> (num_nodes - 1 - i)) & 1 for i in range(num_nodes)]
        
        # 将 [0, 1] 空间转换到 Ising 物理自旋空间 [-1, 1]
        m = np.array([1 if b == 1 else -1 for b in s_bits])
        
        # 5. 根据 PRX 公式 (3) 计算哈密顿量能量
        # E = -I0 * ( 0.5 * sum(J_ij * m_i * m_j) + sum(h_i * m_i) )
        # 注意：矩阵对角线为 0，所以 0.5 * m.T @ J @ m 完美等价于上三角两两耦合之和
        energy_coupling = 0.5 * np.dot(m, np.dot(J, m))
        energy_bias = np.dot(h, m)
        
        # 整体带负号
        total_energy = (energy_coupling + energy_bias)
        
        # 记录逻辑状态 (用于直观展示，将 -1 转换为 0，1 保持 1)
        logic_bits = [1 if val == 1 else 0 for val in m]
        logic_str = "".join(map(str, logic_bits))
        
        results.append({
            'Binary_State (m0 m1 m4 m2 m3)': logic_str,
            'Spin_Configuration': m.tolist(),
            'Coupling_Energy': energy_coupling,
            'Bias_Energy': energy_bias,
            'Total_Hamiltonian (E)': total_energy
        })
    
    # 6. 转换为 DataFrame 方便排序和可视化分析
    df = pd.DataFrame(results)
    df = df.sort_values(by='Total_Hamiltonian (E)').reset_index(drop=True)
    
    # 7. 打印输出分析
    print("=========================================================================")
    print("                     Ising System Energy Landscape                       ")
    print("=========================================================================")
    print(df[['Binary_State (m0 m1 m4 m2 m3)', 'Total_Hamiltonian (E)']].to_string())
    
    # 8. 自动找出能量最低的基态 (Ground States)
    min_energy = df['Total_Hamiltonian (E)'].min()
    ground_states = df[df['Total_Hamiltonian (E)'] == min_energy]
    
    print("\n=========================================================================")
    print(f"检测到的系统基态 (最低哈密顿量 E = {min_energy}):")
    print("=========================================================================")
    for _, row in ground_states.iterrows():
        b_str = row['Binary_State (m0 m1 m4 m2 m3)']
        print(f"状态: {b_str} -> 输入 [m0, m1, m2] = [{b_str[0]}, {b_str[1]}, {b_str[3]}], 辅助位 m4 = {b_str[2]}, 输出 m3 = {b_str[4]}")

if __name__ == "__main__":
    calculate_ising_energy_landscape()