import os
import sys
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/WFC 目录下）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)
import jax.numpy as np
from typing import List,Callable,Dict




def centroid(cells: np.ndarray) -> np.ndarray:
    """
    计算单元格的中心点
    
    Args:
        cells (np.ndarray): 形状为(n, vertices, dim)的数组，
                           n为单元数量，vertices为每个单元的顶点数，dim为维度
        
    Returns:
        np.ndarray: 形状为(n, dim)的数组，每个元素为对应单元的中心点坐标
    """
    return np.mean(cells, axis=1)



def segment_chunk(centroids: np.ndarray, rules: List[Callable[[np.ndarray], np.ndarray]]) -> List[np.ndarray]:
    """
    向量化实现：根据中心点和规则列表划分单元索引
    
    Args:
        centroids (np.ndarray): 形状为(n, dim)的中心点数组
        rules (List[Callable]): 规则列表，每个规则是向量化函数，
                              输入形状为(n, dim)的数组，返回形状为(n,)的布尔数组（是否满足规则）
        
    Returns:
        List[np.ndarray]: 划分后的索引数组列表，每个元素对应一个规则的结果，
        List[]: 不满足任何规则的索引
    """
    n = len(centroids)
    # 初始化掩码：标记所有点未被分配（True表示未分配）
    unassigned = np.ones(n, dtype=bool)
    groups = []
    for rule in rules:
        # 对未分配的点应用当前规则（向量化操作）
        mask = unassigned & rule(centroids)
        # 获取满足条件的索引
        group_indices = np.where(mask)[0]
        groups.append(group_indices)
        # 更新掩码：已分配的点标记为False
        unassigned &= ~mask
    return groups,np.where(unassigned)[0]

def get_region_specific_adj(
    global_adj: Dict,
    region_groups: List[np.ndarray]
) -> List[Dict]:
    """
    为每个划分区域生成独立的邻居稀疏矩阵（排除未划分元素）
    
    Args:
        global_adj: 全局邻居稀疏矩阵（build_hex8_adjacency_with_meshio的输出）
        region_groups: 分块结果（segment_chunk返回的groups），每个元素是区域内单元的全局索引数组
    
    Returns:
        列表，每个元素是对应区域的邻居稀疏矩阵，结构与global_adj一致
    """
    # 转换全局邻接数据为numpy格式（方便索引操作）
    row_ptr = np.asarray(global_adj["row_ptr"])
    col_idx = np.asarray(global_adj["col_idx"])
    directions = global_adj["directions"]
    num_global = global_adj["num_elements"]
    
    region_adj_list = []  # 存储每个区域的邻接矩阵
    
    for region_idx in range(len(region_groups)):
        # 1. 获取当前区域的全局单元索引及数量
        region_global_indices = region_groups[region_idx]
        num_local = len(region_global_indices)
        if num_local == 0:
            region_adj_list.append(None)
            continue
        
        # 2. 构建全局索引→区域内局部索引的映射（加速查询）
        global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(region_global_indices)}
        
        # 3. 筛选区域内的邻接关系
        local_row_ptr = [0]
        local_col_idx = []
        local_directions = []
        local_data = []
        
        # 遍历区域内的每个单元（全局索引）
        for g_row in region_global_indices:
            l_row = global_to_local[g_row]  # 转换为局部行索引
            
            # 从全局邻接矩阵中获取当前单元的所有邻居（全局索引）
            start = row_ptr[g_row]
            end = row_ptr[g_row + 1]
            neighbors_global = col_idx[start:end]
            dirs = directions[start:end]
            
            # 筛选：只保留同样属于当前区域的邻居
            for g_col, dir in zip(neighbors_global, dirs):
                if g_col in global_to_local:  # 邻居在当前区域内
                    l_col = global_to_local[g_col]  # 转换为局部列索引
                    local_col_idx.append(l_col)
                    local_directions.append(dir)
                    local_data.append(1.0)  # 权重保持为1
            
            # 更新行指针（累计当前行的邻居数量）
            local_row_ptr.append(local_row_ptr[-1] + (end - start - (len(neighbors_global) - len(local_col_idx) + local_row_ptr[-1])))
        
        # 4. 构建当前区域的邻接矩阵（转换为JAX数组）
        region_adj = {
            "row_ptr": np.array(local_row_ptr, dtype=np.int32),
            "col_idx": np.array(local_col_idx, dtype=np.int32),
            "data": np.array(local_data, dtype=np.float32),
            "directions": local_directions,
            "num_elements": num_local,
            "global_indices": region_global_indices  # 附加：区域对应的全局索引
        }
        region_adj_list.append(region_adj)
    
    return region_adj_list


if __name__ == "__main__":
    # 1. 读取网格并构建全局邻接矩阵
    mesh_file = "data/msh/box.msh"
    import meshio
    from src.WFC.adjacencyCSR import build_hex8_adjacency_with_meshio
    global_adj = build_hex8_adjacency_with_meshio(mesh_file=mesh_file)
    
    # 2. 提取单元信息计算中心点（假设从mesh中获取单元顶点）
    mesh = meshio.read(mesh_file)
    hex_cells = None
    for cell_block in mesh.cells:
        if cell_block.type == "hexahedron":
            hex_cells = cell_block.data
            break
    # 假设节点坐标存储在mesh.points中，形状为(n_points, 3)
    # 构建单元顶点坐标数组：shape=(n_elements, 8, 3)
    cell_vertices = mesh.points[hex_cells]  # 每个单元的8个顶点坐标
    centroids = centroid(cell_vertices)  # 计算中心点
    
    # 3. 定义分块规则并执行分块（示例规则）
    rules = [
        lambda c: c[:, 0] < 3,  # 规则1：x坐标 < 0.5
        lambda c: c[:, 0] >= 6  # 规则2：x坐标 >= 0.5
    ]
    region_groups, unassigned = segment_chunk(centroids, rules)
    
    # 4. 为每个区域生成独立的邻接矩阵
    region_adjs = get_region_specific_adj(global_adj, region_groups)
    
    # 5. 输出结果示例（查看第一个区域）
    if region_adjs[0] is not None:
        print(f"区域0包含{region_adjs[0]['num_elements']}个单元")
        print(f"区域0的全局索引：{region_adjs[0]['global_indices']}")
        # 查看区域0中第一个单元的邻居
        first_cell_local = 0
        start = region_adjs[0]["row_ptr"][first_cell_local]
        end = region_adjs[0]["row_ptr"][first_cell_local + 1]
        print(f"区域0中局部索引{first_cell_local}的邻居（局部索引）：{region_adjs[0]['col_idx'][start:end]}")
        print(f"对应的方向：{region_adjs[0]['directions'][start:end]}")
    