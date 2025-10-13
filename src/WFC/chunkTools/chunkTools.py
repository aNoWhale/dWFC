import os
import sys
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/WFC 目录下）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)
import jax.numpy as np
from typing import List,Callable,Dict
import meshio

def map_mesh_to_tensor(mesh:meshio.Mesh=None, msh_file_path=None, nx=10, ny=10, nz=10):
    """
    将nx×ny×nz的hex8网格与(nx, ny, nz)张量建立映射关系
    参数:
        msh_file_path: msh网格文件路径
        nx, ny, nz: 张量的三维维度（对应网格在x, y, z方向的划分数量）
    返回:
        形状为(nx, ny, nz)的JAX张量，每个元素为对应网格单元的索引
    """
    # 1. 读取msh文件
    if mesh is None:
        mesh = meshio.read(msh_file_path)
    
    # 2. 提取hex8类型的网格单元
    hex_cells = None
    for cell in mesh.cells:
        if cell.type == "hexahedron":
            hex_cells = cell.data
            break
    
    if hex_cells is None:
        raise ValueError("未在msh文件中找到hex8类型的网格单元")
    
    # 验证网格数量是否与nx×ny×nz一致
    expected_cell_count = nx * ny * nz
    if len(hex_cells) != expected_cell_count:
        raise ValueError(f"网格单元数量应为{expected_cell_count}，实际为{len(hex_cells)}")
    
    # 3. 计算每个网格单元的中心点坐标
    cell_centers = []
    for cell in hex_cells:
        points = mesh.points[cell]
        center = np.mean(points, axis=0)  # 用NumPy计算中心点（高效且不影响JAX特性）
        cell_centers.append(center)
    cell_centers = np.array(cell_centers)  # 形状为(nx*ny*nz, 3)
    
    # 4. 确定网格的空间范围（用于坐标到索引的映射）
    min_coords = np.min(cell_centers, axis=0)  # (xmin, ymin, zmin)
    max_coords = np.max(cell_centers, axis=0)  # (xmax, ymax, zmax)
    
    # 5. 计算每个维度的间隔（根据用户指定的nx, ny, nz）
    dx = (max_coords[0] - min_coords[0]) / nx  # x方向每个网格的间隔
    dy = (max_coords[1] - min_coords[1]) / ny  # y方向每个网格的间隔
    dz = (max_coords[2] - min_coords[2]) / nz  # z方向每个网格的间隔
    
    # 初始化nx×ny×nz的JAX张量
    tensor = np.zeros((nx, ny, nz), dtype=int)
    
    for cell_idx, center in enumerate(cell_centers):
        # 计算在三个维度上的张量索引
        i = int(np.floor((center[0] - min_coords[0]) / dx))
        j = int(np.floor((center[1] - min_coords[1]) / dy))
        k = int(np.floor((center[2] - min_coords[2]) / dz))
        
        # 处理边界情况（避免因浮点精度导致索引越界）
        i = np.clip(i, 0, nx - 1)
        j = np.clip(j, 0, ny - 1)
        k = np.clip(k, 0, nz - 1)
        
        # 使用JAX的.at[]方法赋值（兼容不可变数组特性）
        tensor = tensor.at[i, j, k].set(cell_idx)
    
    return tensor



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



def segment_chunk(centroids: np.ndarray, rules: List[Callable[[np.ndarray], np.ndarray]]):
    """
    向量化实现：根据中心点和规则列表划分单元索引
    
    Args:
        centroids (np.ndarray): 形状为(n, dim)的中心点数组
        rules (List[Callable]): 规则列表，每个规则是向量化函数，
                              输入形状为(n, dim)的数组，返回形状为(n,)的布尔数组（是否满足规则）
        
    Returns:
        List[np.ndarray]: 划分后的索引数组列表，每个元素对应一个规则的结果
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
    chunks: List[np.ndarray]
) -> List[Dict]:
    """
    为每个划分区域生成独立的邻居稀疏矩阵（排除未划分元素）
    
    Args:
        global_adj: 全局邻居稀疏矩阵（build_hex8_adjacency_with_meshio的输出）
        chunks: 分块结果（segment_chunk返回的groups），每个元素是区域内单元的全局索引数组
    
    Returns:
        列表，每个元素是对应区域的邻居稀疏矩阵，结构与global_adj一致
    """
    # 转换全局邻接数据为numpy格式（方便索引操作）
    row_ptr = np.asarray(global_adj["row_ptr"])
    col_idx = np.asarray(global_adj["col_idx"])
    directions = global_adj["directions"]
    num_global = global_adj["num_elements"]
    
    region_adj_list = []  # 存储每个区域的邻接矩阵
    
    for region_idx in range(len(chunks)):
        # 1. 获取当前区域的全局单元索引及数量
        region_global_indices = chunks[region_idx]
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

def make_chunk_mask(core_size,padding_size,cx,cy,cz):
    block_size = core_size + 2 * padding_size  # 单个块的大小
    Nx, Ny, Nz = cx * block_size, cy * block_size, cz * block_size  # 大矩阵总大小
    
    # 1. 创建单个块的mask模板（核心区域为True）
    block_mask = np.zeros((block_size, block_size, block_size), dtype=bool)
    core_slice = slice(padding_size, -padding_size)
    block_mask[core_slice, core_slice, core_slice] = True  # 标记核心区域
    
    # 2. 存储所有块的列表（按x→y→z顺序）
    blocks = []
    code = 0  # 核心区域编号，从0开始
    
    # 3. 生成每个块并添加到列表
    for i in range(cx):  # x方向的块索引
        y_blocks = []  # 存储当前x层中y方向的所有块
        for j in range(cy):  # y方向的块索引
            z_blocks = []  # 存储当前(x,y)位置中z方向的所有块
            for k in range(cz):  # z方向的块索引
                # 创建单个块：填充区域为-1，核心区域为当前code
                padding_tensor = np.full((block_size, block_size, block_size), -1)
                padding_tensor[block_mask] = code  # 核心区域赋值（替代at方法，兼容性更好）
                z_blocks.append(padding_tensor)  # 收集z方向的块
                code += 1  # 编号递增
            # 拼接当前(x,y)位置的所有z方向块（沿z轴拼接）
            y_slice = np.concatenate(z_blocks, axis=2)  # axis=2对应z轴
            y_blocks.append(y_slice)  # 收集y方向的块
        # 拼接当前x层的所有y方向块（沿y轴拼接）
        x_slice = np.concatenate(y_blocks, axis=1)  # axis=1对应y轴
        blocks.append(x_slice)  # 收集x方向的块
    
    # 4. 拼接所有x方向的块（沿x轴拼接），得到最终大矩阵
    result = np.concatenate(blocks, axis=0)  # axis=0对应x轴
    
    return result
    



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
    