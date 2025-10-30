import jax.numpy as jnp
import numpy as np

# def build_grid_adjacency(height, width, connectivity=4):
#     """
#     为规则网格创建带方向的邻接矩阵
    
#     参数:
#         height: 网格高度
#         width: 网格宽度
#         connectivity: 4(4邻域)或8(8邻域)
    
#     返回:
#         与build_adjacency相同格式的CSR邻接矩阵，增加方向信息
#     """
#     num_elements = height * width
    
#     # 1. 创建索引映射
#     def idx(i, j):
#         return i * width + j
    
#     # 2. 准备CSR数据结构
#     row_ptr = np.zeros(num_elements + 1, dtype=np.int32)
#     col_idx = []
#     directions = []  # 存储方向信息
    
#     # 定义方向映射
#     direction_map = {
#         (-1, 0): "up",     # 上邻居
#         (1, 0): "down",    # 下邻居
#         (0, -1): "left",   # 左邻居
#         (0, 1): "right",   # 右邻居
#         (-1, -1): "up_left",    # 左上邻居 (8邻域)
#         (-1, 1): "up_right",    # 右上邻居 (8邻域)
#         (1, -1): "down_left",   # 左下邻居 (8邻域)
#         (1, 1): "down_right"    # 右下邻居 (8邻域)
#     }
    
#     # 3. 遍历所有单元
#     for i in range(height):
#         for j in range(width):
#             current_idx = idx(i, j)
#             neighbors = []
#             neighbor_directions = []  # 存储当前单元邻居的方向
            
#             # 4邻域连接
#             if connectivity == 4:
#                 # 左邻居
#                 if j > 0:
#                     neighbors.append(idx(i, j - 1))
#                     neighbor_directions.append("left")
#                 # 右邻居
#                 if j < width - 1:
#                     neighbors.append(idx(i, j + 1))
#                     neighbor_directions.append("right")
#                 # 上邻居
#                 if i > 0:
#                     neighbors.append(idx(i - 1, j))
#                     neighbor_directions.append("up")
#                 # 下邻居
#                 if i < height - 1:
#                     neighbors.append(idx(i + 1, j))
#                     neighbor_directions.append("down")
            
#             # 8邻域连接
#             elif connectivity == 8:
#                 for di, dj in [(-1, -1), (-1, 0), (-1, 1),
#                                (0, -1),           (0, 1),
#                                (1, -1),  (1, 0),  (1, 1)]:
#                     ni, nj = i + di, j + dj
#                     if 0 <= ni < height and 0 <= nj < width:
#                         neighbors.append(idx(ni, nj))
#                         neighbor_directions.append(direction_map[(di, dj)])
            
#             # 更新CSR数据结构
#             col_idx.extend(neighbors)
#             directions.extend(neighbor_directions)  # 添加方向到列表
#             row_ptr[current_idx + 1] = row_ptr[current_idx] + len(neighbors)
    
#     # 4. 转换为JAX数组
#     return {
#         'row_ptr': jnp.array(row_ptr, dtype=jnp.int32),
#         'col_idx': jnp.array(col_idx, dtype=jnp.int32),
#         'data': jnp.ones(len(col_idx)),  # 所有边权重为1
#         'directions': directions,        # 方向信息列表
#         'num_elements': num_elements
#     }

def build_grid_adjacency(height, width, connectivity=4):
    """
    为规则网格创建带方向的邻接矩阵并计算顶点坐标
    
    参数:
        height: 网格高度
        width: 网格宽度
        connectivity: 4(4邻域)或8(8邻域)
    
    返回:
        字典包含:
        row_ptr, col_idx, data: CSR格式邻接矩阵
        directions: 每条边的方向信息
        num_elements: 单元总数
        vertices: 每个单元四个顶点坐标数组, 形状为[height*width, 4, 2]
                 顶点顺序: [左上, 右上, 右下, 左下]
    """
    num_elements = height * width
    # 创建顶点坐标数组 [网格单元数, 4个顶点, (x,y)]
    vertices = np.zeros((num_elements, 4, 2), dtype=np.float32)

    # 定义顶点顺序：左上, 右上, 右下, 左下 (顺时针)
    vertex_offsets = np.array([
        [0, 0],  # 左上
        [1, 0],  # 右上
        [1, 1],  # 右下
        [0, 1]   # 左下
    ])

    # 填充顶点坐标
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            # 每个顶点的实际位置 (x, y)
            vertices[idx] = vertex_offsets + [j, i]
    
    # 其余部分保持不变（同原函数）
    def idx(i, j):
        return i * width + j
    
    row_ptr = np.zeros(num_elements + 1, dtype=np.int32)
    col_idx = []
    directions = []
    
    direction_map = {
        (-1, 0): "up",     (-1, 1): "up_right",
        (0, 1): "right",   (1, 1): "down_right",
        (1, 0): "down",    (1, -1): "down_left",
        (0, -1): "left",   (-1, -1): "up_left"
    }
    
    for i in range(height):
        for j in range(width):
            current_idx = idx(i, j)
            neighbors = []
            neighbor_directions = []
            
            if connectivity == 4:
                for di, dj, dstr in [(-1,0,"up"), (0,1,"right"), (1,0,"down"), (0,-1,"left")]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbors.append(idx(ni, nj))
                        neighbor_directions.append(dstr)
            
            elif connectivity == 8:
                for di in (-1,0,1):
                    for dj in (-1,0,1):
                        if di == 0 and dj == 0: continue
                        ni, nj = i+di, j+dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbors.append(idx(ni, nj))
                            neighbor_directions.append(direction_map[(di,dj)])
            
            col_idx.extend(neighbors)
            directions.extend(neighbor_directions)
            row_ptr[current_idx+1] = row_ptr[current_idx] + len(neighbors)
    
    return {
        'row_ptr': np.array(row_ptr, dtype=jnp.int32),
        'col_idx': np.array(col_idx, dtype=jnp.int32),
        'data': np.ones(len(col_idx)),
        'directions': directions,
        'num_elements': num_elements,
        'vertices': np.array(vertices)  # 添加顶点坐标
    }
    # return {
    #     'row_ptr': jnp.array(row_ptr, dtype=jnp.int32),
    #     'col_idx': jnp.array(col_idx, dtype=jnp.int32),
    #     'data': jnp.ones(len(col_idx)),
    #     'directions': directions,
    #     'num_elements': num_elements,
    #     'vertices': jnp.array(vertices)  # 添加顶点坐标
    # }

if __name__ == '__main__':
    # 运行函数
    adj = build_grid_adjacency(height=3, width=3, connectivity=4)

    # 验证输出
    print("单元数:", adj['num_elements'])
    print("row_ptr:", adj['row_ptr'])
    print("col_idx:", adj['col_idx'])
    print("邻居关系数:", len(adj['col_idx']))

    # 验证每个单元的邻居
    for i in range(9):
        start = adj['row_ptr'][i]
        end = adj['row_ptr'][i + 1]
        neighbors = adj['col_idx'][start:end]
        direction = adj["directions"][start:end]
        print(f"单元 {i} 的邻居: {neighbors}, 方向： {direction}")
    
        # 获取第二个网格单元(i=0,j=1)的顶点坐标
    unit_index = 1  # 行优先索引
    corners = adj['vertices'][unit_index]

    # 四个顶点坐标分别是:
    top_left     = corners[0]  # (1.0, 0.0)
    top_right    = corners[1]  # (2.0, 0.0)
    bottom_right = corners[2]  # (2.0, 1.0)
    bottom_left  = corners[3]  # (1.0, 1.0)