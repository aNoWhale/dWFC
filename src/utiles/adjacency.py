import jax.numpy as jnp
import numpy as np

def build_grid_adjacency(height, width, connectivity=4):
    """
    为规则网格创建带方向的邻接矩阵
    
    参数:
        height: 网格高度
        width: 网格宽度
        connectivity: 4(4邻域)或8(8邻域)
    
    返回:
        与build_adjacency相同格式的CSR邻接矩阵，增加方向信息
    """
    num_elements = height * width
    
    # 1. 创建索引映射
    def idx(i, j):
        return i * width + j
    
    # 2. 准备CSR数据结构
    row_ptr = np.zeros(num_elements + 1, dtype=np.int32)
    col_idx = []
    directions = []  # 存储方向信息
    
    # 定义方向映射
    direction_map = {
        (-1, 0): "up",     # 上邻居
        (1, 0): "down",    # 下邻居
        (0, -1): "left",   # 左邻居
        (0, 1): "right",   # 右邻居
        (-1, -1): "up_left",    # 左上邻居 (8邻域)
        (-1, 1): "up_right",    # 右上邻居 (8邻域)
        (1, -1): "down_left",   # 左下邻居 (8邻域)
        (1, 1): "down_right"    # 右下邻居 (8邻域)
    }
    
    # 3. 遍历所有单元
    for i in range(height):
        for j in range(width):
            current_idx = idx(i, j)
            neighbors = []
            neighbor_directions = []  # 存储当前单元邻居的方向
            
            # 4邻域连接
            if connectivity == 4:
                # 左邻居
                if j > 0:
                    neighbors.append(idx(i, j - 1))
                    neighbor_directions.append("left")
                # 右邻居
                if j < width - 1:
                    neighbors.append(idx(i, j + 1))
                    neighbor_directions.append("right")
                # 上邻居
                if i > 0:
                    neighbors.append(idx(i - 1, j))
                    neighbor_directions.append("up")
                # 下邻居
                if i < height - 1:
                    neighbors.append(idx(i + 1, j))
                    neighbor_directions.append("down")
            
            # 8邻域连接
            elif connectivity == 8:
                for di, dj in [(-1, -1), (-1, 0), (-1, 1),
                               (0, -1),           (0, 1),
                               (1, -1),  (1, 0),  (1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbors.append(idx(ni, nj))
                        neighbor_directions.append(direction_map[(di, dj)])
            
            # 更新CSR数据结构
            col_idx.extend(neighbors)
            directions.extend(neighbor_directions)  # 添加方向到列表
            row_ptr[current_idx + 1] = row_ptr[current_idx] + len(neighbors)
    
    # 4. 转换为JAX数组
    return {
        'row_ptr': jnp.array(row_ptr, dtype=jnp.int32),
        'col_idx': jnp.array(col_idx, dtype=jnp.int32),
        'data': jnp.ones(len(col_idx)),  # 所有边权重为1
        'directions': directions,        # 方向信息列表
        'num_elements': num_elements
    }



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