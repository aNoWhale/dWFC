import jax.numpy as jnp
import numpy as np

def build_grid_adjacency(height, width, connectivity=4):
    """
    为规则网格创建邻接矩阵

    参数:
        height: 网格高度
        width: 网格宽度
        connectivity: 4(4邻域)或8(8邻域)

    返回:
        与build_adjacency相同格式的CSR邻接矩阵
    """
    num_elements = height * width

    # 1. 创建索引映射
    def idx(i, j):
        return i * width + j

    # 2. 准备CSR数据结构
    row_ptr = np.zeros(num_elements + 1, dtype=np.int32)
    col_idx = []

    # 3. 遍历所有单元
    for i in range(height):
        for j in range(width):
            current_idx = idx(i, j)
            neighbors = []

            # 4邻域连接
            if connectivity == 4:
                # 左邻居
                if j > 0:
                    neighbors.append(idx(i, j - 1))
                # 右邻居
                if j < width - 1:
                    neighbors.append(idx(i, j + 1))
                # 上邻居
                if i > 0:
                    neighbors.append(idx(i - 1, j))
                # 下邻居
                if i < height - 1:
                    neighbors.append(idx(i + 1, j))

            # 8邻域连接
            elif connectivity == 8:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # 跳过自身
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbors.append(idx(ni, nj))

            # 更新CSR数据结构
            col_idx.extend(neighbors)
            row_ptr[current_idx + 1] = row_ptr[current_idx] + len(neighbors)

    # 4. 转换为JAX数组
    return {
        'row_ptr': jnp.array(row_ptr, dtype=jnp.int32),
        'col_idx': jnp.array(col_idx, dtype=jnp.int32),
        'data': jnp.ones(len(col_idx)),  # 所有边权重为1
        'num_elements': num_elements
    }