import numpy as np
import jax.numpy as jnp
from collections import defaultdict


def convert_adj_to_vmap_compatible(original_adj, direction_to_idx=None):
    """
    转换adj为vmap兼容形式，核心：方向字符串→整数映射（解决JAX不支持字符串数组问题）。
    
    参数:
        original_adj: 原始adj字典
        direction_to_idx: 自定义方向→整数映射（如无则自动生成，默认包含up/down/right/left）
            示例：{"up":0, "down":1, "right":2, "left":3, "invalid":4}
    
    返回:
        vmap_adj: 兼容vmap的adj字典，方向已转为整数
    """
    # -------------------------- 1. 初始化方向→整数映射 --------------------------
    # 默认方向映射（包含你的核心方向，可根据需要扩展）
    default_dir_map = {
        "up": 0,
        "down": 2,
        "right": 1,
        "left": 3,
        "invalid": -1,  # 无效方向的整数标识
        "top":4,
        "bottom": 5,
    }
    # 优先使用用户自定义映射，无则用默认
    dir_map = direction_to_idx if direction_to_idx is not None else default_dir_map
    # 反向映射（整数→方向，方便调试）
    idx_to_dir = {v: k for k, v in dir_map.items()}
    
    # -------------------------- 2. 提取并转换核心数据 --------------------------
    row_ptr = original_adj['row_ptr']
    col_idx = original_adj['col_idx']
    directions = original_adj['directions']  # 原始方向字符串列表
    num_elements = original_adj['num_elements']
    
    # 转换为NumPy数组（预处理）
    row_ptr_np = np.array(row_ptr)
    col_idx_np = np.array(col_idx)
    
    # -------------------------- 3. 方向字符串→整数转换 --------------------------
    # 将原始方向字符串转为整数（非法/无效方向映射为dir_map["invalid"]）
    dir_integers = []
    for d in directions:
        # 若方向在映射中，用对应整数；否则用"invalid"的整数
        dir_integers.append(dir_map.get(d, dir_map["invalid"]))
    dir_integers_np = np.array(dir_integers, dtype=np.int32)  # 整数数组（JAX支持）
    
    # -------------------------- 4. 计算最大邻居数（静态长度） --------------------------
    neighbor_counts = row_ptr_np[1:] - row_ptr_np[:-1]
    max_neighbors = int(np.max(neighbor_counts))
    
    # -------------------------- 5. 填充邻居和方向（整数形式） --------------------------
    # 邻居数组：无效值用-1（整数，JAX支持）
    padded_neighbors_np = np.full((num_elements, max_neighbors), fill_value=-1, dtype=np.int32)
    # 方向数组：无效值用dir_map["invalid"]（整数，如4）
    padded_dirs_int_np = np.full((num_elements, max_neighbors), fill_value=dir_map["invalid"], dtype=np.int32)
    
    for i in range(num_elements):
        start = row_ptr_np[i]
        end = row_ptr_np[i+1]
        actual_neighbor_count = end - start
        
        if actual_neighbor_count == 0:
            continue
        
        # 提取实际邻居和整数方向
        actual_neighbors = col_idx_np[start:end]
        actual_dirs_int = dir_integers_np[start:end]
        
        # 填充到统一长度数组
        padded_neighbors_np[i, :actual_neighbor_count] = actual_neighbors
        padded_dirs_int_np[i, :actual_neighbor_count] = actual_dirs_int
    
    # -------------------------- 6. 构建vmap兼容adj --------------------------
    vmap_adj = original_adj.copy()
    vmap_adj.update({
        'max_neighbors': max_neighbors,
        'padded_neighbors': jnp.array(padded_neighbors_np, dtype=jnp.int32),  # 整数数组
        'padded_dirs_int': jnp.array(padded_dirs_int_np, dtype=jnp.int32),    # 方向整数数组（JAX支持）
        'dir_map': dir_map,  # 方向→整数映射表（供后续处理用）
        # 'idx_to_dir': idx_to_dir  # 整数→方向反向映射（调试用）
    })
    
    # 删除原始字符串方向字段（避免后续误用）
    if 'directions' in vmap_adj:
        del vmap_adj['directions']
    
    return vmap_adj

