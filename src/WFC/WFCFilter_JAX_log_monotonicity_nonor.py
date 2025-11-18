import os
import sys
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import tqdm.rich as tqdm
import numpy as np
from src.WFC.gumbelSoftmax import gumbel_softmax
from src.WFC.TileHandler_JAX import TileHandler
from src.WFC.builder import visualizer_2D, Visualizer
from src.WFC.FigureManager import FigureManager


def preprocess_adjacency(adj_csr, tileHandler):
    """保持不变，仅处理邻接结构"""
    dir_int_to_str = tileHandler.dir_int_to_str
    concrete_dirs = [
        dir_str for dir_int, dir_str in dir_int_to_str.items()
        if dir_int != -1 and dir_str != 'isotropy'
    ]
    unique_dirs = list(dict.fromkeys(concrete_dirs))
    dir_mapping = {dir_str: idx for idx, dir_str in enumerate(unique_dirs)}
    
    row_ptr = np.array(adj_csr['row_ptr'])
    col_idx = np.array(adj_csr['col_idx'])
    directions = np.array(adj_csr['directions'])
    
    dir_indices = np.array([dir_mapping[dir_str] for dir_str in directions], dtype=np.int32)
    
    n_cells = len(row_ptr) - 1
    A_np = np.zeros((n_cells, n_cells), dtype=np.float32)
    D_np = np.zeros((n_cells, n_cells), dtype=np.int32)
    
    for j in range(n_cells):
        start = row_ptr[j]
        end = row_ptr[j+1]
        neighbors_j = col_idx[start:end]
        dirs_j = dir_indices[start:end]
        A_np[neighbors_j, j] = 1.0
        D_np[neighbors_j, j] = dirs_j
    
    A = jnp.array(A_np)
    D = jnp.array(D_np)
    return A, D


def get_neighbors(csr, index):
    """保持不变"""
    start = csr['row_ptr'][index]
    end = csr['row_ptr'][index + 1]
    neighbors = csr['col_idx'][start:end]
    neighbors_dirs = csr['directions'][start:end]
    return neighbors, neighbors_dirs


@partial(jax.jit, static_argnames=())
def update_by_neighbors(log_probs, collapse_idx, A, D, dirs_opposite_index, log_compatibility, log_init_probs, key, alpha=0.3,): #0.3 
    n_cells, n_tiles = log_probs.shape
    # 1. 生成软掩码（保持不变）
    collapse_mask = soft_mask(collapse_idx, n_cells)
    collapse_mask = collapse_mask / jnp.sum(collapse_mask)
    
    # 2. 提取邻居和方向
    neighbor_mask = A[:, collapse_idx]  # (n_cells,)：邻居掩码（1表示是邻居，0表示非邻居）
    neighbor_dirs = D[:, collapse_idx].astype(jnp.int32)
    valid_dirs = (neighbor_dirs * neighbor_mask).astype(jnp.int32)
    
    # 3. 对数空间兼容性（已预先转换为log，直接索引）
    opposite_dirs = jnp.take(dirs_opposite_index, valid_dirs, mode='clip')
    log_compat = jnp.take(log_compatibility, opposite_dirs, axis=0)  # 对数空间的兼容性矩阵
    
    # 添加微小噪声（对数空间）
    noise = jax.random.normal(key, log_compat.shape) * 1e-8
    log_compat = jnp.clip(log_compat + noise, -50, 0)  # 限制范围
    
    # 4. 计算更新因子（核心修改：对邻居概率应用自适应变换，弱化低模p的影响）
    # 4.1 先将log空间的邻居概率转回原空间，并结合掩码
    neighbor_probs = jnp.exp(log_probs) * (neighbor_mask[:, None] + 1e-10)  # 原空间：p * mask（非邻居为0）
    
    # 4.2 计算邻居概率的模长（反映整体大小，模接近0表示p普遍较小）
    # 模长定义：欧式距离（对tile维度求和，保持维度用于广播）
    norm_p = jnp.sqrt(jnp.sum(neighbor_probs **2, axis=1, keepdims=True))  # (n_cells, 1)
    
    # 4.3 自适应变换：当模长接近0时，p_i→1（几乎不影响连乘）；模长大时，p_i保持单调性
    # 变换函数：f(p_i) = p_i + (1 - p_i) * exp(-norm_p)
    f_p = neighbor_probs + (1 - neighbor_probs) * jnp.exp(-norm_p)  # 原空间变换
    
    # 4.4 转换回log空间，作为新的邻居概率（避免log(0)，加微小值）
    log_neighbor_probs = jnp.log(f_p + 1e-10)
    log_neighbor_probs = jnp.clip(log_neighbor_probs, -50, 0)  # 限制极端值
    
    # 4.5 对数空间的连乘（加法）与求和（logsumexp）
    log_update_factors = log_compat + log_neighbor_probs[:, None, :]  # 连乘→加法
    # log_sum_factors = jax.scipy.special.logsumexp(log_update_factors, axis=2)  # 对tile维度求和
    # sum_log = jax.scipy.special.logsumexp(log_sum_factors, axis=0)  # 对邻居维度求和
    # sum_log = jnp.clip(sum_log, -50, 0)
    log_sum_factors = jnp.sum(log_update_factors, axis=2)  # 替换logsumexp为sum
    # 对邻居维度：保留所有邻居的贡献，不求和（相乘）
    sum_log = jnp.sum(log_sum_factors, axis=0)  # 替换logsumexp为sum
    sum_log = jnp.clip(sum_log, -50, 0)  # 仍需限制范围避免数值爆炸
    
    # 5. 更新当前单元对数概率
    log_p_updated = log_probs[collapse_idx] + sum_log
    # 混合初始概率（对数空间的加权平均）
    log_p_updated = jnp.log(
        (1 - alpha) * jnp.exp(log_p_updated) + 
        alpha * jnp.exp(log_init_probs[collapse_idx])
    )
    log_p_updated = jnp.clip(log_p_updated, -50, 0)
    
    # 6. 软更新
    return log_probs * (1 - collapse_mask[:, None]) + log_p_updated * collapse_mask[:, None]

@jax.jit
def update_neighbors(log_probs, collapse_idx, A, D, log_compatibility):
    n_cells, n_tiles = log_probs.shape
    # 1. 生成当前单元掩码，获取当前单元的对数概率
    collapse_mask = jnp.zeros(n_cells, dtype=jnp.float32).at[collapse_idx].set(1.0)
    log_p_collapsed = log_probs[collapse_idx]  # 形状：(n_tiles,) → 当前单元的对数概率
    
    # -------------------------- 新增：自适应变换当前单元概率 --------------------------
    # 将对数概率转回原空间（p_collapsed ∈ (0,1)）
    p_collapsed = jnp.exp(log_p_collapsed)  # 形状：(n_tiles,)
    
    # 计算当前单元概率的模长（反映整体大小，模接近0表示p普遍较小）
    # 模长定义：欧式距离（对tile维度求和）
    norm_p = jnp.sqrt(jnp.sum(p_collapsed** 2))  # 标量：当前单元概率的整体模长
    
    # 自适应变换：低模p→1（影响小），高模p保持单调性
    # 变换函数：f(p) = p + (1 - p) * exp(-norm_p)
    f_p_collapsed = p_collapsed + (1 - p_collapsed) * jnp.exp(-norm_p)  # 形状：(n_tiles,)
    
    # 转回log空间（加微小值避免log(0)）
    log_f_p_collapsed = jnp.log(f_p_collapsed + 1e-10)  # 形状：(n_tiles,)
    # ------------------------------------------------------------------------------
    
    # 2. 提取邻居和方向（保持不变）
    neighbor_mask = A[:, collapse_idx]  # 形状：(n_cells,) → 邻居掩码（1=邻居，0=非邻居）
    neighbor_dirs = D[:, collapse_idx]  # 形状：(n_cells,) → 邻居方向
    
    # 3. 对数空间计算邻居更新值（核心：使用变换后的当前单元概率）
    log_compat = jnp.take(log_compatibility, neighbor_dirs, axis=0)  # 形状：(n_cells, n_tiles, n_tiles)
    log_compat = jnp.clip(log_compat, -50, 0)  # 限制范围
    
    # 关键：连乘→加法（使用变换后的log_f_p_collapsed）
    # 原逻辑：log(compat_ij * p_j) = log_compat_ij + log_p_j → 现在替换为log_f_p_collapsed
    # log_p_neigh = log_compat + log_f_p_collapsed[None, None, :]  # 扩展维度：(n_cells, n_tiles, n_tiles)
    # log_p_neigh = jax.scipy.special.logsumexp(log_p_neigh, axis=2)  # 收缩最后一维：(n_cells, n_tiles)
    
    log_p_neigh = log_compat + log_f_p_collapsed[None, None, :]
    log_p_neigh = jnp.sum(log_p_neigh, axis=2)  # 替换logsumexp为sum（不求和，直接累积）
    log_p_neigh = jnp.clip(log_p_neigh, -50, 0)

    # 结合邻居掩码和原有概率（保持不变）
    log_p_neigh = log_p_neigh + log_probs + jnp.log(neighbor_mask[:, None] + 1e-10)
    log_p_neigh = jnp.clip(log_p_neigh, -50, 0)  # 防止极端值
    
    # 4. 软更新邻居对数概率（保持不变）
    return log_probs * (1 - neighbor_mask[:, None]) + log_p_neigh * neighbor_mask[:, None]

@partial(jax.jit, static_argnames=['n_cells'])  # 新增：将n_cells设为静态参数
def soft_mask(index, n_cells, sigma=0.2):
    """生成以index为中心的高斯软掩码，sigma越小越接近硬掩码"""
    x = jnp.arange(n_cells)  # 现在n_cells是具体值，可正常创建数组
    return jax.nn.sigmoid((- (x - index)**2) / (2 * sigma**2))


@jax.jit
def waveFunctionCollapse(init_probs, A, D, dirs_opposite_index, compatibility):
    n_cells, n_tiles = init_probs.shape
    # 1. 转换为对数空间（初始化）
    # 防止log(0)，先clip初始概率
    init_probs_clipped = jnp.clip(init_probs, 1e-10, 1.0)
    log_init_probs = jnp.log(init_probs_clipped)
    log_init_probs = jnp.clip(log_init_probs, -50, 0)  # 限制初始对数范围
    
    # 2. 兼容性矩阵转换为对数空间
    compatibility_clipped = jnp.clip(compatibility, 1e-10, 1.0)  # 兼容性通常是0-1
    log_compatibility = jnp.log(compatibility_clipped)
    log_compatibility = jnp.clip(log_compatibility, -50, 0)  # 限制兼容性对数范围
    # 3. 第一步：所有单元格依据log_init_probs进行update_by_neighbors
    def step1_update(carry, collapse_idx):
        log_probs, current_key = carry
        subkey, next_key = jax.random.split(current_key)
        updated_log_probs = update_by_neighbors(
            log_probs, collapse_idx, A, D, dirs_opposite_index, 
            log_compatibility, log_init_probs,subkey
        )
        return (updated_log_probs,next_key), None
        
    key = jax.random.PRNGKey(0)
    log_carry1 = (log_init_probs,key)  # 初始对数概率
    (log_probs_step1, _), _ = jax.lax.scan(
        step1_update,
        init=log_carry1,
        xs=jnp.arange(n_cells)
    )
    
    # 4. 第二步：每个单元格依据log_probs_step1进行update_neighbors
    def step2_update(carry, collapse_idx):
        log_probs = carry
        updated_log_probs = update_neighbors(
            log_probs, collapse_idx, A, D, log_compatibility
        )
        return updated_log_probs, None
    
    final_log_probs, _ = jax.lax.scan(
        step2_update,
        init=log_probs_step1,
        xs=jnp.arange(n_cells)
    )
    
    # 5. 还原为概率空间（exp转换）
    final_probs = jnp.exp(final_log_probs)
    final_probs = jnp.clip(final_probs, 1e-10, 1.0-1e-10)  # 确保概率在合理范围
    # 最后归一化一次，消除exp带来的微小偏差
    # final_probs = final_probs / jnp.sum(final_probs, axis=-1, keepdims=True)
    
    collapse_list = jnp.arange(n_cells)
    return final_probs, 0, collapse_list
