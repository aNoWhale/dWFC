import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import numpy as np


# def preprocess_adjacency(adj_csr, tileHandler):
#     """处理邻接矩阵和方向，适配可变邻居数"""
#     dir_int_to_str = tileHandler.dir_int_to_str
#     concrete_dirs = [
#         dir_str for dir_int, dir_str in dir_int_to_str.items()
#         if dir_int != -1 and dir_str != 'isotropy'
#     ]
#     unique_dirs = list(dict.fromkeys(concrete_dirs))
#     dir_mapping = {dir_str: idx for idx, dir_str in enumerate(unique_dirs)}
    
#     row_ptr = np.array(adj_csr['row_ptr'])
#     col_idx = np.array(adj_csr['col_idx'])
#     directions = np.array(adj_csr['directions'])
    
#     dir_indices = np.array([dir_mapping[dir_str] for dir_str in directions], dtype=np.int32)
    
#     n_cells = len(row_ptr) - 1
#     A_np = np.zeros((n_cells, n_cells), dtype=np.float32)
#     D_np = np.zeros((n_cells, n_cells), dtype=np.int32)
    
#     for j in range(n_cells):
#         start = row_ptr[j]
#         end = row_ptr[j+1]
#         neighbors_j = col_idx[start:end]
#         dirs_j = dir_indices[start:end]
#         A_np[neighbors_j, j] = 1.0  # 1表示是邻居
#         D_np[neighbors_j, j] = dirs_j
    
#     A = jnp.array(A_np)
#     D = jnp.array(D_np)
#     return A, D


def preprocess_adjacency(adj_csr, tileHandler):
    """修正后的邻接矩阵处理"""
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
        # 修正：正确的索引方式
        A_np[j, neighbors_j] = 1.0  # j的邻居是neighbors_j
        D_np[j, neighbors_j] = dirs_j
    
    A = jnp.array(A_np)
    D = jnp.array(D_np)
    return A, D, len(unique_dirs)  # 返回方向数量用于验证


def get_neighbors(csr, index):
    """获取指定单元的邻居及方向（辅助函数）"""
    start = csr['row_ptr'][index]
    end = csr['row_ptr'][index + 1]
    neighbors = csr['col_idx'][start:end]
    neighbors_dirs = csr['directions'][start:end]
    return neighbors, neighbors_dirs


@partial(jax.jit, static_argnames=['n_cells', 'tau','sigma'])  # 添加tau为静态参数
def soft_mask(index, n_cells, tau=2.0, sigma=1.0):  # tau默认2.0（可配置）
    """接近硬掩码的软掩码：sigma=0.1（陡峭，仅局部影响）"""
    x = jnp.arange(n_cells)
    dist_sq = (x - index) ** 2
    mask = jax.nn.sigmoid(-dist_sq / (2 * sigma**2))  # 陡峭衰减
    mask = mask / jnp.sum(mask)  # 归一化（适配可变邻居数）
    return mask


@partial(jax.jit, static_argnames=['tau'])  # 添加tau静态参数
def single_update_by_neighbors(collapse_idx, key, log_init_probs, A, D, dirs_opposite_index, log_compatibility, alpha=0., tau=2.0):
    n_cells, n_tiles = log_init_probs.shape
    
    # 1. 生成当前单元的局部软掩码（接近硬掩码）
    collapse_mask = soft_mask(collapse_idx, n_cells, tau=tau)[:, None]  # (n_cells, 1)
    
    # 2. 提取邻居掩码（无需获取动态索引，直接用掩码）
    neighbor_mask = A[:, collapse_idx]  # (n_cells,)：1=有效邻居，0=无效
    neighbor_mask_broadcast = neighbor_mask[:, None]  # (n_cells, 1)
    
    # 3. 兼容性计算（用掩码筛选有效邻居，避免动态索引）
    neighbor_dirs = D[:, collapse_idx].astype(jnp.int32)  # (n_cells,)
    opposite_dirs = jnp.take(dirs_opposite_index, neighbor_dirs, mode='clip')  # (n_cells,)
    log_compat = jnp.take(log_compatibility, opposite_dirs, axis=0)  # (n_cells, n_tiles, n_tiles)
    log_compat = log_compat + jnp.log(neighbor_mask_broadcast)[:, None]  # 无效邻居置为 -inf
    log_compat = jnp.clip(log_compat, -50, 0)
    
    # 添加微小噪声（稳定梯度）
    noise = jax.random.normal(key, log_compat.shape) * 1e-8
    log_compat = jnp.clip(log_compat + noise, -50, 0)
    
    # 4. 邻居贡献聚合（带温度系数，log空间适配）
    # 有效邻居的对数概率：无效邻居置为 -inf（log(0)）
    log_neighbor_probs = log_init_probs + jnp.log(neighbor_mask_broadcast)  # (n_cells, n_tiles)
    log_neighbor_probs = jnp.clip(log_neighbor_probs, -50, 0)
    
    # 扩展维度后相加（保持 n_cells 维度，静态形状）
    log_update_factors = log_compat + log_neighbor_probs[:, None, :]  # (n_cells, n_tiles, n_tiles)
    # 第一步聚合：tile维度（普通空间sum(C*p) → logsumexp）
    log_sum_factors = jax.scipy.special.logsumexp(log_update_factors, axis=2)  # (n_cells, n_tiles)
    # 温度系数适配：普通空间(x)^tau → log空间tau * log(x)
    log_tau_sum_factors = tau * log_sum_factors  # (n_cells, n_tiles)
    # 第二步聚合：所有邻居（带温度的求和，logsumexp保证归一化）
    sum_log = jax.scipy.special.logsumexp(log_tau_sum_factors, axis=0)  # (n_tiles,)
    sum_log = jnp.clip(sum_log, -50, 0)
    
    # 5. 更新当前单元概率（混合初始概率 + 归一化）
    log_p_updated = log_init_probs[collapse_idx] + sum_log
    # 归一化：保证概率和为1（关键！温度系数后必须归一化）
    log_p_updated = log_p_updated - jax.scipy.special.logsumexp(log_p_updated)
    # 混合初始概率
    log_p_updated = jnp.log(
        (1 - alpha) * jnp.exp(log_p_updated) + 
        alpha * jnp.exp(log_init_probs[collapse_idx])
    )
    # 再次归一化，避免混合后偏离1
    log_p_updated = log_p_updated - jax.scipy.special.logsumexp(log_p_updated)
    log_p_updated = jnp.clip(log_p_updated, -50, 0)
    
    # 6. 局部软更新
    updated_log_probs = log_init_probs * (1 - collapse_mask) + log_p_updated * collapse_mask
    # 全局归一化（可选，保证所有单元概率和为n_cells）
    updated_log_probs = updated_log_probs - jax.scipy.special.logsumexp(updated_log_probs, axis=1)[:, None]
    return updated_log_probs

@partial(jax.jit, static_argnames=['tau'])  # 添加tau静态参数
def single_update_neighbors(collapse_idx, log_probs, A, D, log_compatibility, tau=2.0):
    n_cells, n_tiles = log_probs.shape
    
    # 1. 提取邻居掩码（避免动态索引）
    neighbor_mask = A[:, collapse_idx]  # (n_cells,)
    neighbor_mask_broadcast = neighbor_mask[:, None]  # (n_cells, 1)
    
    # 2. 邻居方向与兼容性（用掩码过滤）
    neighbor_dirs = D[:, collapse_idx].astype(jnp.int32)  # (n_cells,)
    log_compat = jnp.take(log_compatibility, neighbor_dirs, axis=0)  # (n_cells, n_tiles, n_tiles)
    log_compat = log_compat + jnp.log(neighbor_mask_broadcast)[:, None]  # 无效邻居置为 -inf
    log_compat = jnp.clip(log_compat, -50, 0)
    
    # 3. 邻居贡献聚合（带温度系数）
    log_p_collapsed = log_probs[collapse_idx]  # (n_tiles,)
    log_p_neigh = log_compat + log_p_collapsed[None, None, :]  # (n_cells, n_tiles, n_tiles)
    # 第一步聚合：tile维度
    log_contrib = jax.scipy.special.logsumexp(log_p_neigh, axis=2)  # (n_cells, n_tiles)
    # 温度系数适配
    log_tau_contrib = tau * log_contrib  # (n_cells, n_tiles)
    
    # 4. 邻居概率更新（用掩码过滤无效邻居 + 归一化）
    w = neighbor_mask_broadcast  # (n_cells, 1)
    log_p_prev = log_probs  # (n_cells, n_tiles)
    # 仅有效邻居参与更新（无效邻居因 w=0 保持原概率）
    log_p_updated = jnp.log((1 - w) * jnp.exp(log_p_prev) + w * jnp.exp(log_tau_contrib))
    # 归一化：每个邻居的概率和为1
    log_p_updated = log_p_updated - jax.scipy.special.logsumexp(log_p_updated, axis=1)[:, None]
    log_p_updated = jnp.clip(log_p_updated, -50, 0)
    
    return log_p_updated  # 直接返回更新后的概率（无效邻居未变）



def preprocess_compatibility(compatibility, compat_threshold=1e-3, eps=1e-5):
    """
    预处理兼容性矩阵：逐行乘以行和的倒数（你的思路）
    :param compatibility: 原兼容性矩阵 (T, T)
    :param compat_threshold: 兼容阈值（>阈值视为兼容）
    :param eps: 数值稳定项（避免除0）
    :return: 新兼容性矩阵 C' (T, T)
    """
    print("Preprocessing compatibility matrix...")
    # 步骤1：计算每行的兼容tile数（行和，>阈值视为兼容）
    compat_mask = (compatibility > compat_threshold).astype(jnp.float32)
    row_sum = jnp.sum(compat_mask, axis=-1)  # (T,) 行和向量s
    
    # 步骤2：计算反向权重向量v（行和的倒数，加eps避免除0）
    v = 1.0 / (row_sum + eps)  # (T,)
    
    # 步骤3：逐行乘以v（等价于diag(v)·C）
    new_compatibility = v[:, None] * compatibility  # (T, T)
    
    return new_compatibility


@jax.jit
def waveFunctionCollapse(init_probs, A, D, dirs_opposite_index, compatibility, key, tau=1.0,*args, **kwargs):  # 添加tau参数
    """WFC主函数：用vmap批量处理，适配可变邻居数"""
    n_cells, n_tiles = init_probs.shape
    
    # 1. 初始化对数概率（避免log(0)）
    init_probs_clipped = jnp.clip(init_probs, 1e-5, 1.0)  # 用1e-5作为void阈值
    log_init_probs = jnp.log(init_probs_clipped)
    log_init_probs = jnp.clip(log_init_probs, -11.5, 0)  # log(1e-5)≈-11.5
    # 初始归一化：保证每个单元概率和为1
    log_init_probs = log_init_probs - jax.scipy.special.logsumexp(log_init_probs, axis=1)[:, None]
    
    # 2. 兼容性矩阵转换为对数空间
    compatibility_clipped = jnp.clip(compatibility, 1e-5, 1.0)
    log_compatibility = jnp.log(compatibility_clipped)
    log_compatibility = jnp.clip(log_compatibility, -11.5, 0)
    
    # 3. 第一步：批量更新所有单元（用vmap替代scan）
    subkeys = jax.random.split(key, n_cells)  # 每个单元独立密钥
    # 用vmap并行处理所有单元的update_by_neighbors（传递tau参数）
    batch_updated_step1 = jax.vmap(
        single_update_by_neighbors,
        in_axes=(0, 0, None, None, None, None, None, None, None)  # 新增tau的in_axis=None
    )(
        jnp.arange(n_cells),  # 所有单元索引
        subkeys,              # 批量密钥
        log_init_probs,       # 初始对数概率（共享）
        A, D, dirs_opposite_index, log_compatibility,
        0.,  # alpha参数
        tau  # 温度系数
    )
    # 聚合第一步结果（因迭代独立，取平均叠加）
    log_probs_step1 = jnp.mean(batch_updated_step1, axis=0)
    # 第一步后归一化
    log_probs_step1 = log_probs_step1 - jax.scipy.special.logsumexp(log_probs_step1, axis=1)[:, None]
    
    # 4. 第二步：批量更新邻居（用vmap，传递tau参数）
    batch_updated_step2 = jax.vmap(
        single_update_neighbors,
        in_axes=(0, None, None, None, None, None)  # 新增tau的in_axis=None
    )(
        jnp.arange(n_cells),  # 所有单元索引
        log_probs_step1,      # 第一步结果（共享）
        A, D, log_compatibility,
        tau  # 温度系数
    )
    # 聚合第二步结果
    final_log_probs = jnp.mean(batch_updated_step2, axis=0)
    # 最终归一化
    final_log_probs = final_log_probs - jax.scipy.special.logsumexp(final_log_probs, axis=1)[:, None]
    
    # 5. 转换回概率空间
    final_probs = jnp.exp(final_log_probs)
    final_probs = jnp.clip(final_probs, 1e-5, 1.0)  # 低于阈值视为void
    # 最终概率归一化（保证每个单元概率和为1）
    final_probs = final_probs / jnp.sum(final_probs, axis=1)[:, None]
    return final_probs, 0, jnp.arange(n_cells)