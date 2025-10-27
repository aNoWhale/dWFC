import os
import sys
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/WFC 目录下）
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
import gmsh
import jax
# jax.config.update('jax_disable_jit', True)

import jax.numpy as jnp
from functools import partial


import tqdm.rich as tqdm

import scipy.sparse
import numpy as np


from collections import defaultdict
from scipy.sparse import csr_matrix

from src.WFC.gumbelSoftmax import gumbel_softmax
from src.WFC.shannonEntropy import shannon_entropy
from src.WFC.TileHandler_JAX import TileHandler
from src.WFC.builder import visualizer_2D,Visualizer
from src.WFC.FigureManager import FigureManager



# # @jax.jit
# def get_neighbors(csr, index):
#     """获取指定索引的邻居列表"""
#     start = csr['row_ptr'][index]
#     end = csr['row_ptr'][index + 1]
#     neighbors = csr['col_idx'][start:end]
#     neighbors_dirs = csr['directions'][start:end]
#     return neighbors, neighbors_dirs


@jax.jit
def batch_get_neighbors(vmap_adj, batch_indices):
    """
    批量获取单元的邻居和方向（兼容vmap）。
    适配 convert_adj_to_vmap_compatible 转换后的邻居矩阵结构。
    
    参数:
        vmap_adj: 经 convert_adj_to_vmap_compatible 转换后的邻居矩阵
        batch_indices: 批量单元索引（JAX数组，形状[batch_size]）
    
    返回:
        neighbors_batch: 批量邻居索引（形状[batch_size, max_neighbors]）
        directions_batch: 批量方向整数（形状[batch_size, max_neighbors]），对应dir_map中的整数
    """
    # 从转换后的adj中获取整数形式的邻居和方向（修正键名以匹配转换结果）
    neighbors_batch = vmap_adj['padded_neighbors'][batch_indices]  # 邻居索引（整数）
    directions_batch = vmap_adj['padded_dirs_int'][batch_indices]  # 方向整数（对应dir_map）
    
    return neighbors_batch, directions_batch



@partial(jax.jit, static_argnames=('threshold',))
def collapsed_mask(probabilities, threshold=0.99):
    """
    创建连续掩码标记已坍缩单元
    1 = 未坍缩, 0 = 已坍缩（接近 one-hot）
    """
    max_probs = jnp.max(probabilities, axis=-1, keepdims=True)
    return jax.nn.sigmoid(-1000 * (max_probs - threshold))

@partial(jax.jit, static_argnames=('tau',))
def select_collapse(key, probs, tau=0.03,*args, **kwargs):
    """
    calculate shannon entropy, give out mask by probability, modify shannon entropy by mask, select uncollapsed.
    """
    # 1. 计算各单元熵值
    entropy = shannon_entropy(probs,axis=-1)
    # 2. 标记已坍缩单元
    mask = collapsed_mask(probs).squeeze()
    c_map=kwargs.pop("collapse_map",None)+mask
    # 3. 调整熵值：已坍缩单元赋予高熵值
    # 未坍缩: entropy_adj = entropy
    # 已坍缩: entropy_adj = max_entropy + 1
    max_entropy = jnp.max(entropy)
    entropy_adj = entropy * mask + (1 - mask) * (max_entropy + 100.0)
    entropy_adj = entropy * 1/(c_map+1)
    # 4. 转换为选择概率（最小熵对应最高概率）
    selection_logits = -entropy_adj  # 最小熵->最大logits
    # selection_logits = entropy_adj
    # 5. 使用Gumbel-Softmax采样位置
    flat_logits = selection_logits.reshape(-1)

    collapse_probs = gumbel_softmax(key, flat_logits, tau=tau, hard=True, axis=-1, eps=1e-10)
    collapse_idx = jnp.argmax(collapse_probs)

    return collapse_idx, max_entropy,c_map

# @partial(jax.jit, static_argnames=())
# def select_collapse_by_map(key, map, shape):
#     wait4collapseIndices = jnp.where( map == jnp.max(map))
#     probs = jnp.ones_like(wait4collapseIndices[0])/len(wait4collapseIndices[0])
#     collapse_probs = gumbel_softmax(key, probs, hard=True, axis=-1, eps=1e-10)
#     collapse_idx = jnp.argmax(collapse_probs)
#     return wait4collapseIndices[0][collapse_idx]

# @partial(jax.jit, static_argnames=('batch_size', 'shape', 'max_neighbors'))
# def select_batch_collapse_by_map(key, map, vmap_adj, shape, batch_size=16, max_neighbors=4):
#     """批量选择高优先级坍缩单元（按map最大值筛选，一次选batch_size个）"""
#     max_val = jnp.max(map)
#     mask = map == max_val  # 高优先级单元掩码
    
#     # 1. 获取所有高优先级单元索引（用-1填充无效位置，确保形状固定）
#     all_candidates = jnp.nonzero(mask, size=shape[0], fill_value=-1)[0]
#     valid_count = jnp.sum(mask)
#     valid_count = jnp.maximum(valid_count, 1)  # 避免除以0
    
#     # 2. 随机选batch_size个（从有效候选中采样，重复也不影响，后续过滤已坍缩）
#     subkey, key = jax.random.split(key)
#     random_offsets = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=valid_count)
#     batch_indices = all_candidates[random_offsets]
#     candidates = jnp.unique(batch_indices)

#     # 3. 过滤无效索引（-1），用第一个有效索引兜底
#     # 或许可以直接使用-1？
#     # first_valid = jnp.argmax(mask)
#     # batch_indices = jnp.where(batch_indices == -1, first_valid, batch_indices)
    
#     return candidates

@partial(jax.jit, static_argnames=('shape', 'batch_size', 'max_neighbors'))
def select_batch_collapse_by_map(
    key, map_arr, vmap_adj, shape, batch_size=16, max_neighbors=4
):
     # 1. 获取高优先级有效候选（无-1）
    max_val = jnp.max(map_arr)
    mask = map_arr == max_val
    valid_candidates = jnp.nonzero(mask)[0]
    valid_count = jnp.maximum(len(valid_candidates), 1)  # 至少1个候选
    
    # 2. 采样并去重（候选池大小为batch_size*2）
    subkey, key = jax.random.split(key)
    offsets = jax.random.randint(subkey, (batch_size * 2,), 0, valid_count)
    candidates = jnp.unique(valid_candidates[offsets])
    num_candidates = len(candidates)  # 动态候选数量
    
    # 3. 批量获取邻居（固定形状[num_candidates, max_neighbors]，含-1填充）
    neighbors_batch, _ = batch_get_neighbors(vmap_adj, candidates)  # 无需动态切片，保留-1
    
    # 4. 用lax.scan筛选无重叠邻居的单元
    init_state = (
        jnp.full((batch_size,), -1, dtype=jnp.int32),  # 已选中单元
        jnp.full((batch_size * max_neighbors,), -1, dtype=jnp.int32),  # 已选中的所有邻居（固定长度）
        jnp.array(0, dtype=jnp.int32)  # 已选数量
    )
    
    def scan_fn(state, i):
        selected, all_neighbors, count = state
        candidate = candidates[i]
        candidate_neigh = neighbors_batch[i]  # 含-1的固定长度邻居数组
        
        # 条件1：若已选满，直接跳过
        if count >= batch_size:
            return state, None
        
        # 过滤候选邻居中的-1（有效邻居）
        valid_candidate_neigh = candidate_neigh[candidate_neigh != -1]
        
        # 过滤已选中邻居中的-1（有效邻居）
        valid_all_neighbors = all_neighbors[all_neighbors != -1]
        
        # 条件2：判断邻居是否重叠
        has_overlap = jnp.any(jnp.isin(valid_candidate_neigh, valid_all_neighbors))
        
        # 不重叠则加入选中列表
        new_count = jnp.where(has_overlap, count, count + 1)
        new_selected = selected.at[count].set(jnp.where(has_overlap, selected[count], candidate))
        
        # 更新已选中的邻居集合（追加有效邻居，用-1填充剩余位置）
        neighbor_start = count * max_neighbors
        new_all_neighbors = all_neighbors.at[neighbor_start:neighbor_start+len(valid_candidate_neigh)].set(
            jnp.where(has_overlap, all_neighbors[neighbor_start:neighbor_start+len(valid_candidate_neigh)], valid_candidate_neigh)
        )
        
        return (new_selected, new_all_neighbors, new_count), None
    
    # 执行扫描（遍历所有候选）
    final_state, _ = jax.lax.scan(scan_fn, init_state, jnp.arange(num_candidates))
    selected_indices, _, final_count = final_state
    
    # 返回有效选中单元（长度≤batch_size）
    return selected_indices[:final_count]


@partial(jax.jit, static_argnames=('batch_size', 'shape'))
def select_batch_collapse(key, collapse_map, adj_csr, shape, batch_size=8):
    # 1. 选Top-K个高优先级单元（K是batch_size的2倍，留足筛选空间）
    topk_vals, topk_indices = jax.lax.top_k(collapse_map, k=batch_size * 2)
    
    # 2. 构建非邻居掩码：确保选中的单元彼此不是邻居
    def is_non_neighbor(idx1, idx2):
        # 检查idx1是否在idx2的邻居列表中
        start = adj_csr['row_ptr'][idx2]
        end = adj_csr['row_ptr'][idx2 + 1]
        neighbors = jax.lax.dynamic_slice(adj_csr['col_idx'], (start,), (end-start,))
        return jnp.all(neighbors != idx1)
    
    # 3. 筛选出互不相邻的单元，组成最终批量
    batch_indices = []
    for idx in topk_indices:
        if len(batch_indices) >= batch_size:
            break
        # 检查当前idx是否与已选单元都非邻居
        non_conflict = jnp.all(jnp.array([is_non_neighbor(idx, b_idx) for b_idx in batch_indices]))
        if non_conflict:
            batch_indices.append(idx)
    
    # 4. 不足batch_size时用-1填充（后续处理跳过）
    batch_indices = jnp.array(batch_indices + [-1]*(batch_size - len(batch_indices)))
    return batch_indices


@partial(jax.jit, static_argnames=('shape',))  # shape作为静态参数（编译时已知的元组，如(100,)）
def select_collapse_by_map(key, map, shape):
    #不可微，梯度在此处会绕过，即initprobs不会影响坍缩决策
    # 1. 计算最大值和有效掩码（哪些位置是候选索引）
    max_val = jnp.max(map)
    mask = map == max_val  # 布尔数组，形状为shape（静态已知）
    
    # 2. 获取固定长度的索引数组（无效位置用-1填充，形状与map一致）
    # 确保索引数组长度固定（shape[0]），满足JIT对静态形状的要求
    all_indices = jnp.nonzero(mask, size=shape[0], fill_value=-1)[0]  # 一维数组，长度=shape[0]
    
    # 3. 计算有效索引的数量（动态值，但不影响数组形状）
    valid_count = jnp.sum(mask)
    valid_count = jnp.maximum(valid_count, 1)  # 避免除以0（兜底逻辑）
    
    # # 4. 生成固定长度的概率数组：有效位置=1/valid_count，无效位置=0
    # # 概率数组形状固定（与map一致），JIT可编译
    # probs = jnp.where(mask, 1.0 / valid_count, 0.0)  # 无效位置概率为0，确保不会被选中
    
    # # 5. Gumbel-softmax选择索引（在固定形状数组上操作）
    # collapse_probs = gumbel_softmax(key, probs, hard=True, axis=-1, eps=1e-10)
    # selected_pos = jnp.argmax(collapse_probs)  # 选中的是概率数组中的位置（范围在0~shape[0]-1，静态已知）
    
    # # 6. 从固定长度索引数组中取结果（安全访问，形状固定）
    # collapse_idx = all_indices[selected_pos]
    # 4. 生成随机整数索引（范围[0, valid_count-1]）
    # 注意：randint的high参数可以是动态值，JIT兼容
    subkey, key = jax.random.split(key)  # 拆分密钥，避免随机状态重复
    random_offset = jax.random.randint(subkey, shape=(), minval=0, maxval=valid_count)
    
    # 5. 从有效索引中随机选择（利用固定长度数组的前valid_count个元素）
    collapse_idx = all_indices[random_offset]
    
    # 兜底：若极端情况选中无效索引（-1），强制取第一个有效索引
    first_valid_pos = jnp.argmax(mask)  # 第一个有效索引的位置
    collapse_idx = jnp.where(collapse_idx == -1, all_indices[first_valid_pos], collapse_idx)
    
    return collapse_idx

@partial(jax.jit, static_argnames=("tileNum",))
def update_by_neighbors(probs, collapse_id, neighbors, tileNum, dirs_opposite_index, compatibility):
    def update_single(neighbor_prob,opposite_dire_index):
        p_c = jnp.einsum("...ij,...j->...i", compatibility[opposite_dire_index], neighbor_prob) # (d,i,j) (j,)-> (d,i,j) (1,1,j)->(d,i,1)->(d,i)
        norm = jnp.sum(jnp.abs(p_c), axis=-1, keepdims=True)
        return p_c / jnp.where(norm == 0, 1.0, norm)

    neighbor_probs = probs[neighbors]
    neighbor_probs = jnp.atleast_1d(neighbor_probs)
    opposite_indices = jnp.array(dirs_opposite_index)
    batch_update = jax.vmap(update_single)
    update_factors = batch_update(neighbor_probs, opposite_indices)
    cumulative_factor = jnp.prod(update_factors, axis=0)
    p = probs[collapse_id] * cumulative_factor
    norm = jnp.sum(jnp.abs(p), axis=-1, keepdims=True)
    p = p / jnp.where(norm == 0, 1.0, norm)
    return probs.at[collapse_id].set(p)

@partial(jax.jit, static_argnames=())
def batch_update_by_neighbors(probs, collapse_ids, neighbors_batch, dirs_opposite_index_batch, compatibility):
    """批量更新多个坍缩单元的概率（基于各自邻居）"""
    def update_single(collapse_id, neighbors, dirs_opposite_index):
        """单个单元的更新逻辑（复用原逻辑）"""
        def update_neighbor_prob(neighbor, opposite_dire_index):
            neighbor_prob=probs[neighbor]
            p_c = jnp.einsum("...ij,...j->...i", compatibility[opposite_dire_index], neighbor_prob)
            norm = jnp.sum(jnp.abs(p_c), axis=-1, keepdims=True)
            p = p_c / jnp.where(norm == 0, 1.0, norm)
            return p
        neighbors = jnp.atleast_1d(neighbors)
        update_factors = jax.vmap(update_neighbor_prob)(neighbors, dirs_opposite_index)
        mask = (neighbors == -1)[:,None]
        update_factors = jnp.where(mask, 1.0, update_factors)
        cumulative_factor = jnp.prod(update_factors, axis=0)
        p = probs[collapse_id] * cumulative_factor
        norm = jnp.sum(jnp.abs(p), axis=-1, keepdims=True)
        p = p / jnp.where(norm == 0, 1.0, norm)
        return p  # 仅返回更新后的概率，不直接set
    
    # 用vmap批量处理多个坍缩单元
    updated_probs_batch = jax.vmap(update_single)(
        collapse_ids, neighbors_batch, dirs_opposite_index_batch
    )

    # 批量更新全局概率
    #TODO 此处应该修改为乘法以应对针对相同单元格的更新？ 如果是这样的话就要加一步归一化
    return probs.at[collapse_ids].set(updated_probs_batch)



@jax.jit
def update_neighbors(probs, neighbors, dirs_index, p_collapsed, compatibility):
    def vectorized_update(neighbor_prob, dir_idx):
        p_neigh = jnp.einsum("...ij,...j->...i", compatibility, p_collapsed)
        p_neigh = jnp.einsum("...i,...i->...i", p_neigh, neighbor_prob)
        norm = jnp.sum(jnp.abs(p_neigh), axis=-1, keepdims=True)
        p_neigh = p_neigh / jnp.where(norm == 0, 1.0, norm)
        return jnp.clip(p_neigh[dir_idx], 0, 1)
    dirs_index=jnp.array(dirs_index)
    updated_probs = jax.vmap(vectorized_update)(probs[neighbors], dirs_index)
    return probs.at[neighbors].set(updated_probs)

@jax.jit
def batch_update_neighbors(probs, neighbors_batch, dirs_index_batch, p_collapsed_batch, compatibility):
    """批量更新多个坍缩单元的邻居概率"""
    def update_single_neighbors(neighbors, dirs_index, p_collapsed):
        """单个坍缩单元的邻居更新逻辑（复用原逻辑）"""
        def vectorized_update(neighbor_prob, dir_idx):
            p_neigh = jnp.einsum("...ij,...j->...i", compatibility, p_collapsed)
            p_neigh = jnp.einsum("...i,...i->...i", p_neigh, neighbor_prob)
            norm = jnp.sum(jnp.abs(p_neigh), axis=-1, keepdims=True)
            p_neigh = p_neigh / jnp.where(norm == 0, 1.0, norm)
            return jnp.clip(p_neigh[dir_idx], 0, 1)
        
        dirs_index = jnp.array(dirs_index)
        updated_probs = jax.vmap(vectorized_update)(probs[neighbors], dirs_index)
        return neighbors, updated_probs  # 返回邻居索引和对应更新值
    
    # 1. 批量处理每个坍缩单元的邻居更新
    neighbors_list, updated_probs_list = jax.vmap(update_single_neighbors)(
        neighbors_batch, dirs_index_batch, p_collapsed_batch
    )

    # 2. 展平邻居和更新值（处理不同单元的邻居数量差异）
    flat_neighbors = neighbors_list.reshape(-1)
    flat_updated = updated_probs_list.reshape(-1, probs.shape[-1])
    mask = (flat_neighbors == -1)[:,None]
    flat_updated = jnp.where(mask, 1, flat_updated)
    # # 3. 批量更新全局概率（重复邻居会被多次更新，符合WFC逻辑）
    probs = probs.at[flat_neighbors].multiply(flat_updated)
    norm = jnp.sum(jnp.abs(probs), axis=-1, keepdims=True)
    probs = probs / jnp.where(norm == 0, 1.0, norm)
    return probs
    # return flat_neighbors,flat_updated


@partial(jax.jit, static_argnames=('tau','max_rerolls','zero_threshold','k'))   
def collapse(subkey, probs, max_rerolls=3, zero_threshold=-1e5, k=1000.0, tau=1e-3):
    near_zero_mask = jax.nn.sigmoid(k * (zero_threshold - probs))
    initial_gumbel = gumbel_softmax(subkey, probs, tau=tau, hard=False, axis=-1)
    key, subkey = jax.random.split(subkey)
    chosen_near_zero = jnp.sum(initial_gumbel * near_zero_mask)
    should_reroll = jax.nn.sigmoid(k * (chosen_near_zero - 0.5))
    CONVERGENCE_THRESHOLD = 0.01
    # def continue_loop(should_reroll):
    #     return should_reroll > CONVERGENCE_THRESHOLD
    final_gumbel, total_rerolls, key = jax.lax.cond(
        should_reroll > CONVERGENCE_THRESHOLD,
        true_fun=lambda: full_while_loop(
            subkey, probs, initial_gumbel, near_zero_mask,
            max_rerolls, zero_threshold, k, tau, should_reroll
        ),
        false_fun=lambda: (initial_gumbel, jnp.array(0.0), subkey)
    )
    return final_gumbel, key

@partial(jax.jit, static_argnames=('tau', 'max_rerolls', 'zero_threshold', 'k'))
def batch_collapse(subkeys, probs_batch, tau=1e-3, max_rerolls=3, zero_threshold=-1e-5, k=1000):
    """批量坍缩多个单元（每个单元独立坍缩）"""
    # 用vmap并行调用原collapse函数，每个单元对应一个subkey
    collapsed_probs, _ = jax.vmap(
        lambda key, p: collapse(key, p, tau=tau, max_rerolls=max_rerolls, zero_threshold=zero_threshold, k=k)
    )(subkeys, probs_batch)
    return collapsed_probs


@partial(jax.jit, static_argnames=('max_rerolls','zero_threshold','k','tau'))
def full_while_loop(
    subkey, probs, initial_gumbel, near_zero_mask,
    max_rerolls, zero_threshold, k, tau, initial_should_reroll
):
    """用while_loop实现重选循环，直到重选概率≤阈值或达到最大次数"""
    CONVERGENCE_THRESHOLD = 0.01
    
    # 定义循环状态：(重选次数, 当前采样结果, 随机密钥, 当前重选概率)
    initial_state = (
        jnp.array(1.0),  # 初始重选次数（已执行1次初始检查）
        initial_gumbel,  # 当前gumbel采样
        subkey,          # 随机密钥
        initial_should_reroll  # 初始重选概率（>阈值）
    )
    def cond_fn(state):
        reroll_count, _, _, current_should_reroll = state
        # 继续循环的条件：重选次数<上限 且 重选概率>阈值
        return (reroll_count < max_rerolls) & (current_should_reroll > CONVERGENCE_THRESHOLD)
    
    # 循环体函数：更新状态（执行一次重选）
    def body_fn(state):
        reroll_count, current_gumbel, key, _ = state
        new_gumbel = gumbel_softmax(key, probs, tau=tau, hard=False, axis=-1)
        new_key, subkey = jax.random.split(key)
        
        # 计算新的重选概率
        chosen_near_zero = jnp.sum(new_gumbel * near_zero_mask)
        new_should_reroll = jax.nn.sigmoid(k * (chosen_near_zero - 0.5))
        
        # 混合新旧采样并归一化
        # mixed_gumbel = 0.8 * new_gumbel + (0.2) * current_gumbel
        mixed_gumbel = new_gumbel
        mixed_gumbel = mixed_gumbel / (jnp.sum(mixed_gumbel, axis=-1, keepdims=True) + 1e-8)
        
        # 更新重选次数（累计有效重选）
        new_reroll_count = reroll_count + jnp.clip(new_should_reroll, 0, 1)
        
        # 返回更新后的状态
        return (new_reroll_count, mixed_gumbel, new_key, new_should_reroll)
    
    # 执行while_loop
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    
    # 提取最终结果
    total_rerolls, final_gumbel, key, _ = final_state
    return final_gumbel, total_rerolls, key


def waveFunctionCollapseBatch(init_probs, adj_csr, tileHandler: TileHandler,vmap_adj=None ,plot: bool | str = False, batch_size=1, *args, **kwargs) -> jnp.ndarray:
    key = jax.random.PRNGKey(0)
    num_elements = init_probs.shape[0]
    probs = init_probs
    collapse_map = jnp.ones(probs.shape[0])
    key, subkey = jax.random.split(key)
    init = jax.random.randint(subkey, shape=(), minval=0, maxval=probs.shape[0])
    collapse_map = collapse_map.at[init].multiply(10)
    should_stop = False
    collapse_list = [-1]  # 记录批量坍缩索引
    max_neighbors = vmap_adj['max_neighbors']
    # 初始化可视化（保持原逻辑）
    if plot == "2d":
        visualizer: Visualizer = kwargs.pop("visualizer", None)
        visualizer.add_frame(probs=probs)
    
    pbar = tqdm.tqdm(total=num_elements, desc="collapsing", unit="tiles")
    
    while not should_stop:
        # 1. 概率归一化（保持原逻辑）
        norm = jnp.sum(jnp.abs(probs), axis=-1, keepdims=True)
        probs = probs / jnp.where(norm == 0, 1.0, norm)
        
        # 2. 批量选择坍缩单元（替换原单单元选择）
        key, subkey = jax.random.split(key)
        batch_indices = select_batch_collapse_by_map(subkey, collapse_map, vmap_adj, collapse_map.shape[0], batch_size=batch_size, max_neighbors=max_neighbors,)
        # 过滤已坍缩单元（collapse_map=0的单元跳过）
        valid_mask = collapse_map[batch_indices] > 0
        valid_batch = batch_indices[valid_mask]
        if len(valid_batch) == 0:
            if jnp.max(collapse_map) < 1:
                should_stop = True
            break
        
        # 3. 批量获取邻居和方向（用vmap并行处理）
        neighbors_batch, dirs_index_batch = batch_get_neighbors(vmap_adj, valid_batch)
        
        
        # 5. 批量计算反向方向索引（替换原for循环）
        def get_opposite_dirs(dirs_index):
            """单个单元的反向方向计算"""
            def single_opposite(idx):
                return tileHandler.get_opposite_index_by_index(idx)
            return jax.vmap(single_opposite)(dirs_index)
        dirs_opposite_index_batch = jax.vmap(get_opposite_dirs)(dirs_index_batch)
        
        #TODO 填充的invalid值-1会引发异常行为，应当加入mask

        # 6. 批量更新坍缩单元的概率（替换原单单元update_by_neighbors） 已处理-1
        probs = batch_update_by_neighbors(
            probs, valid_batch, neighbors_batch, dirs_opposite_index_batch, tileHandler.compatibility
        )
        
        # 7. 批量坍缩单元（替换原单单元collapse）
        key, subkey = jax.random.split(key)
        batch_subkeys = jax.random.split(subkey, len(valid_batch))  # 每个单元独立密钥
        p_collapsed_batch = batch_collapse(
            batch_subkeys, probs[valid_batch], tau=1e-3, max_rerolls=3, zero_threshold=-1e-5, k=1000
        )

        # 批量更新坍缩后的概率
        probs = probs.at[valid_batch].set(jnp.clip(p_collapsed_batch, 0, 1))
        collapse_list.extend(jnp.unique(valid_batch).tolist())  # 记录批量索引
        
        # 8. 批量更新collapse_map（标记已坍缩+提升邻居优先级）
        collapse_map = collapse_map.at[valid_batch].multiply(0)  # 标记已坍缩
        
        unique_neighbors = jnp.unique(neighbors_batch,axis=-2) # 展平邻居并去重，避免重复更新
        flat_neighbors = unique_neighbors.reshape(-1)
        collapse_map = collapse_map.at[flat_neighbors].multiply(10)  # 提升邻居优先级
        
        # 9. 批量更新邻居概率（替换原单单元update_neighbors）
        probs = batch_update_neighbors(
            probs, neighbors_batch, dirs_index_batch, p_collapsed_batch, tileHandler.compatibility
        )
        # probs = probs.at[neighbors].multiply(update_factors)
        
        # 10. 可视化和进度条更新（适配批量）
        if plot == '2d':
            visualizer.add_frame(probs=probs)
        pbar.update(len(valid_batch))  # 按批量大小更新进度
        
        # 停止条件：所有单元已坍缩
        if jnp.max(collapse_map) < 1:
            should_stop = True
            break
        if pbar.n >= pbar.total:
            pbar.set_description_str("fixing high entropy")
    
    pbar.close()
    return probs, 0, collapse_list



if __name__ == "__main__":
    from src.utiles.adjacency import build_grid_adjacency
    height = 10
    width = 10
    adj=build_grid_adjacency(height=height, width=width, connectivity=4)

    # from src.utiles.generateMsh import generate_cube_hex_mesh
    # msh_name='box.msh'
    # from jax_fem.generate_mesh import box_mesh_gmsh
    # Nx,Ny,Nz=2,2,2
    # meshio_mesh = box_mesh_gmsh(
    #     Nx=Nx,
    #     Ny=Ny,
    #     Nz=Nz,
    #     domain_x=1.0,
    #     domain_y=1.0,
    #     domain_z=1.0,
    #     data_dir=f"data",
    #     ele_type='HEX8',
    # )
    # from src.WFC.adjacencyCSR import build_hex8_adjacency_with_meshio
    # adj = build_hex8_adjacency_with_meshio(f'data/msh/{msh_name}')

    tileHandler = TileHandler(typeList=['a','b','c','d','e'],direction=(('up',"down"),("left","right"),), direction_map={"up": 0, "down": 2, "left": 3, "right": 1})
    from src.dynamicGenerator.TileImplement.Dimension2.LinePath import LinePath
    tileHandler.register(typeName='a',class_type=LinePath(['da-bc','cen-cd'],color='blue'))
    tileHandler.register(typeName='b',class_type=LinePath(['ab-cd','cen-da'],color='green'))
    tileHandler.register(typeName='c',class_type=LinePath(['da-bc','cen-ab'],color='yellow'))
    tileHandler.register(typeName='d',class_type=LinePath(['ab-cd','cen-bc'],color='red'))
    tileHandler.register(typeName='e',class_type=LinePath(['da-bc','ab-cd'],color='magenta'))

    tileHandler.selfConnectable(typeName="e",direction='isotropy',value=1)


    tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','d','b'],direction='down',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','d','a'],direction='left',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','b','a'],direction='right',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName='c',direction='up',value=1,dual=True)


    tileHandler.setConnectiability(fromTypeName='b',toTypeName=['e','d','a','c'],direction='left',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='b',toTypeName=['e','d','a','b'],direction='up',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='b',toTypeName=['e','d','c','b'],direction='down',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='b',toTypeName='d',direction='right',value=1,dual=True)


    tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','d','a','b'],direction='up',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','d','a','c'],direction='left',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','a','b','c'],direction='right',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',direction='down',value=1,dual=True)


    tileHandler.setConnectiability(fromTypeName='d',toTypeName=['e','c','a','b'],direction='right',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='d',toTypeName=['e','a','b','d'],direction='up',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='d',toTypeName=['e','c','b','d'],direction='down',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='d',toTypeName='b',direction='left',value=1,dual=True)


    tileHandler.setConnectiability(fromTypeName='e',toTypeName='a',direction='up',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='e',toTypeName='b',direction='right',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='e',toTypeName='c',direction='down',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='e',toTypeName='d',direction='left',value=1,dual=True)

    # tileHandler.setConnectiability(fromTypeName='a',toTypeName='b',direction='left',value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='a',toTypeName='d',direction='right',value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='b',toTypeName='c',direction='up',value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='b',toTypeName='c',direction='down',value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='c',toTypeName='b',direction='left',value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='c',toTypeName='d',direction='right',value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='d',toTypeName='c',direction='up',value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='d',toTypeName='c',direction='down',value=0,dual=True)
    # print(f"tileHandler:\n {tileHandler}")
    tileHandler.constantlize_compatibility()

    num_elements = adj['num_elements']
    numTypes = tileHandler.typeNum
    init_probs = jnp.ones((num_elements ,numTypes)) / numTypes # (n_elements, n_types)
    from src.utiles.generateMsh import generate_grid_vertices_vectorized
    figureManager=FigureManager(figsize=(10,10))
    visualizer=Visualizer(tileHandler=tileHandler,points=adj['vertices'],figureManager=figureManager)
    grid= generate_grid_vertices_vectorized(width+1,height+1)

    from src.WFC.GPUTools.batchAdjCSR import convert_adj_to_vmap_compatible
    vmap_adj=convert_adj_to_vmap_compatible(adj)
    probs,max_entropy, collapse_list=waveFunctionCollapseBatch(init_probs,adj,tileHandler, vmap_adj=vmap_adj,batch_size=8,plot='2d',points=adj['vertices'],figureManger=figureManager,visualizer=visualizer)
    wfc = lambda p:waveFunctionCollapseBatch(p,adj,tileHandler,vmap_adj=vmap_adj,plot='2d',points=adj['vertices'],figureManger=figureManager,visualizer=visualizer)
    def loss_fn(init_probs):
            probs, _, collapse_list = wfc(init_probs)
            cell_max_p = jnp.max(probs, axis=-1) + 1e-8
            modified_e = -cell_max_p * jnp.log2(cell_max_p) + (1 - cell_max_p)
            mean = jnp.mean(modified_e)
            ec = mean / 0.1 - 1
            return ec
    
    visualizer.collapse_list=collapse_list
    visualizer.draw()
    visualizer_2D(tileHandler=tileHandler,probs=probs,points=adj['vertices'],figureManager=figureManager,epoch='end')
    pattern = jnp.argmax(probs, axis=-1, keepdims=False).reshape(width,height)
    name_pattern = tileHandler.pattern_to_names(pattern)
    print(f"pattern: \n{name_pattern}")
    print(f"max entropy: {max_entropy}")
