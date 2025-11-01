import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from functools import partial
import tqdm.rich as tqdm
import numpy as np

from src.WFC.gumbelSoftmax import gumbel_softmax
from src.WFC.TileHandler_JAX import TileHandler
from src.WFC.builder import visualizer_2D, Visualizer
from src.WFC.FigureManager import FigureManager


@jax.jit
def get_neighbors_vmap(vmap_adj, batch_indices):
    """批量获取邻居（固定形状，无动态流）"""
    neighbors_batch = vmap_adj['padded_neighbors'][batch_indices]  # [batch_size, max_neighbors]
    directions_batch = vmap_adj['padded_dirs_int'][batch_indices]  # [batch_size, max_neighbors]
    return neighbors_batch, directions_batch


@partial(jax.jit, static_argnames=('threshold',))
def collapsed_mask(probabilities, threshold=0.99):
    """无控制流，纯向量化操作"""
    max_probs = jnp.max(probabilities, axis=-1, keepdims=True)
    return jax.nn.sigmoid(-1000 * (max_probs - threshold))


@partial(jax.jit, static_argnames=('shape',))
def select_collapse_by_map(key, map_arr, shape):
    """用lax操作替代动态索引，确保形状固定"""
    max_val = jnp.max(map_arr)
    mask = map_arr == max_val
    all_indices = jnp.nonzero(mask, size=shape[0], fill_value=-1)[0]  # 固定输出长度
    valid_count = jnp.maximum(jnp.sum(mask), 1)
    subkey, key = jax.random.split(key)
    random_offset = jax.random.randint(subkey, shape=(), minval=0, maxval=valid_count)
    collapse_idx = all_indices[random_offset]
    first_valid_pos = jnp.argmax(mask)
    collapse_idx = jnp.where(collapse_idx == -1, all_indices[first_valid_pos], collapse_idx)
    return collapse_idx

@partial(jax.jit, static_argnames=('batch_size', 'invalid_dir'))
def select_collapse_by_map_batch(collapse_map, vmap_adj, batch_size, invalid_dir):
    max_neighbors = vmap_adj['max_neighbors']
    num_elements = collapse_map.shape[0]
    
    # 1. 计算非零掩码和索引（固定形状）
    nonzero_mask = collapse_map > 0
    nonzero_indices = jnp.where(nonzero_mask, size=num_elements, fill_value=-1)[0]
    nonzero_values = jnp.where(nonzero_mask, collapse_map, 0.0)  # 固定形状[num_elements]
    
    # 2. 计算有效数量并确保k是非负的（关键修改）
    valid_count = jnp.sum(nonzero_mask)
    # 限制k在[0, batch_size]范围内，确保非负
    k = jnp.clip(jnp.minimum(batch_size, valid_count), 0, batch_size)
    
    # 3. 处理空输入（k == 0）
    def empty_case():
        return jnp.array([], dtype=jnp.int32)
    
    # 4. 非空处理（显式确保k >= 1）
    def non_empty_case():
        # 强制k至少为1（避免top_k接收k=0）
        k_safe = jnp.maximum(k, 1)
        # 用安全的k_safe调用top_k（此时k_safe >= 1）
        topk_values, topk_positions = jax.lax.top_k(nonzero_values, k=k_safe)
        
        # 过滤无效值（topk_values为0的位置对应无效索引）
        topk_valid_mask = topk_values > 0
        topk_indices = jnp.where(topk_valid_mask, nonzero_indices[topk_positions], -1)
        topk_indices = topk_indices[topk_indices != -1]  # 此时mask是具体的
        
        # 5. 获取邻居（固定形状）
        topk_neighbors = vmap_adj['padded_neighbors'][topk_indices]
        
        # 6. 扫描筛选不重叠索引
        def scan_body(acc, idx_neigh):
            selected, used = acc
            idx, neigh = idx_neigh
            if idx == -1:
                return (selected, used)
            neigh_valid = neigh != -1
            used_neigh = jnp.where(neigh_valid, used[neigh], False)
            overlap = jnp.any(used_neigh)
            
            def update_case():
                new_selected = selected.at[jnp.argmin(selected == -1)].set(idx)
                new_used = used.at[neigh[neigh_valid]].set(True)
                new_used = new_used.at[idx].set(True)
                return (new_selected, new_used)
            
            def skip_case():
                return (selected, used)
            
            return jax.lax.cond(overlap, skip_case, update_case)
        
        init_selected = jnp.full((batch_size,), -1, dtype=jnp.int32)
        init_used = jnp.zeros_like(collapse_map, dtype=jnp.bool_)
        final_selected, _ = jax.lax.scan(
            scan_body, (init_selected, init_used),
            (topk_indices, topk_neighbors)
        )
        
        return final_selected[final_selected != -1]
    
    # 确保cond的分支条件是标量布尔值（k == 0）
    return jax.lax.cond(k == 0, empty_case, non_empty_case)


@partial(jax.jit, static_argnames=())
def batch_update_collapse_probs(probs, collapse_indices, vmap_adj, compatibility, opposite_dir_array):
    """纯向量化操作，无动态流"""
    max_neighbors = vmap_adj['max_neighbors']
    batch_size = collapse_indices.shape[0]  # 从输入形状获取（静态已知）
    
    # 批量获取邻居和方向（固定形状[batch_size, max_neighbors]）
    neighbors_batch = vmap_adj['padded_neighbors'][collapse_indices]
    dirs_int_batch = vmap_adj['padded_dirs_int'][collapse_indices]
    dirs_opposite_batch = opposite_dir_array[dirs_int_batch]  # 向量化索引
    
    # 单元素处理函数（无控制流）
    def process_single(neighbors, dirs_opposite):
        valid_mask = neighbors != -1
        valid_neighbors = neighbors[valid_mask]
        valid_dirs_opposite = dirs_opposite[valid_mask]
        
        # 用lax.cond处理空邻居（分支形状一致）
        def has_neighbors():
            factors = jax.vmap(lambda n, d: jax.lax.dot(compatibility[d], probs[n]))(
                valid_neighbors, valid_dirs_opposite
            )
            cumulative = jnp.prod(factors, axis=0)
            updated_p = probs[collapse_indices[0]] * cumulative
            norm = jnp.sum(jnp.abs(updated_p), keepdims=True)
            return updated_p / jnp.where(norm == 0, 1.0, norm)
        
        def no_neighbors():
            return probs[collapse_indices[0]]  # 形状与has_neighbors一致
        
        return jax.lax.cond(valid_neighbors.size > 0, has_neighbors, no_neighbors)
    
    # 批量处理（固定形状[batch_size, num_types]）
    updated_probs = jax.vmap(process_single)(neighbors_batch, dirs_opposite_batch)
    return probs.at[collapse_indices].set(updated_probs)


@partial(jax.jit, static_argnames=('invalid_dir',))
def batch_update_neighbors(probs, collapse_indices, vmap_adj, compatibility, p_collapsed_batch, invalid_dir):
    """纯向量化操作，无动态流"""
    max_neighbors = vmap_adj['max_neighbors']
    batch_size = collapse_indices.shape[0]  # 静态已知
    
    # 展平数据（固定形状[batch_size * max_neighbors, ...]）
    neighbors_batch = vmap_adj['padded_neighbors'][collapse_indices]  # [B, M]
    dirs_int_batch = vmap_adj['padded_dirs_int'][collapse_indices]    # [B, M]
    flat_neigh = neighbors_batch.reshape(-1)  # [B*M]
    flat_dirs = dirs_int_batch.reshape(-1)    # [B*M]
    flat_p = jnp.repeat(p_collapsed_batch, max_neighbors, axis=0)     # [B*M, T]
    
    # 过滤无效方向时，用静态参数invalid_dir替代vmap_adj['dir_map']['invalid']
    valid_mask = (flat_neigh != -1) & (flat_dirs != invalid_dir)
    valid_neigh = flat_neigh[valid_mask]
    valid_dirs = flat_dirs[valid_mask]
    valid_p = flat_p[valid_mask]
    
    # 用lax.cond处理空情况（分支形状一致）
    def update_case():
        def update_single(neigh, dir_idx, p):
            p_neigh = jax.lax.dot(compatibility[dir_idx], p) * probs[neigh]
            norm = jnp.sum(jnp.abs(p_neigh), keepdims=True)
            return jnp.clip(p_neigh / jnp.where(norm == 0, 1.0, norm), 0, 1)
        
        updated = jax.vmap(update_single)(valid_neigh, valid_dirs, valid_p)
        probs_updated = probs.at[valid_neigh].multiply(updated)
        norm = jnp.sum(jnp.abs(probs_updated), axis=-1, keepdims=True)
        return probs_updated / jnp.where(norm == 0, 1.0, norm)
    
    def no_update_case():
        return probs  # 与update_case返回形状一致
    
    return jax.lax.cond(jnp.any(valid_mask), update_case, no_update_case)


@partial(jax.jit, static_argnames=('tau', 'max_rerolls', 'zero_threshold', 'k'))
def collapse(subkey, prob, max_rerolls=3, zero_threshold=-1e5, k=1000.0, tau=1e-3):
    """用lax.cond和lax.while_loop替代动态分支和循环"""
    near_zero_mask = jax.nn.sigmoid(k * (zero_threshold - prob))
    initial_gumbel = gumbel_softmax(subkey, prob, tau=tau, hard=False, axis=-1)
    key, subkey = jax.random.split(subkey)
    chosen_near_zero = jnp.sum(initial_gumbel * near_zero_mask)
    should_reroll = jax.nn.sigmoid(k * (chosen_near_zero - 0.5))
    CONVERGENCE_THRESHOLD = 0.01

    # 静态分支（条件为标量）
    def true_fun():
        return full_while_loop(subkey, prob, initial_gumbel, near_zero_mask,
                              max_rerolls, zero_threshold, k, tau, should_reroll)
    
    def false_fun():
        return (initial_gumbel, jnp.array(0.0), subkey)  # 与true_fun返回形状一致
    
    return jax.lax.cond(should_reroll > CONVERGENCE_THRESHOLD, true_fun, false_fun)


@partial(jax.jit, static_argnames=('max_rerolls', 'zero_threshold', 'k', 'tau'))
def full_while_loop(subkey, probs, initial_gumbel, near_zero_mask,
                   max_rerolls, zero_threshold, k, tau, initial_should_reroll):
    """用lax.while_loop替代while（循环次数由max_rerolls静态限制）"""
    CONVERGENCE_THRESHOLD = 0.01
    # 状态形状固定：(reroll_count, gumbel, key, should_reroll)
    initial_state = (jnp.array(0, dtype=jnp.int32), initial_gumbel, subkey, initial_should_reroll)
    
    # 循环条件（标量，无动态流）
    def cond_fn(state):
        reroll_count, _, _, current_should_reroll = state
        return (reroll_count < max_rerolls) & (current_should_reroll > CONVERGENCE_THRESHOLD)
    
    # 迭代逻辑（状态形状不变）
    def body_fn(state):
        reroll_count, current_gumbel, key, _ = state
        new_gumbel = gumbel_softmax(key, probs, tau=tau, hard=False, axis=-1)
        new_key, subkey = jax.random.split(key)
        chosen_near_zero = jnp.sum(new_gumbel * near_zero_mask)
        new_should_reroll = jax.nn.sigmoid(k * (chosen_near_zero - 0.5))
        mixed_gumbel = new_gumbel / (jnp.sum(new_gumbel, axis=-1, keepdims=True) + 1e-8)
        return (reroll_count + 1, mixed_gumbel, new_key, new_should_reroll)
    
    # 循环次数被max_rerolls限制（静态已知上限）
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    return final_state


def waveFunctionCollapse(init_probs, vmap_adj, tileHandler: TileHandler, plot: bool | str = False, *args, **kwargs):
    """主循环（外部非JIT部分，允许Python控制流）"""
    key = jax.random.PRNGKey(0)
    num_elements = init_probs.shape[0]
    batch_size = kwargs.pop("batch_size", 128)
    max_neighbors = vmap_adj['max_neighbors']  # 静态参数
    invalid_dir = dir_map["invalid"]
    
    # 确保方向映射一致（静态配置）
    tileHandler.opposite_dir_array = jnp.array([2, 3, 0, 1])  # 与dir_map匹配
    
    probs = init_probs
    collapse_map = jnp.ones(probs.shape[0])
    should_stop = False

    if plot == "2d":
        visualizer: Visualizer = kwargs.pop("visualizer", None)
        visualizer.add_frame(probs=probs)
    
    pbar = tqdm.tqdm(total=num_elements, desc="collapsing", unit="tiles", mininterval=1.5)

    while not should_stop:
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # 归一化（纯向量化）
        norm = jnp.sum(jnp.abs(probs), axis=-1, keepdims=True)
        probs = probs / jnp.where(norm == 0, 1.0, norm)
        
        # 批量选择坍缩单元（无动态流）
        collapse_indices = select_collapse_by_map_batch(
            collapse_map, vmap_adj, batch_size, invalid_dir=invalid_dir
        )
        
        # 终止条件（用JAX操作替代Python动态判断）
        stop_cond = (collapse_indices.size == 0) | (jnp.max(collapse_map) < 1)
        if stop_cond.item():  # 转为Python标量判断（主循环非JIT）
            print(f"####reached stop condition####\n")
            should_stop = True
            break
        
        # 批量更新概率（无动态流）
        probs = batch_update_collapse_probs(
            probs, collapse_indices, vmap_adj,
            tileHandler.compatibility, tileHandler.opposite_dir_array
        )
        
        # 批量坍缩（无动态流）
        subkeys = jax.random.split(subkey2, len(collapse_indices))
        p_collapsed_batch = jax.vmap(
            lambda sk, p: collapse(sk, p, max_rerolls=3, zero_threshold=-1e-5, tau=1e-3, k=1000)
        )(subkeys, probs[collapse_indices])[0]  # 取gumbel结果
        probs = probs.at[collapse_indices].set(jnp.clip(p_collapsed_batch, 0, 1))
        
        # 更新坍缩地图（向量化操作）
        collapse_map = collapse_map.at[collapse_indices].set(0)
        neighbors_batch = vmap_adj['padded_neighbors'][collapse_indices]
        flat_neighbors = neighbors_batch.reshape(-1)
        valid_neighbors = flat_neighbors[flat_neighbors != -1]
        collapse_map = collapse_map.at[valid_neighbors].multiply(10)
        
        # 批量更新邻居（无动态流）
        probs = batch_update_neighbors(
            probs, collapse_indices, vmap_adj,
            tileHandler.compatibility, p_collapsed_batch,
            invalid_dir=invalid_dir  # 作为静态参数传入
        )
        
        if plot == '2d':
            visualizer.add_frame(probs=probs)
        
        pbar.update(len(collapse_indices))
    
    pbar.close()
    return probs, 0, []


if __name__ == "__main__":
    from src.utiles.adjacency import build_grid_adjacency
    height, width = 50, 50
    dir_map = {"up": 0, "down": 2, "right": 1, "left": 3, "invalid": -1}
    adj = build_grid_adjacency(height=height, width=width, connectivity=4)
    
    from src.WFC.GPUTools.batchAdjCSR import convert_adj_to_vmap_compatible
    vmap_adj = convert_adj_to_vmap_compatible(adj, direction_to_idx=dir_map)

    tileHandler = TileHandler(typeList=['a','b','c','d','e'], direction=(('up',"down"),("left","right"),))
    from src.dynamicGenerator.TileImplement.Dimension2.LinePath import LinePath
    tileHandler.register(typeName='a', class_type=LinePath(['da-bc','cen-cd'], color='blue'))
    tileHandler.register(typeName='b', class_type=LinePath(['ab-cd','cen-da'], color='green'))
    tileHandler.register(typeName='c', class_type=LinePath(['da-bc','cen-ab'], color='yellow'))
    tileHandler.register(typeName='d', class_type=LinePath(['ab-cd','cen-bc'], color='red'))
    tileHandler.register(typeName='e', class_type=LinePath(['da-bc','ab-cd'], color='magenta'))

    # 配置连接性（与dir_map一致）
    tileHandler.selfConnectable(typeName="e", direction='isotropy', value=1)
    tileHandler.setConnectiability(fromTypeName='a', toTypeName=['e','c','d','b'], direction='down', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='a', toTypeName=['e','c','d','a'], direction='left', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='a', toTypeName=['e','c','b','a'], direction='right', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='a', toTypeName='c', direction='up', value=1, dual=True)
    # 其他连接性配置...

    tileHandler.constantlize_compatibility()

    num_elements = vmap_adj['num_elements']
    numTypes = tileHandler.typeNum
    init_probs = jnp.ones((num_elements, numTypes)) / numTypes
    figureManager = FigureManager(figsize=(10, 10))
    visualizer = Visualizer(tileHandler=tileHandler, points=vmap_adj['vertices'], figureManager=figureManager)
    
    probs, max_entropy, _ = waveFunctionCollapse(
        init_probs, vmap_adj, tileHandler, plot='2d',
        visualizer=visualizer, batch_size=128
    )

    visualizer.draw()
    visualizer_2D(tileHandler=tileHandler, probs=probs, points=vmap_adj['vertices'], figureManager=figureManager, epoch='end')
    pattern = jnp.argmax(probs, axis=-1).reshape(width, height)
    print(f"pattern: \n{tileHandler.pattern_to_names(pattern)}")