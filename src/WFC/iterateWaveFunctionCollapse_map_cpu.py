import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import jax
jax.config.update('jax_platform_name', 'cpu')  # 强制使用CPU
jax.config.update('jax_disable_jit', True)     # 禁用JIT避免追踪问题

import jax.numpy as jnp
from functools import partial

import tqdm.rich as tqdm

import numpy as np

from src.WFC.gumbelSoftmax import gumbel_softmax
from src.WFC.TileHandler_JAX import TileHandler
from src.WFC.builder import visualizer_2D, Visualizer
from src.WFC.FigureManager import FigureManager


def get_neighbors(csr, index):
    start = csr['row_ptr'][index]
    end = csr['row_ptr'][index + 1]
    neighbors = csr['col_idx'][start:end]
    neighbors_dirs = csr['directions'][start:end]
    return neighbors, neighbors_dirs


def collapsed_mask(probabilities, threshold=0.99):
    max_probs = jnp.max(probabilities, axis=-1, keepdims=True)
    return jax.nn.sigmoid(-1000 * (max_probs - threshold))


def select_collapse_by_map(key, collapse_map):
    max_val = jnp.max(collapse_map)
    mask = collapse_map == max_val
    all_indices = jnp.nonzero(mask)[0]
    if len(all_indices) == 0:
        return 0
    subkey, _ = jax.random.split(key)
    random_idx = jax.random.randint(subkey, shape=(), minval=0, maxval=len(all_indices))
    return all_indices[random_idx]


def update_by_neighbors_align(probs, collapse_id, neighbors, dirs_opposite_index, compatibility):
    """使用jax.lax.cond替代Python的if判断"""
    def update_neighbor_prob(neighbor, opposite_dire_index):
        # 用jax.lax.cond处理追踪变量的条件判断
        return jax.lax.cond(
            neighbor == -1,  # 条件：邻居无效
            lambda: jnp.ones(probs.shape[-1]),  # 无效时返回全1（不影响更新）
            lambda: jnp.einsum("...ij,...j->...i", compatibility[opposite_dire_index], probs[neighbor])
        )
    
    neighbors = jnp.atleast_1d(neighbors)
    dirs_opposite_index = jnp.atleast_1d(dirs_opposite_index)
    update_factors = jax.vmap(update_neighbor_prob)(neighbors, dirs_opposite_index)
    
    # 对有效邻居的结果进行归一化（无效邻居的结果是1，不影响乘积）
    def normalize(factor):
        norm = jnp.sum(jnp.abs(factor), axis=-1, keepdims=True)
        return factor / jnp.where(norm == 0, 1.0, norm)
    update_factors = jax.vmap(normalize)(update_factors)
    
    cumulative_factor = jnp.prod(update_factors, axis=0)
    p = probs[collapse_id] * cumulative_factor
    norm = jnp.sum(jnp.abs(p), axis=-1, keepdims=True)
    return p / jnp.where(norm == 0, 1.0, norm)


def update_neighbors(probs, neighbors, dirs_index, p_collapsed, compatibility):
    """使用jax.lax.cond替代Python的if判断"""
    def vectorized_update(neighbor_prob, dir_idx, neighbor):
        # 用jax.lax.cond处理追踪变量的条件判断
        return jax.lax.cond(
            neighbor == -1,  # 条件：邻居无效
            lambda: neighbor_prob,  # 无效时不更新
            lambda: jnp.clip(
                jnp.einsum("...i,...i->...i", 
                           jnp.einsum("...ij,...j->...i", compatibility, p_collapsed)[dir_idx],
                           neighbor_prob
                          ),
                0, 1
            )
        )
    
    valid_neighbors = neighbors
    valid_dirs = dirs_index
    # 传递原始邻居索引用于判断有效性
    updated_probs = jax.vmap(vectorized_update)(probs[valid_neighbors], valid_dirs, valid_neighbors)
    return probs.at[valid_neighbors].set(updated_probs)


def collapse(subkey, prob, max_rerolls=3, zero_threshold=-1e5, k=1000.0, tau=1e-3):
    near_zero_mask = jax.nn.sigmoid(k * (zero_threshold - prob))
    initial_gumbel = gumbel_softmax(subkey, prob, tau=tau, hard=False, axis=-1)
    chosen_near_zero = jnp.sum(initial_gumbel * near_zero_mask)
    should_reroll = jax.nn.sigmoid(k * (chosen_near_zero - 0.5))
    CONVERGENCE_THRESHOLD = 0.01
    
    if should_reroll > CONVERGENCE_THRESHOLD:
        for _ in range(max_rerolls):
            subkey, _ = jax.random.split(subkey)
            new_gumbel = gumbel_softmax(subkey, prob, tau=tau, hard=False, axis=-1)
            chosen_near_zero = jnp.sum(new_gumbel * near_zero_mask)
            if jax.nn.sigmoid(k * (chosen_near_zero - 0.5)) <= CONVERGENCE_THRESHOLD:
                initial_gumbel = new_gumbel
                break
    
    return jnp.clip(initial_gumbel, 0, 1)


def waveFunctionCollapse(init_probs, adj_csr, tileHandler: TileHandler, plot: bool | str = False, *args, **kwargs) -> jnp.ndarray:
    key = jax.random.PRNGKey(0)
    num_elements = init_probs.shape[0]
    max_neighbors = kwargs.pop("max_neighbors", 4)

    probs = init_probs
    collapse_map = jnp.ones(probs.shape[0])
    should_stop = False

    if plot == "2d":
        visualizer: Visualizer = kwargs.pop("visualizer", None)
        visualizer.add_frame(probs=probs)
    
    pbar = tqdm.tqdm(total=num_elements, desc="collapsing", unit="tiles")
    collapse_list = [-1]

    while not should_stop:
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # 概率归一化
        solid_mask = np.sum(probs,axis=-1) > 0.3
        norm = jnp.sum(jnp.abs(probs), axis=-1, keepdims=True)
        probs = probs / jnp.where(norm == 0, 1.0, norm)
        
        # 选择单个坍缩单元
        collapse_idx = select_collapse_by_map(subkey1, collapse_map)
        
        # 停止条件
        if jnp.max(collapse_map) < 1:
            print(f"####reached stop condition####\n")
            should_stop = True
            break
        if solid_mask[collapse_idx]:

            # 获取邻居和方向信息
            neighbors, neighbors_dirs = get_neighbors(adj_csr, collapse_idx)
            neighbors_dirs_index = tileHandler.get_index_by_direction(neighbors_dirs)
            neighbors_dirs_opposite_index = tileHandler.opposite_dir_array[jnp.array(neighbors_dirs_index)]
            
            # 对齐邻居数组
            def align_single_array(arr, m, fill_value=-1):
                arr = jnp.array(arr)
                pad_length = max(0, m - len(arr))
                return jnp.pad(arr, (0, pad_length), mode='constant', constant_values=fill_value)
            
            neighbors_aligned = align_single_array(neighbors, max_neighbors)
            dirs_index_aligned = align_single_array(neighbors_dirs_index, max_neighbors)
            dirs_opposite_aligned = align_single_array(neighbors_dirs_opposite_index, max_neighbors)

            # 更新坍缩单元概率
            p_updated = update_by_neighbors_align(
                probs, collapse_idx, neighbors_aligned, dirs_opposite_aligned, tileHandler.compatibility
            )
            probs = probs.at[collapse_idx].set(p_updated)
            
            # 执行坍缩
            p_collapsed = collapse(subkey2, probs[collapse_idx])
            probs = probs.at[collapse_idx].set(p_collapsed)
   
            # 更新邻居概率
            probs = update_neighbors(probs, neighbors_aligned, dirs_index_aligned, p_collapsed, tileHandler.compatibility)
        else:
            probs = probs.at[collapse_idx].multiply(0)

        # 更新坍缩记录和掩码
        collapse_list.append(collapse_idx)
        collapse_map = collapse_map.at[collapse_idx].set(0)
        valid_neighbors = neighbors_aligned[neighbors_aligned != -1]
        collapse_map = collapse_map.at[valid_neighbors].multiply(10)


        # 可视化
        if plot == '2d' and visualizer is not None:
            visualizer.add_frame(probs=probs)

        pbar.update(1)
        if pbar.n >= pbar.total:
            pbar.set_description_str("fixing high entropy")
    
    pbar.close()
    return probs, 0, collapse_list


if __name__ == "__main__":
    from src.utiles.adjacency import build_grid_adjacency
    height = 50
    width = 50
    adj = build_grid_adjacency(height=height, width=width, connectivity=4)

    # 初始化瓦片处理器
    tileHandler = TileHandler(typeList=['a','b','c','d','e'], direction=(('up',"down"),("left","right"),))
    from src.dynamicGenerator.TileImplement.Dimension2.LinePath import LinePath
    tileHandler.register(typeName='a', class_type=LinePath(['da-bc','cen-cd'], color='blue'))
    tileHandler.register(typeName='b', class_type=LinePath(['ab-cd','cen-da'], color='green'))
    tileHandler.register(typeName='c', class_type=LinePath(['da-bc','cen-ab'], color='yellow'))
    tileHandler.register(typeName='d', class_type=LinePath(['ab-cd','cen-bc'], color='red'))
    tileHandler.register(typeName='e', class_type=LinePath(['da-bc','ab-cd'], color='magenta'))

    # 设置瓦片连接性
    tileHandler.selfConnectable(typeName="e", direction='isotropy', value=1)
    tileHandler.setConnectiability(fromTypeName='a', toTypeName=['e','c','d','b'], direction='down', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='a', toTypeName=['e','c','d','a'], direction='left', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='a', toTypeName=['e','c','b','a'], direction='right', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='a', toTypeName='c', direction='up', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='b', toTypeName=['e','d','a','c'], direction='left', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='b', toTypeName=['e','d','a','b'], direction='up', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='b', toTypeName=['e','d','c','b'], direction='down', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='b', toTypeName='d', direction='right', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='c', toTypeName=['e','d','a','b'], direction='up', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='c', toTypeName=['e','d','a','c'], direction='left', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='c', toTypeName=['e','a','b','c'], direction='right', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='c', toTypeName='a', direction='down', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='d', toTypeName=['e','c','a','b'], direction='right', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='d', toTypeName=['e','a','b','d'], direction='up', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='d', toTypeName=['e','c','b','d'], direction='down', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='d', toTypeName='b', direction='left', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='e', toTypeName='a', direction='up', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='e', toTypeName='b', direction='right', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='e', toTypeName='c', direction='down', value=1, dual=True)
    tileHandler.setConnectiability(fromTypeName='e', toTypeName='d', direction='left', value=1, dual=True)

    tileHandler.constantlize_compatibility()

    # 初始化概率分布
    num_elements = adj['num_elements']
    numTypes = tileHandler.typeNum
    init_probs = jnp.ones((num_elements, numTypes)) / numTypes

    # 可视化配置
    figureManager = FigureManager(figsize=(10,10))
    visualizer = Visualizer(tileHandler=tileHandler, points=adj['vertices'], figureManager=figureManager)

    # 运行WFC
    probs, max_entropy, collapse_list = waveFunctionCollapse(
        init_probs, adj, tileHandler, plot='2d', 
        points=adj['vertices'], visualizer=visualizer, max_neighbors=4
    )

    # 结果输出
    visualizer.collapse_list = collapse_list
    visualizer.draw()
    visualizer_2D(tileHandler=tileHandler, probs=probs, points=adj['vertices'], figureManager=figureManager, epoch='end')
    pattern = jnp.argmax(probs, axis=-1, keepdims=False).reshape(width, height)
    name_pattern = tileHandler.pattern_to_names(pattern)
    print(f"pattern: \n{name_pattern}")
    print(f"max entropy: {max_entropy}")