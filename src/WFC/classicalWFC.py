import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import jax
jax.config.update('jax_platform_name', 'cpu')  # 强制使用CPU
# jax.config.update('jax_disable_jit', True)     # 禁用JIT避免追踪问题

import jax.numpy as jnp
from functools import partial

import tqdm.rich as tqdm

import numpy as np

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


def select_collapse_by_map (key,collapse_map):
    max_map = jnp.max(collapse_map)
    choices = jnp.where(collapse_map == max_map)[0]
    idx = jax.random.choice(key, choices)
    return idx


@partial(jax.jit, static_argnames=())
def update_by_neighbors(probs, collapse_id, neighbors, dirs_opposite_index, compatibility):
    def update_single(neighbor_prob,opposite_dire_index):
        # print(f"neighbor_prob.shape:{neighbor_prob.shape}")
        p_c = jnp.einsum("...ij,...j->...i", compatibility[opposite_dire_index], neighbor_prob) # (d,i,j) (j,)-> (d,i,j) (1,1,j)->(d,i,1)->(d,i)
        # p_neigh = jnp.einsum("...i,...i->...i",p_neigh,neighbor_prob) #(d,i) (i,) ->(d,i) (1,i) -> (d,i) 

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
    


@jax.jit
def update_neighbors(probs, neighbors, dirs_index, p_collapsed, compatibility):
    # 定义向量化更新函数（包含方向选择）
    def vectorized_update(neighbor_prob, dir_idx):
        p_neigh = jnp.einsum("...ij,...j->...i", compatibility, p_collapsed)
        p_neigh = jnp.einsum("...i,...i->...i", p_neigh, neighbor_prob)
        norm = jnp.sum(jnp.abs(p_neigh), axis=-1, keepdims=True)
        p_neigh = p_neigh / jnp.where(norm == 0, 1.0, norm)
        return jnp.clip(p_neigh[dir_idx], 0, 1)
    # print(f"probs[neighbors].shape:{probs[neighbors].shape}")
    dirs_index=jnp.array(dirs_index)
    # print(f"dirx_index.shape:{dirs_index.shape}")
    updated_probs = jax.vmap(vectorized_update)(probs[neighbors], dirs_index)
    return probs.at[neighbors].set(updated_probs)


def collapse( key,prob):
    zeros=jnp.zeros_like(prob)
    max_val = jnp.max(prob)
    max_indices = jnp.where(prob == max_val)[0]  
    random_idx = jax.random.choice(key, max_indices)
    return zeros.at[random_idx].set(1)


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
        
        
        solid_mask = np.sum(probs,axis=-1) > 0.4

        # 概率归一化
        norm = jnp.sum(jnp.abs(probs), axis=-1, keepdims=True)
        probs = probs / jnp.where(norm == 0, 1.0, norm)
        
        # 选择单个坍缩单元
        collapse_idx = select_collapse_by_map(subkey1,collapse_map)
        
        # 停止条件
        if jnp.max(collapse_map) < 1:
            print(f"####reached stop condition####\n")
            should_stop = True
            break

        # 获取邻居和方向信息
        neighbors, neighbors_dirs = get_neighbors(adj_csr, collapse_idx)
        neighbors_dirs_index = tileHandler.get_index_by_direction(neighbors_dirs)
        neighbors_dirs_opposite_index = tileHandler.opposite_dir_array[jnp.array(neighbors_dirs_index)]
        
        if solid_mask[collapse_idx]:

            # 更新坍缩单元概率
            probs = update_by_neighbors(
                probs, collapse_idx, neighbors, neighbors_dirs_opposite_index, tileHandler.compatibility
            )
            
            # 执行坍缩
            p_collapsed = collapse(subkey2,probs[collapse_idx])
            probs = probs.at[collapse_idx].set(p_collapsed)
   
            # 更新邻居概率
            probs = update_neighbors(probs, neighbors, neighbors_dirs_index, p_collapsed, tileHandler.compatibility)
        else:
            probs = probs.at[collapse_idx].multiply(0)

        # 更新坍缩记录和掩码
        collapse_list.append(collapse_idx)
        collapse_map = collapse_map.at[collapse_idx].set(0)
        collapse_map = collapse_map.at[neighbors].multiply(10)


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
    height = 5
    width = 5
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