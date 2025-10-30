import os
import sys
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/WFC 目录下）
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import jax
# jax.config.update('jax_disable_jit', True)

import jax.numpy as jnp
from functools import partial


import tqdm.rich as tqdm


import numpy as np

from src.WFC.gumbelSoftmax import gumbel_softmax
from src.WFC.TileHandler_JAX import TileHandler
from src.WFC.builder import visualizer_2D,Visualizer
from src.WFC.FigureManager import FigureManager



# @jax.jit
def get_neighbors(csr, index):
    """获取指定索引的邻居列表"""
    start = csr['row_ptr'][index]
    end = csr['row_ptr'][index + 1]
    neighbors = csr['col_idx'][start:end]
    neighbors_dirs = csr['directions'][start:end]
    return neighbors, neighbors_dirs


@partial(jax.jit, static_argnames=('threshold',))
def collapsed_mask(probabilities, threshold=0.99):
    """
    创建连续掩码标记已坍缩单元
    1 = 未坍缩, 0 = 已坍缩（接近 one-hot）
    """
    max_probs = jnp.max(probabilities, axis=-1, keepdims=True)
    return jax.nn.sigmoid(-1000 * (max_probs - threshold))

@partial(jax.jit, static_argnames=('shape',))  # shape作为静态参数（编译时已知的元组，如(100,)）
def select_collapse_by_map(key, map, shape):
    max_val = jnp.max(map)
    mask = map == max_val  # 布尔数组，形状为shape（静态已知）
    all_indices = jnp.nonzero(mask, size=shape[0], fill_value=-1)[0]  # 一维数组，长度=shape[0]
    valid_count = jnp.sum(mask)
    valid_count = jnp.maximum(valid_count, 1)  # 避免除以0（兜底逻辑）
    subkey, key = jax.random.split(key)  # 拆分密钥，避免随机状态重复
    random_offset = jax.random.randint(subkey, shape=(), minval=0, maxval=valid_count)
    collapse_idx = all_indices[random_offset]
    first_valid_pos = jnp.argmax(mask)  # 第一个有效索引的位置
    collapse_idx = jnp.where(collapse_idx == -1, all_indices[first_valid_pos], collapse_idx)
    
    return collapse_idx

# @partial(jax.jit, static_argnames=('shape', 'adj_csr'))  # shape作为静态参数（编译时已知的元组，如(100,)）
# def select_collapse_by_map_batch(collapse_map, adj_csr, batch_size):
#     topk_vals, topk_indices = jax.lax.top_k(collapse_map, k=batch_size)
#     exist_neighbor_set=set()
#     collapse_idx = []
#     for ind in topk_indices:
#         neighbors, _ = get_neighbors(adj_csr, ind)
#         now_neighbors_set = set(np.asarray(nei).item() for nei in neighbors)
#         now_neighbors_set.add(ind.item())
#         if not bool(exist_neighbor_set & now_neighbors_set): #如果不具有相同邻居 且不是直接邻居
#             exist_neighbor_set |= now_neighbors_set
#             collapse_idx.append(ind)

#     # selected_array = jnp.full((batch_size,), -1, dtype=jnp.int32)
#     # selected_array = selected_array.at[:len(selected)].set(jnp.array(selected, dtype=jnp.int32))
#     # return selected_array
#     return collapse_idx

def select_collapse_by_map_batch(collapse_map, adj_csr, batch_size):
    # 核心1：筛选非零索引及对应值
    nonzero_indices = jnp.nonzero(collapse_map)[0]
    nonzero_values = collapse_map[nonzero_indices]
    k = min(batch_size, len(nonzero_values))
    if k == 0:
        return []
    
    # 核心2：取top k非零索引（按值排序）
    _, topk_in_nonzero = jax.lax.top_k(nonzero_values, k=k)
    topk_indices = nonzero_indices[topk_in_nonzero]
    
    # 核心3：筛选邻居不重叠的索引
    exist_neighbors = set()
    collapse_idx = []
    for ind in topk_indices:
        neighbors, _ = get_neighbors(adj_csr, ind)
        curr_neighbors = set(np.asarray(nei).item() for nei in neighbors) | {ind.item()}
        if not exist_neighbors & curr_neighbors:
            exist_neighbors.update(curr_neighbors)
            collapse_idx.append(ind)
    
    return collapse_idx


@partial(jax.jit, static_argnames=())
def update_by_neighbors(probs, collapse_id, neighbors, dirs_index, dirs_opposite_index, compatibility):
    def update_single(neighbor_prob,opposite_dire_index):
        # p_c = jnp.einsum("...ij,...j->...i", compatibility[opposite_dire_index], neighbor_prob) # (d,i,j) (j,)-> (d,i,j) (1,1,j)->(d,i,1)->(d,i)
        p_c = jax.lax.dot(compatibility[opposite_dire_index], neighbor_prob, precision=jax.lax.Precision.HIGH)
        return p_c
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
def update_by_neighbors_align(probs, collapse_id, neighbors, dirs_opposite_index,compatibility):
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
        return p.squeeze()  # 仅返回更新后的概率，不直接set


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
def collapse(subkey, prob, max_rerolls=3, zero_threshold=-1e5, k=1000.0, tau=1e-3):
    near_zero_mask = jax.nn.sigmoid(k * (zero_threshold - prob))
    initial_gumbel = gumbel_softmax(subkey, prob, tau=tau, hard=False, axis=-1)
    key, subkey = jax.random.split(subkey)
    chosen_near_zero = jnp.sum(initial_gumbel * near_zero_mask)
    should_reroll = jax.nn.sigmoid(k * (chosen_near_zero - 0.5))
    CONVERGENCE_THRESHOLD = 0.01
    # def continue_loop(should_reroll):
    #     return should_reroll > CONVERGENCE_THRESHOLD
    final_gumbel, total_rerolls, key = jax.lax.cond(
        should_reroll > CONVERGENCE_THRESHOLD,
        true_fun=lambda: full_while_loop(
            subkey, prob, initial_gumbel, near_zero_mask,
            max_rerolls, zero_threshold, k, tau, should_reroll
        ),
        false_fun=lambda: (initial_gumbel, jnp.array(0.0), subkey)
    )
    return final_gumbel

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



def waveFunctionCollapse(init_probs,adj_csr, tileHandler: TileHandler,plot:bool|str=False,*args,**kwargs)->jnp.ndarray:
    """a WFC function

    Args:
        init_probs (array): (n_cells,n_tileTypes)
        adj_csr (Dict): a dict with neighbor CSR and directions CSR
        tileHandler (TileHandler): a TileHandler to deal with Tiles
        plot (bool|str, optional): '2d' or '3d' or False, give out result during the iter whether or not. Defaults to False.
    
    Kwargs:
        points (array): vertices of cells

    Returns:
        jnp.ndarray: probs (n_cells,n_tileTypes)
    """
    key = jax.random.PRNGKey(0)
    num_elements = init_probs.shape[0]
    max_neighbors = kwargs.pop("max_neighbors",4)
    batch_size=kwargs.pop("batch_size",8)

    # 初始化概率分布
    probs=init_probs
    collapse_map = jnp.ones(probs.shape[0])
    

    should_stop = False
    if plot=="2d":
        visualizer:Visualizer=kwargs.pop("visualizer",None)
        visualizer.add_frame(probs=probs)
    pbar = tqdm.tqdm(total=num_elements, desc="collapsing", unit="tiles")
    collapse_list=[-1]
    while should_stop is False:
        key, subkey1, subkey2 = jax.random.split(key, 3)
        norm = jnp.sum(jnp.abs(probs), axis=-1, keepdims=True)
        probs = probs / jnp.where(norm == 0, 1.0, norm)
        # collapse_idx, max_entropy, collapse_map = select_collapse(subkey, probs, tau=1e-3,collapse_map=collapse_map,)
        # collapse_idx = select_collapse_by_map(subkey1, collapse_map, collapse_map.shape)
        collapse_idx_list = select_collapse_by_map_batch(collapse_map,adj_csr,batch_size=batch_size)
        if jnp.max(collapse_map) < 1: 
            print(f"####reached stop condition####\n")
            should_stop=True
            break

        # 获取该单元的邻居
        neighbors_list =[]
        neighbors_dirs_index_list = []
        neighbors_dirs_opposite_index_list = []
        for cid in collapse_idx_list:
            neighbors, neighbors_dirs = get_neighbors(adj_csr, cid)
            neighbors_dirs_index = tileHandler.get_index_by_direction(neighbors_dirs)
            neighbors_list.append(neighbors)
            neighbors_dirs_index_list.append(neighbors_dirs_index)
            # 根据邻居更新坍缩单元格的概率
            neighbors_dirs_opposite_index = tileHandler.opposite_dir_array[jnp.array(neighbors_dirs_index)]
            neighbors_dirs_opposite_index_list.append(neighbors_dirs_opposite_index)
        #邻居可能形状不整齐
        
        def align_array(input_list, m, fill_value=-1):
            result = jnp.array([jnp.pad(jnp.array(sub), (0, m - len(sub)), mode='constant', constant_values=-1) for sub in input_list])
            return result
        
        n=len(collapse_idx_list)
        collapse_array = jnp.array(collapse_idx_list)
        neighbors_array = align_array(neighbors_list,max_neighbors)
        neighbors_dirs_index_array = align_array(neighbors_dirs_index_list,max_neighbors)
        neighbors_dirs_opposite_index_array = align_array(neighbors_dirs_opposite_index_list,max_neighbors)


        # probs = update_by_neighbors(probs, collapse_idx, neighbors, neighbors_dirs_index,neighbors_dirs_opposite_index, tileHandler.compatibility)
        p = jax.lax.map(lambda para:update_by_neighbors_align(probs,para[0],para[1],para[2],tileHandler.compatibility), (collapse_array[:,None], neighbors_array,neighbors_dirs_opposite_index_array))
        # p = update_by_neighbors_align(probs, collapse_idx, neighbors, neighbors_dirs_opposite_index)
        probs = probs.at[collapse_array].set(p)
        
        # 坍缩选定的单元
        subkeys = jax.random.split(subkey2, n)
        p_collapsed = jax.lax.map(lambda pa: collapse(pa[0],pa[1], max_rerolls=3, zero_threshold=-1e-5, tau=1e-3,k=1000),(subkeys,probs[collapse_array]))
        # p_collapsed = collapse(subkey=subkey2, prob=probs[collapse_idx], max_rerolls=3, zero_threshold=-1e-5, tau=1e-3,k=1000)
        probs = probs.at[collapse_array].set(jnp.clip(p_collapsed,0,1))

        collapse_list.append(collapse_idx_list)
        # 更新map
        collapse_map = collapse_map.at[collapse_array].multiply(0)
        collapse_map = collapse_map.at[neighbors_array].multiply(10)
        # 更新邻居的概率
        # probs = update_neighbors(probs, neighbors, neighbors_dirs_index, p_collapsed, tileHandler.compatibility)
        probs = batch_update_neighbors(probs,neighbors_array,neighbors_dirs_index_array,p_collapsed, tileHandler.compatibility)

        if plot is not False:
            if plot == '2d':
                visualizer.add_frame(probs=probs)
            if plot == "3d":
                #TODO 3D visualizer here
                pass
        pbar.update(len(collapse_idx_list))
        if pbar.n > pbar.total:
            pbar.set_description_str("fixing high entropy")
    pbar.close()
    return probs, 0, collapse_list

if __name__ == "__main__":
    from src.utiles.adjacency import build_grid_adjacency
    height = 50
    width = 50
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

    tileHandler = TileHandler(typeList=['a','b','c','d','e'],direction=(('up',"down"),("left","right"),))
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
    # from src.WFC.GPUTools.batchAdjCSR import convert_adj_to_vmap_compatible
    # vmap_adj=convert_adj_to_vmap_compatible(adj)
    probs,max_entropy, collapse_list=waveFunctionCollapse(init_probs,adj,tileHandler,plot='2d',points=adj['vertices'],figureManger=figureManager,visualizer=visualizer,batch_size=128,max_neighbors=4)


    wfc = lambda p:waveFunctionCollapse(p,adj,tileHandler,plot='2d',points=adj['vertices'],figureManger=figureManager,visualizer=visualizer,batch_size=1,max_neighbors=4)
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
