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
from src.WFC.TileHandler import TileHandler
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
    #TODO 如果一次坍缩多个，这个mask可能会失效
    max_probs = jnp.max(probabilities, axis=-1, keepdims=True)
    return jax.nn.sigmoid(-1000 * (max_probs - threshold))

@partial(jax.jit, static_argnames=('tau',))
def select_collapse(key, probs, tau=0.03):
    """
    calculate shannon entropy, give out mask by probability, modify shannon entropy by mask, select uncollapsed.
    """
    # 1. 计算各单元熵值
    entropy = shannon_entropy(probs,axis=-1)

    # 2. 标记已坍缩单元
    mask = collapsed_mask(probs).squeeze()

    # 3. 调整熵值：已坍缩单元赋予高熵值
    # 未坍缩: entropy_adj = entropy
    # 已坍缩: entropy_adj = max_entropy + 1
    max_entropy = jnp.max(entropy)
    # print(f"max entropy: {max_entropy}", )
    entropy_adj = entropy * mask + (1 - mask) * (max_entropy + 10.0)

    # 4. 转换为选择概率（最小熵对应最高概率）
    selection_logits = -entropy_adj  # 最小熵->最大logits
    # print('modified logits:\n',selection_logits)
    # 5. 使用Gumbel-Softmax采样位置
    flat_logits = selection_logits.reshape(-1)

    collapse_probs = gumbel_softmax(key, flat_logits, tau=tau, hard=True, axis=-1, eps=1e-10)
    collapse_idx = jnp.argmax(collapse_probs)

    return collapse_idx, max_entropy


@partial(jax.jit, static_argnames=())
def update_by_neighbors(probs, collapse_id, neighbors, dirs_index, dirs_opposite_index, compatibility):
    """向量化更新邻居概率"""
    def update_single(neighbor_prob,opposite_dire_index):
        p_c = jnp.einsum("...ij,...j->...i", compatibility[opposite_dire_index], neighbor_prob) # (d,i,j) (j,)-> (d,i,j) (1,1,j)->(d,i,1)->(d,i)
        # p_neigh = jnp.einsum("...i,...i->...i",p_neigh,neighbor_prob) #(d,i) (i,) ->(d,i) (1,i) -> (d,i) 

        norm = jnp.sum(jnp.abs(p_c), axis=-1, keepdims=True)
        return p_c / jnp.where(norm == 0, 1.0, norm)
    p=probs[collapse_id]
    for neighbor,di,odi in zip(neighbors,dirs_index, dirs_opposite_index):
        p=p*update_single(probs[neighbor],odi)
    norm = jnp.sum(jnp.abs(p), axis=-1, keepdims=True)
    p = p / jnp.where(norm == 0, 1.0, norm)
    probs=probs.at[collapse_id].set(p)
    return probs

@partial(jax.jit, static_argnames=())
def update_neighbors(probs, neighbors, dirs_index ,p_collapsed, compatibility):
    """向量化更新邻居概率"""
    def update_single(neighbor_prob):
        p_neigh = jnp.einsum("...ij,...j->...i", compatibility, p_collapsed) # (d,i,j) (j,)-> (d,i,j) (1,1,j)->(d,i,1)->(d,i)
        p_neigh = jnp.einsum("...i,...i->...i",p_neigh,neighbor_prob) #(d,i) (i,) ->(d,i) (1,i) -> (d,i) 
        # print(f"compatibiliy @ p_collapsed * neighbor_prob:\n{jnp.einsum('...ij,...j->...i',compatibility, p_collapsed)} * {neighbor_prob}")
        # print(f'p_neigh: {p_neigh}')
        norm = jnp.sum(jnp.abs(p_neigh), axis=-1, keepdims=True)
        p_neigh = p_neigh / jnp.where(norm == 0, 1.0, norm)
        return jnp.clip(p_neigh, 0, 1)
    for neighbor,dirs in zip(neighbors,dirs_index):
        prob=update_single(probs[neighbor])
        # print(f"update neighbor {neighbor} with {prob}")
        probs = probs.at[neighbor].set(prob[dirs])
    return probs

# def collapse(subkey,probs,max_rerolls=3,zero_threshold=1e-5,k=1000.0,tau=1e-3):
#     near_zero_mask = jax.nn.sigmoid(k * (zero_threshold - probs)) #大于threshold ~=0， 小于threshold ~= 1
#     # 声明空间
#     gumbel_output = probs * 0  # 初始化为全零
    
#     def body_fn(state, i):
#         reroll_count, gumbel, subkey = state
#         # 当前步骤的Gumbel采样
#         current_gumbel = gumbel_softmax(subkey,probs,tau=tau,hard=True,axis=-1,eps=1e-10) # 输出(0...,1,...,0)
#         # 检测是否选择了接近零的项
#         chosen_near_zero = jnp.sum(current_gumbel * near_zero_mask) # 如果选择到了接近零项， sum ~= 1
#         should_reroll = jax.nn.sigmoid(k * (chosen_near_zero - 0.5)) # 以0.5为分界，大于0.5 ~=1， 小于0.5 ~=0
        
#         # 重新选择逻辑：混合新旧采样
#         new_gumbel = (
#             (1 - should_reroll) * current_gumbel + 
#             should_reroll * gumbel
#         ) # no reroll ~= current, reroll ~= gumbel
#         new_gumbel = new_gumbel/jnp.sum(new_gumbel,axis=-1,keepdims=False)
#         # 更新状态
#         new_reroll = reroll_count + should_reroll
#         key, sub_key = jax.random.split(subkey)
#         return (new_reroll, new_gumbel, key), None
#     initial_state = (jnp.array(0.0), gumbel_output, subkey) # status, inputs
#     (total_rerolls, final_gumbel, key), _ = jax.lax.scan( body_fn, initial_state, jnp.arange(max_rerolls) ) # 
    
#     # 最终确保至少有一个有效选择
#     return final_gumbel, total_rerolls, key

@partial(jax.jit, static_argnames=('tau','max_rerolls','zero_threshold','k'))   
def collapse(subkey, probs, max_rerolls=3, zero_threshold=1e-5, k=1000.0, tau=1e-3):
    near_zero_mask = jax.nn.sigmoid(k * (zero_threshold - probs))
    
    # 初始化为有效采样（避免全零问题）
    initial_gumbel = gumbel_softmax(subkey, probs, tau=tau, hard=False, axis=-1)
    key, subkey = jax.random.split(subkey)
    
    # 检查初始采样是否有效
    chosen_near_zero = jnp.sum(initial_gumbel * near_zero_mask)
    should_reroll = jax.nn.sigmoid(k * (chosen_near_zero - 0.5))
    
    # 收敛阈值：当重选概率<0.01时视为稳定
    CONVERGENCE_THRESHOLD = 0.01
    
    # 使用cond实现提前终止
    def continue_scan(should_reroll):
        return should_reroll > CONVERGENCE_THRESHOLD

    # 完整的循环扫描（只有初始需要重选时才执行）
    final_gumbel, total_rerolls, key = jax.lax.cond(
        continue_scan(should_reroll),
        true_fun=lambda: full_scan_loop(subkey, probs, initial_gumbel, near_zero_mask, 
                                       max_rerolls, zero_threshold, k, tau),
        false_fun=lambda: (initial_gumbel, jnp.array(0.0), subkey)
    )
     
    return final_gumbel, total_rerolls, key

@partial(jax.jit, static_argnames=('max_rerolls','zero_threshold','k','tau'))
def full_scan_loop(subkey, probs, initial_gumbel, near_zero_mask, 
                max_rerolls, zero_threshold, k, tau):
    """执行完整的重选扫描循环"""
    def body_fn(state, i):
        reroll_count, gumbel, subkey = state
        current_gumbel = gumbel_softmax(subkey, probs, tau=tau, hard=False, axis=-1)
        key, subkey = jax.random.split(subkey)
        
        # 检测是否需要重选
        chosen_near_zero = jnp.sum(current_gumbel * near_zero_mask)
        should_reroll = jax.nn.sigmoid(k * (chosen_near_zero - 0.5))
        
        # 混合新旧采样
        new_gumbel = (1 - should_reroll) * current_gumbel + should_reroll * gumbel
        new_gumbel = new_gumbel / (jnp.sum(new_gumbel, axis=-1, keepdims=True) + 1e-8)
        
        # 更新重选计数（只累计实际发生的重选）
        new_reroll = reroll_count + jnp.clip(should_reroll, 0, 1)
        
        # 返回更新后的状态
        return (new_reroll, new_gumbel, key), should_reroll
    
    initial_state = (jnp.array(1.0), initial_gumbel, subkey)
    
    # 执行扫描，同时跟踪收敛情况
    (total_rerolls, final_gumbel, key), reroll_probs = jax.lax.scan(
        body_fn, 
        initial_state, 
        jnp.arange(max_rerolls)
    )
    
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
    num_elements = adj_csr['num_elements']

    # 初始化概率分布
    probs=init_probs

    should_stop = False
    if plot=="2d":
        # visualizer_2D(tileHandler=tileHandler,probs=probs, points=kwargs.get('points'), figureManager=figureManger,epoch=0)
        visualizer:Visualizer=kwargs.pop("visualizer",None)
        visualizer.add_frame(probs=probs)
    pbar = tqdm.tqdm(total=num_elements, desc="collapsing", unit="tiles")
    id_pattern=None
    while should_stop is False:
        # 归一化
        norm = jnp.sum(jnp.abs(probs), axis=-1, keepdims=True)
        probs = probs / jnp.where(norm == 0, 1.0, norm)
        # 选择要坍缩的单元（最小熵单元）
        # print(f"#epoch: {pbar.n}")
        key, subkey = jax.random.split(key)
        collapse_idx,max_entropy = select_collapse(subkey, probs, tau=1e-3)
        # print(f"max entorpy: {max_entropy}")
        pbar.update(1)
        if max_entropy<0.3:
            should_stop=True
            print(f"####entropy reached stop condition with max entropy {max_entropy}####\n")
            break
        # 获取该单元的邻居
        neighbors, neighbors_dirs = get_neighbors(adj_csr, collapse_idx)
        neighbors_dirs_index = tileHandler.get_index_by_direction(neighbors_dirs) #邻居所在的方向
        # print(f"collapse_idx: {collapse_idx}", )
        # print(f"neighbors: {neighbors}")

        # 根据邻居更新坍缩单元格的概率
        neighbors_dirs_opposite_index=[]
        for index in neighbors_dirs_index:
            direction=tileHandler.get_direction_by_index(index)
            opposite_dire=tileHandler.get_opposite_direction_by_direction(direction)
            neighbors_dirs_opposite_index.append(tileHandler.get_index_by_direction(opposite_dire))
        probs = update_by_neighbors(probs, collapse_idx, neighbors, neighbors_dirs_index,neighbors_dirs_opposite_index, tileHandler.compatibility)
        # print(f"updated by neighbors: \n{probs}")
        
        
        # 坍缩选定的单元
        key, subkey = jax.random.split(key)
        # p_collapsed = gumbel_softmax(subkey,probs[collapse_idx],tau=1e-3,hard=True,axis=-1,eps=1e-10)
        p_collapsed, _ , key = collapse(subkey=subkey, probs=probs[collapse_idx], max_rerolls=3, zero_threshold=1e-5, tau=1e-3,k=1000)
        # print(f"p_collapsed:{p_collapsed}")
        probs = probs.at[collapse_idx].set(jnp.clip(p_collapsed,0,1))

        # 更新邻居的概率
        probs = update_neighbors(probs, neighbors, neighbors_dirs_index, p_collapsed, tileHandler.compatibility)
        # print(f"updated neighbors: \n{probs}")

        if plot is not False:
            if plot == '2d':
                # visualizer_2D(tileHandler=tileHandler,probs=probs, points=kwargs.get('points'), figureManager=figureManger,epoch=pbar.n)
                visualizer.add_frame(probs=probs)
            if plot == "3d":
                #TODO 3D visualizer here
                pass
        if pbar.n > pbar.total:
            pbar.set_description_str("fixing high entropy")
        # print(f"probs: \n{probs}",)
        # print("epoch end\n")
        # 然后再计算香农熵选择下一个
    pbar.close()
    return probs, max_entropy, id_pattern

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

    num_elements = adj['num_elements']
    numTypes = tileHandler.typeNum
    init_probs = jnp.ones((num_elements ,numTypes)) / numTypes # (n_elements, n_types)
    from src.utiles.generateMsh import generate_grid_vertices_vectorized
    figureManager=FigureManager(figsize=(10,10))
    visualizer=Visualizer(tileHandler=tileHandler,points=adj['vertices'],figureManager=figureManager)
    grid= generate_grid_vertices_vectorized(width+1,height+1)
    probs,max_entropy, _=waveFunctionCollapse(init_probs,adj,tileHandler,plot='2d',points=adj['vertices'],figureManger=figureManager,visualizer=visualizer)
    visualizer.draw()
    # visualizer_2D(tileHandler=tileHandler,probs=probs,points=adj['vertices'],figureManager=figureManger,epoch='end')
    pattern = jnp.argmax(probs, axis=-1, keepdims=False).reshape(width,height)
    name_pattern = tileHandler.pattern_to_names(pattern)
    print(f"pattern: \n{name_pattern}")
    print(f"max entropy: {max_entropy}")
