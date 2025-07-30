import os
import sys
import gmsh
import jax
jax.config.update('jax_disable_jit', True)

import jax.numpy as jnp
from functools import partial
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/WFC 目录下）
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import tqdm as tqdm

import scipy.sparse
import numpy as np


from collections import defaultdict
from scipy.sparse import csr_matrix

from src.WFC.gumbelSoftmax import gumbel_softmax
from src.WFC.shannonEntropy import shannon_entropy
from src.WFC.TileHandler import TileHandler
from src.WFC.builder import visualizer_2D
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
    print(f"max entropy: {max_entropy}", )
    entropy_adj = entropy * mask + (1 - mask) * (max_entropy + 1.0)

    # 4. 转换为选择概率（最小熵对应最高概率）
    selection_logits = -entropy_adj  # 最小熵->最大logits
    print('modified logits:\n',selection_logits)
    # 5. 使用Gumbel-Softmax采样位置
    flat_logits = selection_logits.reshape(-1)

    collapse_probs = gumbel_softmax(key, flat_logits, tau=tau, hard=True, axis=-1, eps=1e-10)
    collapse_idx = jnp.argmax(collapse_probs)

    return collapse_idx, max_entropy


@partial(jax.jit, static_argnames=())
def update_neighbors(probs, neighbors, dirs_index ,p_collapsed, compatibility):
    """向量化更新邻居概率"""
    def update_single(neighbor_prob):
        p_neigh = jnp.einsum("...ij,...j->...i", compatibility, p_collapsed) # (d,i,j) (j,)-> (d,i,j) (1,1,j)->(d,i,1)->(d,i)
        p_neigh = jnp.einsum("...i,...i->...i",p_neigh,neighbor_prob) #(d,i) (i,) ->(d,i) (1,i) -> (d,i) 
        # print(f"compatibiliy @ p_collapsed * neighbor_prob:\n{jnp.einsum('...ij,...j->...i',compatibility, p_collapsed)} * {neighbor_prob}")
        # print(f'p_neigh: {p_neigh}')
        norm = jnp.sum(jnp.abs(p_neigh), axis=-1, keepdims=True)
        return p_neigh / jnp.where(norm == 0, 1.0, norm)
    for neighbor,dirs in zip(neighbors,dirs_index):
        prob=update_single(probs[neighbor])
        # print(f"update neighbor {neighbor} with {prob}")
        probs = probs.at[neighbor].set(prob[dirs])
    return probs


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
    visualizer_2D(tileHandler=tileHandler,probs=probs, points=kwargs.get('points'), figureManager=figureManger,epoch=0)
    pbar = tqdm.tqdm(total=num_elements, desc="collpasing", unit="tiles")
    while should_stop is False:
        pbar.update(1)
        # 选择要坍缩的单元（最小熵单元）
        key, subkey = jax.random.split(key)
        collapse_idx,max_entropy = select_collapse(subkey, probs, tau=0.01)
        # print(f"max entorpy: {max_entropy}")
        if max_entropy<0.2:
            should_stop=True
            print("####entropy reached stop condition####\n")
            break
        # 坍缩选定的单元
        key, subkey = jax.random.split(key)
        p_collapsed = gumbel_softmax(subkey,probs[collapse_idx],tau=1.0,hard=True,axis=-1,eps=1e-10)
        probs = probs.at[collapse_idx].set(p_collapsed)

        # 获取该单元的邻居
        neighbors, neighbors_dirs = get_neighbors(adj_csr, collapse_idx)

        neighbors_dirs_index = tileHandler.get_index_by_direction(neighbors_dirs) #邻居所在的方向
        print(f"collapse_idx: {collapse_idx}", )
        print(f"neighbors: {neighbors}")
        # 更新邻居的概率场
        probs = update_neighbors(probs, neighbors, neighbors_dirs_index, p_collapsed, tileHandler.compatibility)
        if plot is not False:
            if plot == '2d':
                visualizer_2D(tileHandler=tileHandler,probs=probs, points=kwargs.get('points'), figureManager=figureManger,epoch=pbar.n)
            if plot == "3d":
                #TODO 3D visualizer here
                pass
        
        if pbar.n > pbar.total:
            pbar.set_description_str("trying fix conflicts")
        print(f"probs: \n{probs}",)
        print("epoch end\n")
        # 然后再计算香农熵选择下一个
    pbar.close()
    return probs

if __name__ == "__main__":
    from src.utiles.adjacency import build_grid_adjacency
    height = 3
    width = 3
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
    tileHandler.register(typeName='a',class_type=LinePath(['da-bc','cen-cd']))
    tileHandler.register(typeName='b',class_type=LinePath(['ab-cd','cen-da']))
    tileHandler.register(typeName='c',class_type=LinePath(['da-bc','cen-ab']))
    tileHandler.register(typeName='d',class_type=LinePath(['ab-cd','cen-bc']))
    tileHandler.register(typeName='e',class_type=LinePath(['da-bc','ab-cd']))
    tileHandler.selfConnectable(typeName="a",direction='left',value=1)
    tileHandler.selfConnectable(typeName='b',direction="up",value=1)
    tileHandler.selfConnectable(typeName="c",direction="left",value=1)
    tileHandler.selfConnectable(typeName="d",direction="up",value=1)
    tileHandler.selfConnectable(typeName="e",direction='isotropy',value=1)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','d','b'],direction='down',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName='c',direction='up',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='b',toTypeName=['e','d','a','c'],direction='left',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='b',toTypeName='d',direction='right',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='e',toTypeName='a',direction='up',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='e',toTypeName='b',direction='right',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='e',toTypeName='c',direction='down',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='e',toTypeName='d',direction='left',value=1,dual=True)


    print(f"tileHandler:\n {tileHandler}")

    num_elements = adj['num_elements']
    numTypes = tileHandler.typeNum
    init_probs = jnp.ones((num_elements ,numTypes)) / numTypes # (n_elements, n_types)
    from src.utiles.generateMsh import generate_grid_vertices_vectorized
    figureManger=FigureManager()
    grid= generate_grid_vertices_vectorized(width+1,height+1)
    probs=waveFunctionCollapse(init_probs,adj,tileHandler,plot=True,points=adj['vertices'],figureManger=figureManger)
    pattern = jnp.argmax(probs, axis=-1, keepdims=False).reshape(width,height)
    name_pattern = tileHandler.pattern_to_names(pattern)
    print(f"pattern: \n{name_pattern}")
