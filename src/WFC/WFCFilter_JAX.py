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

def preprocess_adjacency(adj_csr, tileHandler):
    """
    修复：从 TileHandler 中正确提取方向字符串列表，避免访问不存在的 'direction' 属性
    """
    # 步骤1：从 TileHandler 提取方向字符串（基于整数→字符串映射）
    # 排除 'isotropy'（-1），仅保留具体方向
    dir_int_to_str = tileHandler.dir_int_to_str
    concrete_dirs = [
        dir_str for dir_int, dir_str in dir_int_to_str.items()
        if dir_int != -1 and dir_str != 'isotropy'  # 过滤 isotropy
    ]
    # 去重并保持顺序
    unique_dirs = list(dict.fromkeys(concrete_dirs))
    # 创建方向字符串→整数映射
    dir_mapping = {dir_str: idx for idx, dir_str in enumerate(unique_dirs)}
    
    # 步骤2：提取CSR数据（用NumPy处理）
    row_ptr = np.array(adj_csr['row_ptr'])
    col_idx = np.array(adj_csr['col_idx'])
    directions = np.array(adj_csr['directions'])  # 原始方向字符串数组
    
    # 步骤3：将方向字符串转为整数（NumPy操作）
    dir_indices = np.array([dir_mapping[dir_str] for dir_str in directions], dtype=np.int32)
    
    # 步骤4：构建邻接矩阵A和方向矩阵D（NumPy稠密矩阵）
    n_cells = len(row_ptr) - 1  # 总单元数 = row_ptr长度 - 1
    A_np = np.zeros((n_cells, n_cells), dtype=np.float32)  # 邻接矩阵
    D_np = np.zeros((n_cells, n_cells), dtype=np.int32)    # 方向矩阵
    
    for j in range(n_cells):
        start = row_ptr[j]
        end = row_ptr[j+1]
        neighbors_j = col_idx[start:end]
        dirs_j = dir_indices[start:end]
        A_np[neighbors_j, j] = 1.0  # 标记邻居关系
        D_np[neighbors_j, j] = dirs_j  # 存储方向整数索引
    
    # 步骤5：转为JAX数组
    A = jnp.array(A_np)
    D = jnp.array(D_np)
    return A, D


# @jax.jit
def get_neighbors(csr, index):
    """获取指定索引的邻居列表"""
    start = csr['row_ptr'][index]
    end = csr['row_ptr'][index + 1]
    neighbors = csr['col_idx'][start:end]
    neighbors_dirs = csr['directions'][start:end]
    return neighbors, neighbors_dirs





@partial(jax.jit, static_argnames=())
def update_by_neighbors(probs, collapse_idx, A, D, dirs_opposite_index, compatibility, init_probs, alpha=0.3):
    n_cells, n_tiles = probs.shape
    # 1. 生成当前单元的软掩码
    collapse_mask = soft_mask(collapse_idx, n_cells)  # 软掩码
    collapse_mask = collapse_mask / jnp.sum(collapse_mask)  # 归一化权重
    
    # 2. 提取邻居和方向
    neighbor_mask = A[:, collapse_idx]  # (n_cells,)
    neighbor_dirs = D[:, collapse_idx].astype(jnp.int32)  # 确保整数类型
    valid_dirs = (neighbor_dirs * neighbor_mask).astype(jnp.int32)  # 有效方向索引
    
    # 3. 计算兼容性（添加可微噪声打破常数壁垒）
    opposite_dirs = jnp.take(dirs_opposite_index, valid_dirs, mode='clip')
    compat = jnp.take(compatibility, opposite_dirs, axis=0)
    key = jax.random.PRNGKey(jnp.mod(collapse_idx, 1000).astype(jnp.int32))  # 避免固定的key
    noise = jax.random.normal(key, compat.shape) * 1e-6  # 减小噪声幅度
    compat = jnp.clip(compat + noise, 1e-8, 1e8)  # 双向截断
    
    # 4. 计算更新因子（增强数值稳定性）
    neighbor_probs = probs * neighbor_mask[:, None]
    update_factors = jnp.einsum('cij, cj -> ci', compat, neighbor_probs)
    # 关键：限制update_factors的范围，避免log输入过小
    update_factors = jnp.clip(update_factors, 1e-10, 1e10)  # 缩小范围，避免极端值
    # 计算log时进一步增加安全边际
    log_factors = jnp.log(update_factors + (1 - neighbor_mask[:, None]) + 1e-10)
    # 限制log的总和，避免exp溢出（e^50 ≈ 1e22，已足够大）
    sum_log = jnp.clip(jnp.sum(log_factors, axis=0), -50, 50)  # 核心：防止exp溢出
    cumulative_factor = jnp.exp(sum_log)  # 此时不会产生inf

    # 5. 更新当前单元概率（限制范围）
    p_updated = probs[collapse_idx] * cumulative_factor
    p_updated = jnp.clip(p_updated, 1e-10, 1e10)  # 避免p_updated为0或inf
    # 混合初始概率（保持梯度）
    p_updated = (1 - alpha) * p_updated + alpha * init_probs[collapse_idx]
    # 归一化时处理极端值
    # p_updated = normalize_minmax(p_updated, axis=-1)
    p_updated = normalize(p_updated)  # 归一化

    
    
    # 6. 软更新
    return probs * (1 - collapse_mask[:, None]) + p_updated * collapse_mask[:, None]




@jax.jit
def update_neighbors(probs, collapse_idx, A, D, compatibility):
    n_cells, n_tiles = probs.shape
    # 1. 生成当前单元的掩码（仅collapse_idx位置为1）
    collapse_mask = jnp.zeros(n_cells, dtype=jnp.float32).at[collapse_idx].set(1.0)  # (n_cells,)
    p_collapsed = probs[collapse_idx]  # 当前单元的概率
    
    # 2. 提取当前单元的邻居（通过邻接矩阵）
    neighbor_mask = A[:, collapse_idx]  # (n_cells,)，邻居位置为1
    neighbor_dirs = D[:, collapse_idx]  # (n_cells,)，邻居的方向索引
    
    # 3. 计算邻居的更新值（限制范围）
    compat = jnp.take(compatibility, neighbor_dirs, axis=0)
    compat = jnp.clip(compat, 1e-10, 1e10)  # 兼容性限制
    p_neigh = jnp.einsum('cij, j -> ci', compat, p_collapsed)
    p_neigh = jnp.clip(p_neigh, 1e-10, 1e10)  # 邻居概率限制
    p_neigh = p_neigh * probs * neighbor_mask[:, None]
    p_neigh = normalize(p_neigh)  # 归一化

    
    # 4. 软更新邻居概率
    return probs * (1 - neighbor_mask[:, None]) + p_neigh * neighbor_mask[:, None]


@jax.jit
def soft_normalize(p, temp=1.0):  # 增大temp（如1.0），降低指数敏感度
    p = jnp.clip(p, -50, 50)  # 截断极端值，避免exp溢出（e^50 ≈ 1e22，仍可控）
    p_exp = jnp.exp(p / temp)
    return p_exp / (jnp.sum(p_exp, axis=-1, keepdims=True) + 1e-8)  # +1e-8避免除以零


@partial(jax.jit, static_argnames=('axis',))
def normalize_minmax(x, axis=None, epsilon=1e-10):
    """增强版Min-Max归一化，处理极端值"""
    # 替换NaN和inf为有限值
    x = jnp.nan_to_num(x, nan=0.0, posinf=1e10, neginf=1e-10)
    min_val = jnp.min(x, axis=axis, keepdims=True)
    max_val = jnp.max(x, axis=axis, keepdims=True)
    # 处理max == min的情况（避免除以0）
    range_val = jnp.where(max_val == min_val, 1.0, max_val - min_val)
    return (x - min_val) / (range_val + epsilon)

@jax.jit
def soft_max(p, temp=0.5):  # 温度从1.0降至0.5
    p = jnp.clip(p, -50, 50)
    p_exp = jnp.exp(p / temp)
    return p_exp / (jnp.sum(p_exp, axis=-1, keepdims=True) + 1e-8)

def normalize(p):
    # normalize_minmax(p, axis=-1)
    return soft_max(p, temp=0.5)


@partial(jax.jit, static_argnames=('n_cells',))
def soft_mask(index, n_cells, sigma=0.2):
    """生成以index为中心的高斯软掩码，sigma越小越接近硬掩码"""
    x = jnp.arange(n_cells)
    return jax.nn.sigmoid((- (x - index)**2) / (2 * sigma**2))



@jax.jit
def waveFunctionCollapse(init_probs, A, D, dirs_opposite_index, compatibility):
    n_cells, n_tiles = init_probs.shape
    # 检查并替换初始概率中的异常值
    init_probs = jnp.nan_to_num(init_probs, nan=1e-10, posinf=1e10, neginf=1e-10)
    init_probs = jnp.clip(init_probs, 1e-10, 1.0)  # 概率应在[0,1]附近
    
    # 检查兼容性矩阵（确保非负）
    compatibility = jnp.clip(compatibility, 1e-10, 1e10)  # 兼容性不能为负

    # 第一步：所有单元格依据init_probs进行update_by_neighbors
    def step1_update(carry, collapse_idx):
        probs = carry
        # 使用原始的update_by_neighbors函数
        updated_probs = update_by_neighbors(
            probs, collapse_idx, A, D, dirs_opposite_index, compatibility, init_probs
        )
        return updated_probs, None
    
    # 初始化第一步的概率
    probs_step1 = init_probs
    # 用scan遍历所有单元格，执行update_by_neighbors
    probs_step1, _ = jax.lax.scan(
        step1_update,
        init=probs_step1,
        xs=jnp.arange(n_cells)
    )
    
    # 第二步：每个单元格依据probs_step1进行update_neighbors
    def step2_update(carry, collapse_idx):
        probs = carry
        # 使用原始的update_neighbors函数
        updated_probs = update_neighbors(probs, collapse_idx, A, D, compatibility)
        return updated_probs, None
    
    # 用scan遍历所有单元格，执行update_neighbors
    final_probs, _ = jax.lax.scan(
        step2_update,
        init=probs_step1,
        xs=jnp.arange(n_cells)
    )
    
    collapse_list = jnp.arange(n_cells)
    return final_probs, 0, collapse_list






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
    probs,max_entropy, collapse_list=waveFunctionCollapse(init_probs,adj,tileHandler,plot='2d',points=adj['vertices'],figureManger=figureManager,visualizer=visualizer)
    wfc = lambda p:waveFunctionCollapse(p,adj,tileHandler,plot='2d',points=adj['vertices'],figureManger=figureManager,visualizer=visualizer)
    def loss_fn(init_probs):
            probs, _, collapse_list = wfc(init_probs)
            cell_max_p = jnp.max(probs, axis=-1) + 1e-8
            modified_e = -cell_max_p * jnp.log2(cell_max_p) + (1 - cell_max_p)
            mean = jnp.mean(modified_e)
            ec = mean / 0.1 - 1
            return ec
    
    visualizer.collapse_list=collapse_list
    visualizer.draw()
    visualizer_2D(tileHandler=tileHandler,probs=probs,points=adj['vertices'],figureManager=figureManager,epoch='end') # pyright: ignore[reportArgumentType]
    pattern = jnp.argmax(probs, axis=-1, keepdims=False).reshape(width,height)
    name_pattern = tileHandler.pattern_to_names(pattern)
    print(f"pattern: \n{name_pattern}")
    print(f"max entropy: {max_entropy}")
