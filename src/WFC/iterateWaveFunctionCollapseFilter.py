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





@partial(jax.jit, static_argnames=())
def update_by_neighbors(probs, collapse_idx, A, D, dirs_opposite_index, compatibility, init_probs, alpha=0.5):
    n_cells, n_tiles = probs.shape
    # 1. 生成当前单元的软掩码
    collapse_mask = soft_mask(collapse_idx, n_cells, sigma=0.5)  # 软掩码
    collapse_mask = collapse_mask / jnp.sum(collapse_mask)  # 归一化权重
    
    # 2. 提取邻居和方向
    neighbor_mask = A[:, collapse_idx]  # (n_cells,)
    neighbor_dirs = D[:, collapse_idx].astype(jnp.int32)  # 确保整数类型
    valid_dirs = (neighbor_dirs * neighbor_mask).astype(jnp.int32)  # 有效方向索引
    
    # 3. 计算兼容性（添加可微噪声打破常数壁垒）
    opposite_dirs = jnp.take(dirs_opposite_index, valid_dirs, mode='clip')
    compat = jnp.take(compatibility, opposite_dirs, axis=0)
    noise = jax.random.normal(jax.random.PRNGKey(42), compat.shape) * 1e-5  # 可微噪声
    compat = compat + noise  # 避免兼容性矩阵为严格常数
    
    # 4. 计算更新因子
    neighbor_probs = probs * neighbor_mask[:, None]
    update_factors = jnp.einsum('cij, cj -> ci', compat, neighbor_probs)
    cumulative_factor = jnp.prod(update_factors + (1 - neighbor_mask[:, None]), axis=0)
    
    # 5. 更新当前单元概率（强化初始概率的依赖）
    p_updated = probs[collapse_idx] * cumulative_factor
    # 关键：通过混合初始概率增强梯度传递（替代原无效梯度注入）
    p_updated = (1 - alpha) * p_updated + alpha * init_probs[collapse_idx]  # 提高alpha权重
    p_updated = soft_normalize(p_updated, temp=0.5)  # 放宽归一化
    
    # 6. 软更新
    return probs * (1 - collapse_mask[:, None]) + p_updated * collapse_mask[:, None]


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


@jax.jit
def update_neighbors(probs, collapse_idx, A, D, compatibility):
    n_cells, n_tiles = probs.shape
    # 1. 生成当前单元的掩码（仅collapse_idx位置为1）
    collapse_mask = jnp.zeros(n_cells, dtype=jnp.float32).at[collapse_idx].set(1.0)  # (n_cells,)
    p_collapsed = probs[collapse_idx]  # 当前单元的概率
    
    # 2. 提取当前单元的邻居（通过邻接矩阵）
    neighbor_mask = A[:, collapse_idx]  # (n_cells,)，邻居位置为1
    neighbor_dirs = D[:, collapse_idx]  # (n_cells,)，邻居的方向索引
    
    # 3. 计算邻居的更新值（矩阵乘法实现）
    compat = jnp.take(compatibility, neighbor_dirs, axis=0)  # (n_cells, n_tiles, n_tiles)
    p_neigh = jnp.einsum('cij, j -> ci', compat, p_collapsed)  # (n_cells, n_tiles)
    p_neigh = p_neigh * probs * neighbor_mask[:, None]  # 仅更新邻居
    p_neigh = soft_normalize(p_neigh, temp=0.1)  # 替换为软归一化
    
    # 4. 软更新邻居概率
    return probs * (1 - neighbor_mask[:, None]) + p_neigh * neighbor_mask[:, None]



def sequential_collapse_idx(step, num_elements):
    """按固定顺序返回当前坍缩单元的索引（第step步坍缩第step个单元）"""
    return jnp.clip(step, 0, num_elements - 1)  # 确保索引不越界


def soft_normalize(p, temp=0.1):
    """带温度的平滑归一化，temp越小越接近硬归一化，越大梯度越平滑"""
    p_exp = jnp.exp(p / temp)
    return p_exp / (jnp.sum(p_exp, axis=-1, keepdims=True) + 1e-8)

def soft_mask(index, n_cells, sigma=0.5):
    """生成以index为中心的高斯软掩码，sigma越小越接近硬掩码"""
    x = jnp.arange(n_cells)
    return jax.nn.sigmoid((- (x - index)**2) / (2 * sigma**2))



def waveFunctionCollapse(init_probs, adj_csr, tileHandler: TileHandler, loop, plot: bool | str = False, *args, **kwargs) -> jnp.ndarray:
    n_cells, n_tiles = init_probs.shape
    probs = init_probs.copy()
    collapse_list = []
    
    # 预构建邻接矩阵和方向矩阵（仅一次）
    A, D = preprocess_adjacency(adj_csr, tileHandler)
    dirs_opposite_index = tileHandler.opposite_dir_array
    
    pbar = tqdm.tqdm(total=n_cells, desc="collapsing", unit="tiles")
    
    for collapse_idx in range(n_cells):
        collapse_list.append(collapse_idx)
        
        probs = update_by_neighbors(
                    probs, collapse_idx, A, D, dirs_opposite_index, 
                    tileHandler.compatibility, init_probs=init_probs # 传入初始概率
                )
        
        # 2. 更新邻居概率（用邻接矩阵）
        probs = update_neighbors(
            probs, collapse_idx, A, D, 
            tileHandler.compatibility
        )
        
        pbar.update(1)
    
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
    visualizer_2D(tileHandler=tileHandler,probs=probs,points=adj['vertices'],figureManager=figureManager,epoch='end')
    pattern = jnp.argmax(probs, axis=-1, keepdims=False).reshape(width,height)
    name_pattern = tileHandler.pattern_to_names(pattern)
    print(f"pattern: \n{name_pattern}")
    print(f"max entropy: {max_entropy}")
