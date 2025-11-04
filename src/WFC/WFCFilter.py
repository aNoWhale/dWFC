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
def update_by_neighbors(probs, collapse_idx, A, D, dirs_opposite_index, compatibility, init_probs, alpha=0.3):
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
    compat = jnp.clip(compat + noise, 1e-8, None)  # 确保≥1e-8
    
    # 4. 计算更新因子
    neighbor_probs = probs * neighbor_mask[:, None]
    update_factors = jnp.einsum('cij, cj -> ci', compat, neighbor_probs)
    # cumulative_factor = jnp.prod(update_factors + (1 - neighbor_mask[:, None]), axis=0)
    update_factors = jnp.clip(update_factors, 1e-8, None)  # 确保≥1e-8
    # 重新计算log_factors，增强安全边际
    log_factors = jnp.log(update_factors + (1 - neighbor_mask[:, None]) + 1e-8)
    cumulative_factor = jnp.exp(jnp.sum(log_factors, axis=0))  # 指数还原为乘积 
    
    # 5. 更新当前单元概率（强化初始概率的依赖）
    p_updated = probs[collapse_idx] * cumulative_factor
    # 关键：通过混合初始概率增强梯度传递（替代原无效梯度注入）
    p_updated = (1 - alpha) * p_updated + alpha * init_probs[collapse_idx]  # 提高alpha权重
    # p_updated = soft_normalize(p_updated, temp=0.5)  # 放宽归一化
    p_updated = normalize_minmax(p_updated, axis=-1)
    
    
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
    # p_neigh = soft_normalize(p_neigh, temp=0.1)  # 替换为软归一化
    p_neigh = normalize_minmax(p_neigh, axis=-1)
    
    # 4. 软更新邻居概率
    return probs * (1 - neighbor_mask[:, None]) + p_neigh * neighbor_mask[:, None]



def sequential_collapse_idx(step, num_elements):
    """按固定顺序返回当前坍缩单元的索引（第step步坍缩第step个单元）"""
    return jnp.clip(step, 0, num_elements - 1)  # 确保索引不越界


def soft_normalize(p, temp=1.0):  # 增大temp（如1.0），降低指数敏感度
    p = jnp.clip(p, -50, 50)  # 截断极端值，避免exp溢出（e^50 ≈ 1e22，仍可控）
    p_exp = jnp.exp(p / temp)
    return p_exp / (jnp.sum(p_exp, axis=-1, keepdims=True) + 1e-8)  # +1e-8避免除以零

def normalize_minmax(x, axis=None, epsilon=1e-8):
    """
    Min-Max 归一化: (x - min) / (max - min)
    将数据缩放到[0, 1]范围
    参数:
        x: jax.numpy.ndarray，输入数据
        axis: 归一化的轴（如axis=0为按列归一化，None为全局归一化）
        epsilon: 小值，避免除以0
    返回:
        归一化后的数据
    """
    min_val = jnp.min(x, axis=axis, keepdims=True)  # 计算最小值
    max_val = jnp.max(x, axis=axis, keepdims=True)  # 计算最大值
    return (x - min_val) / (max_val - min_val + epsilon)  # 避免最大值等于最小值导致的除零错误


def soft_mask(index, n_cells, sigma=0.5):
    """生成以index为中心的高斯软掩码，sigma越小越接近硬掩码"""
    x = jnp.arange(n_cells)
    return jax.nn.sigmoid((- (x - index)**2) / (2 * sigma**2))

def sync_waveFunctionCollapse(init_probs, adj_csr, tileHandler, loop):
    """同步更新版本的WFC（仅用于反向传播）"""
    n_cells, n_tiles = init_probs.shape
    probs = init_probs.copy()
    A, D = preprocess_adjacency(adj_csr, tileHandler)
    dirs_opposite_index = tileHandler.opposite_dir_array
    compatibility = tileHandler.compatibility

    for _ in range(loop):  # 用loop控制同步迭代次数
        # 同步更新所有单元（基于当前probs的原始值）
        all_updated = jax.vmap(
            lambda idx: compute_sync_update(probs, idx, A, D, dirs_opposite_index, compatibility, init_probs)
        )(jnp.arange(n_cells))
        # 应用更新
        masks = jnp.eye(n_cells)[:, :, None]
        probs = jnp.sum(masks * all_updated[:, None, :] + (1 - masks) * probs[None, :, :], axis=0)
    return probs

def compute_sync_update(probs, collapse_idx, A, D, dirs_opposite_index, compatibility, init_probs, alpha=0.3):
    """同步更新中单个单元的计算（复用update_by_neighbors逻辑，但基于原始probs）"""
    n_cells, n_tiles = probs.shape
    collapse_mask = soft_mask(collapse_idx, n_cells, sigma=0.5)
    collapse_mask = collapse_mask / jnp.sum(collapse_mask)
    
    neighbor_mask = A[:, collapse_idx]
    neighbor_dirs = D[:, collapse_idx].astype(jnp.int32)
    valid_dirs = (neighbor_dirs * neighbor_mask).astype(jnp.int32)
    
    opposite_dirs = jnp.take(dirs_opposite_index, valid_dirs, mode='clip')
    compat = jnp.take(compatibility, opposite_dirs, axis=0)
    noise = jax.random.normal(jax.random.PRNGKey(42), compat.shape) * 1e-4  # 增强噪声
    compat = jnp.clip(compat + noise, 1e-8, None)  # 确保≥1e-8
    
    neighbor_probs = probs * neighbor_mask[:, None]  # 基于原始probs（非更新后的值）
    update_factors = jnp.einsum('cij, cj -> ci', compat, neighbor_probs)
    # cumulative_factor = jnp.prod(update_factors + (1 - neighbor_mask[:, None]), axis=0)
    update_factors = jnp.clip(update_factors, 1e-8, None)  # 确保≥1e-8
    # 重新计算log_factors，增强安全边际
    log_factors = jnp.log(update_factors + (1 - neighbor_mask[:, None]) + 1e-8)
    cumulative_factor = jnp.exp(jnp.sum(log_factors, axis=0))  # 指数还原为乘积
    
    p_updated = probs[collapse_idx] * cumulative_factor
    p_updated = (1 - alpha) * p_updated + alpha * init_probs[collapse_idx]  # 强化初始依赖
    # p_updated = soft_normalize(p_updated, temp=1.0)  # 放宽归一化
    p_updated = normalize_minmax(p_updated,axis=-1)
    return p_updated



@jax.custom_vjp
def waveFunctionCollapse(init_probs, A, D, dirs_opposite_index, compatibility) -> jnp.ndarray:
    n_cells, n_tiles = init_probs.shape
    probs = init_probs.copy()
    collapse_list = []
    
    # 打印初始probs的均值（应随init_probs变化）
    print("初始probs均值：", jnp.mean(probs))
    
    pbar = tqdm.tqdm(total=n_cells, desc="collapsing", unit="tiles")
    for collapse_idx in range(n_cells):
        collapse_list.append(collapse_idx)
        # 保存更新前的probs
        probs_before = probs.copy()
        # 第一步更新：update_by_neighbors
        probs = update_by_neighbors(
            probs, collapse_idx, A, D, dirs_opposite_index, 
            compatibility, init_probs=init_probs
        )
        # 打印第一步更新后的变化
        # print(f"第{collapse_idx}步 update_by_neighbors 变化量：", 
        #       jnp.linalg.norm(probs - probs_before))
        
        # 第二步更新：update_neighbors
        probs_before2 = probs.copy()
        probs = update_neighbors(probs, collapse_idx, A, D, compatibility)
        # 打印第二步更新后的变化
        # print(f"第{collapse_idx}步 update_neighbors 变化量：", 
        #       jnp.linalg.norm(probs - probs_before2))
        
        pbar.update(1)
    pbar.close()
    
    # 打印最终probs的均值（应随初始probs变化）
    print("最终probs均值：", jnp.mean(probs))
    return probs, 0, collapse_list





def waveFunctionCollapse_fwd(init_probs, A, D, dirs_opposite_index, compatibility):
    probs, max_entropy, collapse_list = waveFunctionCollapse(
        init_probs, A, D, dirs_opposite_index, compatibility
    )
    # 残留值保存n_cells（WFC正向迭代次数），供反向传播使用
    n_cells = init_probs.shape[0]  # 从输入形状获取单元数
    residuals = (init_probs, A, D, dirs_opposite_index, compatibility, n_cells)
    return (probs, max_entropy, collapse_list), residuals


def waveFunctionCollapse_bwd(residuals, grad_output):
    # 解析残留值，获取WFC正向迭代次数n_cells
    init_probs, A, D, dirs_opposite_index, compatibility, n_cells = residuals
    grad_probs, _, _ = grad_output
    
    # 同步更新次数 = WFC正向迭代次数n_cells（确保梯度与正向逻辑匹配）
    def sync_wfc(init_p):
        probs = init_p.copy()
        for _ in range(3):  # 用n_cells控制反向同步次数
            all_updated = jax.vmap(
                lambda idx: compute_sync_update(
                    probs, idx, A, D, dirs_opposite_index, compatibility, init_p
                )
            )(jnp.arange(n_cells))
            masks = jnp.eye(n_cells)[:, :, None]
            probs = jnp.sum(masks * all_updated[:, None, :] + (1 - masks) * probs[None, :, :], axis=0)
        return probs
    
    grad_init = jax.grad(lambda p: jnp.sum(sync_wfc(p) * grad_probs))(init_probs)
    # 梯度结构与原始参数匹配（5个参数）
    grads = (grad_init, None, None, None, None)
    return grads


# 4. 绑定正向/反向函数
waveFunctionCollapse.defvjp(waveFunctionCollapse_fwd, waveFunctionCollapse_bwd)


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
