import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# os.environ["JAX_PLATFORMS"] = "cpu"
import jax
# jax.config.update('jax_platform_name', 'cpu')  # 强制使用CPU
# jax.config.update('jax_disable_jit', True)     # 调试时开启

import jax.numpy as jnp
from functools import partial

import tqdm.rich as tqdm

import numpy as np

# 假设以下模块是您项目中已实现的核心模块
from src.WFC.TileHandler_JAX import TileHandler
from src.WFC.builder import visualizer_2D, Visualizer
from src.WFC.FigureManager import FigureManager


def get_neighbors(csr, index, tileHandler):
    """修复：提前将方向字符串转为整数索引，避免返回字符串"""
    start = csr['row_ptr'][index]
    end = csr['row_ptr'][index + 1]
    # 邻居索引：转为JAX数组（数值类型）
    neighbors = csr['col_idx'][start:end]
    neighbors = jnp.atleast_1d(jnp.array(neighbors, dtype=jnp.int32))
    # 方向字符串→整数索引（核心修复：避免字符串传入JAX）
    dirs_str = csr['directions'][start:end]
    dirs_index = tileHandler.get_index_by_direction(dirs_str)
    dirs_index = jnp.atleast_1d(jnp.array(dirs_index, dtype=jnp.int32))
    return neighbors, dirs_index


def collapsed_mask(probabilities, threshold=0.99):
    """保留原坍缩掩码逻辑"""
    max_probs = jnp.max(probabilities, axis=-1, keepdims=True)
    return jax.nn.sigmoid(-1000 * (max_probs - threshold))


def select_collapse_by_map(key, collapse_map):
    """保留collapse_map熵选择核心：选值最大的节点（熵最小）"""
    max_map = jnp.max(collapse_map)
    if max_map < 1e-6:  # 所有节点已坍缩
        return -1
    choices = jnp.where(collapse_map == max_map)[0]
    idx = jax.random.choice(key, choices)
    return int(idx)  # 转为Python int，避免维度问题


def detect_contradiction(probs, collapse_idx, neighbors=None):
    """完善矛盾检测：当前节点全0 / 全局无有效节点 / 邻居节点全0"""
    # 1. 检查当前节点概率是否全0
    node_prob_sum = jnp.sum(jnp.abs(probs[collapse_idx]))
    if node_prob_sum < 1e-6:
        return True
    # 2. 检查邻居节点是否全0（关键补充）
    if neighbors is not None and len(neighbors) > 0:
        neighbor_sums = jnp.sum(jnp.abs(probs[neighbors]), axis=-1)
        if jnp.all(neighbor_sums < 1e-6):
            return True
    # 3. 检查所有未坍缩节点是否都无有效概率（全局矛盾）
    global_prob_sum = jnp.sum(jnp.abs(probs), axis=-1)
    valid_nodes = jnp.sum(global_prob_sum > 1e-6)
    if valid_nodes == 0:
        return True
    return False


@partial(jax.jit, static_argnames=())
def update_by_neighbors(probs, collapse_id, neighbors, dirs_opposite_index, compatibility):
    """完全保留原邻居更新逻辑，仅处理数值类型"""
    def update_single(neighbor_prob, opposite_dire_index):
        p_c = jnp.einsum("...ij,...j->...i", compatibility[opposite_dire_index], neighbor_prob)
        norm = jnp.sum(jnp.abs(p_c), axis=-1, keepdims=True)
        return p_c / jnp.where(norm == 0, 1.0, norm)
    
    # 仅处理数值数组，无字符串
    neighbor_probs = jnp.atleast_1d(probs[neighbors])
    opposite_indices = jnp.atleast_1d(dirs_opposite_index)
    
    if neighbor_probs.size == 0 or opposite_indices.size == 0:
        return probs
    
    batch_update = jax.vmap(update_single)
    update_factors = batch_update(neighbor_probs, opposite_indices)
    cumulative_factor = jnp.prod(update_factors, axis=0)
    
    p = probs[collapse_id] * cumulative_factor
    norm = jnp.sum(jnp.abs(p), axis=-1, keepdims=True)
    p = p / jnp.where(norm == 0, 1.0, norm)
    return probs.at[collapse_id].set(p)


@jax.jit
def update_neighbors(probs, neighbors, dirs_index, p_collapsed, compatibility):
    """完全保留原坍缩后更新邻居逻辑，仅处理数值类型"""
    def vectorized_update(neighbor_prob, dir_idx):
        p_neigh = jnp.einsum("...ij,...j->...i", compatibility, p_collapsed)
        p_neigh = jnp.einsum("...i,...i->...i", p_neigh, neighbor_prob)
        norm = jnp.sum(jnp.abs(p_neigh), axis=-1, keepdims=True)
        p_neigh = p_neigh / jnp.where(norm == 0, 1.0, norm)
        return jnp.clip(p_neigh[dir_idx], 0, 1)
    
    # 仅处理数值数组，无字符串
    neighbor_probs = jnp.atleast_1d(probs[neighbors])
    dirs_index = jnp.atleast_1d(dirs_index)
    
    if neighbor_probs.size == 0 or dirs_index.size == 0:
        return probs
    
    updated_probs = jax.vmap(vectorized_update)(neighbor_probs, dirs_index)
    return probs.at[neighbors].set(updated_probs)


def collapse_max_prob(key, prob, backtrack=False):
    """经典WFC坍缩：优先选概率最高的瓦片（回溯时生成全新随机种子）"""
    if backtrack:
        # 核心修复：回溯时生成全新随机种子，避免重复选择
        key = jax.random.PRNGKey(jax.random.randint(key, (), 0, 1000000))
        subkey = key
    else:
        subkey = key
    
    zeros = jnp.zeros_like(prob)
    max_val = jnp.max(prob)
    max_indices = jnp.where(prob == max_val)[0]  
    # 若只有一个选项，直接选；多个则随机（回溯时换随机）
    if len(max_indices) == 1:
        random_idx = max_indices[0]
    else:
        random_idx = jax.random.choice(subkey, max_indices)
    return zeros.at[random_idx].set(1), key


def waveFunctionCollapse(init_probs, adj_csr, tileHandler: TileHandler, plot: bool | str = False, *args, **kwargs) -> jnp.ndarray:
    """经典WFC迭代流程 + 修复后的collapse_map + 可生效的回溯机制"""
    key = jax.random.PRNGKey(0)
    num_elements = init_probs.shape[0]
    max_neighbors = kwargs.pop("max_neighbors", 4)
    max_backtracks = kwargs.pop("max_backtracks", 100)  # 最大回溯次数
    max_stack_depth = kwargs.pop("max_stack_depth", 50)  # 状态栈最大深度（避免内存溢出）

    # 初始化：保留原collapse_map（熵选择核心）
    probs = jnp.array(init_probs, dtype=jnp.float32)
    collapse_map = jnp.ones(probs.shape[0])  # 1=初始熵权重，值越大优先级越高
    should_stop = False
    backtrack_count = 0  # 回溯计数器
    state_stack = []     # 回溯状态栈：保存(probs, collapse_map, key, collapse_idx)

    if plot == "2d":
        visualizer: Visualizer = kwargs.pop("visualizer", None)
        visualizer.add_frame(probs=probs)
    
    pbar = tqdm.tqdm(total=num_elements, desc="classic WFC (collapse_map + backtrack)", unit="tiles")
    collapse_list = [-1]

    # 提前获取反向方向数组（避免重复转换）
    opposite_dir_array = jnp.array(tileHandler.opposite_dir_array, dtype=jnp.int32)
    compatibility = jnp.array(tileHandler.compatibility, dtype=jnp.float32)

    # 经典WFC核心迭代循环（修复后的回溯逻辑）
    while not should_stop:
        # 检查最大回溯次数，避免死循环
        if backtrack_count >= max_backtracks:
            print(f"#### 达到最大回溯次数({max_backtracks})，停止迭代 ####\n")
            should_stop = True
            break

        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # 1. 概率归一化（保留原逻辑）
        norm = jnp.sum(jnp.abs(probs), axis=-1, keepdims=True)
        probs = probs / jnp.where(norm == 0, 1.0, norm)
        
        # 2. 经典WFC + collapse_map熵选择：选熵最小（collapse_map最大）的节点
        collapse_idx = select_collapse_by_map(subkey1, collapse_map)
        
        # 3. 停止条件：所有节点已坍缩（collapse_map无有效值）
        if collapse_idx == -1:
            print(f"#### collapse_map无有效节点，停止迭代 ####\n")
            should_stop = True
            break

        # 4. 保留原solid_mask逻辑（概率阈值过滤）
        solid_mask = jnp.max(probs, axis=-1) > 0

        # 5. 获取邻居和方向索引（核心修复：直接获取整数索引，无字符串）
        neighbors, dirs_index = get_neighbors(adj_csr, collapse_idx, tileHandler)
        # 计算反向方向索引（邻居→当前节点的方向）
        dirs_opposite_index = opposite_dir_array[dirs_index]
        
        # 6. 核心修复：仅当要修改状态时，才保存纯净状态到回溯栈
        state_saved = False
        if solid_mask[collapse_idx]:
            # 保存当前纯净状态（未更新、未坍缩、未修改collapse_map）
            current_state = (
                jnp.copy(probs),          # 概率状态（未更新）
                jnp.copy(collapse_map),   # 熵权重状态（未修改）
                key,                      # 随机种子
                collapse_idx              # 待坍缩节点
            )
            # 限制栈深度，避免内存溢出
            if len(state_stack) >= max_stack_depth:
                state_stack.pop(0)  # 移除最早的状态
            state_stack.append(current_state)
            state_saved = True
            
            # 7. 邻居先更新当前节点概率
            probs = update_by_neighbors(
                probs, collapse_idx, neighbors, dirs_opposite_index, compatibility
            )
            
            # 8. 检测更新后是否出现矛盾（传入neighbors，完善检测）
            if detect_contradiction(probs, collapse_idx, neighbors):
                print(f"#### 节点{collapse_idx}出现矛盾，触发回溯 ({backtrack_count+1}/{max_backtracks}) ####")
                # 恢复上一状态（增加栈非空判断）
                if state_stack and state_saved:
                    probs, collapse_map, key, collapse_idx = state_stack.pop()
                    # 核心修复：降低该节点优先级，避免重复选择
                    collapse_map = collapse_map.at[collapse_idx].set(0.1)
                    state_saved = False
                backtrack_count += 1
                # 跳过本次循环，重新选择节点
                continue
            
            # 9. 经典WFC坍缩：优先选概率最高的瓦片（回溯时换随机）
            p_collapsed, key = collapse_max_prob(subkey2, probs[collapse_idx], backtrack=(backtrack_count>0))
            probs = probs.at[collapse_idx].set(p_collapsed)
       
            # 10. 坍缩后更新邻居概率
            probs = update_neighbors(probs, neighbors, dirs_index, p_collapsed, compatibility)
            
            # 11. 再次检测邻居更新后是否矛盾（传入neighbors）
            if detect_contradiction(probs, collapse_idx, neighbors):
                print(f"#### 邻居更新后节点{collapse_idx}出现矛盾，触发回溯 ({backtrack_count+1}/{max_backtracks}) ####")
                # 恢复上一状态（增加栈非空判断）
                if state_stack and state_saved:
                    probs, collapse_map, key, collapse_idx = state_stack.pop()
                    # 核心修复：降低该节点优先级，避免重复选择
                    collapse_map = collapse_map.at[collapse_idx].set(0.1)
                    state_saved = False
                backtrack_count += 1
                continue
        else:
            # 保留原逻辑：非solid节点概率置0
            probs = probs.at[collapse_idx].multiply(0)

        # 12. collapse_map核心更新逻辑（熵权重调整）
        collapse_list.append(collapse_idx)
        collapse_map = collapse_map.at[collapse_idx].set(0)  # 已坍缩节点熵权重置0
        collapse_map = collapse_map.at[neighbors].multiply(10)  # 邻居熵权重提升（优先级提高）

        # 可视化（保留原逻辑）
        if plot == '2d' and visualizer is not None:
            visualizer.add_frame(probs=probs)

        # 进度更新（保留原逻辑）
        pbar.update(1)
        if pbar.n >= pbar.total:
            pbar.set_description_str("fixing high entropy")
    
    pbar.close()
    print(f"#### 迭代结束，总回溯次数：{backtrack_count} ####")
    return probs, 0, collapse_list


if __name__ == "__main__":
    from src.utiles.adjacency import build_grid_adjacency
    height = 5
    width = 5
    adj = build_grid_adjacency(height=height, width=width, connectivity=4)

    # 初始化瓦片处理器（保留原逻辑）
    tileHandler = TileHandler(typeList=['a','b','c','d','e'], direction=(('up',"down"),("left","right"),))
    from src.dynamicGenerator.TileImplement.Dimension2.LinePath import LinePath
    tileHandler.register(typeName='a', class_type=LinePath(['da-bc','cen-cd'], color='blue'))
    tileHandler.register(typeName='b', class_type=LinePath(['ab-cd','cen-da'], color='green'))
    tileHandler.register(typeName='c', class_type=LinePath(['da-bc','cen-ab'], color='yellow'))
    tileHandler.register(typeName='d', class_type=LinePath(['ab-cd','cen-bc'], color='red'))
    tileHandler.register(typeName='e', class_type=LinePath(['da-bc','ab-cd'], color='magenta'))

    # 设置瓦片连接性（保留原逻辑）
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

    # 初始化概率分布（保留原逻辑）
    num_elements = adj['num_elements']
    numTypes = tileHandler.typeNum
    init_probs = jnp.ones((num_elements, numTypes)) / numTypes

    # 可视化配置（保留原逻辑）
    figureManager = FigureManager(figsize=(10,10))
    visualizer = Visualizer(tileHandler=tileHandler, points=adj['vertices'], figureManager=figureManager)

    # 运行修复后的经典WFC（collapse_map + 可生效的回溯）
    probs, max_entropy, collapse_list = waveFunctionCollapse(
        init_probs, adj, tileHandler, plot='2d', 
        points=adj['vertices'], visualizer=visualizer, 
        max_neighbors=4, max_backtracks=200, max_stack_depth=50
    )

    # 结果输出（保留原逻辑）
    visualizer.collapse_list = collapse_list
    visualizer.draw()
    visualizer_2D(tileHandler=tileHandler, probs=probs, points=adj['vertices'], figureManager=figureManager, epoch='end')
    pattern = jnp.argmax(probs, axis=-1, keepdims=False).reshape(width, height)
    name_pattern = tileHandler.pattern_to_names(pattern)
    print(f"pattern: \n{name_pattern}")
    print(f"max entropy: {max_entropy}")