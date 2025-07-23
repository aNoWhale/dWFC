
import gmsh
import jax
# jax.config.update('jax_disable_jit', True)

import jax.numpy as jnp
from functools import partial

from src.WFC.gumbelSoftmax import gumbel_softmax
from src.WFC.shannonEntropy import shannon_entropy
from src.WFC.TileHandler import TileHandler

import tqdm

import scipy.sparse
import numpy as np


def build_adjacency(model, element_dim=3):
    """
    builds a JAX compatible adjacency matrix from gmsh model
    element_dim: 2face 或3volume
    """
    # gmsh.initialize()
    # gmsh.open(mesh_file)
    # 1. 获取所有单元并建立索引映射
    all_elements = model.getEntities(element_dim)
    num_elements = len(all_elements)

    # 创建标签到连续索引的映射
    tag_to_idx = {tag: idx for idx, (_, tag) in enumerate(all_elements)}

    # 2. 准备CSR数据结构
    row_ptr = np.zeros(num_elements + 1, dtype=np.int32)
    col_idx = []

    # 3. 遍历所有单元收集邻接关系
    for idx, (dim, tag) in enumerate(all_elements):
        # 获取边界实体
        boundaries = model.getBoundary(
            [(dim, tag)],
            oriented=False,
            combined=False,
            recursive=True
        )

        # 收集邻居标签
        neighbors = set()
        for b_dim, b_tag in boundaries:
            adj_tags = model.getEntitiesForPhysicalGroup(b_dim, b_tag) or []
            neighbors.update(adj_tags)

        # 移除自身并转换为索引
        neighbors.discard(tag)
        neighbor_indices = [tag_to_idx[t] for t in neighbors if t in tag_to_idx]

        # 更新CSR数据结构
        col_idx.extend(neighbor_indices)
        row_ptr[idx + 1] = row_ptr[idx] + len(neighbor_indices)

    # 4. 转换为JAX数组
    return {
        'row_ptr': jnp.array(row_ptr, dtype=jnp.int32),
        'col_idx': jnp.array(col_idx, dtype=jnp.int32),
        'data': jnp.ones(len(col_idx)),  # 所有边权重为1
        'num_elements': num_elements
    }

def build_adjacency_gmsh(mesh_file, element_dim=3):
    gmsh.initialize()
    gmsh.open(mesh_file)
    adj_csr = build_adjacency(gmsh.model)
    gmsh.finalize()
    return adj_csr

# @jax.jit
def get_neighbors(csr, index):
    """获取指定索引的邻居列表"""
    start = csr['row_ptr'][index]
    end = csr['row_ptr'][index + 1]
    return csr['col_idx'][start:end]

@partial(jax.jit, static_argnames=('threshold',))
def collapsed_mask(probabilities, threshold=0.99):
    """
    创建连续掩码标记已坍缩单元
    1 = 未坍缩, 0 = 已坍缩（接近 one-hot）
    """
    max_probs = jnp.max(probabilities, axis=-1, keepdims=True)
    return jax.nn.sigmoid(-1000 * (max_probs - threshold))

@partial(jax.jit, static_argnames=('tau','stopThreshold'))
def select_collapse(key, probs, tau=0.1, stopThreshold=0.1):
    """
    calculate shannon entropy, give out mask by probability, modify shannon entropy by mask, select uncollapsed.
    """
    # 1. 计算各单元熵值
    entropy = shannon_entropy(probs)

    # 2. 标记已坍缩单元
    mask = collapsed_mask(probs)

    # 3. 调整熵值：已坍缩单元赋予高熵值
    # 未坍缩: entropy_adj = entropy
    # 已坍缩: entropy_adj = max_entropy + 1
    max_entropy = jnp.max(entropy)
    should_stop = max_entropy < stopThreshold
    entropy_adj = entropy * mask + (1 - mask) * (max_entropy + 1.0)

    # 4. 转换为选择概率（最小熵对应最高概率）
    selection_logits = -entropy_adj  # 最小熵->最大logits

    # 5. 使用Gumbel-Softmax采样位置
    flat_logits = selection_logits.reshape(-1)

    collapse_probs = gumbel_softmax(key, flat_logits, tau=tau, hard=True, axis=-1, eps=1e-10)
    collapse_idx = np.argmax(collapse_probs)

    return collapse_idx, should_stop


@partial(jax.jit, static_argnames=())
def update_neighbors(probs, neighbors, p_collapsed, compatibility):
    """向量化更新邻居概率"""
    def update_single(neighbor_prob):
        # 简化einsum：'ij,j,il->il' -> 'j,j->' 但保持维度
        support = jnp.dot(compatibility, p_collapsed)
        p_neigh = neighbor_prob * support
        norm = jnp.sum(p_neigh, axis=-1, keepdims=True)
        return p_neigh / jnp.where(norm == 0, 1.0, norm)

    for neighbor in neighbors:
        prob=update_single(probs[neighbor])
        probs.at[neighbor].set(prob)
    # return jax.vmap(update_single)(probs[neighbors])
    return probs


def waveFunctionCollapse(adj_csr, tileHandler: TileHandler)->jnp.ndarray:

    numTypes = tileHandler.typeNum
    key = jax.random.PRNGKey(0)
    num_elements = adj_csr['num_elements']

    # 初始化概率分布
    init_probs = jnp.ones((num_elements, numTypes)) / numTypes # (n_elements,n_types)
    probs=init_probs

    should_stop = False

    pbar = tqdm.tqdm(total=num_elements, desc="坍缩中", unit="项")
    while should_stop is False:
        # 选择要坍缩的单元（最小熵单元）
        # entropies = shannon_entropy(probs)
        key, subkey = jax.random.split(key)
        # collapse_idx = gumbel_softmax(subkey,entropies,tau=1.0,hard=True,axis=-1,eps=1e-10)
        collapse_idx,should_stop = select_collapse(subkey, probs, tau=0.1)
        should_stop=should_stop.item() if type(should_stop) is jnp.ndarray else should_stop
        # 坍缩选定的单元
        key, subkey = jax.random.split(key)
        p_collapsed = gumbel_softmax(subkey,probs[collapse_idx],tau=1.0,hard=True,axis=-1,eps=1e-10)
        probs = probs.at[collapse_idx].set(p_collapsed)

        # 获取该单元的邻居
        neighbors = get_neighbors(adj_csr, collapse_idx)
        # print(f"坍缩单元 {collapse_idx} 的邻居: {neighbors}")
        #更新邻居的概率场
        probs=update_neighbors(probs, neighbors, p_collapsed, tileHandler.compatibility)
        # 更新进度
        pbar.update(1)
        # 然后再计算香农熵选择下一个
    pbar.close()
    return probs

if __name__ == "__main__":
    from src.utiles.adjacency import build_grid_adjacency
    adj=build_grid_adjacency(height=100, width=100, connectivity=4)

    tileHandler = TileHandler(typeList=['a','b','c',])
    tileHandler.selfConnectable(typeName=['a','c'],value=1)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName='b',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='b',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',value=1,dual=True)
    print(tileHandler)

    probs=waveFunctionCollapse(adj,tileHandler)
    pattern = jnp.argmax(probs, axis=-1, keepdims=True)
    print(probs)