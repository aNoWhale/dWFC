import warnings
from typing import List

import gmsh
import jax
import jax.numpy as jnp
from functools import partial

from src.WFC.gumbelSoftmax import gumbel_softmax
from src.WFC.shannonEntropy import shannon_entropy
from src.WFC.TileHandler import TileHandler


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


def get_neighbors(csr, index):
    """获取指定索引的邻居列表"""
    start = csr['row_ptr'][index]
    end = csr['row_ptr'][index + 1]
    return csr['col_idx'][start:end]


def waveFunctionCollapse(mesh_file, tileHandler: TileHandler):
    gmsh.initialize()
    gmsh.open(mesh_file)
    adj_csr = build_adjacency(gmsh.model)
    gmsh.finalize()

    numTypes = tileHandler.typeNum
    key = jax.random.PRNGKey(0)
    num_elements = adj_csr['num_elements']

    # 初始化概率分布
    init_probs = jnp.ones((num_elements, numTypes)) / numTypes # (n_elements,n_types)

    probs=init_probs


    # 选择要坍缩的单元（最小熵单元）
    entropies = shannon_entropy(probs)
    key, subkey = jax.random.split(key)
    collapse_idx = gumbel_softmax(subkey,entropies,tau=1.0,hard=True,axis=-1,eps=1e-10)

    # 坍缩选定的单元
    key, subkey = jax.random.split(key)
    p_collapsed = gumbel_softmax(subkey,probs[collapse_idx],tau=1.0,hard=True,axis=-1,eps=1e-10)
    probs = probs.at[collapse_idx].set(p_collapsed)

    # 获取该单元的邻居
    neighbors = get_neighbors(adj_csr, collapse_idx)
    print(f"坍缩单元 {collapse_idx} 的邻居: {neighbors}")
    for neighbor_index in neighbors:
        p_neigh = jnp.einsum('ij,jl,il->il',tileHandler.compatibility, p_collapsed, probs[neighbor_index])
        norm = jax.linalg.norm(jnp.abs(p_neigh), ord=1, axis=-1, keepdims=True)
        p_neigh = p_neigh / jnp.where(norm == 0, 1.0, norm)
        probs = probs.at[neighbor_index].set(p_neigh)

    # 然后再计算香农熵选择下一个
