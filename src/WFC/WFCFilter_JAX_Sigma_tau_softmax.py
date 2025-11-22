import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import numpy as np


def preprocess_adjacency(adj_csr, tileHandler):
    """修正后的邻接矩阵处理（用户原始逻辑）"""
    dir_int_to_str = tileHandler.dir_int_to_str
    concrete_dirs = [
        dir_str for dir_int, dir_str in dir_int_to_str.items()
        if dir_int != -1 and dir_str != 'isotropy'
    ]
    unique_dirs = list(dict.fromkeys(concrete_dirs))
    dir_mapping = {dir_str: idx for idx, dir_str in enumerate(unique_dirs)}
    
    row_ptr = np.array(adj_csr['row_ptr'])
    col_idx = np.array(adj_csr['col_idx'])
    directions = np.array(adj_csr['directions'])
    
    dir_indices = np.array([dir_mapping[dir_str] for dir_str in directions], dtype=np.int32)
    
    n_cells = len(row_ptr) - 1
    A_np = np.zeros((n_cells, n_cells), dtype=np.float32)
    D_np = np.zeros((n_cells, n_cells), dtype=np.int32)
    
    for j in range(n_cells):
        start = row_ptr[j]
        end = row_ptr[j+1]
        neighbors_j = col_idx[start:end]
        dirs_j = dir_indices[start:end]
        A_np[j, neighbors_j] = 1.0
        D_np[j, neighbors_j] = dirs_j
    
    A = jnp.array(A_np)
    D = jnp.array(D_np)
    return A, D


# def get_neighbors(csr, index):
#     """获取指定单元的邻居及方向（辅助函数）"""
#     start = csr['row_ptr'][index]
#     end = csr['row_ptr'][index + 1]
#     neighbors = csr['col_idx'][start:end]
#     neighbors_dirs = csr['directions'][start:end]
#     return neighbors, neighbors_dirs


@partial(jax.jit, static_argnames=['n_cells', 'sigma'])
def soft_mask(index, n_cells, tau=2.0, sigma=1.0):
    """接近硬掩码的软掩码：sigma=0.1（陡峭，仅局部影响）"""
    x = jnp.arange(n_cells)
    dist_sq = (x - index) ** 2
    mask = jax.nn.sigmoid(-dist_sq / (2 * sigma**2))
    mask = mask / jnp.sum(mask)
    return mask

@partial(jax.jit, static_argnames=['sigma', 'neighbor_radius'])
def spatial_soft_mask(target_cell_idx, cell_centers, sigma=0.1, neighbor_radius=1.2):
    """基于单元中心坐标的3D软掩码（无排序依赖，缓解梯度消失）"""
    target_center = cell_centers[target_cell_idx]
    dists = jnp.linalg.norm(cell_centers - target_center[None, :], axis=1,ord=1) #应该比2范数要好
    in_radius = jax.nn.sigmoid(-(dists - neighbor_radius) / sigma)
    distance_decay = jax.nn.sigmoid(-dists**2 / (2 * sigma**2))
    mask = in_radius * distance_decay
    
    mask_sum = jnp.sum(mask)
    mask = jnp.where(mask_sum == 0, mask, mask / mask_sum)
    return mask


def compute_cell_centers(cell_vertices):
    """从单元顶点坐标计算单元中心（与单元排序无关）"""
    cell_centers = jnp.mean(cell_vertices, axis=1,keepdims=False)
    # cell_centers = cell_centers / jnp.max(jnp.abs(cell_centers))
    return cell_centers


@partial(jax.jit)
def single_update_by_neighbors(collapse_idx, key, init_probs, cell_centers, A, D, dirs_opposite_index, compatibility, alpha=0., tau=2.0):
    n_cells, n_tiles = init_probs.shape
    eps = 1e-12  # 数值稳定性参数
    
    # 1. 生成空间软掩码
    collapse_mask = spatial_soft_mask(
        target_cell_idx=collapse_idx,
        cell_centers=cell_centers,
        sigma=0.1,
        neighbor_radius=1.2
    )[:, None]  # (n_cells, 1)
    
    # 2. 提取当前单元的邻居掩码和方向（维度修正）
    neighbor_mask = A[collapse_idx, :]  # (n_cells,)
    neighbor_mask_broadcast = neighbor_mask[:, None]  # (n_cells, 1)
    neighbor_dirs = D[collapse_idx, :].astype(jnp.int32)  # (n_cells,)
    
    # 3. 兼容性矩阵取值（关键：compatibility是(n_dirs, n_tiles, n_tiles)）
    opposite_dirs = jnp.take(dirs_opposite_index, neighbor_dirs, mode='clip')  # (n_cells,)
    compat = jnp.take(compatibility, opposite_dirs, axis=0)  # (n_cells, n_tiles, n_tiles)
    
    # 过滤无效邻居（置为0，不影响求和）
    compat = compat * neighbor_mask_broadcast[:, None]  # (n_cells, n_tiles, n_tiles)
    # compat = jnp.clip(compat, eps, 1.0)  # 避免0值
    
    # 添加微小噪声稳定梯度
    # noise = jax.random.normal(key, compat.shape) * 1e-9
    # compat = jnp.clip(compat + noise, eps, 1.0)
    
    # 4. 邻居概率提取（维度：n_cells × n_tiles）
    neighbor_probs = init_probs * neighbor_mask_broadcast  # (n_cells, n_tiles)
    # neighbor_probs = jnp.clip(neighbor_probs, eps, 1.0)
    
    # 5. 贡献计算（维度匹配：n_cells × n_tiles × n_tiles）
    # compat (n_cells, n_tiles, n_tiles) × neighbor_probs (n_cells, 1, n_tiles)
    # update_factors = compat * neighbor_probs[:, None, :]  # (n_cells, n_tiles, n_tiles)
    # sum_factors = jnp.sum(update_factors, axis=2)  # (n_cells, n_tiles)
    sum_factors = jnp.einsum('...ijk,...ikl->...ijl', compat, neighbor_probs[...,None]).squeeze(axis=-1)
    
    # 6. 温度系数 + Softmax（维度：n_cells × n_tiles）
    tau_sum_factors = sum_factors
    # tau_sum_factors = sum_factors ** (1.0 / tau)  # 温度系数（反向缩放，保持原效果）
    # tau_sum_factors = jax.nn.softmax(tau_sum_factors, axis=-2)  # 沿tile维度Softmax
    # tau_sum_factors = jnp.clip(tau_sum_factors, eps, 1.0)
    # jax.debug.print("tau_sum_factors:\n{a}",a=tau_sum_factors)
    
    # 7. 聚合所有邻居贡献（关键：axis=0 → 结果为(n_tiles,)）
    sum_contrib = jnp.sum(tau_sum_factors, axis=0)  # (n_tiles,)
    # jax.debug.print("sum_contrib:\n{a}",a=sum_contrib)
    sum_contrib = jnp.clip(sum_contrib, eps, None)
    
    # 8. 更新当前单元概率（维度匹配：n_tiles,）
    p_updated = init_probs[collapse_idx] * sum_contrib  # (n_tiles,)
    # p_updated = p_updated / jnp.sum(p_updated,axis=-1)  # 归一化
    
    # 混合初始概率
    p_updated = (1 - alpha) * p_updated + alpha * init_probs[collapse_idx]
    p_updated = p_updated / jnp.sum(p_updated)  # 再次归一化
    p_updated = jnp.clip(p_updated, eps, 1.0)
    
    # 9. 局部软更新（维度：n_cells × n_tiles）
    updated_probs = init_probs * (1 - collapse_mask) + p_updated * collapse_mask
    updated_probs = updated_probs / jnp.sum(updated_probs, axis=1)[:, None]
    updated_probs = jnp.clip(updated_probs, eps, 1.0) #有待商榷
    return updated_probs

@partial(jax.jit)
def single_update_neighbors(collapse_idx, step1_probs, A, D, compatibility, tau=2.0):
    n_cells, n_tiles = step1_probs.shape
    eps = 1e-10
    
    # 1. 提取邻居掩码和方向
    neighbor_mask = A[collapse_idx, :]  # (n_cells,)
    neighbor_mask_broadcast = neighbor_mask[:, None]  # (n_cells, 1)
    neighbor_dirs = D[collapse_idx, :].astype(jnp.int32)  # (n_cells,)
    
    # 2. 兼容性矩阵取值
    compat = jnp.take(compatibility, neighbor_dirs, axis=0)  # (n_cells, n_tiles, n_tiles)
    compat = compat * neighbor_mask_broadcast[..., None]  # 过滤无效邻居
    # compat = jnp.clip(compat, eps, 1.0)
    
    # 3. 贡献计算
    p_collapsed = step1_probs[collapse_idx]  # (n_tiles,)
    # compat (n_cells, n_tiles, n_tiles) × p_collapsed (1, 1, n_tiles)
    p_neigh = compat * p_collapsed[None, None, :]  # (n_cells, n_tiles, n_tiles)
    # p_neigh = jnp.einsum("...ijk,...jkl->...ijl",compat,p_collapsed[None, None, :,None]).squeeze(axis=-1)
    contrib = jnp.sum(p_neigh, axis=2)  # (n_cells, n_tiles)
    
    # 4. 温度系数 + Softmax
    tau_contrib = contrib
    # tau_contrib = contrib ** (1.0 / tau)
    # tau_contrib = jax.nn.softmax(tau_contrib, axis=1)  # 沿tile维度Softmax
    # tau_contrib = jnp.clip(tau_contrib, eps, 1.0)
    
    # 5. 更新邻居概率
    w = neighbor_mask_broadcast  # (n_cells, 1)
    p_prev = step1_probs  # (n_cells, n_tiles)
    p_updated = (1 - w) * p_prev + w * tau_contrib
    p_updated = p_updated / jnp.sum(p_updated, axis=1)[:, None]
    p_updated = jnp.clip(p_updated, eps, 1.0) #有待商榷
    
    return p_updated


def preprocess_compatibility(compatibility, compat_threshold=1e-3, eps=1e-5):
    """预处理兼容性矩阵（适配n_dirs × n_tiles × n_tiles维度）"""
    print("Preprocessing compatibility matrix...")
    # compatibility shape: (n_dirs, n_tiles, n_tiles)
    n_dirs, n_tiles, _ = compatibility.shape
    compat_mask = (compatibility > compat_threshold).astype(jnp.float32)  # (n_dirs, n_tiles, n_tiles)
    
    # 逐行（每个方向×每个tile）计算行和
    row_sum = jnp.sum(compat_mask, axis=-1)  # (n_dirs, n_tiles)
    v = 1.0 / (row_sum + eps)  # (n_dirs, n_tiles)
    
    # 逐行乘以权重
    new_compatibility = v[:, :, None] * compatibility  # (n_dirs, n_tiles, n_tiles)
    # new_compatibility = jnp.clip(new_compatibility, eps, 1.0)
    return new_compatibility


@partial(jax.jit)
def waveFunctionCollapse(init_probs, A, D, dirs_opposite_index, compatibility, key, cell_centers, tau=0.1,*args, **kwargs):
    """WFC主函数：用vmap批量处理，适配可变邻居数（普通空间版本）"""
    n_cells, n_tiles = init_probs.shape
    eps = 1e-10
    
    # 1. 初始化概率（确保归一化和数值稳定性）
    init_probs_norm = init_probs
    # init_probs_clipped = jnp.clip(init_probs, eps, 1.0)
    # init_probs_norm = init_probs_clipped / jnp.sum(init_probs_clipped, axis=1)[:, None]
    # 2. 兼容性矩阵预处理（确保数值稳定性）
    compatibility_clipped = compatibility
    # compatibility_clipped = jnp.clip(compatibility, eps, 1.0)
    
    # 3. 第一步：批量更新所有单元
    subkeys = jax.random.split(key, n_cells)
    batch_updated_step1 = jax.vmap(
        single_update_by_neighbors,
        in_axes=(0, 0, None, None, None, None, None, None, None, None)
    )(
        jnp.arange(n_cells),
        subkeys,
        init_probs_norm,
        cell_centers,
        A, D, dirs_opposite_index, compatibility_clipped,
        0.,
        tau
    )
    # probs_step1 = jnp.mean(batch_updated_step1, axis=0)
    # probs_step1 = probs_step1 / jnp.sum(probs_step1, axis=1)[:, None]
    # probs_step1 = jnp.clip(probs_step1, eps, 1.0)
    
    # ========== 核心修改：加权求和聚合（替换原mean） ==========
    # 3.1 计算每个坍缩中心到所有cell的空间L1距离（贴合之前的spatial_soft_mask）
    collapse_indices = jnp.arange(n_cells)[:, None]  # (n_cells, 1) → 坍缩中心轴扩维
    # cell_centers[collapse_indices] → (n_cells, n_cells, 3)，每个坍缩中心对应的所有cell坐标
    distances = jnp.linalg.norm(
        cell_centers[collapse_indices] - cell_centers,  # 坍缩中心坐标 - 目标cell坐标
        axis=2,  # 沿坐标维度（3维）计算范数
        ord=1    # L1范数，与spatial_soft_mask保持一致
    )  # 结果：(n_cells, n_cells) → [坍缩中心, 目标cell]的距离
    
    # 3.2 将距离转化为权重（距离越近，权重越高）
    # sigma控制权重衰减速度：sigma越小，近邻权重越集中；越大越均匀
    sigma_weight = 0.1  
    distance_weights = jax.nn.softmax(-distances / sigma_weight, axis=0)  # 沿坍缩中心轴归一化，权重和为1
    distance_weights = jnp.clip(distance_weights, eps, 1.0)  # 数值稳定
    
    # 3.3 权重扩维适配tile维度（batch_updated_step1是(n_cells, n_cells, n_tiles)）
    weights_expanded = distance_weights[:, :, None]  # (n_cells, n_cells, 1)
    
    # 3.4 加权求和聚合（沿坍缩中心轴，axis=0）
    weighted_updates = batch_updated_step1 * weights_expanded  # 逐元素加权
    probs_step1 = jnp.sum(weighted_updates, axis=0)  # (n_cells, n_tiles) → 聚合后
    
    # 3.5 归一化+数值裁剪（保证概率分布合法）
    probs_step1 = probs_step1 / jnp.sum(probs_step1, axis=-1)[:, None]
    probs_step1 = jnp.clip(probs_step1, eps, 1.0)
    # ========== 加权求和聚合结束 ==========

    # 4. 第二步：批量更新邻居
    batch_updated_step2 = jax.vmap(
        single_update_neighbors,
        in_axes=(0, None, None, None, None, None)
    )(
        jnp.arange(n_cells),
        probs_step1,
        A, D, compatibility_clipped,
        tau
    )
    # final_probs = jnp.mean(batch_updated_step2, axis=0)
    # final_probs = final_probs / jnp.sum(final_probs, axis=1)[:, None]
    # final_probs = jnp.clip(final_probs, eps, 1.0)
    # === step2加权求和（复用step1的权重） ===
    weighted_updates_step2 = batch_updated_step2 * weights_expanded  # 复用权重，无需重算
    final_probs = jnp.sum(weighted_updates_step2, axis=0)
    # 归一化+数值裁剪（保证概率合法）
    final_probs = final_probs / jnp.sum(final_probs, axis=1)[:, None]
    final_probs = jnp.clip(final_probs, eps, 1.0)
    
    return final_probs, 0, jnp.arange(n_cells)


# ========== 测试模块（保持不变） ==========
class MockTileHandler:
    """模拟TileHandler类（用于测试）"""
    def __init__(self):
        self.dir_int_to_str = {0: 'back', 1: 'front', 2: 'bottom', 3: 'top', 4: 'left', 5: 'right'}


def test_adjacency_matrix():
    """测试1：邻接矩阵邻居关系验证"""
    print("="*50)
    print("测试1：邻接矩阵邻居关系验证")
    adj_csr = {
        'row_ptr': [0, 3, 6, 9, 12, 15, 18, 21, 24],
        'col_idx': [1, 2, 4, 0, 3, 5, 0, 3, 6, 1, 2, 7, 0, 5, 6, 1, 4, 7, 2, 4, 7, 3, 5, 6],
        'directions': ['right', 'top', 'front'] * 8
    }
    tile_handler = MockTileHandler()
    
    A, D = preprocess_adjacency(adj_csr, tile_handler)
    cell_0_neighbors = jnp.where(A[0, :] == 1)[0]
    print(f"单元0的邻居索引: {cell_0_neighbors}")
    print(f"单元0的邻居方向: {D[0, cell_0_neighbors]}")
    print(f"单元0的邻居数量: {len(cell_0_neighbors)}")
    
    assert len(cell_0_neighbors) == 3, f"单元0应有3个邻居，实际{len(cell_0_neighbors)}个"
    print("✅ 邻接矩阵邻居关系验证通过")
    print("="*50)


def test_cell_centers():
    """测试2：单元中心计算验证"""
    print("\n" + "="*50)
    print("测试2：单元中心计算验证")
    cell_vertices = jnp.array([
        [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]],
        [[1,0,0], [2,0,0], [2,1,0], [1,1,0], [1,0,1], [2,0,1], [2,1,1], [1,1,1]],
        [[0,1,0], [1,1,0], [1,2,0], [0,2,0], [0,1,1], [1,1,1], [1,2,1], [0,2,1]],
        [[1,1,0], [2,1,0], [2,2,0], [1,2,0], [1,1,1], [2,1,1], [2,2,1], [1,2,1]],
        [[0,0,1], [1,0,1], [1,1,1], [0,1,1], [0,0,2], [1,0,2], [1,1,2], [0,1,2]],
        [[1,0,1], [2,0,1], [2,1,1], [1,1,1], [1,0,2], [2,0,2], [2,1,2], [1,1,2]],
        [[0,1,1], [1,1,1], [1,2,1], [0,2,1], [0,1,2], [1,1,2], [1,2,2], [0,2,2]],
        [[1,1,1], [2,1,1], [2,2,1], [1,2,1], [1,1,2], [2,1,2], [2,2,2], [1,2,2]],
    ])
    
    cell_centers = compute_cell_centers(cell_vertices)
    print(f"单元0中心坐标: {cell_centers[0]}")
    print(f"单元1中心坐标: {cell_centers[1]}")
    assert cell_centers.shape == (8, 3), f"单元中心形状应为(8,3)，实际{cell_centers.shape}"
    print("✅ 单元中心计算验证通过")
    print("="*50)


def test_wfc_run():
    """测试3：WFC完整运行验证（普通空间版本）"""
    print("\n" + "="*50)
    print("测试3：WFC完整运行验证")
    # 1. 基础参数
    n_cells = 8
    n_tiles = 3    # 3种Tile
    n_dirs = 6     # 6个方向（back/front/bottom/top/left/right）
    tile_handler = MockTileHandler()
    
    # 2. 模拟CSR邻接数据
    adj_csr = {
        'row_ptr': [0, 3, 6, 9, 12, 15, 18, 21, 24],
        'col_idx': [1, 2, 4, 0, 3, 5, 0, 3, 6, 1, 2, 7, 0, 5, 6, 1, 4, 7, 2, 4, 7, 3, 5, 6],
        'directions': ['right', 'top', 'front'] * 8
    }
    
    # 3. 预处理邻接矩阵
    A, D = preprocess_adjacency(adj_csr, tile_handler)
    
    # 4. 构造兼容性矩阵（关键：维度n_dirs × n_tiles × n_tiles）
    base_compat = jnp.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.1],
        [0.0, 0.1, 0.9]
    ])
    # 6个方向共享相同的兼容性规则（可根据需求自定义）
    compatibility = jnp.tile(base_compat, (n_dirs, 1, 1))  # (6, 3, 3)
    compatibility = preprocess_compatibility(compatibility)
    
    # 5. 构造单元中心
    cell_vertices = jnp.array([
        [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]] for _ in range(n_cells)
    ])
    cell_centers = compute_cell_centers(cell_vertices)
    
    # 6. 初始概率（均匀分布）
    init_probs = jnp.ones((n_cells, n_tiles)) / n_tiles
    
    # 7. 方向反向索引（back↔front, bottom↔top, left↔right）
    dirs_opposite_index = jnp.array([1,0,3,2,5,4])  # (6,)
    
    # 8. 随机密钥
    key = jax.random.PRNGKey(42)
    
    # 9. 运行WFC
    final_probs, _, _ = waveFunctionCollapse(
        init_probs=init_probs,
        A=A,
        D=D,
        dirs_opposite_index=dirs_opposite_index,
        compatibility=compatibility,
        key=key,
        cell_centers=cell_centers,
        tau=1.0
    )
    
    # 10. 验证结果
    print(f"WFC输出概率形状: {final_probs.shape}")
    print(f"单元0的最终概率: {final_probs[0]}")
    print(f"单元0概率和: {jnp.sum(final_probs[0]):.4f}")
    print(f"所有单元概率和: {jnp.sum(final_probs, axis=1)}")
    
    # 断言验证
    assert final_probs.shape == (n_cells, n_tiles), f"输出形状应为({n_cells},{n_tiles})，实际{final_probs.shape}"
    assert jnp.allclose(jnp.sum(final_probs, axis=1), 1.0, atol=1e-3), "每个单元概率和应≈1"
    print("✅ WFC完整运行验证通过")
    print("="*50)


if __name__ == "__main__":
    # 运行所有测试
    test_adjacency_matrix()
    test_cell_centers()
    test_wfc_run()
    
    print("\n🎉 所有测试通过！WFC算法（普通空间版本）运行正常")