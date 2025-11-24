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


def get_neighbors(csr, index):
    """获取指定单元的邻居及方向（辅助函数）"""
    start = csr['row_ptr'][index]
    end = csr['row_ptr'][index + 1]
    neighbors = csr['col_idx'][start:end]
    neighbors_dirs = csr['directions'][start:end]
    return neighbors, neighbors_dirs


@partial(jax.jit, static_argnames=['n_cells', 'sigma'])
def soft_mask(index, n_cells, tau=2.0, sigma=1.0):
    """接近硬掩码的软掩码：sigma=0.1（陡峭，仅局部影响）"""
    x = jnp.arange(n_cells)
    dist_sq = (x - index) ** 2
    mask = jax.nn.sigmoid(-dist_sq / (2 * sigma**2))
    mask = mask / jnp.sum(mask)
    return mask

@partial(jax.jit, static_argnames=['sigma', 'neighbor_radius'])
def spatial_soft_mask(target_cell_idx, cell_centers, sigma=0.1, neighbor_radius=1.0):
    """基于单元中心坐标的3D软掩码（无排序依赖，缓解梯度消失）"""
    target_center = cell_centers[target_cell_idx]
    dists = jnp.linalg.norm(cell_centers - target_center[None, :], axis=1)
    in_radius = jax.nn.sigmoid(-(dists - neighbor_radius) / sigma)
    distance_decay = jax.nn.sigmoid(-dists**2 / (2 * sigma**2))
    mask = in_radius * distance_decay
    
    mask_sum = jnp.sum(mask)
    mask = jnp.where(mask_sum == 0, mask, mask / mask_sum)
    return mask


def compute_cell_centers(cell_vertices):
    """从单元顶点坐标计算单元中心（与单元排序无关）"""
    cell_centers = jnp.mean(cell_vertices, axis=1)
    cell_centers = cell_centers / jnp.max(jnp.abs(cell_centers))
    return cell_centers


@partial(jax.jit)
def single_update_by_neighbors(collapse_idx, key, log_init_probs, cell_centers, A, D, dirs_opposite_index, log_compatibility, alpha=0., tau=2.0):
    n_cells, n_tiles = log_init_probs.shape
    
    # 1. 生成空间软掩码
    collapse_mask = spatial_soft_mask(
        target_cell_idx=collapse_idx,
        cell_centers=cell_centers,
        sigma=0.1,
        neighbor_radius=1.0
    )[:, None]  # (n_cells, 1)
    
    # 2. 提取当前单元的邻居掩码和方向（维度修正）
    neighbor_mask = A[collapse_idx, :]  # (n_cells,)
    neighbor_mask_broadcast = neighbor_mask[:, None]  # (n_cells, 1)
    neighbor_dirs = D[collapse_idx, :].astype(jnp.int32)  # (n_cells,)
    
    # 3. 兼容性矩阵取值（关键：log_compatibility是(n_dirs, n_tiles, n_tiles)）
    opposite_dirs = jnp.take(dirs_opposite_index, neighbor_dirs, mode='clip')  # (n_cells,)
    log_compat = jnp.take(log_compatibility, opposite_dirs, axis=0)  # (n_cells, n_tiles, n_tiles)
    
    # 过滤无效邻居（置为-inf，不影响logsumexp）
    log_compat = log_compat + jnp.log(neighbor_mask_broadcast)[:, None]  # (n_cells, n_tiles, n_tiles)
    log_compat = jnp.clip(log_compat, -50, 0)
    
    # 添加微小噪声稳定梯度
    noise = jax.random.normal(key, log_compat.shape) * 1e-8
    log_compat = jnp.clip(log_compat + noise, -50, 0)
    
    # 4. 邻居概率提取（维度：n_cells × n_tiles）
    log_neighbor_probs = log_init_probs + jnp.log(neighbor_mask_broadcast)  # (n_cells, n_tiles)
    log_neighbor_probs = jnp.clip(log_neighbor_probs, -50, 0)
    
    # 5. 贡献计算（维度匹配：n_cells × n_tiles × n_tiles）
    log_update_factors = log_compat + log_neighbor_probs[:, None, :]  # (n_cells, n_tiles, n_tiles)
    log_sum_factors = jax.scipy.special.logsumexp(log_update_factors, axis=2)  # (n_cells, n_tiles)
    
    # 6. 温度系数 + Softmax（维度：n_cells × n_tiles）
    log_tau_sum_factors = tau * log_sum_factors
    log_tau_sum_factors = jax.nn.log_softmax(log_tau_sum_factors, axis=1)  # 沿tile维度Softmax
    
    # 7. 聚合所有邻居贡献（关键：axis=0 → 结果为(n_tiles,)）
    sum_log = jax.scipy.special.logsumexp(log_tau_sum_factors, axis=0)  # (n_tiles,)
    sum_log = jnp.clip(sum_log, -50, 0)
    
    # 8. 更新当前单元概率（维度匹配：n_tiles,）
    log_p_updated = log_init_probs[collapse_idx] + sum_log  # (n_tiles,)
    log_p_updated = log_p_updated - jax.scipy.special.logsumexp(log_p_updated)  # 归一化
    
    # 混合初始概率
    log_p_updated = jnp.log(
        (1 - alpha) * jnp.exp(log_p_updated) + 
        alpha * jnp.exp(log_init_probs[collapse_idx])
    )
    log_p_updated = log_p_updated - jax.scipy.special.logsumexp(log_p_updated)  # 再次归一化
    log_p_updated = jnp.clip(log_p_updated, -50, 0)
    
    # 9. 局部软更新（维度：n_cells × n_tiles）
    updated_log_probs = log_init_probs * (1 - collapse_mask) + log_p_updated * collapse_mask
    updated_log_probs = updated_log_probs - jax.scipy.special.logsumexp(updated_log_probs, axis=1)[:, None]
    return updated_log_probs

@partial(jax.jit)
def single_update_neighbors(collapse_idx, log_probs, A, D, log_compatibility, tau=2.0):
    n_cells, n_tiles = log_probs.shape
    
    # 1. 提取邻居掩码和方向
    neighbor_mask = A[collapse_idx, :]  # (n_cells,)
    neighbor_mask_broadcast = neighbor_mask[:, None]  # (n_cells, 1)
    neighbor_dirs = D[collapse_idx, :].astype(jnp.int32)  # (n_cells,)
    
    # 2. 兼容性矩阵取值
    log_compat = jnp.take(log_compatibility, neighbor_dirs, axis=0)  # (n_cells, n_tiles, n_tiles)
    log_compat = log_compat + jnp.log(neighbor_mask_broadcast)[:, None]  # 过滤无效邻居
    log_compat = jnp.clip(log_compat, -50, 0)
    
    # 3. 贡献计算
    log_p_collapsed = log_probs[collapse_idx]  # (n_tiles,)
    log_p_neigh = log_compat + log_p_collapsed[None, None, :]  # (n_cells, n_tiles, n_tiles)
    log_contrib = jax.scipy.special.logsumexp(log_p_neigh, axis=2)  # (n_cells, n_tiles)
    
    # 4. 温度系数 + Softmax
    log_tau_contrib = tau * log_contrib
    log_tau_contrib = jax.nn.log_softmax(log_tau_contrib, axis=1)  # 沿tile维度Softmax
    
    # 5. 更新邻居概率
    w = neighbor_mask_broadcast  # (n_cells, 1)
    log_p_prev = log_probs  # (n_cells, n_tiles)
    log_p_updated = jnp.log((1 - w) * jnp.exp(log_p_prev) + w * jnp.exp(log_tau_contrib))
    log_p_updated = log_p_updated - jax.scipy.special.logsumexp(log_p_updated, axis=1)[:, None]
    log_p_updated = jnp.clip(log_p_updated, -50, 0)
    
    return log_p_updated


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
    return new_compatibility


@partial(jax.jit)
def waveFunctionCollapse(init_probs, A, D, dirs_opposite_index, compatibility, key, cell_centers, tau=1.0,*args, **kwargs):
    """WFC主函数：用vmap批量处理，适配可变邻居数（完全保留用户核心逻辑）"""
    n_cells, n_tiles = init_probs.shape
    
    # 1. 初始化对数概率
    init_probs_clipped = jnp.clip(init_probs, 1e-5, 1.0)
    log_init_probs = jnp.log(init_probs_clipped)
    log_init_probs = jnp.clip(log_init_probs, -11.5, 0)
    log_init_probs = log_init_probs - jax.scipy.special.logsumexp(log_init_probs, axis=1)[:, None]
    
    # 2. 兼容性矩阵转换为对数空间（关键：维度n_dirs × n_tiles × n_tiles）
    compatibility_clipped = jnp.clip(compatibility, 1e-5, 1.0)
    log_compatibility = jnp.log(compatibility_clipped)
    log_compatibility = jnp.clip(log_compatibility, -11.5, 0)
    
    # 3. 第一步：批量更新所有单元
    subkeys = jax.random.split(key, n_cells)
    batch_updated_step1 = jax.vmap(
        single_update_by_neighbors,
        in_axes=(0, 0, None, None, None, None, None, None, None, None)
    )(
        jnp.arange(n_cells),
        subkeys,
        log_init_probs,
        cell_centers,
        A, D, dirs_opposite_index, log_compatibility,
        0.,
        tau
    )
    log_probs_step1 = jnp.mean(batch_updated_step1, axis=0)
    log_probs_step1 = log_probs_step1 - jax.scipy.special.logsumexp(log_probs_step1, axis=1)[:, None]
    
    # 4. 第二步：批量更新邻居
    batch_updated_step2 = jax.vmap(
        single_update_neighbors,
        in_axes=(0, None, None, None, None, None)
    )(
        jnp.arange(n_cells),
        log_probs_step1,
        A, D, log_compatibility,
        tau
    )
    final_log_probs = jnp.mean(batch_updated_step2, axis=0)
    final_log_probs = final_log_probs - jax.scipy.special.logsumexp(final_log_probs, axis=1)[:, None]
    
    # 5. 转换回概率空间
    final_probs = jnp.exp(final_log_probs)
    final_probs = jnp.clip(final_probs, 1e-5, 1.0)
    final_probs = final_probs / jnp.sum(final_probs, axis=1)[:, None]
    return final_probs, 0, jnp.arange(n_cells)


# ========== 测试模块（修正兼容性矩阵维度） ==========
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
    """测试3：WFC完整运行验证（修正兼容性矩阵维度）"""
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
    
    print("\n🎉 所有测试通过！WFC算法运行正常")