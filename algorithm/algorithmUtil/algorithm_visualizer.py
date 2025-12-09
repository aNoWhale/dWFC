# wfc_vis_universal.py
import os
import numpy as np
import matplotlib.pyplot as plt

# 全局可视化配置（统一风格）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 7   # 自适应网格大小，缩小标题
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5

def plot_wfc_grid(
    probs: np.ndarray,
    grid_size: int,
    iteration: int = 0,
    n_tiles: int = 3,
    save_dir: str = "wfc_plots",
    show_values: bool = True,
    return_fig: bool = False,
    fig_scale: float = 2.0  # 画布缩放系数（grid_size=5→10x10，5*2=10）
):
    """
    即插即用的通用WFC概率网格可视化函数（支持任意网格大小：3×3/4×4/5×5/6×6等）
    
    Args:
        probs: 概率矩阵，形状必须为 (grid_size×grid_size, n_tiles)
        grid_size: 网格边长（如3=3×3，5=5×5）
        iteration: 迭代次数（标题/文件名用）
        n_tiles: Tile数量（默认3）
        save_dir: 图片保存目录（自动创建）
        show_values: 是否显示概率数值
        return_fig: 是否返回matplotlib fig/axes对象（自定义用）
        fig_scale: 画布缩放系数（每个网格单元的画布大小，默认2.0）
    
    Returns:
        若return_fig=True，返回(fig, axes)；否则返回None
    """
    # 1. 严格参数校验（鲁棒性保障）
    n_cells = grid_size * grid_size
    if probs.shape[0] != n_cells:
        raise ValueError(
            f"概率矩阵行数需等于网格总单元数 {grid_size}×{grid_size}={n_cells}，"
            f"当前形状：{probs.shape}"
        )
    if probs.shape[1] != n_tiles:
        raise ValueError(
            f"概率矩阵列数需等于n_tiles={n_tiles}，当前：{probs.shape[1]}"
        )
    
    # 2. 自动创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. 固定配色（适配任意tile数量）
    tile_colors = plt.cm.Set3(np.linspace(0, 1, n_tiles))
    
    # 4. 动态计算画布大小（适配不同网格）
    fig_size = (grid_size * fig_scale, grid_size * fig_scale)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=fig_size)
    fig.suptitle(f'WFC Iteration {iteration} - {grid_size}×{grid_size} Grid Probability', fontsize=10)
    
    # 处理单网格（grid_size=1）的特殊情况
    if grid_size == 1:
        axes = np.array([[axes]])
    
    # 5. 遍历每个单元绘制柱状图（通用逻辑，适配任意网格）
    for cell_idx in range(n_cells):
        # 计算单元在网格中的行列位置
        row = cell_idx // grid_size
        col = cell_idx % grid_size
        ax = axes[row, col]
        
        # 提取当前单元的概率分布
        cell_probs = probs[cell_idx]
        
        # 绘制柱状图（极简风格，适配小尺寸）
        bars = ax.bar(
            range(n_tiles),
            cell_probs,
            color=tile_colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.3,  # 细边框，适配小网格
            width=0.8       # 柱子宽度，避免重叠
        )
        
        # 子图样式配置（自适应网格大小）
        ax.set_title(f'Cell ({row},{col})', fontsize=6)
        ax.set_xlim(-0.5, n_tiles - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(n_tiles))
        ax.set_xticklabels([f'T{i}' for i in range(n_tiles)])
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(axis='y', alpha=0.2, linewidth=0.3)
        
        # 可选：显示概率数值（适配小网格）
        if show_values:
            font_size = max(4, int(6 - grid_size/2))  # 网格越大，字体越小
            for bar, prob in zip(bars, cell_probs):
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    bar.get_height() + 0.01,
                    f'{prob:.2f}',
                    ha='center', va='bottom',
                    fontsize=font_size
                )
    
    # 6. 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 预留总标题空间
    save_path = os.path.join(save_dir, f'wfc_{grid_size}x{grid_size}_iter_{iteration}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)  # 高分辨率，适配大网格
    print(f"✅ WFC可视化图已保存：{save_path}")
    
    # 7. 返回或关闭画布
    if return_fig:
        return fig, axes
    else:
        plt.close()
        return None

def batch_plot_wfc_grid(
    all_probs: dict,
    grid_size: int,
    n_tiles: int = 3,
    save_dir: str = "wfc_plots_batch"
):
    """
    批量绘制多次迭代的通用网格图（支持任意网格大小）
    Args:
        all_probs: 字典 {迭代次数: 概率矩阵}
        grid_size: 网格边长
        n_tiles: Tile数量
        save_dir: 保存目录
    """
    for iter_num, probs in all_probs.items():
        plot_wfc_grid(
            probs=probs,
            grid_size=grid_size,
            iteration=iter_num,
            n_tiles=n_tiles,
            save_dir=save_dir
        )