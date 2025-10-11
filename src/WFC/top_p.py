import os
import sys
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/WFC 目录下）
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import jax
import jax.numpy as jnp


def top_p(logits, p=0.9, temperature=1.0):
    """
    JAX 实现的 top-p 采样（修复均匀分布问题）
    
    参数:
        logits: 模型输出的 logits，形状为 (vocab_size,)
        p: 核采样的概率阈值
        temperature: 温度参数，控制分布的平滑程度
    
    返回:
        selected_indices: 选中的token索引
        selected_probs: 选中token的概率（已归一化）
    """
    # 应用温度缩放
    logits = logits / temperature
    
    # 计算概率分布
    probs = jax.nn.softmax(logits)
    
    # 对概率排序（降序）并获取排序索引
    sorted_probs, sorted_indices = jnp.sort(probs)[::-1], jnp.argsort(probs)[::-1]
    
    # 计算累积概率
    cumulative_probs = jnp.cumsum(sorted_probs)
    
    # 找到累积概率首次超过p的索引（核心修复）
    # 1. 标记累积概率超过p的位置
    exceeds_p = cumulative_probs >= p
    # 2. 找到第一个超过p的索引（若全不超过p，则取最后一个索引）
    k = jnp.argmax(exceeds_p)  # 首次超过p的索引
    # 3. 处理所有累积概率都不超过p的情况（如p=1.0）
    k = jnp.where(exceeds_p.any(), k, len(cumulative_probs) - 1)
    
    # 生成mask：包含从0到k的所有索引（确保累积和超过p）
    mask = jnp.arange(len(cumulative_probs)) <= k
    
    # 筛选出符合条件的token及其概率
    selected_probs = sorted_probs[mask]
    selected_indices = sorted_indices[mask]
    
    # 重新归一化概率
    selected_probs = selected_probs / jnp.sum(selected_probs)
    
    return selected_indices, selected_probs