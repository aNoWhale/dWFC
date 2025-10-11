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
    JAX 实现的 top-p 采样
    
    参数:
        logits: 模型输出的 logits，形状为 (vocab_size,)
        p: 核采样的概率阈值
        temperature: 温度参数，控制分布的平滑程度
    
    返回:
        采样的 token 索引
    """
    # 应用温度缩放
    logits = logits / temperature
    
    # 计算概率分布
    probs = jax.nn.softmax(logits)
    
    # 对概率排序（降序）并获取排序索引
    sorted_probs, sorted_indices = jnp.sort(probs)[::-1], jnp.argsort(probs)[::-1]
    
    # 计算累积概率
    cumulative_probs = jnp.cumsum(sorted_probs)
    
    # 找到累积概率超过 p 的最小索引
    # 保留累积概率 <= p 的所有 token，再加上下一个 token 使其总和超过 p
    mask = cumulative_probs <= p
    # 确保至少保留一个 token
    mask = jnp.logical_or(mask, jnp.arange(len(mask)) == jnp.argmax(mask))
    
    # 筛选出符合条件的 token 及其概率
    selected_probs = sorted_probs[mask]
    selected_indices = sorted_indices[mask]
    
    # 重新归一化概率
    selected_probs = selected_probs / jnp.sum(selected_probs)
    
    # 采样
    return selected_indices, selected_probs