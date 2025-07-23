import jax
import jax.numpy as jnp

@jax.jit
def shannon_entropy(probs: jnp.ndarray)->jnp.ndarray:
    """
    shannon entropy
    :param probs: probabilities
    :return:
    """
    probs = jnp.clip(probs, 1e-8, 1 - 1e-8)
    entropy = -jnp.sum(probs * jnp.log2(probs))
    return entropy







# 运行测试
if __name__ == "__main__":
    # 测试函数
    def test_shannon_entropy():
        # 测试1: 等概率分布
        uniform_probs = jnp.array([0.5, 0.5])
        entropy_uniform = shannon_entropy(uniform_probs)
        print(f"测试1 - 等概率分布 [0.5, 0.5]: {entropy_uniform:.4f} bits (应为 1.0 bits)")
        assert jnp.isclose(entropy_uniform, 1.0), "等概率分布测试失败"

        # 测试2: 确定性事件
        certain_probs = jnp.array([1.0])
        entropy_certain = shannon_entropy(certain_probs)
        print(f"测试2 - 确定性事件 [1.0]: {entropy_certain:.8f} bits (应为 0.0 bits)")
        assert jnp.isclose(entropy_certain, 0.000), "确定性事件测试失败"

        # 测试3: 有偏分布
        biased_probs = jnp.array([0.7, 0.3])
        entropy_biased = shannon_entropy(biased_probs)
        expected_biased = -(0.7 * jnp.log2(0.7) + 0.3 * jnp.log2(0.3))
        print(f"测试3 - 有偏分布 [0.7, 0.3]: {entropy_biased:.4f} bits (应为 {expected_biased:.4f} bits)")
        assert jnp.isclose(entropy_biased, expected_biased, atol=1e-4), "有偏分布测试失败"

        # 测试4: 多类别分布
        multi_probs = jnp.array([0.25, 0.25, 0.25, 0.25])
        entropy_multi = shannon_entropy(multi_probs)
        print(f"测试4 - 多类别分布 [0.25, 0.25, 0.25, 0.25]: {entropy_multi:.4f} bits (应为 2.0 bits)")
        assert jnp.isclose(entropy_multi, 2.0), "多类别分布测试失败"

        # 测试5: 带零值的概率分布
        zero_probs = jnp.array([0.0, 0.8, 0.2])
        entropy_zero = shannon_entropy(zero_probs)
        expected_zero = -(0.8 * jnp.log2(0.8) + 0.2 * jnp.log2(0.2))
        print(f"测试5 - 带零值的分布 [0.0, 0.8, 0.2]: {entropy_zero:.4f} bits (应为 {expected_zero:.4f} bits)")
        assert jnp.isclose(entropy_zero, expected_zero, atol=1e-4), "带零值分布测试失败"

        # 测试6: 大型分布
        big_probs = jnp.ones(100) / 100
        entropy_big = shannon_entropy(big_probs)
        print(f"测试6 - 大型分布 [100个0.01]: {entropy_big:.4f} bits (应为 {-jnp.log2(1 / 100):.4f} bits)")
        assert jnp.isclose(entropy_big, jnp.log2(100)), "大型分布测试失败"

        # 测试7: JIT 编译能力
        entropy_jitted = jax.jit(shannon_entropy)
        result_jit = entropy_jitted(biased_probs)
        print(f"测试7 - JIT编译结果: {result_jit:.4f} bits (与普通调用一致)")
        assert jnp.isclose(result_jit, entropy_biased), "JIT编译测试失败"

        # 测试8: 梯度计算
        def entropy_wrapper(x):
            return shannon_entropy(jax.nn.softmax(x))

        x = jnp.array([1.0, 2.0])
        grad_entropy = jax.grad(entropy_wrapper)(x)
        print("测试8 - 熵梯度:", grad_entropy)
        assert jnp.all(jnp.isfinite(grad_entropy)), "梯度计算测试失败"

        # 测试9: 批量计算
        batch_probs = jnp.array([
            [0.5, 0.5],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.75, 0.25]
        ])
        expected_batch = jnp.array([
            shannon_entropy(batch_probs[0]),
            shannon_entropy(batch_probs[1]),
            shannon_entropy(batch_probs[2]),
            shannon_entropy(batch_probs[3])
        ])
        actual_batch = jax.vmap(shannon_entropy)(batch_probs)
        print(f"测试9 - 批量计算:", actual_batch)
        assert jnp.allclose(actual_batch, expected_batch), "批量计算测试失败"

        print("\n✅ 所有测试通过！")
    # test_shannon_entropy()
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.arange(1,1001)
    for i in x:
        probs = np.ones(i)/i
        plt.scatter(i, np.array(np.max(shannon_entropy(probs))))
    plt.show()