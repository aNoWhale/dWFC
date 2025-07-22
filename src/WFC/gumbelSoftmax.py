from functools import partial

import jax
import jax.numpy as jnp

def sample_gumbel(key: jax.random.PRNGKey, shape: tuple, eps: float = 1e-10) -> jnp.ndarray:
    """
    生成 Gumbel 噪声

    参数:
        key: JAX 随机数生成器密钥
        shape: 输出张量的形状
        eps: 数值稳定性常数

    返回:
        Gumbel 噪声张量
    """
    u = jax.random.uniform(key, shape, minval=0, maxval=1)
    return -jnp.log(-jnp.log(u + eps) + eps)


@partial(jax.jit, static_argnames=('hard', 'tau', 'axis', 'eps'))
def gumbel_softmax(
        key: jax.random.PRNGKey,
        logits: jnp.ndarray,
        tau: float = 1.0,
        hard: bool = False,
        axis: int = -1,
        eps: float = 1e-10
) -> jnp.ndarray:
    gumbels = sample_gumbel(key, logits.shape, eps)
    noisy_logits = logits + gumbels
    tempered_logits = noisy_logits / tau
    y_soft = jax.nn.softmax(tempered_logits, axis=axis)
    def hard_case(y_soft):
        y_hard = jax.nn.one_hot(jnp.argmax(tempered_logits, axis=axis), logits.shape[axis])
        return y_hard - jax.lax.stop_gradient(y_soft) + y_soft

    return jax.lax.cond(
        hard,
        hard_case,  # 如果hard=True
        lambda y: y,y_soft  # 如果hard=False
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def test_gumbel_softmax():
        # 初始化随机密钥
        key = jax.random.PRNGKey(42)

        # 创建一组测试 logits (batch_size=5, num_classes=3)
        logits = jnp.array([
            [1.0, 1.0, 1.0],  # 均匀分布
            [1.0, 2.0, 3.0],  # 偏向第2类
            [3.0, 2.0, 1.0],  # 偏向第0类
            [-1.0, 2.0, -3.0],  # 偏向第1类
            [10.0, 1.0, 0.1]  # 强偏向第0类
        ])

        # 计算对应的概率分布
        probs = jax.nn.softmax(logits, axis=-1)

        print("original:")
        for i, p in enumerate(probs):
            print(f"sample {i}: {p} (entropy: {-jnp.sum(p * jnp.log2(p + 1e-10)):.3f} bits)")

        # 测试不同温度下软采样的行为
        temperatures = [1.0, 0.5, 0.1, 0.01]
        num_samples = 1000  # 每个设置的采样次数

        # 绘制不同温度下的分布变化
        plt.figure(figsize=(15, 10))

        for sample_idx in range(3):  # 测试前三组 logits
            for i, tau in enumerate(temperatures):
                # 采样函数
                sample_fn = lambda k: gumbel_softmax(k, logits[sample_idx], tau=tau, hard=False)

                # 生成多个样本密钥
                keys = jax.random.split(key, num_samples)
                key, _ = jax.random.split(jax.random.PRNGKey(i))

                # 批量采样
                samples = jax.vmap(sample_fn)(keys)

                # 计算平均样本分布
                mean_samples = jnp.mean(samples, axis=0)

                # 打印结果
                print(f"\n样本 #{sample_idx} (τ={tau}) - 真实概率: {probs[sample_idx]}, 平均样本: {mean_samples}")

                # 绘制分布
                plt.subplot(3, len(temperatures), sample_idx * len(temperatures) + i + 1)
                plt.bar(range(probs.shape[1]), mean_samples, alpha=0.7)
                plt.plot(range(probs.shape[1]), probs[sample_idx], 'ro--', label='real probability')
                plt.ylim(0, 1)
                plt.title(f"sample #{sample_idx}, τ={tau}")
                plt.xlabel("types")
                plt.ylabel("average probability")
                if i == 0:
                    plt.legend()

        plt.tight_layout()
        plt.savefig("gumbel_softmax_temp_effect.png")
        plt.show()

        # 测试硬采样行为
        key, subkey = jax.random.split(key)
        hard_samples = gumbel_softmax(subkey, logits, tau=0.1, hard=True)
        print("\n硬采样结果 (τ=0.1):")
        print(hard_samples)

        # 测试梯度存在性（针对硬采样）
        def loss_fn(logits, key):
            samples = gumbel_softmax(key, logits, tau=0.5, hard=True)
            # 使用简单的奖励函数（偏好第0类）
            reward = samples[..., 0]
            return jnp.mean(reward)  # 最大化奖励

        # 计算梯度
        key, subkey = jax.random.split(key)
        grad_fn = jax.grad(lambda l: loss_fn(l, subkey))
        gradients = grad_fn(logits)

        print("\n梯度结果 (硬采样):")
        print(gradients)

        # 验证低温下的离散行为
        print("\n低温离散行为 (τ=0.01):")
        key, subkey = jax.random.split(key)
        low_temp_samples = gumbel_softmax(subkey, logits, tau=0.01, hard=True)
        print(low_temp_samples)

        # 验证硬采样是否真正是one-hot
        print("\n硬采样是否为one-hot:")
        one_hot_check = jnp.all(jnp.sum(low_temp_samples, axis=-1) == 1) & \
                        jnp.all(jnp.max(low_temp_samples, axis=-1) == 1)
        print("所有行都是one-hot向量:", one_hot_check)

        # 批量采样示例
        key, *subkeys = jax.random.split(key, 10)
        batch_samples = jax.vmap(lambda k: gumbel_softmax(k, logits[0], tau=0.5))(jnp.array(subkeys))
        print("\n批量采样结果 (5次采样):")
        print(batch_samples[:5])


    def test_forward_and_backward():
        # 初始化随机密钥
        key = jax.random.PRNGKey(42)

        # 创建测试 logits
        logits = jnp.array([
            [1.0, 2.0, 3.0],  # 偏向第2类
            [3.0, 2.0, 1.0],  # 偏向第0类
            [0.5, 0.5, 0.5]  # 均匀分布
        ])

        # 计算对应的概率分布
        probs = jax.nn.softmax(logits, axis=-1)

        print("\n=== 前向传播测试 ===")

        # 测试软模式前向传播
        soft_key, _ = jax.random.split(key)
        soft_samples = gumbel_softmax(soft_key, logits, tau=1.0, hard=False)
        print("\n软模式输出 (τ=1.0):")
        print(soft_samples)

        # 验证软模式输出是概率分布
        print("\n软模式输出验证:")
        print("每行和:", jnp.sum(soft_samples, axis=1))  # 应接近1.0
        print("是否在[0,1]区间:", jnp.all(soft_samples >= 0) and jnp.all(soft_samples <= 1))

        # 测试硬模式前向传播
        hard_key, _ = jax.random.split(key)
        hard_samples = gumbel_softmax(hard_key, logits, tau=0.1, hard=True)
        print("\n硬模式输出 (τ=0.1):")
        print(hard_samples)

        # 验证硬模式输出是one-hot向量
        print("\n硬模式输出验证:")
        print("每行和:", jnp.sum(hard_samples, axis=1))  # 应精确为1.0
        print("最大值:", jnp.max(hard_samples, axis=1))  # 应精确为1.0
        print("最小值:", jnp.min(hard_samples, axis=1))  # 应精确为0.0

        print("\n=== 反向传播测试 ===")

        # 定义损失函数
        def loss_fn(logits, key, tau, hard):
            # 使用Gumbel-Softmax采样
            samples = gumbel_softmax(key, logits, tau=tau, hard=hard)

            # 简单损失函数：偏好第0类
            target = jnp.array([1.0, 0.0, 0.0])
            return jnp.mean(jnp.sum(samples * target, axis=-1))

        # 测试软模式反向传播
        print("\n软模式反向传播 (τ=1.0):")
        soft_loss_fn = lambda l, k: loss_fn(l, k, tau=1.0, hard=False)
        soft_grad_fn = jax.grad(soft_loss_fn)

        soft_key, _ = jax.random.split(key)
        soft_gradients = soft_grad_fn(logits, soft_key)
        print("梯度值:")
        print(soft_gradients)

        # 验证软模式梯度非零
        print("\n软模式梯度验证:")
        print("梯度范数:", jnp.linalg.norm(soft_gradients))
        print("是否非零:", jnp.any(jnp.abs(soft_gradients) > 1e-5))

        # 测试硬模式反向传播
        print("\n硬模式反向传播 (τ=0.1):")
        hard_loss_fn = lambda l, k: loss_fn(l, k, tau=0.1, hard=True)
        hard_grad_fn = jax.grad(hard_loss_fn)

        hard_key, _ = jax.random.split(key)
        hard_gradients = hard_grad_fn(logits, hard_key)
        print("梯度值:")
        print(hard_gradients)

        # 验证硬模式梯度非零
        print("\n硬模式梯度验证:")
        print("梯度范数:", jnp.linalg.norm(hard_gradients))
        print("是否非零:", jnp.any(jnp.abs(hard_gradients) > 1e-5))

        # 比较两种模式的梯度
        print("\n软模式与硬模式梯度比较:")
        print("梯度差异范数:", jnp.linalg.norm(soft_gradients - hard_gradients))

        # 测试梯度随温度变化
        print("\n不同温度下的梯度变化:")
        for tau in [2.0, 1.0, 0.5, 0.1]:
            loss_fn_tau = lambda l, k: loss_fn(l, k, tau=tau, hard=False)
            grad_fn_tau = jax.grad(loss_fn_tau)

            tau_key, _ = jax.random.split(key)
            gradients = grad_fn_tau(logits, tau_key)

            print(f"τ={tau}: 梯度范数={jnp.linalg.norm(gradients):.4f}")

        # 测试梯度方向是否正确
        print("\n梯度方向验证:")
        # 增加logits[0]的值应该增加损失（因为我们偏好第0类）
        perturbed_logits = logits.at[0, 0].add(0.1)
        original_loss = loss_fn(logits, key, tau=1.0, hard=False)
        perturbed_loss = loss_fn(perturbed_logits, key, tau=1.0, hard=False)

        print(f"原始损失: {original_loss:.4f}, 扰动后损失: {perturbed_loss:.4f}")
        print("损失变化方向:", "增加" if perturbed_loss > original_loss else "减少")

        # 检查梯度符号
        grad_sign = jnp.sign(soft_gradients[0, 0])
        expected_sign = 1 if perturbed_loss > original_loss else -1
        print(f"梯度符号匹配: {grad_sign == expected_sign}")


    test_gumbel_softmax()
    print("\n\n=== 前向传播与反向传播测试 ===")
    test_forward_and_backward()