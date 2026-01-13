from typing import List, Dict
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from functools import partial

class SigmaInterpreter:
    def __init__(self, p, Emax=70.e3, nu=0.3, tau=0.5, beta=10.0) -> None:
        self.p: np.ndarray = np.array(p)
        self.Emax = Emax
        self.Emin = 1e-3 * self.Emax
        self.nu = nu
        self.tau = tau
        self.beta = beta

    def __call__(self, u_grad, weights, *args, **kwargs):
        """
        计算各向同性材料应力
        输入维度：
        - u_grad: (..., 3, 3) 位移梯度张量
        - weights: (..., n_materials) 材料权重
        输出维度：
        - sigma: (..., 3, 3) 应力张量
        """
        # 1. 提取基本参数
        nu = self.nu
        Emin = self.Emin
        Emax = self.Emax
        p = self.p
        tau = self.tau
        beta = self.beta
        
        # 2. 计算应变张量（保持原有维度）
        u_grad_T = np.swapaxes(u_grad, -1, -2)
        epsilon = 0.5 * (u_grad + u_grad_T)  # (..., 3, 3)
        
        # 3. 权重处理 - 兼容高维输入
        # weights形状：(..., n_materials)
        upper = smooth_heaviside((weights - tau), beta) * (weights ** p)
        weights_sum = np.sum(upper, axis=-1, keepdims=True)
        weightsDMO = upper / (weights_sum + 1e-12)  # 防止除零
        
        # 4. 计算等效弹性模量
        # weightsDMO: (..., n_materials)
        # 对材料维度求和，得到每个位置的等效模量
        # weight_sum = np.sum(weightsDMO, axis=-1)  # (...)
        E = Emin + np.sum((Emax - Emin) * weights**p ) # (...)
        
        # 5. 计算拉梅常数
        # 为后续计算方便，扩展E的维度
        E = E[..., np.newaxis, np.newaxis]  # (..., 1, 1)
        
        mu = E / (2.0 * (1.0 + nu))  # 剪切模量 (..., 1, 1)
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # 拉梅第一参数 (..., 1, 1)
        
        # 6. 计算应变张量的迹
        trace_epsilon = np.trace(epsilon, axis1=-2, axis2=-1)  # (...)
        trace_epsilon = trace_epsilon[..., np.newaxis, np.newaxis]  # (..., 1, 1)
        
        # 7. 计算应力张量 (各向同性)
        # sigma_ij = lambda * trace(epsilon) * delta_ij + 2 * mu * epsilon_ij
        identity = np.eye(3)  # (3, 3)
        
        # 维度扩展以匹配广播
        lam_trace = lam * trace_epsilon  # (..., 1, 1)
        mu_epsilon = 2.0 * mu * epsilon  # (..., 3, 3)
        
        # 添加单位张量的贡献
        # 注意：lam_trace * identity 会广播为 (..., 3, 3)
        sigma = lam_trace * identity + mu_epsilon
        
        return sigma
    
    def stress_isotropic_vectorized(self, epsilon, E):
        """
        向量化的各向同性应力计算函数
        输入：
        - epsilon: (..., 3, 3) 应变张量
        - E: (...) 弹性模量
        输出：
        - sigma: (..., 3, 3) 应力张量
        """
        nu = self.nu
        
        # 扩展维度以进行广播
        E_expanded = E[..., np.newaxis, np.newaxis]  # (..., 1, 1)
        
        # 计算拉梅常数
        mu = E_expanded / (2.0 * (1.0 + nu))  # 剪切模量
        lam = E_expanded * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # 拉梅第一参数
        
        # 计算应变张量的迹
        trace_epsilon = np.trace(epsilon, axis1=-2, axis2=-1)  # (...)
        trace_epsilon = trace_epsilon[..., np.newaxis, np.newaxis]  # (..., 1, 1)
        
        # 计算应力
        identity = np.eye(3)
        sigma = lam * trace_epsilon * identity + 2.0 * mu * epsilon
        
        return sigma

    def __repr__(self) -> str:
        return f"""SigmaInterpreter (3D Isotropic)
p: {self.p}
Emax: {self.Emax}, Emin: {self.Emin}, nu: {self.nu}
tau: {self.tau}, beta: {self.beta}"""


@jax.jit
def smooth_heaviside(x, beta=10.0):
    """
    平滑阶跃函数
    输入: 任意形状
    输出: 相同形状
    """
    return 1.0 / (1.0 + np.exp(-beta * x))


# 测试函数
def test_high_dimensional():
    """测试高维输入"""
    interpreter = SigmaInterpreter(p=[3.0, 4.0, 3.0])
    
    # 测试用例1: 单个点
    print("测试用例1: 单个点")
    u_grad = np.array([
        [0.01, 0.002, 0.001],
        [0.002, 0.005, 0.003],
        [0.001, 0.003, 0.004]
    ])
    weights = np.array([0.2, 0.5, 0.3])
    sigma = interpreter(u_grad, weights)
    print(f"位移梯度形状: {u_grad.shape}")
    print(f"权重形状: {weights.shape}")
    print(f"应力形状: {sigma.shape}")
    print(f"应力:\n{sigma}")
    print()
    
    # 测试用例2: 批量点 (cells, 3, 3)
    print("测试用例2: 批量点 (cells=2)")
    u_grad_batch = np.array([
        [[0.01, 0.002, 0.001],
         [0.002, 0.005, 0.003],
         [0.001, 0.003, 0.004]],
        
        [[0.015, 0.001, 0.002],
         [0.001, 0.004, 0.002],
         [0.002, 0.002, 0.003]]
    ])
    weights_batch = np.array([
        [0.2, 0.5, 0.3],
        [0.1, 0.6, 0.3]
    ])
    sigma_batch = interpreter(u_grad_batch, weights_batch)
    print(f"位移梯度形状: {u_grad_batch.shape}")
    print(f"权重形状: {weights_batch.shape}")
    print(f"应力形状: {sigma_batch.shape}")
    print()
    
    # 测试用例3: 更高维 (cells, quads, 3, 3)
    print("测试用例3: 更高维 (cells=2, quads=3)")
    n_cells = 2
    n_quads = 3
    n_materials = 3
    
    # 生成随机但合理的数据
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)
    
    u_grad_high = jax.random.normal(key1, (n_cells, n_quads, 3, 3)) * 0.01
    weights_high = jax.random.uniform(key2, (n_cells, n_quads, n_materials))
    # 归一化权重
    weights_high = weights_high / np.sum(weights_high, axis=-1, keepdims=True)
    
    sigma_high = interpreter(u_grad_high, weights_high)
    print(f"位移梯度形状: {u_grad_high.shape}")
    print(f"权重形状: {weights_high.shape}")
    print(f"应力形状: {sigma_high.shape}")
    
    # 检查对称性
    sigma_T = np.swapaxes(sigma_high, -1, -2)
    symmetry_error = np.max(np.abs(sigma_high - sigma_T))
    print(f"应力对称性最大误差: {symmetry_error:.2e}")
    
    return sigma_high


if __name__ == "__main__":
    interpreter = SigmaInterpreter(p=[3.0, 4.0, 3.0])
    print(interpreter)
    print("\n" + "="*50 + "\n")
    
    # 运行测试
    sigma_result = test_high_dimensional()