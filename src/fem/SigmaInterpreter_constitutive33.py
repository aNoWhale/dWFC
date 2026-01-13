from typing import List, Dict
import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)

import jax.numpy as np
from jax import vmap
import os
import json
from functools import partial

class SigmaInterpreter:
    def __init__(self, p) -> None:
        self.p: np.array = np.array(p)
        self.Emax = 70.e3  # 弹性模量上限 (MPa)
        self.Emin = 1e-3 * self.Emax  # 弹性模量下限
        self.nu = 0.3  # 泊松比

    def __call__(self, u_grad, weights, *args, **kwargs):
        nu = self.nu
        Emin = self.Emin
        Emax = self.Emax
        p = self.p
        tau = 0.5
        beta = 10.
        
        # 权重平滑与归一化（保持原有逻辑）
        upper = smooth_heaviside((weights - tau), beta) * weights**p
        weightsDMO = upper / np.sum(upper, axis=-1, keepdims=True)
    

        # 计算当前弹性模量（权重加权后的有效模量）
        E = Emin + np.sum((Emax - Emin) * weightsDMO,axis=-1)
        # # 扩展E到(...,1,1)，适配后续3x3张量的广播
        # E = E[..., np.newaxis, np.newaxis]
        
        # 提取三维应变张量（对称化）
        u_grad_T=np.swapaxes(u_grad,-1,-2)
        epsilon = 0.5 * (u_grad + u_grad_T)  # 3x3 应变张量，保证对称性
        # jax.debug.print("epsilon.shape{a}",a=epsilon.shape)
        eps11 = epsilon[..., 0, 0]
        eps22 = epsilon[..., 1, 1]
        eps33 = epsilon[..., 2, 2]
        eps12 = epsilon[..., 0, 1]
        eps13 = epsilon[..., 0, 2]
        eps23 = epsilon[..., 1, 2]
        
        # 三维各向同性材料胡克定律
        # 体积模量与剪切模量组合形式，避免数值奇异
        factor1 = E / ((1 + nu) * (1 - 2 * nu))  # 正应力项系数
        factor2 = E / (1 + nu)  # 切应力项系数
        
        # 正应力分量
        sig11 = factor1 * ((1 - nu) * eps11 + nu * (eps22 + eps33))
        sig22 = factor1 * ((1 - nu) * eps22 + nu * (eps11 + eps33))
        sig33 = factor1 * ((1 - nu) * eps33 + nu * (eps11 + eps22))
        
        # 切应力分量
        sig12 = factor2 * eps12
        sig13 = factor2 * eps13
        sig23 = factor2 * eps23
        # # jax.debug.print("sig12.shape{a}",a=sig12.shape)
        # # 构造3x3 应力张量
        # # 修复3：改用stack保证维度顺序为(...,3,3)（原方式是(3,3,...)）
        # sigma = np.zeros((*epsilon.shape[:-2], 3, 3))
        # sigma = sigma.at[...,0,0].set(sig11)
        # sigma = sigma.at[...,0,1].set(sig12)
        # sigma = sigma.at[...,0,2].set(sig13)

        # sigma = sigma.at[...,1,0].set(sig12)
        # sigma = sigma.at[...,1,1].set(sig22)
        # sigma = sigma.at[...,1,2].set(sig23)

        # sigma = sigma.at[...,2,0].set(sig13)
        # sigma = sigma.at[...,2,1].set(sig23)
        # sigma = sigma.at[...,2,2].set(sig33)
        sigma = np.stack([np.stack([sig11,sig12,sig13],axis=-1),
                          np.stack([sig12,sig22,sig23],axis=-1),
                          np.stack([sig13,sig23,sig33],axis=-1)],axis=-2)

        return sigma

    def __repr__(self) -> str:
        line = "SigmaInterpreter for 3D problems\n"
        line += f"p:{self.p}\n"
        line += f"Emax:{self.Emax}, Emin:{self.Emin}, nu:{self.nu}"
        return line

@jax.jit
def heaviside(x: np.ndarray, beta: float = 10.0) -> np.ndarray:
    """
    简化的Heaviside投影函数（保留原有实现）
    """
    return np.tanh(beta * 0.5) + np.tanh(beta * (x - 0.5)) / (
        np.tanh(beta * 0.5) + np.tanh(beta * 0.5))

@jax.jit
def smooth_heaviside(x, beta=10.):
    """
    平滑阶跃函数（保留原有实现）
    """
    return 1 / (1 + np.exp(-beta * x))

if __name__ == "__main__":
    # 初始化解释器（权重维度与维度无关）
    sigmaInterpreter = SigmaInterpreter(p=np.array([3., 4., 3.]))
    print(sigmaInterpreter)
    
    # 三维位移梯度张量（3x3）
    u_grad_3d = np.array([
        [0.01, 0.002, 0.001],
        [0.002, 0.005, 0.003],
        [0.001, 0.003, 0.004]
    ])
    
    # 权重数组（与原测试一致）
    weights = np.array([0.2, 0.5, 0.3])
    
    # 计算三维应力张量
    sigma_3d = sigmaInterpreter(u_grad_3d, weights)
    print("\n三维应力张量:")
    print(sigma_3d)