# 储存 tile 的种类 及相应的simga
# 在程序开始运行时按照tileHandler注册的tile构建type->sigma的字典
# sigma来自json或其他RVE得到的文件
from typing import List,Dict
import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)


import jax.numpy as np
from jax import vmap
import os
import json
from functools import partial

class SigmaInterpreter:
    def __init__(self,p) -> None:
        self.p:np.array=np.array(p)
        self.Emax= 70.e3
        self.Emin= 1e-3*self.Emax
        self.nu=0.3


    def __call__(self, u_grad, weights, *args, **kwargs):
        nu = self.nu
        Emin= self.Emin
        Emax= self.Emax
        p = self.p
        tau=0.5
        beta=10.
        upper = smooth_heaviside((weights-tau),beta)*weights**p # (tiles,)*(tiles,) ** (tiles )
        weightsDMO = upper/np.sum(upper,axis=-1,keepdims=True)
        # weights = weights / np.linalg.norm(weights, ord=1,axis=-1)  # 归一化
        E = Emin + np.sum((Emax - Emin)*weights**p)
        epsilon = 0.5*(u_grad + u_grad.T)
        eps11 = epsilon[...,0, 0]
        eps22 = epsilon[...,1, 1]
        eps12 = epsilon[...,0, 1]
        sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22) 
        sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
        sig12 = E/(1 + nu)*eps12
        sigma = np.array([[sig11, sig12], [sig12, sig22]])
        return sigma

    # def __call__(self, u_grad, weights, *args, **kwargs):
    #     nu = self.nu
    #     Emin = self.Emin
    #     Emax = self.Emax
    #     p = self.p
        
    #     # 归一化权重
    #     weights = weights / np.linalg.norm(weights, ord=1, axis=-1, keepdims=True)
        
    #     # 计算弹性模量E
    #     # weights形状假设为(..., tiletypes)
    #     # 对最后一个维度求和得到每个点的E
    #     E = Emin + np.sum((Emax - Emin) * weights**p, axis=-1)
        
    #     # 计算应变张量 epsilon = 0.5*(u_grad + u_grad^T)
    #     # 对于2D情况，最后两个维度是(2,2)
    #     # 使用转置交换最后两个轴
    #     u_grad_T = np.swapaxes(u_grad, -1, -2)
    #     epsilon = 0.5 * (u_grad + u_grad_T)
        
    #     # 提取应变分量
    #     eps11 = epsilon[..., 0, 0]
    #     eps22 = epsilon[..., 1, 1]
    #     eps12 = epsilon[..., 0, 1]
        
    #     # 计算应力分量
    #     sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
    #     sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
    #     sig12 = E / (1 + nu) * eps12
        
    #     # 构建应力张量，兼容任意前导维度
    #     # 先获取输出张量的形状
    #     output_shape = eps11.shape + (2, 2)
        
    #     # 创建输出张量
    #     sigma = np.zeros(output_shape, dtype=sig11.dtype)
        
    #     # 填充应力分量
    #     sigma.at[..., 0, 0].set( sig11)
    #     sigma.at[..., 1, 1].set( sig22)
    #     sigma.at[..., 0, 1].set( sig12)
    #     sigma.at[..., 1, 0].set( sig12)  # 对称分量
        
    #     return sigma




    def __repr__(self) -> str:
        line="SigmaInterpreter for 2D problems\n"
        line+=f"p:{self.p}\n"
        line+=f"Emax:{self.Emax}, Emin:{self.Emin}, nu:{self.nu}"
        return line
    

@jax.jit
def heaviside(x: np.ndarray, beta: float = 10.0) -> np.ndarray:
    """
    简化的Heaviside投影函数。
    
    参数:
        x: 输入数组
        beta: 投影锐度参数
    返回:
        投影后的数组
    """
    return np.tanh(beta * 0.5) + np.tanh(beta * (x - 0.5)) / (
        np.tanh(beta * 0.5) + np.tanh(beta * 0.5))


@jax.jit
def smooth_heaviside(x,beta=10.):
    """
    greater the beta, sharper the curve.
    """
    return 1/(1+np.exp(-beta*x))



if __name__ == "__main__":
    sigmaInterpreter=SigmaInterpreter(p=np.array([3.,4.,3.]),) #3,4 445 544 
    print(sigmaInterpreter)
    a=sigmaInterpreter(np.array([[0.01,0.002],[0.002,0.005]]),np.array([0.2,0.5,0.3]))