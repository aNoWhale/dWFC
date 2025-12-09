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

    # @partial(jax.jit, static_argnames=())
    def __call__(self, u_grad, weights, *args, **kwargs):
        nu = self.nu
        Emin= self.Emin
        Emax= self.Emax
        p = self.p
        weights = weights / np.linalg.norm(weights, ord=1,axis=-1)  # 归一化
        E = Emin + np.sum((Emax - Emin)*weights**p)
        epsilon = 0.5*(u_grad + u_grad.T)
        eps11 = epsilon[0, 0]
        eps22 = epsilon[1, 1]
        eps12 = epsilon[0, 1]
        sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22) 
        sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
        sig12 = E/(1 + nu)*eps12
        sigma = np.array([[sig11, sig12], [sig12, sig22]])
        return sigma

    def __repr__(self) -> str:
        line="SigmaInterpreter for 2D problems\n"
        line+=f"p:{self.p}\n"
        line+=f"Emax:{self.Emax}, Emin:{self.Emin}, nu:{self.nu}"
        return line
if __name__ == "__main__":
    sigmaInterpreter=SigmaInterpreter(p=np.array([3.,4.,3.]),) #3,4 445 544 
    print(sigmaInterpreter)
    a=sigmaInterpreter(np.array([[0.01,0.002],[0.002,0.005]]),np.array([0.2,0.5,0.3]))