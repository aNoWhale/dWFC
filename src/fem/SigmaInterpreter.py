# 储存 tile 的种类 及相应的simga
# 在程序开始运行时按照tileHandler注册的tile构建type->sigma的字典
# sigma来自json或其他RVE得到的文件
from typing import List,Dict
import jax
import jax.numpy as np
from jax import vmap
import os
import json
from functools import partial

class SigmaInterpreter:
    def __init__(self,typeList:List,folderPath:str=None,*args,**kwargs) -> None:
        """SigmaInterpreter class was designed to provide sigma for fem

        Args:
            typeList (List): the typelist from TileHandler
            folderPath (str): path where has propretries, the fielename should be the same as the type
        """
        self.typeList = typeList
        self.folderPath = folderPath
        self.C_dict:Dict[str,np.ndarray] = {}
        self.C:np.ndarray=None
        self.debug=kwargs.get("debug",False)
        if not self.debug:
            self._buildCDict() 
            self._buildCache() 
    

    
    # @partial(jax.jit, static_argnames=())
    def __call__(self, u_grad, weights, *args, **kwargs):
        # if self.debug:
        #     return stress(u_grad)

        # 1. 概率 → 标量密度（外部顺序）
        x_e = np.dot(weights, self.a_aux[self.inv_order])  # 用排序后的辅助坐标

        # 2. 区间选择（在排序空间进行）
        k = np.searchsorted(self.a_aux, x_e, side='right') - 1
        k = np.clip(k, 0, self.a_aux.shape[0] - 2)

        # 3. 映射回原始索引
        orig_k = self.inv_order[k]
        orig_k1 = self.inv_order[k + 1]

        # 4. 局部线性坐标 + Ordered-RAMP
        beta = 1.  #刚才是1
        a_k, a_k1 = self.a_aux[k], self.a_aux[k + 1]
        xi = (x_e - a_k) / (a_k1 - a_k + 1e-12)
        xi_p = xi / (1 + beta * (1 - xi))

        # 5. 矩阵插值（用原始顺序的 C）
        C_k = self.C[orig_k]
        C_k1 = self.C[orig_k1]
        C_eff = C_k + xi_p[...,None,None] * (C_k1 - C_k)

        return stress_anisotropic(C_eff, u_grad)

    def __repr__(self) -> str:
        if not hasattr(self, "order"):
            return "<SigmaInterpreter: 尚未初始化缓存>"

        header = f"{'idx':>3} {'type':<12} {'order':>5} {'|C|':>10}"
        bar = "-" * len(header)
        lines = [header, bar]

        c_norm = np.linalg.norm(self.C, axis=(1, 2))
        for ext_idx, typ in enumerate(self.typeList):
            # 安全地把“外部索引”映射到“内部排序序号”
            int_idx = int(np.asarray(self.order == ext_idx).argmax())
            lines.append(f"{ext_idx:>3} {typ:<12} {int_idx:>5} {c_norm[ext_idx]:>10.3e}")

        return "\n".join(lines)
    
    
    def _buildCDict(self):
        C_list = []
        for mat_type in self.typeList:
            file_path = os.path.join(self.folderPath, f"{mat_type}.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                C_j = np.array(data)
                C_list.append(C_j)
                print(f"Loaded C for * {mat_type} * from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        self.C = np.array(C_list)  # 保持外部顺序

    def _buildCache(self):
        # 1. 计算代表标量（外部顺序）
        c_j = np.linalg.norm(self.C, axis=(1, 2))

        # 2. 生成排序索引（升序）
        self.order = np.argsort(c_j)  # 内部用：单调索引 按c_j从小到大排序，索引跟C一样
        self.inv_order = np.argsort(self.order)  # 回映射：内部k -> 外部idx

        # 3. 生成排序后的辅助坐标（仅用于区间查找）
        self.a_aux = (c_j[self.order] - c_j[self.order][0]) / (
            c_j[self.order][-1] - c_j[self.order][0]+1e-12
        )


def stress( u_grad, *args, **kwargs):
    Emax = 3.5e9   # 杨氏模量 [Pa] (3.5 GPa)
    nu = 0.36      # 泊松比
    E = Emax
    # 计算应变张量
    epsilon = 0.5 * (u_grad + u_grad.T)
    # 计算材料参数
    mu = E / (2 * (1 + nu))        # 剪切模量
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # 拉梅第一参数
    # 计算应变张量的迹
    trace_epsilon = epsilon[0, 0] + epsilon[1, 1] + epsilon[2, 2]
    # 初始化应力张量
    sigma = np.zeros((3, 3))
    
    diag_indices = np.diag_indices(3)
    sigma = sigma.at[diag_indices].set(lam * trace_epsilon + 2 * mu * epsilon[diag_indices])
    
    triu_indices = np.triu_indices(3, k=1)
    sigma = sigma.at[triu_indices].set(2 * mu * epsilon[triu_indices])
    
    tril_indices = np.tril_indices(3, k=-1)
    sigma = sigma.at[tril_indices].set(2 * mu * epsilon[tril_indices])
    # print(f"sigma:{sigma}")
    return sigma

def stress_anisotropic( C, u_grad, *args, **kwargs):
    """
    计算三维各向异性材料的应力张量
    
    参数:
    u_grad: 位移梯度张量 (3x3 numpy数组)
    
    返回:
    sigma: 应力张量 (3x3 numpy数组)
    """
    # 计算应变张量
    u_grad_t = np.transpose(u_grad, axes=(*range(u_grad.ndim-2), -1, -2))  # 保持前n-2维不变，交换最后两维
    epsilon = 0.5 * (u_grad + u_grad_t)
    # 将应变张量转换为Voigt符号表示 (6x1向量)
    # Voigt符号: [ε11, ε22, ε33, 2ε23, 2ε13, 2ε12]
    epsilon_voigt = np.stack([
        epsilon[..., 0, 0],  # ε11，形状: (...)
        epsilon[..., 1, 1],  # ε22
        epsilon[..., 2, 2],  # ε33
        2 * epsilon[..., 1, 2],  # 2ε23
        2 * epsilon[..., 0, 2],  # 2ε13
        2 * epsilon[..., 0, 1]   # 2ε12
    ], axis=-1)  # 堆叠为 (..., 6)，最后一维为Voigt分量
    
    # 计算应力向量 (Voigt符号)
    # sigma_voigt = np.dot(C, epsilon_voigt)
    sigma_voigt = np.einsum("...ij,...j->...i",C,epsilon_voigt)
    
    # 将应力向量转换回张量形式
    # 初始化应力张量: (..., 3, 3)
    sigma = np.zeros((*epsilon.shape[:-2], 3, 3), dtype=sigma_voigt.dtype)
    
    # 填充对角元素
    sigma = sigma.at[..., 0, 0].set(sigma_voigt[..., 0])  # σ11
    sigma = sigma.at[..., 1, 1].set(sigma_voigt[..., 1])  # σ22
    sigma = sigma.at[..., 2, 2].set(sigma_voigt[..., 2])  # σ33
    
    # 填充对称非对角元素
    sigma = sigma.at[..., 1, 2].set(sigma_voigt[..., 3])  # σ23
    sigma = sigma.at[..., 2, 1].set(sigma_voigt[..., 3])  # σ32（对称）
    sigma = sigma.at[..., 0, 2].set(sigma_voigt[..., 4])  # σ13
    sigma = sigma.at[..., 2, 0].set(sigma_voigt[..., 4])  # σ31（对称）
    sigma = sigma.at[..., 0, 1].set(sigma_voigt[..., 5])  # σ12
    sigma = sigma.at[..., 1, 0].set(sigma_voigt[..., 5])  # σ21（对称）
    return sigma
  