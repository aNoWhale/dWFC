from typing import List, Dict
import jax
import jax.numpy as np
from jax import vmap
import os
import json

class SigmaInterpreter:
    def __init__(self, typeList: List, folderPath: str = None, *args, **kwargs) -> None:
        """SigmaInterpreter class was designed to provide sigma for fem

        Args:
            typeList (List): the typelist from TileHandler
            folderPath (str): path where has properties, the filename should be the same as the type
        """
        self.typeList = typeList
        self.folderPath = folderPath
        self.C_dict: Dict[str, np.ndarray] = {}
        self.C: np.ndarray = None
        self.debug = kwargs.get("debug", False)
        if not self.debug:
            self._buildCDict()

    def __call__(self, u_grad, weights, *args, **kwargs) -> None:
        if self.debug:
            return stress(u_grad)
        else:
            p = 3  # 惩罚因子
            q = 1  # RAMP形状参数
            eps = 1e-6
            Cmin = eps * np.max(self.C)  # 最小刚度（避免奇异性）
            
            # 计算权重的p次方
            wm = np.power(weights, p)
            
            # ---------------------- ordered-RAMP 核心修改 ----------------------
            # 1. 获取权重排序索引（升序）
            # 对最后一个维度（材料/单元维度）进行排序
            sort_indices = np.argsort(wm, axis=-1)
            
            # 2. 按索引对权重进行排序
            sorted_wm = np.take_along_axis(wm, sort_indices, axis=-1)
            
            # 3. 对排序后的权重应用RAMP插值函数
            sorted_ramp = sorted_wm / (1 + q * (1 - sorted_wm))  # ordered-RAMP插值公式
            
            # 4. 将排序后的插值结果还原到原始顺序
            ramp_factor = np.zeros_like(sorted_ramp)
            ramp_factor = np.put_along_axis(ramp_factor, sort_indices, sorted_ramp, axis=-1)
            # -------------------------------------------------------------------
            
            # 计算插值后的刚度矩阵
            C = Cmin + np.einsum("...n,...nij->...ij", ramp_factor, (self.C - Cmin))
            return stress_anisotropic(C, u_grad)

    def _buildCDict(self):
        C = []
        for mat_type in self.typeList:
            file_path = os.path.join(self.folderPath, f"{mat_type}.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                self.C_dict[mat_type] = np.array(data)
                C.append(data)
                print(f"Loaded C for * {mat_type} * from {file_path}")
                
            except FileNotFoundError:
                print(f"Warning: File not found for type * {mat_type} * at {file_path}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        self.C = np.array(C)

def stress(u_grad, *args, **kwargs):
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
    return sigma

def stress_anisotropic(C, u_grad, *args, **kwargs):
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
    sigma_voigt = np.einsum("...ij,...j->...i", C, epsilon_voigt)
    
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
