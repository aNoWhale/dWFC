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
    def __init__(self, typeList:List,folderPath:str=None,*args,** kwargs) -> None:
        self.typeList = typeList
        self.folderPath = folderPath
        self.EVG:np.ndarray=None # array [['E11', 'E22', 'E33', 'V12', 'V13', 'V23', 'V21', 'V31', 'V32', 'G12', 'G13', 'G23'],[...],...]
        self.debug=kwargs.get("debug",False)
        if not self.debug:
            self._buildEVG()  # 构建包含void的刚度矩阵列表
            self.Cp_list=vmap(simp_stiffness_matrix,in_axes=(0,0,None))
    

    # @partial(jax.jit, static_argnames=())
    def __call__(self, u_grad, weights, *args, **kwargs):
        if self.debug:
            return stress(u_grad, weights.squeeze(axis=-1))
        
        Cp = self.Cp_list(self.EVG, weights,3.)
        C_eff = np.sum(Cp,axis=-3,keepdims=False)
        return stress_anisotropic(C_eff, u_grad)

    def __repr__(self) -> str:
        if not self.debug:
            header = f"{'idx':>3}  {'type':>5}"
            bar = "-" * len(header)
            lines = [header, bar]
            for ext_idx, typ in enumerate(self.typeList):
                # 安全地把“外部索引”映射到“内部排序序号”
                lines.append(f"{ext_idx:>3}  {typ:<12}")
            return "\n".join(lines)
        if self.debug:
            return "debug mode"
    

    def _buildEVG(self):
        EVG_list = []
        required_keys = ['E11', 'E22', 'E33', 'V12', 'V13', 'V23', 'V21', 'V31', 'V32', 'G12', 'G13', 'G23']
        for mat_type in self.typeList:
            file_path = os.path.join(self.folderPath, f"{mat_type}.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for key in required_keys:
                        if key not in data:
                            raise ValueError(f"JSON数据缺少必要参数: {key},at {file_path}")
                    # 提取材料参数
                    E11, E22, E33 = data['E11'], data['E22'], data['E33']
                    V12, V13, V23 = data['V12'], data['V13'], data['V23']
                    V21, V31, V32 = data['V21'], data['V31'], data['V32']
                    G12, G13, G23 = data['G12'], data['G13'], data['G23']
                    EVG = np.array([E11,E22,E33,V12,V13,V23,V21,V31,V32,G12,G13,G23])
                    EVG_list.append(EVG)
                    print(f"Loaded EVG for * {mat_type} * from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        self.EVG = np.array(EVG_list)  # 形状：(tileNum+1, 6, 6)（含void）


def stress( u_grad, theta, *args, **kwargs):
    Emax = 70e3  # 杨氏模量 [Pa] (3.5 GPa)
    nu = 0.3      # 泊松比
    Emin= 1e-3*Emax
    penal=3
    E = Emin + (Emax - Emin)*theta**penal
    # 计算应变张量
    u_grad_t = np.transpose(u_grad, axes=(*range(u_grad.ndim-2), -1, -2))
    epsilon = 0.5 * (u_grad + u_grad_t)
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
  

def simp_stiffness_matrix(EVG:np.ndarray, rho, p=3):
    """
    对E参数应用SIMP惩罚并构建正交各向异性材料的刚度矩阵
    
    参数:
        EVG : array ['E11', 'E22', 'E33', 'V12', 'V13', 'V23', 'V21', 'V31', 'V32', 'G12', 'G13', 'G23']
        rho (float): 密度变量，范围[0,1]
        p (float): SIMP惩罚因子，默认3
    
    返回:
        np.ndarray: 6x6的刚度矩阵
    """
    # 检查输入参数完整性
    E11=EVG[0]
    E22=EVG[1]
    E33=EVG[2]
    V12=EVG[3]
    V13=EVG[4]
    V23=EVG[5]
    V21=EVG[6]
    V31=EVG[7]
    V32=EVG[8]
    G12=EVG[9]
    G13=EVG[10]
    G23=EVG[11]
    
    # SIMP惩罚：对弹性模量E应用ρ^p惩罚
    E11_p = (rho ** p) * E11  # 惩罚后的E11
    E22_p = (rho ** p) * E22  # 惩罚后的E22
    E33_p = (rho ** p) * E33  # 惩罚后的E33
    G12_p = (rho ** p) * G12
    G13_p = (rho ** p) * G13
    G23_p = (rho ** p) * G23


    # 构建6x6柔度矩阵S（正交各向异性材料）
    S = np.zeros((6, 6), dtype=np.float64)
    # 正应力-正应变关系（前3x3块）
    S=S.at[0, 0].set( 1 / E11_p)          # S11
    S=S.at[0, 1].set( -V12 / E11_p)       # S12
    S=S.at[0, 2].set( -V13 / E11_p )      # S13
    S=S.at[1, 0].set( -V21 / E22_p )      # S21（应与S12对称）
    S=S.at[1, 1].set( 1 / E22_p )         # S22
    S=S.at[1, 2].set( -V23 / E22_p)       # S23
    S=S.at[2, 0].set( -V31 / E33_p )      # S31（应与S13对称）
    S=S.at[2, 1].set( -V32 / E33_p  )     # S32（应与S23对称）
    S=S.at[2, 2].set( 1 / E33_p )         # S33
    # 剪切应力-剪切应变关系（对角项）
    S=S.at[3, 3].set( 1 / G23 ) # S44（对应2-3方向剪切）
    S=S.at[4, 4].set( 1 / G13 ) # S55（对应1-3方向剪切）
    S=S.at[5, 5].set( 1 / G12 ) # S66（对应1-2方向剪切）
    
    # 强制柔度矩阵对称（消除输入误差）
    S = (S + S.T) / 2
    
    # 刚度矩阵 = 柔度矩阵的逆 ,如果奇异的话没有逆就会出问题
    stiffness_matrix = np.linalg.inv(S)
    
    return stiffness_matrix


