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
        self.p= np.array(kwargs.pop('p',np.array([3.]*len(self.typeList))))
        if not self.debug:
            self._buildEVG()  # 构建包含void的刚度矩阵列表
    

    # @partial(jax.jit, static_argnames=())
    def __call__(self, u_grad, weights, *args, **kwargs):
        if self.debug:
            return stress(u_grad, weights.squeeze(axis=-1))
        

        Cp_list = simp_stiffness_matrix(self.EVG,weights,self.p)
        # Cp_list = hpdmo_stiffness_matrix(self.EVG,weights,self.p,*args)

        C_eff = np.sum(Cp_list,axis=-3,keepdims=False)
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
  

def simp_stiffness_matrix(EVG:np.ndarray, rho, p):
    """
    对E参数应用SIMP惩罚并构建正交各向异性材料的刚度矩阵
    
    参数:
        EVG : array ['E11', 'E22', 'E33', 'V12', 'V13', 'V23', 'V21', 'V31', 'V32', 'G12', 'G13', 'G23']
        rho (float): 密度变量，范围[0,1]
        p (float): SIMP惩罚因子，默认3
    
    返回:
        np.ndarray: 6x6的刚度矩阵
    """
    q=p
    
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
    eps = 1e-10

    E11_min=eps*E11
    E22_min=eps*E22
    E33_min=eps*E33
    V12_min=eps*V12
    V13_min=eps*V13
    V23_min=eps*V23
    V21_min=eps*V21
    V31_min=eps*V31
    V32_min=eps*V32
    G12_min=eps*G12
    G13_min=eps*G13
    G23_min=eps*G23

    # SIMP惩罚：对弹性模量E应用ρ^p惩罚
    E11_p = E11_min + (rho ** p) * (E11-E11_min) 
    E22_p = E22_min + (rho ** p) * (E22-E22_min) 
    E33_p = E33_min + (rho ** p) * (E33-E33_min)
    G12_p = G12_min + (rho ** p) * (G12-G12_min)
    G13_p = G13_min + (rho ** p) * (G13-G13_min)
    G23_p = G23_min + (rho ** p) * (G23-G23_min)
    V12_p = V12_min + (rho ** q) * (V12-V12_min)
    V13_p = V13_min + (rho ** q) * (V13-V13_min)
    V23_p = V23_min + (rho ** q) * (V23-V23_min)
    V21_p = V21_min + (rho ** q) * (V21-V21_min)
    V31_p = V31_min + (rho ** q) * (V31-V31_min)
    V32_p = V32_min + (rho ** q) * (V32-V32_min)

    return compose_stiffness_matrix(np.array([E11_p,E22_p,E33_p,V12_p,V13_p,V23_p,V21_p,V31_p,V32_p,G12_p,G13_p,G23_p]))


def hpdmo_stiffness_matrix(EVGs:np.ndarray, rhos, ps, beta=3):
    tau=0.5
    upper = smooth_heaviside((rhos-tau),beta)*rhos**ps # (tiles,)*(tiles,) ** (tiles )
    # weightsDMO = upper/np.sum(upper,axis=-1,keepdims=False)
    weightsDMO  = upper
    #EVGs (tiles,12)
    E11=EVGs[...,0][:,None] #(tiles, 1)
    E22=EVGs[...,1][:,None]
    E33=EVGs[...,2][:,None]
    V12=EVGs[...,3][:,None]
    V13=EVGs[...,4][:,None]
    V23=EVGs[...,5][:,None]
    V21=EVGs[...,6][:,None]
    V31=EVGs[...,7][:,None]
    V32=EVGs[...,8][:,None]
    G12=EVGs[...,9][:,None]
    G13=EVGs[...,10][:,None]
    G23=EVGs[...,11][:,None]
    eps = 1e-10

    E11_min=eps*E11 #(tiles, )
    E22_min=eps*E22
    E33_min=eps*E33
    V12_min=eps*V12
    V13_min=eps*V13
    V23_min=eps*V23
    V21_min=eps*V21
    V31_min=eps*V31
    V32_min=eps*V32
    G12_min=eps*G12
    G13_min=eps*G13
    G23_min=eps*G23

    # SIMP惩罚：对弹性模量E应用ρ^p惩罚
    E11_p = E11_min + (weightsDMO[:,None]) * (E11-E11_min) # #(tiles, 1)+ (tiles, 1) * (tiles, 1)
    E22_p = E22_min + (weightsDMO[:,None]) * (E22-E22_min) 
    E33_p = E33_min + (weightsDMO[:,None]) * (E33-E33_min)
    G12_p = G12_min + (weightsDMO[:,None]) * (G12-G12_min)
    G13_p = G13_min + (weightsDMO[:,None]) * (G13-G13_min)
    G23_p = G23_min + (weightsDMO[:,None]) * (G23-G23_min)
    V12_p = V12_min + (weightsDMO[:,None]) * (V12-V12_min)
    V13_p = V13_min + (weightsDMO[:,None]) * (V13-V13_min)
    V23_p = V23_min + (weightsDMO[:,None]) * (V23-V23_min)
    V21_p = V21_min + (weightsDMO[:,None]) * (V21-V21_min)
    V31_p = V31_min + (weightsDMO[:,None]) * (V31-V31_min)
    V32_p = V32_min + (weightsDMO[:,None]) * (V32-V32_min)
    EVGs_p= np.concatenate([E11_p,E22_p,E33_p,V12_p,V13_p,V23_p,V21_p,V31_p,V32_p,G12_p,G13_p,G23_p],axis=-1) #(tiles, 12)
    return compose_stiffness_matrix(EVGs_p,EVGs_p.shape[0])


def compose_stiffness_matrix(EVGs:np.ndarray,tiles=1):
    # 构建6x6柔度矩阵S（正交各向异性材料）
    S = np.zeros((tiles,6, 6), dtype=np.float64)
    E11=EVGs[...,0] #(tiles, )
    E22=EVGs[...,1]
    E33=EVGs[...,2]
    V12=EVGs[...,3]
    V13=EVGs[...,4]
    V23=EVGs[...,5]
    V21=EVGs[...,6]
    V31=EVGs[...,7]
    V32=EVGs[...,8]
    G12=EVGs[...,9]
    G13=EVGs[...,10]
    G23=EVGs[...,11]
    # 正应力-正应变关系（前3x3块）
    S=S.at[...,0, 0].set( 1 / E11)          # S11
    S=S.at[...,0, 1].set( -V12 / E11)       # S12
    S=S.at[...,0, 2].set( -V13 / E11 )      # S13
    S=S.at[...,1, 0].set( -V21 / E22 )      # S21（应与S12对称）
    S=S.at[...,1, 1].set( 1 / E22 )         # S22
    S=S.at[...,1, 2].set( -V23 / E22)       # S23
    S=S.at[...,2, 0].set( -V31 / E33 )      # S31（应与S13对称）
    S=S.at[...,2, 1].set( -V32 / E33  )     # S32（应与S23对称）
    S=S.at[...,2, 2].set( 1 / E33 )         # S33
    # 剪切应力-剪切应变关系（对角项）
    S=S.at[...,3, 3].set( 1 / G23 ) # S44（对应2-3方向剪切）
    S=S.at[...,4, 4].set( 1 / G13 ) # S55（对应1-3方向剪切）
    S=S.at[...,5, 5].set( 1 / G12 ) # S66（对应1-2方向剪切）
    
    # 强制柔度矩阵对称（消除输入误差）
    ST = np.transpose(S,[0,2,1])
    S = (S + ST) / 2
    
    # 刚度矩阵 = 柔度矩阵的逆 ,如果奇异的话没有逆就会出问题
    stiffness_matrix = jax.vmap(lambda s:np.linalg.inv(s),)(S)
    
    return stiffness_matrix


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
    return 1/1+np.exp(-beta*x)