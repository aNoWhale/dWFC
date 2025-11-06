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
        self.C_dict:Dict[str,np.ndarray] = {}
        self.C:np.ndarray=None  # 包含用户材料 + void材料
        self.debug=kwargs.get("debug",False)
        self.void = np.array(void_C(E0=1,nu=0.3,eps=1e-9))  # void的刚度矩阵
        
        # 关键：在内部材料列表中添加"void"作为隐含材料（用户无需传入）
        self.internal_typeList = self.typeList + ["void"]  # 内部材料列表：用户材料 + void
        
        if not self.debug:
            self._buildCDict()  # 构建包含void的刚度矩阵列表
            self._buildCache()  # 排序包含void的材料
    

    
    # @partial(jax.jit, static_argnames=())
    def __call__(self, u_grad, weights, *args, **kwargs):
        if self.debug:
            return stress(u_grad)
        
        # 1. 自动计算void概率：1 - 其他材料概率总和（确保权重和为1）
        sum_weights = np.sum(weights, axis=-1, keepdims=True)  # 外部材料概率和：(1)
        void_weight = 1.0 - sum_weights  # void概率：(n,1)
        void_weight = np.clip(void_weight, 0.0, 1.0)  # 避免负值
        full_weights = np.concatenate([weights, void_weight], axis=-1)  # 完整权重：(tileNum+1)（含void）
        
        # 2. 计算加权特征值x_e（包含void的概率）
        x_e = np.sum(full_weights * self.a_aux[self.inv_order], axis=-1)  # (1,)
        
        # 3. 区间选择（自然包含void的区间）
        k = np.searchsorted(self.a_aux, x_e, side='right') - 1
        k = np.clip(k, 0, self.a_aux.shape[0] - 2)  # 避免越界
        
        # 4. 映射回原始材料索引（含void）
        orig_k = self.inv_order[k]
        orig_k1 = self.inv_order[k + 1]
        
        # 5. Ordered-RAMP插值 + SIMP惩罚
        a_k, a_k1 = self.a_aux[k], self.a_aux[k + 1]
        xi = (x_e - a_k) / (a_k1 - a_k + 1e-12)  # 0~1
        p = 3.0  # SIMP惩罚因子
        xi_p = xi ** p  # 惩罚后的插值系数
        
        # 6. 刚度插值（自动包含void的刚度）
        C_k = self.C[orig_k]  # 可能是void或其他材料
        C_k1 = self.C[orig_k1]
        C_eff = C_k + xi_p[..., None, None] * (C_k1 - C_k)  # 无需额外加void，因C_k可能已是void
        
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
        # 1. 加载用户传入的材料（外部材料）
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
        
        # 2. 追加void材料到内部刚度列表（作为最后一个材料）
        C_list.append(self.void)
        self.C = np.array(C_list)  # 形状：(tileNum+1, 6, 6)（含void）

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
  
import numpy as onp
def void_C(E0=1.0, nu=0.3, eps=1e-9):
    """返回 6×6 各向同性刚度矩阵（接近 void）"""
    E = eps * E0                      # 弹性模量缩小
    C = onp.zeros((6, 6))
    # 对角块
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    for i in range(3):
        C[i, i] = lam + 2 * mu
        for j in range(3):
            if i != j:
                C[i, j] = lam
    # 剪切块
    for i in range(3, 6):
        C[i, i] = mu
    return C