# 储存 tile 的种类 及相应的simga
# 在程序开始运行时按照tileHandler注册的tile构建type->sigma的字典
# sigma来自json或其他RVE得到的文件
from typing import List,Dict
import jax
import jax.numpy as np
import os
import json
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
    

    def __call__(self,u_grad,weights,*args,**kwargs) -> None:

        if self.debug:
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
                # print(f"sigma:{sigma}")
                return sigma
            return stress(u_grad)
        else:
            def stress_anisotropic(C, u_grad, *args, **kwargs):
                """
                计算三维各向异性材料的应力张量
                
                参数:
                u_grad: 位移梯度张量 (3x3 numpy数组)
                
                返回:
                sigma: 应力张量 (3x3 numpy数组)
                """
                  
                # 计算应变张量
                epsilon = 0.5 * (u_grad + u_grad.T)
                
                # 将应变张量转换为Voigt符号表示 (6x1向量)
                # Voigt符号: [ε11, ε22, ε33, 2ε23, 2ε13, 2ε12]
                epsilon_voigt = np.array([
                    epsilon[0, 0],
                    epsilon[1, 1],
                    epsilon[2, 2],
                    2 * epsilon[1, 2],
                    2 * epsilon[0, 2],
                    2 * epsilon[0, 1]
                ])
                
                # 计算应力向量 (Voigt符号)
                sigma_voigt = np.dot(C, epsilon_voigt)
                
                # 将应力向量转换回张量形式
                sigma = np.zeros((3, 3))  # 用JAX创建3x3零矩阵
                # 填充对角元素
                sigma = sigma.at[0, 0].set(sigma_voigt[0])  # σ11
                sigma = sigma.at[1, 1].set(sigma_voigt[1])  # σ22
                sigma = sigma.at[2, 2].set(sigma_voigt[2])  # σ33
                # 填充对称的非对角元素（利用应力张量对称性）
                sigma = sigma.at[1, 2].set(sigma_voigt[3])  # σ23
                sigma = sigma.at[2, 1].set(sigma_voigt[3])  # σ32（对称）
                sigma = sigma.at[0, 2].set(sigma_voigt[4])  # σ13
                sigma = sigma.at[2, 0].set(sigma_voigt[4])  # σ31（对称）
                sigma = sigma.at[0, 1].set(sigma_voigt[5])  # σ12
                sigma = sigma.at[1, 0].set(sigma_voigt[5])  # σ21（对称）
                
                return sigma
            # jax.debug.print("weight: {a}", a=weights)
            penalty=3
            return stress_anisotropic(np.einsum("n,nij->ij",weights**penalty,self.C),u_grad)


    def _buildCDict(self):
        C=[]
        for mat_type in self.typeList:
            file_path = os.path.join(self.folderPath, f"{mat_type}.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # 将数据转换为JAX数组并存入字典
                self.C_dict[mat_type] = np.array(data)
                C.append(data)
                print(f"Loaded C for * {mat_type} * from {file_path}")
                
            except FileNotFoundError:
                print(f"Warning: File not found for type * {mat_type} * at {file_path}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        self.C=np.array(C)
