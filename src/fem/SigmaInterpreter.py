# 储存 tile 的种类 及相应的simga
# 在程序开始运行时按照tileHandler注册的tile构建type->sigma的字典
# sigma来自json或其他RVE得到的文件
from typing import List,Dict
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
        self.sigma:Dict[str,np.ndarray] = {}
        self.debug=kwargs.get("debug",False)
        if not self.debug:
            self._buildSigmaDict() 
    

    def __call__(self,u_grad,require,*args,**kwargs) -> None:
        if type(require) == int:
            t = self.typeList[require]
        if type(require) == str:
            t=require
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
                
                # 使用at语法设置对角线元素（法向应力）
                diag_indices = np.diag_indices(3)
                sigma = sigma.at[diag_indices].set(lam * trace_epsilon + 2 * mu * epsilon[diag_indices])
                
                # 使用at语法设置上三角非对角线元素（剪应力）
                triu_indices = np.triu_indices(3, k=1)
                sigma = sigma.at[triu_indices].set(2 * mu * epsilon[triu_indices])
                
                # 使用at语法设置下三角非对角线元素（剪应力）
                tril_indices = np.tril_indices(3, k=-1)
                sigma = sigma.at[tril_indices].set(2 * mu * epsilon[tril_indices])
                # print(f"sigma:{sigma}")
                return sigma
            return stress(u_grad)
        return self.sigma[t]


    def _buildSigmaDict(self):
        for mat_type in self.typeList:
            file_path = os.path.join(self.folderPath, f"{mat_type}.json")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # 将数据转换为JAX数组并存入字典
                self.sigma[mat_type] = np.array(data)
                print(f"Loaded sigma for {mat_type} from {file_path}")
                
            except FileNotFoundError:
                print(f"Warning: File not found for type {mat_type} at {file_path}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
