import warnings
from typing import List
import jax.numpy as jnp

class TileHandler:
    def __init__(self, typeList:List[str]):
        warnings.warn( NotImplemented)
        self.typeList = typeList #单元类型表
        self.typeNum = len(typeList)
        self.compatibility = jnp.zeros((len(typeList), len(typeList))) # 兼容性矩阵
    def setConnectiability(self,fromTypeName:str,toTypeName:str,direction,value):
        pass
    def selfConnectable(self,typeList:List[str]):
        pass