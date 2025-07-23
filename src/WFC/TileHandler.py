from __future__ import annotations

import warnings
from typing import List, Dict

import jax
import jax.numpy as jnp
import numpy as np
class TileHandler:
    def __init__(self, typeList:List[str]):
        self.typeList = typeList #单元类型表
        self.typeNum = len(typeList)
        self._compatibility = np.zeros((len(typeList), len(typeList))) # 兼容性矩阵,1为兼容，0为不兼容
        # 创建名称到索引的映射字典
        self.name_to_index: Dict[str, int] = self._create_name_index_map()
        # 创建索引到名称的映射字典
        self.index_to_name: Dict[int, str] = self._create_index_name_map()
    def __repr__(self):
        return (f'types: {self.typeList},\n'
                f'compatibility:\n'
                f'{self.compatibility},\n')

    @property
    def compatibility(self):
        return self._compatibility

    def _create_name_index_map(self) -> Dict[str, int]:
        """创建从类型名称到索引的映射字典"""
        return {name: idx for idx, name in enumerate(self.typeList)}

    def _create_index_name_map(self) -> Dict[int, str]:
        """创建从索引到类型名称的映射字典"""
        return {idx: name for idx, name in enumerate(self.typeList)}

    def get_name_by_index(self, index: int) -> str:
        """根据索引获取类型名称"""
        try:
            return self.index_to_name[index]
        except KeyError:
            raise ValueError(f"索引 '{index}' 超出范围 (0-{self.typeNum-1})") from None

    def _get_index_by_name(self, name: str) -> int:
        """根据类型名称获取索引 (O(1)时间复杂度的快速查询)"""
        try:
            return self.name_to_index[name]
        except KeyError:
            raise ValueError(f"类型名称 '{name}' 不存在于类型列表中") from None

    def setConnectiability(self,fromTypeName:str,toTypeName:str|List[str],direction=None,value=1,dual=True):
        toTypeName=toTypeName if type(toTypeName) is List else list(toTypeName)
        j=self._get_index_by_name(fromTypeName)
        for toName in toTypeName:
            i=self._get_index_by_name(toName)
            self._compatibility[i,j]=value
            if dual:
                self._compatibility[j,i]=value
        pass

    def selfConnectable(self,typeName:str|List[str],direction=None,value=1):
        typeName = typeName if type(typeName) is List else list(typeName)
        for name in typeName:
            self.setConnectiability(fromTypeName=name,toTypeName=name,direction=direction,value=value)
        pass

    def pattern_to_names(self, pattern) -> np.ndarray:
        name_array = np.array(self.typeList)
        return name_array[pattern]


if __name__ == '__main__':
    tileHandler = TileHandler(typeList=['a','b','c',])
    tileHandler.selfConnectable(typeName=['a','c'],value=1)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName='b',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='b',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',value=1,dual=True)
    print(tileHandler)