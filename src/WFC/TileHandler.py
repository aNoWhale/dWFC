from __future__ import annotations

import warnings
from typing import List, Dict, Callable

import jax
import jax.numpy as jnp
import numpy as np

from src.WFC.Tile import Tile

class TileHandler:
    def __init__(self, *args,**kwargs):
        typeList=kwargs.pop('typeList',[])
        self.typeList = typeList
        self.typeMethod:Dict[str,Tile]={}
         #单元类型表
        self.typeNum = len(self.typeList)
        self._compatibility = np.zeros((len(self.typeList), len(self.typeList))) # 兼容性矩阵,1为兼容，0为不兼容
        # 创建名称到索引的映射字典
        self.name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(self.typeList)}
        # 创建索引到名称的映射字典
        self.index_to_name: Dict[int, str] = {idx: name for idx, name in enumerate(self.typeList)}

    def __repr__(self):
        type_method = ''
        for i, name in enumerate(self.typeMethod.keys()):
            type_method += f"{name} -> {self.typeMethod[name]}\n"
        return (f'type -> method: \n{type_method}'
                f'{len(self.typeMethod)} of {len(self.typeList)} types has method\n\n'
                f'compatibility:\n'
                f'{self.compatibility},\n')

    def register(self,typeName:str|List[str],class_type:Callable|List[Callable]) -> None:
        typeName=typeName if type(typeName) is list else list(typeName)
        class_type=class_type if type(class_type) is list else [class_type]
        assert len(typeName)==len(class_type)
        update_index=False
        for i,name in enumerate(typeName):
            self.typeMethod[name]=class_type[i]
            if name not in self.typeList:
                self.typeList.append(name) #添加到typelist
                self._compatibility = np.pad(self._compatibility, pad_width=((0, 1), (0, 1)), mode='constant', constant_values=0) # 更新兼容矩阵
                update_index=True
        if update_index:
            # 更新两个索引
            self.name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(self.typeList)}
            self.index_to_name: Dict[int, str] = {idx: name for idx, name in enumerate(self.typeList)}


    @property
    def compatibility(self):
        return self._compatibility


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
    class Test:
        def __init__(self):
            pass
        def call(self):
            print('call')
    from src.dynamicGenerator.TileImplement.Cube import Cubic, BCC, BCCZ, FCC, FCCZ
    tileHandler = TileHandler(typeList=['a','b','c',])
    tileHandler.register(['d','e'],[Cubic,BCC])
    tileHandler.selfConnectable(typeName=['a','c'],value=1)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName='b',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='b',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',value=1,dual=True)
    cube_points = [
        [0, 0, 0],  # 底面左前角
        [1, 0, 0],  # 底面右前角
        [1, 0, 1],  # 顶面右前角
        [0, 0, 1],  # 顶面左前角
        [0, 1, 0],  # 底面左后角
        [1, 1, 0],  # 底面右后角
        [1, 1, 1],  # 顶面右后角
        [0, 1, 1]  # 顶面左后角
    ]
    tileHandler.typeMethod['d'].build(cube_points)

    print(tileHandler)