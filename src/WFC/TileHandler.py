from __future__ import annotations

import os
import sys
import warnings
from typing import List, Dict, Callable

import jax
import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from src.WFC.Tile import Tile

class TileHandler:
    def __init__(self, *args,**kwargs):
        """deal with tiles
        \n:param: direction:str|List[str] -> name of directions,default:['isotropy']
        \n:param: typeList:str|List[str] -> name of types
        """
        self.directionList:List[str] = kwargs.pop("direction",['isotropy'])
        self.typeList = kwargs.pop('typeList',[])


        self.typeNum = len(self.typeList)
        self.directionNum=len(self.directionList)
        self.typeMethod:Dict[str,Tile]={}
        
        self._name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(self.typeList)}
        self._index_to_name: Dict[int, str] = {idx: name for idx, name in enumerate(self.typeList)}

        self._dire_to_index: Dict[str, int] = {dire: idx for idx, dire in enumerate(self.directionList)}
        self._index_to_dire: Dict[int, str] = {idx: dire for idx, dire in enumerate(self.directionList)}

        self._compatibility = np.zeros((self.directionNum,len(self.typeList), len(self.typeList))) # 兼容性矩阵,1为兼容，0为不兼容

    def __repr__(self):
        type_method = ''
        for i, name in enumerate(self.typeMethod.keys()):
            type_method += f"{name} -> {self.typeMethod[name]}\n"
        compatibilities = ''
        for j, direction in enumerate(self.directionList):
            compatibilities += f"{direction}:\n{self.compatibility[self.get_index_by_direction(direction),:,:]}\n"
        return (f'type -> method: \n{type_method}'
                f'{len(self.typeMethod)} of {len(self.typeList)} types has method\n\n'
                f'compatibility:\n'
                f'{compatibilities},\n')


    def register(self,typeName:str|List[str],class_type:Callable|List[Callable]) -> None:
        typeName=typeName if type(typeName) is list else list(typeName)
        class_type=class_type if type(class_type) is list else [class_type]
        assert len(typeName)==len(class_type)
        update_index=False
        for i,name in enumerate(typeName):
            self.typeMethod[name]=class_type[i]
            if name not in self.typeList:
                self.typeList.append(name) #添加到typelist
                self._compatibility = np.pad(self._compatibility, pad_width=((0,0),(0, 1), (0, 1)), mode='constant', constant_values=0) # 更新兼容矩阵
                update_index=True
        if update_index:
            # 更新两个索引
            self._name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(self.typeList)}
            self._index_to_name: Dict[int, str] = {idx: name for idx, name in enumerate(self.typeList)}


    @property
    def compatibility(self):
        return self._compatibility


    def get_name_by_index(self, index: int) -> str:
        try:
            return self._index_to_name[index]
        except KeyError:
            raise ValueError(f"索引 '{index}' 超出范围 (0-{self.typeNum-1})") from None

    def get_index_by_name(self, name: str) -> int:
        try:
            return self._name_to_index[name]
        except KeyError:
            raise ValueError(f"类型名称 '{name}' 不存在于类型列表中") from None

    def get_direction_by_index(self, index: int) -> str:
        try:
            return self._index_to_dire[index]
        except KeyError:
            raise ValueError(f"索引 '{index}' 超出范围 (0-{self.directionNum-1})") from None

    def get_index_by_direction(self, direction: str) -> int:
        try:
            return self._dire_to_index[direction]
        except KeyError:
            raise ValueError(f"类型名称 '{direction}' 不存在于类型列表中") from None


    def setConnectiability(self,fromTypeName:str,toTypeName:str|List[str],direction:str|List[str]='isotropy',value=1,dual=True):
        j = self.get_index_by_name(fromTypeName)
        toTypeName = toTypeName if type(toTypeName) is list else list(toTypeName)
        direction = self.directionList if direction not in self.directionList else direction
        print(f"{type(direction)}")
        direction = direction if type(direction) is list else [direction]

        for toName in toTypeName:
            for dName in direction:
                i=self.get_index_by_name(toName)
                d=self.get_index_by_direction(dName)
                self._compatibility[d,i,j]=value
                if dual:
                    self._compatibility[d,j,i]=value

    def selfConnectable(self,typeName:str|List[str],direction:str|List[str]='isotropy',value=1):
        typeName = typeName if type(typeName) is list else list(typeName)
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
    from src.dynamicGenerator.TileImplement.Cube import CF
    tileHandler = TileHandler(typeList=['a','b','c',],direction=['right','left',"top","bottom","front","back"])
    tileHandler.register(['d','e'],[CF,CF])
    tileHandler.selfConnectable(typeName=['a','c'],value=1)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName='b',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='b',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='e',toTypeName=['a','b','c','d'],direction='back',value=1,dual=True)

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
    # tileHandler.typeMethod['d'].build(cube_points)

    print(tileHandler)