from __future__ import annotations

import os
import sys
import warnings
from typing import List, Dict, Callable, Tuple

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
        directionPair:Tuple[Tuple[str,str]] = kwargs.pop('direction',None)
        self.typeList = kwargs.pop('typeList',[]) # every tile type will be saved here
        self.oppositeDirection:Dict[str,str] = {}
        self.direction_map=kwargs.pop("direction_map",{})
        self.reverse_direction_map = {v: k for k, v in self.direction_map.items()}  # 整数→字符串反向映射

        directionList=[]
        if directionPair is not None:
            for pair in directionPair:
                self.oppositeDirection[pair[0]]=pair[1]
                self.oppositeDirection[pair[1]]=pair[0]
                directionList.append(pair[0])
                directionList.append(pair[1])

        self.directionList:List[str] = ['isotropy'] if directionPair is None else directionList
        self.typeNum = len(self.typeList)
        self.directionNum=len(self.directionList)
        self.typeMethod:Dict[str,Tile]={}
        
        self._name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(self.typeList)}
        self._index_to_name: Dict[int, str] = {idx: name for idx, name in enumerate(self.typeList)}

        self.dire_to_index: Dict[str, int] = {dire: idx for idx, dire in enumerate(self.directionList)}
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
    
    def constantlize_compatibility(self):
        self._compatibility = jnp.array(self._compatibility)


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
    
    def get_opposite_direction_by_direction(self, direction: str) -> str:
        return self.oppositeDirection[direction]


    def get_index_by_direction(self, direction: str | List[str]) -> int | List[int]:
        """
        获取方向的索引
        :param direction: 方向名称或方向名称列表
        :return: 方向索引或索引列表
        """

        if isinstance(direction, (int, jnp.integer)):
            direction = self.reverse_direction_map[dir_val]
        else:
            direction = direction  # 若已是字符串则直接使用
        # 处理单个方向的情况
        if isinstance(direction, str):
            try:
                return self.dire_to_index[direction]
            except KeyError:
                raise ValueError(f"方向名称 '{direction}' 不存在于方向列表中") from None
        # 处理方向列表的情况
        elif isinstance(direction, list):
            try:
                return [self.dire_to_index[d] for d in direction]
            except KeyError as e:
                raise ValueError(f"方向名称 '{e.args[0]}' 不存在于方向列表中") from None
        # 处理不支持的输入类型
        else:
            raise TypeError(f"参数类型必须是str或list[str]，但输入的是{type(direction)}")


    def setConnectiability(self,fromTypeName:str,toTypeName:str|List[str],direction:str|List[str]='isotropy',value=1,dual=True):
        """_summary_

        Args:
            fromTypeName (str): _description_
            toTypeName (str | List[str]): _description_
            direction (str | List[str], optional): _description_. Defaults to 'isotropy'.
            value (int, optional): _description_. Defaults to 1.
            dual (bool, optional): example: from A to B left dual=True means A's left can connect B's right and B's right can connect A's left. 
                                            if False, only from A to B left will be added. Defaults to True.
        """
        #TODO 在具有方向时的,isotropy应当同时修改所有方向，
        j = self.get_index_by_name(fromTypeName)
        toTypeName = toTypeName if type(toTypeName) is list else [toTypeName]
        direction = self.directionList if direction not in self.directionList else direction
        # print(f"{type(direction)}")
        direction = direction if type(direction) is list else [direction]
        direction = self.directionList if (len(self.directionList) != 1 and direction==['isotropy']) else direction
        for toName in toTypeName:
            for dName in direction:
                i=self.get_index_by_name(toName)
                d=self.get_index_by_direction(dName)
                self._compatibility[d,i,j]=value
                if dual:
                    odName = self.oppositeDirection[dName]
                    od =self.get_index_by_direction(odName)
                    self._compatibility[od,j,i]=value
                    # self._compatibility[d,j,i]=value

    def selfConnectable(self,typeName:str|List[str],direction:str|List[str]='isotropy',value=1):
        typeName = typeName if type(typeName) is list else list(typeName)
        for name in typeName:
            self.setConnectiability(fromTypeName=name,toTypeName=name,direction=direction,value=value,dual=True)
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
    tileHandler = TileHandler(typeList=['a','b','c','d'],direction=(('up',"down"),("left","right")))
    tileHandler.setConnectiability(fromTypeName='a',toTypeName="b",direction="left",value=1,dual=True)
    tileHandler.selfConnectable(typeName='c',value=1)
    # tileHandler.register(['d','e'],[CF,CF])
    # tileHandler.selfConnectable(typeName=['a','c'],value=1)
    # tileHandler.setConnectiability(fromTypeName='a',toTypeName='b',value=1,dual=True)
    # tileHandler.setConnectiability(fromTypeName='c',toTypeName='b',value=1,dual=True)
    # tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',value=1,dual=True)
    # tileHandler.setConnectiability(fromTypeName='e',toTypeName=['a','b','c','d'],direction='back',value=1,dual=True)
    # tileHandler.typeMethod['d'].build(cube_points)

    print(tileHandler)