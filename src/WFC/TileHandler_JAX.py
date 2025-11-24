from __future__ import annotations

import os
import sys
import warnings
import inspect
from typing import List, Dict, Callable, Tuple, Union

import jax
# jax.config.update('jax_disable_jit', True)

import jax.numpy as jnp
import numpy as np

# 导入依赖（确保 Tile 类路径正确）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from src.WFC.Tile import Tile


class TileHandler:
    def __init__(self, *args, **kwargs):
        """
        以整数方向为核心，兼容旧代码字符串方向输入
        关键修改：`self.opposite_dir_int` 改为 JAX 数组 `self.opposite_dir_array`
        """
        # -------------------------- 1. 方向初始化（整数为核心，强制保留isotropy） --------------------------
        direction_pair: Tuple[Tuple[str, str]] = kwargs.pop('direction', None)
        self.custom_dir_map: Dict[str, int] = kwargs.pop('direction_map', {})  # 用户自定义映射
        
        # 方向基础数据结构（移除原字典，新增数组）
        self.dir_int_to_str: Dict[int, str] = {}  # 整数→方向字符串（原生字典，标量用）
        self.str_to_dir_int: Dict[str, Union[int, List[int]]] = self.custom_dir_map  # 字符串→整数/列表
        self.direction_list: List[int] = []  # 所有具体方向整数列表（原生列表）

        # 步骤1：初始化具体方向（up/down/left/right 等）
        if direction_pair is not None:
            for pair in direction_pair:
                dir1_str, dir2_str = pair[0], pair[1]
                # 处理方向1
                dir1_int = self.str_to_dir_int.get(dir1_str)
                if dir1_int is None or isinstance(dir1_int, list):
                    dir1_int = len(self.direction_list)
                    self.str_to_dir_int[dir1_str] = dir1_int
                if dir1_int not in self.dir_int_to_str:
                    self.dir_int_to_str[dir1_int] = dir1_str
                    self.direction_list.append(dir1_int)
                # 处理方向2
                dir2_int = self.str_to_dir_int.get(dir2_str)
                if dir2_int is None or isinstance(dir2_int, list):
                    dir2_int = len(self.direction_list)
                    self.str_to_dir_int[dir2_str] = dir2_int
                if dir2_int not in self.dir_int_to_str:
                    self.dir_int_to_str[dir2_int] = dir2_str
                    self.direction_list.append(dir2_int)
            # 排序具体方向列表
            self.direction_list = sorted(self.direction_list)
        else:
            pass

        # -------------------------- 核心修改：反向方向数组（替代原字典） --------------------------
        # 1. 确定数组长度（最大方向整数 + 1，确保所有方向整数可索引）
        self.max_dir_int = max(self.dir_int_to_str.keys(), default=-1)+1  # 包括 -1（isotropy）
        # 2. 初始化 JAX 数组：默认值 -1（无效反向），后续填充有效映射
        self.opposite_dir_array = jnp.full((self.max_dir_int + 1,), fill_value=-1, dtype=jnp.int32)
        # 3. 填充反向方向映射（从方向对中提取）
        if direction_pair is not None:
            for pair in direction_pair:
                dir1_str, dir2_str = pair[0], pair[1]
                dir1_int = self.str_to_dir_int[dir1_str]
                dir2_int = self.str_to_dir_int[dir2_str]
                # 双向填充反向映射
                self.opposite_dir_array = self.opposite_dir_array.at[dir1_int].set(dir2_int)
                self.opposite_dir_array = self.opposite_dir_array.at[dir2_int].set(dir1_int)
        # 4. 处理 isotropy（-1）的反向（自身）
        if -1 in self.dir_int_to_str:  # 确保 isotropy 已初始化
            self.opposite_dir_array = self.opposite_dir_array.at[-1].set(-1)  # -1 的反向是 -1
        # 5. 无具体方向时的兼容（isotropy 反向自身）
        if not self.direction_list:
            self.opposite_dir_array = self.opposite_dir_array.at[0].set(0)

        # 步骤2：强制添加 isotropy 映射（保持原逻辑）
        self.str_to_dir_int['isotropy'] = self.direction_list if self.direction_list else [0]
        self.dir_int_to_str[-1] = 'isotropy'  # -1 作为 isotropy 标识

        # 步骤3：无具体方向时的兼容逻辑
        if not self.direction_list:
            self.direction_list = [0]
            self.dir_int_to_str[0] = 'isotropy'
            self.str_to_dir_int['isotropy'] = [0]

        # -------------------------- 2. 瓷砖类型初始化（保持原逻辑） --------------------------
        self.typeList: List[str] = kwargs.pop('typeList', [])
        self.typeMethod: Dict[str, Tile] = {}
        self.typeNum: int = len(self.typeList)
        self._name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(self.typeList)}
        self._index_to_name: Dict[int, str] = {idx: name for idx, name in enumerate(self.typeList)}

        # -------------------------- 3. 兼容性矩阵初始化（保持原逻辑） --------------------------
        self.directionNum: int = len(self.direction_list)
        self._compatibility: np.ndarray|jnp.ndarray = np.zeros(
            (self.directionNum, self.typeNum, self.typeNum), dtype=np.float32
        )

    def __repr__(self) -> str:
        """打印关键信息（反向方向显示数组映射）"""
        type_method_lines = [f"  {name} → {self.typeMethod[name]}" for name, inst in self.typeMethod.items()]
        type_method_str = "\n".join(type_method_lines) if type_method_lines else "  无已注册实例"

        compat_lines = []
        for dir_int in self.direction_list:
            dir_str = self.dir_int_to_str[dir_int]
            compat_mat = self._compatibility[dir_int, :, :]
            compat_lines.append(f"  {dir_str} (int:{dir_int}):\n{compat_mat}")
        compat_str = "\n".join(compat_lines)

        # 反向方向：从数组转换为字典格式（便于阅读）
        opposite_map_str = {}
        for dir_int in self.dir_int_to_str.keys():
            if dir_int >= 0 and dir_int < len(self.opposite_dir_array):  # 有效索引
                opposite_int = self.opposite_dir_array[dir_int].item()
                opposite_map_str[dir_int] = opposite_int if opposite_int != -1 else "invalid"
        opposite_map_str[-1] = self.opposite_dir_array[-1].item()  # 单独显示 isotropy

        dir_map_str = {k: v for k, v in self.str_to_dir_int.items() if k != 'isotropy'}
        dir_map_str['isotropy'] = self.str_to_dir_int['isotropy']

        return (
            f"TileHandler 信息:\n"
            f"1. 瓷砖类型（共{self.typeNum}种）:\n"
            f"   已注册实例（共{len(self.typeMethod)}种）:\n{type_method_str}\n"
            f"   所有类型名称: {self.typeList}\n\n"
            f"2. 方向配置:\n"
            f"   具体方向列表（共{self.directionNum}个）: {self.direction_list}\n"
            f"   方向映射（字符串→整数/列表）: {dir_map_str}\n"
            f"   反向方向（整数→整数，数组映射）: {opposite_map_str}\n\n"
            f"3. 兼容性矩阵（方向→类型×类型）:\n{compat_str}\n"
        )

    # -------------------------- 4. 瓷砖类型注册（保持原逻辑） --------------------------
    def register(
        self, 
        typeName: Union[str, List[str]], 
        class_type: Union[Callable, List[Callable], Tile, List[Tile]]
    ) -> None:
        type_names = typeName if isinstance(typeName, list) else [typeName]
        class_types = class_type if isinstance(class_type, list) else [class_type]
        assert len(type_names) == len(class_types), "类型名称和类/实例数量必须匹配"

        update_index = False
        for name, cls_or_inst in zip(type_names, class_types):
            if inspect.isclass(cls_or_inst) and issubclass(cls_or_inst, Tile):
                tile_instance = cls_or_inst()
            elif isinstance(cls_or_inst, Tile):
                tile_instance = cls_or_inst
            else:
                raise TypeError(f"class_type 必须是 Tile 子类或实例，输入：{type(cls_or_inst)}")

            if name not in self.typeList:
                self.typeList.append(name)
                self.typeNum += 1
                self._compatibility = np.pad(
                    self._compatibility,
                    pad_width=((0, 0), (0, 1), (0, 1)),
                    mode='constant', constant_values=0.0
                )
                update_index = True

            self.typeMethod[name] = tile_instance

        if update_index:
            self._name_to_index = {name: idx for idx, name in enumerate(self.typeList)}
            self._index_to_name = {idx: name for idx, name in enumerate(self.typeList)}

    # -------------------------- 5. 兼容性矩阵操作（保持原逻辑） --------------------------
    @property
    def compatibility(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._compatibility

    def constantlize_compatibility(self) -> None:
        # self._compatibility = jnp.maximum(jnp.array(self._compatibility, dtype=jnp.float32), 1e-10)
        self._compatibility = jnp.array(self._compatibility, dtype=jnp.float32)


    # -------------------------- 6. 类型索引转换（保持原逻辑） --------------------------
    def get_name_by_index(self, index: int) -> str:
        try:
            return self._index_to_name[index]
        except KeyError:
            raise ValueError(f"类型索引 '{index}' 超出范围（0~{self.typeNum-1}）") from None

    def get_index_by_name(self, name: str) -> int:
        try:
            return self._name_to_index[name]
        except KeyError:
            raise ValueError(f"类型名称 '{name}' 不存在（已注册：{list(self._name_to_index.keys())}）") from None

    # -------------------------- 7. 方向转换（核心：数组替代字典查询反向方向） --------------------------
    def _to_dir_int(
        self, 
        direction: Union[str, int, jnp.integer, List[str], List[int]]
    ) -> Union[int, List[int]]:
        if isinstance(direction, list):
            return [self._to_dir_int(d) for d in direction]
        
        if isinstance(direction, (int, jnp.integer)):
            dir_int = int(direction)
            if dir_int not in self.dir_int_to_str and dir_int != -1:
                raise ValueError(f"整数方向 '{dir_int}' 不存在（有效：{list(self.dir_int_to_str.keys())}）") from None
            return dir_int
        
        elif isinstance(direction, str):
            if direction not in self.str_to_dir_int:
                raise ValueError(f"字符串方向 '{direction}' 不存在（有效：{list(self.str_to_dir_int.keys())}）") from None
            result = self.str_to_dir_int[direction]
            return result if isinstance(result, list) else int(result)
        
        else:
            raise TypeError(f"方向类型必须是 str/int/list，输入：{type(direction)}") from None

    def get_direction_by_index(self, index: Union[int, jnp.integer, List[int]]) -> Union[str, List[str]]:
        if isinstance(index, list):
            return [self.get_direction_by_index(idx) for idx in index]
        
        idx = int(index)
        if idx == -1:
            return 'isotropy'
        try:
            return self.dir_int_to_str[idx]
        except KeyError:
            raise ValueError(f"整数方向 '{idx}' 不存在（有效：{list(self.dir_int_to_str.keys())}）") from None

    def get_index_by_direction(self, direction: Union[str, int, List[str], List[int]]) -> Union[int, List[int]]:
        return self._to_dir_int(direction)

    def get_opposite_direction_by_direction(
        self, 
        direction: Union[str, int, List[str], List[int]]
    ) -> Union[str, int, List[str], List[int]]:
        if isinstance(direction, list):
            return [self.get_opposite_direction_by_direction(d) for d in direction]
        
        dir_val = self._to_dir_int(direction)
        
        # 处理 isotropy（反向仍是自身）
        if isinstance(dir_val, list) or dir_val == -1:
            return 'isotropy' if isinstance(direction, str) else -1
        
        # 核心修改：数组索引替代字典查询反向方向
        opposite_int = self.opposite_dir_array[dir_val].item()  # 数组索引获取反向整数
        if opposite_int == -1:
            raise ValueError(f"方向 '{dir_val}' 无有效反向（检查方向初始化）") from None
        
        return self.get_direction_by_index(opposite_int) if isinstance(direction, str) else opposite_int

    def get_opposite_index_by_index(self, index: Union[int, jnp.integer]) -> int:
        """通过方向整数直接获取反向整数（数组索引，兼容 JAX）"""
        opposite_int = self.opposite_dir_array[index]
        return opposite_int

    # -------------------------- 8. 连接性设置（核心：数组替代字典查询反向方向） --------------------------
    def setConnectiability(
        self, 
        fromTypeName: str, 
        toTypeName: Union[str, List[str]], 
        direction: Union[str, int, List[str], List[int]] = 'isotropy',
        value: float = 1.0, 
        dual: bool = True
    ) -> None:
        from_type_idx = self.get_index_by_name(fromTypeName)
        to_type_names = toTypeName if isinstance(toTypeName, list) else [toTypeName]
        to_type_idxs = [self.get_index_by_name(name) for name in to_type_names]
        
        dir_ints = self._to_dir_int(direction)
        dir_ints = dir_ints if isinstance(dir_ints, list) else [dir_ints]
        dir_ints = [d for d in dir_ints if d in self.direction_list]
        if not dir_ints:
            warnings.warn(f"无有效方向可设置（输入：{direction}，有效：{self.direction_list}）")
            return

        for to_idx in to_type_idxs:
            for dir_int in dir_ints:
                self._compatibility[dir_int, to_idx, from_type_idx] = value
                
                # 核心修改：数组索引获取反向方向（替代原字典）
                if dual:
                    opposite_dir_int = self.opposite_dir_array[dir_int].item()  # 数组索引
                    if opposite_dir_int == -1:
                        raise ValueError(f"方向 '{dir_int}' 无有效反向，无法启用双向兼容") from None
                    self._compatibility[opposite_dir_int, from_type_idx, to_idx] = value

    def selfConnectable(
        self, 
        typeName: Union[str, List[str]], 
        direction: Union[str, int, List[str], List[int]] = 'isotropy',
        value: float = 1.0
    ) -> None:
        type_names = typeName if isinstance(typeName, list) else [typeName]
        for name in type_names:
            self.setConnectiability(fromTypeName=name, toTypeName=name, direction=direction, value=value, dual=True)

    # -------------------------- 9. 图案转换（保持原逻辑） --------------------------
    def pattern_to_names(self, pattern: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
        name_array = np.array(self.typeList)
        pattern_np = np.array(pattern) if isinstance(pattern, jnp.ndarray) else pattern
        return name_array[pattern_np]


# -------------------------- 测试代码（验证反向方向数组功能） --------------------------
if __name__ == '__main__':
    # 1. 初始化（自定义方向映射）
    tileHandler = TileHandler(
        typeList=['a', 'b', 'c', 'd', 'e'],
        direction=(('up', "down"), ("left", "right")),
        direction_map={"up": 0, "down": 2, "left": 3, "right": 1}
    )
    print("="*50)
    print("1. 初始化后TileHandler信息（反向方向为数组）：")
    print(tileHandler)

    # 2. 测试反向方向数组查询
    print("="*50)
    print("2. 反向方向数组测试：")
    print(f"   方向 0（up）的反向：{tileHandler.opposite_dir_array[0].item()}（应=2）")
    print(f"   方向 2（down）的反向：{tileHandler.opposite_dir_array[2].item()}（应=0）")
    print(f"   方向 3（left）的反向：{tileHandler.opposite_dir_array[3].item()}（应=1）")
    print(f"   isotropy（-1）的反向：{tileHandler.opposite_dir_array[-1].item()}（应=-1）")

    # 3. 测试方向转换（反向方向）
    print("="*50)
    print("3. 方向反向转换测试：")
    print(f"   'up' 的反向：{tileHandler.get_opposite_direction_by_direction('up')}（应=down）")
    print(f"   整数 1（right）的反向：{tileHandler.get_opposite_direction_by_direction(1)}（应=3）")
    print(f"   'left' 的反向：{tileHandler.get_opposite_direction_by_direction('left')}（应=right）")

    # 4. 测试兼容性设置（双向兼容，依赖反向数组）
    tileHandler.setConnectiability(fromTypeName='a', toTypeName='b', direction='up', value=1.0, dual=True)
    print("="*50)
    print("4. 双向兼容性测试：")
    a_idx = tileHandler.get_index_by_name('a')
    b_idx = tileHandler.get_index_by_name('b')
    up_dir = 0
    down_dir = tileHandler.opposite_dir_array[up_dir].item()
    print(f"   a→b（up 方向，int={up_dir}）：{tileHandler.compatibility[up_dir, b_idx, a_idx]}（应=1.0）")
    print(f"   b→a（down 方向，int={down_dir}）：{tileHandler.compatibility[down_dir, a_idx, b_idx]}（应=1.0）")

    # 5. 测试 JAX 兼容性（反向数组为 JAX 类型）
    print("="*50)
    print(f"5. 反向方向数组 JAX 类型：{type(tileHandler.opposite_dir_array)}（应=jax.numpy.ndarray）")