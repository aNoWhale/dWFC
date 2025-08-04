
import os
import sys
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/WFC 目录下）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
print(f"{project_root}")
sys.path.append(project_root)

from src.dynamicGenerator.TileImplement.Cube import *
import numpy as np
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
import itertools
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.BOPAlgo import BOPAlgo_Options,BOPAlgo_GlueFull,BOPAlgo_GlueOff,BOPAlgo_GlueShift, BOPAlgo_Builder


import tqdm





if __name__ == '__main__':


    # cubic = Cubic.build(cube_points)
    # bcc = BCC.build(cube_points)
    # bccz = BCCZ.build(cube_points)
    # fcc = FCC.build(cube_points)
    # fccz = FCCZ.build(cube_points)

    # cube_points_1 = [
    #     [0, 0, 0],  # 底面左前角
    #     [1, 0, 0],  # 底面右前角
    #     [1, 0, 1],  # 顶面右前角
    #     [0, 0, 1],  # 顶面左前角
    #     [0, 1, 0],  # 底面左后角
    #     [1, 1, 0],  # 底面右后角
    #     [1, 1, 1],  # 顶面右后角
    #     [0, 1, 1]  # 顶面左后角
    # ]
    # cube_points_2 = [[x + 1, y, z ] for x, y, z in cube_points_1]
    # cube_points_3 = [[x, y + 1, z ] for x, y, z in cube_points_1]
    # cube_points_4 = [[x, y, z + 1] for x, y, z in cube_points_1]
    # cube_points_5 = [[x + 1, y + 1, z ] for x, y, z in cube_points_1]
    # cube_points_6 = [[x, y + 1, z + 1] for x, y, z in cube_points_1]
    # cube_points_7 = [[x + 1, y, z + 1] for x, y, z in cube_points_1]
    # cube_points_8 = [[x + 1, y + 1, z + 1] for x, y, z in cube_points_1]
    # rs1 = Cubic.build(cube_points_1)
    # rs2 = Cubic.build(cube_points_2)
    # rs3 = Cubic.build(cube_points_3)
    # rs4 = Cubic.build(cube_points_4)
    # rs5 = Cubic.build(cube_points_5)
    # rs6 = Cubic.build(cube_points_6)
    # rs7 = Cubic.build(cube_points_7)
    # rs8 = Cubic.build(cube_points_8)
    # result_shape = BRepAlgoAPI_Fuse(rs2, rs1).Shape()
    # result_shape = BRepAlgoAPI_Fuse(rs3, result_shape).Shape()
    # result_shape = BRepAlgoAPI_Fuse(rs4, result_shape).Shape()
    # result_shape = BRepAlgoAPI_Fuse(rs5, result_shape).Shape()
    # result_shape = BRepAlgoAPI_Fuse(rs6, result_shape).Shape()
    # result_shape = BRepAlgoAPI_Fuse(rs7, result_shape).Shape()
    # result_shape = BRepAlgoAPI_Fuse(rs8, result_shape).Shape()



    base_pts = [
        [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
        [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]
    ]

    shifts = itertools.product(range(4), repeat=3)  # 100×100×100 位移


    # 用生成器+增量 fuse，避免一次性占满内存
    fuse = BRepAlgoAPI_Fuse()
    fuse.SetRunParallel(True)  # 开启并行计算
    fuse.SetNonDestructive(True)  # 非破坏性模式
    fuse.SetGlue(BOPAlgo_GlueOff) # 粘合

    builder = BOPAlgo_Builder()
    builder.SetRunParallel(True)
    # 初始化结果形状
    result_shape = None
    shifts = [(dx, dy, dz) for dx in range(4) for dy in range(4) for dz in range(4)]

    # 创建可复用的列表对象
    args_list = TopTools_ListOfShape()
    tools_list = TopTools_ListOfShape()

    with tqdm.tqdm(total=len(shifts), desc="Fusing cubes", unit="cube") as pbar:
        for dx, dy, dz in shifts:
            cube = FCCZ.build([[x + dx, y + dy, z + dz] for x, y, z in base_pts])
            
            if result_shape is None:
                # 第一个立方体直接赋值
                result_shape = cube
            else:
                # 准备参数
                args_list = TopTools_ListOfShape()
                tools_list = TopTools_ListOfShape()
                # 设置Arguments（要添加的新立方体）
                args_list.Append(cube)
                
                # 设置Tools（当前已融合的结果）
                tools_list.Append(result_shape)
                builder.AddArgument(cube)
                del args_list
                del tools_list
            pbar.update(1)
        builder.Perform()
        result_shape=builder.Shape()
    Cubic.write_stp('cubes.stp', result_shape)





