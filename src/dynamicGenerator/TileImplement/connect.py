import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)


from src.dynamicGenerator.TileImplement.Cube import *
import numpy as np
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
import itertools
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.BOPAlgo import BOPAlgo_Options
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

    fuse_processor = BRepAlgoAPI_Fuse()
    fuse_processor.SetRunParallel(True)  # 关键优化：开启并行计算
    result_shape = None
    pbar = tqdm.tqdm(total=4**3, desc="", unit="")
    for dx, dy, dz in shifts:
        cube = FCCZ.build([[x + dx, y + dy, z + dz] for x, y, z in base_pts])
        if result_shape is None:
            result_shape = cube
        else:
            fuse_processor(cube, result_shape)
            fuse_processor.Build()
            if not fuse_processor.IsDone():
                print(f"Warning: Fusion failed at ({dx},{dy},{dz})")
            else:
                result_shape = fuse_processor.Shape()
        pbar.update(1)
    pbar.close()
    Cubic.write_stp('cubes.stp', result_shape)





