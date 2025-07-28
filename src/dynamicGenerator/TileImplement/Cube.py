from src.dynamicGenerator.TileImplement.CubeTile import CubeTile
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
import numpy as np


class CF(CubeTile):
    """
    Cylindrical frame(圆柱框架)
    """
    @classmethod
    def build(cls, points, *args, **kwargs):
        filename = "cylindrical_frame.stp"
        pts = points[:]
        # 12 条边（顶点索引）
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (5, 6), (6, 7), (7, 4)
        ]
        RADIUS = CubeTile.RADIUS
        result_shape = CubeTile.build_cylinder(pts, edges, RADIUS)
        result_shape = CubeTile.add_sphere(pts, RADIUS, result_shape)
        CubeTile.write_stp(filename, result_shape)
        return result_shape

class BCC(CubeTile):
    """
    Body-centered cubic (BCC)
    体心立方
    """
    @classmethod
    def build(cls, points, *args, **kwargs):
        filename = "body_centered_cubic.stp"
        pts = points[:]
        bc = np.mean(pts, axis=0).tolist()
        pts.append(bc) #[8]: 体心坐标
        edges = [(i, 8) for i in range(8)] # 将所有顶点与体心连接
        RADIUS = CubeTile.RADIUS
        result_shape = CubeTile.build_cylinder(pts, edges, RADIUS)
        result_shape = CubeTile.add_sphere(pts, RADIUS, result_shape)
        CubeTile.write_stp(filename, result_shape)
        return result_shape





if __name__ == "__main__":
    #cube=Cube()
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
    # 生成立方体框架结构并保存为STP文件
    BCC.build(cube_points)
    CF.build(cube_points)
    # if result:
    #     print(f"立方体框架结构已成功生成并保存到: {result}")
    # else:
    #     print("生成立方体框架结构失败")