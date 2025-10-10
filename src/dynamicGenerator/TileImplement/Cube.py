from src.dynamicGenerator.TileImplement.CubeTile import CubeTile
import numpy as np
from typing import List


class Cubic(CubeTile):
    """
    Cubic
    立方体
    """
    @classmethod
    def build(cls, points, *args, **kwargs):
        filename = "Cubic.stp"
        pts = points.tolist()
        # 12 条边（顶点索引）
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (5, 6), (6, 7), (7, 4)
        ]
        RADIUS = CubeTile.RADIUS
        result_shape = CubeTile.build_cylinder(pts, edges, RADIUS)
        result_shape = CubeTile.add_sphere(pts, RADIUS, result_shape)
        # CubeTile.write_stp(filename, result_shape)
        return result_shape

class BCC(CubeTile):
    """
    Body-centered cubic (BCC)
    体心立方
    """
    @classmethod
    def build(cls, points, *args, **kwargs):
        filename = "BCC.stp"
        pts = points
        bc = np.mean(pts, axis=0).tolist()
        pts.append(bc) #[8]: 体心坐标
        edges = [(i, 8) for i in range(8)] # 将所有顶点与体心连接
        RADIUS = CubeTile.RADIUS
        result_shape = CubeTile.build_cylinder(pts, edges, RADIUS)
        result_shape = CubeTile.add_sphere(pts, RADIUS, result_shape)
        # CubeTile.write_stp(filename, result_shape)
        return result_shape



class BCCZ(CubeTile):
    """
    BCC with vertical strut in z-axis (BCCZ)
    Z轴支撑体心立方
    """

    @classmethod
    def build(cls, points, *args, **kwargs):
        filename = "BCCZ.stp"
        pts = points.tolist()
        bc = np.mean(pts, axis=0).tolist()
        pts.append(bc)  # [8]: 体心坐标
        edges = [(i, 8) for i in range(8)]  # 将所有顶点与体心连接
        edges = edges + [(0, 3), (1, 2), (5, 6), (4, 7)]
        RADIUS = CubeTile.RADIUS
        result_shape = CubeTile.build_cylinder(pts, edges, RADIUS)
        result_shape = CubeTile.add_sphere(pts, RADIUS, result_shape)
        CubeTile.write_stp(filename, result_shape)
        return result_shape


class FCC(CubeTile):
    """
    Face-centered cubic (FCC)
    面心立方
    """
    @classmethod
    def build(cls, points:List, *args, **kwargs):
        filename = "FCC.stp"
        pts = points
        # 12 条边（顶点索引）
        edges = [
            (0, 2), (1, 3), (1, 4), (0, 5),
            (4, 6), (5, 7), (2, 7), (3, 6),
            (0, 7), (3, 4), (1, 6), (2, 5)
        ]
        RADIUS = CubeTile.RADIUS
        result_shape = CubeTile.build_cylinder(pts, edges, RADIUS)
        result_shape = CubeTile.add_sphere(pts, RADIUS, result_shape)
        # CubeTile.write_stp(filename, result_shape)
        return result_shape


class FCCZ(CubeTile):
    """
    FCC with vertical strut in z-axis (FCCZ)
    Z轴支撑面心立方
    """
    @classmethod
    def build(cls, points, *args, **kwargs):
        filename = "FCCZ.stp"
        pts = points.tolist()
        edges = [
            (0, 2), (1, 3), (1, 4), (0, 5),
            (4, 6), (5, 7), (2, 7), (3, 6),
            (0, 7), (3, 4), (1, 6), (2, 5),
            (0, 3), (1, 2), (5, 6), (4, 7)
        ]
        RADIUS = CubeTile.RADIUS
        result_shape = CubeTile.build_cylinder(pts, edges, RADIUS)
        result_shape = CubeTile.add_sphere(pts, RADIUS, result_shape)
        # CubeTile.write_stp(filename, result_shape)
        return result_shape










if __name__ == "__main__":
    #cube=Cube()
    cube_points =  np.array([
        [0, 0, 0],  # 底面左前角
        [1, 0, 0],  # 底面右前角
        [1, 0, 1],  # 顶面右前角
        [0, 0, 1],  # 顶面左前角
        [0, 1, 0],  # 底面左后角
        [1, 1, 0],  # 底面右后角
        [1, 1, 1],  # 顶面右后角
        [0, 1, 1]  # 顶面左后角
    ])
    # 生成立方体框架结构并保存为STP文件
    Cubic.build(cube_points)
    BCC.build(cube_points)
    BCCZ.build(cube_points)
    FCC.build(cube_points)
    FCCZ.build(cube_points)

    # if result:
    #     print(f"立方体框架结构已成功生成并保存到: {result}")
    # else:
    #     print("生成立方体框架结构失败")