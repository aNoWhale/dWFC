# src/dynamicGenerator/Cube_cq.py
import numpy as np
from src.dynamicGenerator.TileImplement.CubeTile_cq import CubeTile_cq   # 注意改 import


# ---------- 各具体结构 ----------
class Cubic(CubeTile_cq):
    """简单立方框架"""
    @classmethod
    def build(cls, points, *_, **__):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底 & 顶
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (5, 6), (6, 7), (7, 4)
        ]
        r = cls.RADIUS
        shape = cls._cylinder_between(points, edges, r)
        shape = cls._sphere_at(points, r, shape)
        cls.write_stp("Cubic.step", shape)
        return shape


class BCC(CubeTile_cq):
    """体心立方"""
    @classmethod
    def build(cls, points, *_, **__):
        pts = points.copy()
        bc = np.mean(pts, axis=0).tolist()
        pts.append(bc)  # 第 8 号：体心
        edges = [(i, 8) for i in range(8)]
        r = cls.RADIUS
        shape = cls.build_cylinder(pts, edges, r)
        shape = cls.add_sphere(pts, r, shape)
        cls.write_stp("BCC.step", shape)
        return shape


class BCCZ(CubeTile_cq):
    """BCC + 垂直支撑"""
    @classmethod
    def build(cls, points, *_, **__):
        pts = points.copy()
        bc = np.mean(pts, axis=0).tolist()
        pts.append(bc)
        edges = [(i, 8) for i in range(8)]
        edges += [(0, 3), (1, 2), (5, 6), (4, 7)]
        r = cls.RADIUS
        shape = cls.build_cylinder(pts, edges, r)
        shape = cls.add_sphere(pts, r, shape)
        cls.write_stp("BCCZ.step", shape)
        return shape


class FCC(CubeTile_cq):
    """面心立方"""
    @classmethod
    def build(cls, points, *_, **__):
        edges = [
            (0, 2), (1, 3), (1, 4), (0, 5),
            (4, 6), (5, 7), (2, 7), (3, 6),
            (0, 7), (3, 4), (1, 6), (2, 5)
        ]
        r = cls.RADIUS
        shape = cls.build_cylinder(points, edges, r)
        shape = cls.add_sphere(points, r, shape)
        cls.write_stp("FCC.step", shape)
        return shape


class FCCZ(CubeTile_cq):
    """FCC + 垂直支撑"""
    @classmethod
    def build(cls, points, *_, **__):
        edges = [
            (0, 2), (1, 3), (1, 4), (0, 5),
            (4, 6), (5, 7), (2, 7), (3, 6),
            (0, 7), (3, 4), (1, 6), (2, 5),
            (0, 3), (1, 2), (5, 6), (4, 7)
        ]
        r = cls.RADIUS
        shape = cls.build_cylinder(points, edges, r)
        shape = cls.add_sphere(points, r, shape)
        cls.write_stp("FCCZ.step", shape)
        return shape


# ---------- 直接运行 ----------
if __name__ == "__main__":
    cube_points = np.array([
        [0, 0, 0],  # 底面左前角
        [1, 0, 0],  # 底面右前角
        [1, 0, 1],  # 顶面右前角
        [0, 0, 1],  # 顶面左前角
        [0, 1, 0],  # 底面左后角
        [1, 1, 0],  # 底面右后角
        [1, 1, 1],  # 顶面右后角
        [0, 1, 1]  # 顶面左后角
    ])
    Cubic.build(cube_points)
