# src/dynamicGenerator/CubeTile_cq.py
import cadquery as cq
from src.WFC.Tile import Tile   # 你的原始基类，保持不动
import numpy as np


class CubeTile_cq(Tile):
    """
    生成一个立方框架并保存为 STEP（CadQuery 版）
    顶点顺序与原 OCC 版完全一致：
      0 底面左前
      1 底面右前
      2 顶面右前
      3 顶面左前
      4 底面左后
      5 底面右后
      6 顶面右后
      7 顶面左后
    """
    RADIUS = 0.03

    @property
    def properties(self):
        return {"strength": 10000, "dieralect": 200}

    # ---------- 内部静态工具 ----------
    @staticmethod
    def _cylinder_between(p0, p1, r):
        vec = np.array(p1) - np.array(p0)
        length = np.linalg.norm(vec)
        if length < 1e-9:
            return None

        direction = vec / length
        z_axis = np.array([0, 0, 1])

        rot_axis = np.cross(z_axis, direction)
        cos_ang = np.dot(z_axis, direction)
        rot_angle = 0
        if np.linalg.norm(rot_axis) > 1e-9:
            rot_angle = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            rot_axis = tuple(rot_axis / np.linalg.norm(rot_axis))

        cyl = cq.Workplane().cylinder(length, r, centered=(True, True, False))
        # 关键：把 p0 变成 tuple
        cyl = cyl.translate(tuple(p0))

        if abs(rot_angle) > 1e-3:
            cyl = cyl.rotate((0, 0, 0), rot_axis, rot_angle)
        return cyl

    @staticmethod
    def _sphere_at(p, r):
        return cq.Workplane().sphere(r).translate(tuple(p))
    # ---------- 供子类调用的静态方法 ----------
    @staticmethod
    def build_cylinder(points, edges, radius):
        """
        points: list[list[float]]  8 个顶点
        edges:  list[tuple[int,int]] 12 条边的索引
        radius: float
        return: cq.Workplane 合并后的形状
        """
        frame = None
        for i, j in edges:
            cyl = CubeTile_cq._cylinder_between(points[i], points[j], radius)
            if cyl is None:
                continue
            frame = cyl if frame is None else frame.union(cyl)
        return frame

    @staticmethod
    def add_sphere(points, radius, frame):
        """
        points: list[list[float]] 顶点坐标
        radius: float  球半径
        frame:  cq.Workplane  已有形状
        return: cq.Workplane  合并后的形状
        """
        for p in points:
            sph = CubeTile_cq._sphere_at(p, radius)
            frame = frame.union(sph)
        return frame

    @staticmethod
    def write_stp(filename, shape):
        """把 CadQuery 对象写成 STEP"""
        cq.exporters.export(shape, filename)
        print(f"STEP 已写出：{filename}")