from src.WFC.Tile import Tile
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static

class CubeTile(Tile):
    """一个立方体"""
    RADIUS = 0.02
    @property
    def property(self):
        return {"strength": 10000,
                "dieralect": 200}

    @staticmethod
    def build_cylinder(points, edges, radius):
        pts = [gp_Pnt(x, y, z) for x, y, z in points]
        RADIUS = radius
        result_shape = None
        for i, j in edges:
            p_start = pts[i]
            p_end = pts[j]
            vec = gp_Vec(p_start, p_end)
            length = vec.Magnitude()
            if length < 1e-6:
                continue

            # 圆柱坐标系：原点在起点，Z 轴指向终点
            axis = gp_Ax2(p_start, gp_Dir(vec))
            cyl = BRepPrimAPI_MakeCylinder(axis, RADIUS, length).Shape()

            # 第一次把 cyl 存下来；后续添加并合并
            if result_shape is None:
                result_shape = cyl
            else:
                result_shape = BRepAlgoAPI_Fuse(result_shape, cyl).Shape()

        return result_shape


    @staticmethod
    def write_stp(filename, shape):
        step_writer = STEPControl_Writer()
        Interface_Static.SetCVal("write.step.schema", "AP203")
        step_writer.Transfer(shape, STEPControl_AsIs)
        status = step_writer.Write(filename)
        if status == IFSelect_RetDone:
            print(f"STEP 已写出：{filename}")
        else:
            raise AssertionError("写文件失败")
        return filename





