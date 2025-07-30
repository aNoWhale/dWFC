from src.WFC.Tile import Tile
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.TopTools import TopTools_ListOfShape



class CubeTile(Tile):
    """生成一个立方结构并保存为STP格式

      Args:
          points: 立方体8个角的坐标点列表，格式为[[x1,y1,z1], [x2,y2,z2], ...]
                 按照以下顺序排列：
                 [0]: 底面左前角 (x_min, y_min, z_min)
                 [1]: 底面右前角 (x_max, y_min, z_min)
                 [2]: 顶面右前角 (x_max, y_min, z_max)
                 [3]: 顶面左前角 (x_min, y_min, z_max)
                 [4]: 底面左后角 (x_min, y_max, z_min)
                 [5]: 底面右后角 (x_max, y_max, z_min)
                 [6]: 顶面右后角 (x_max, y_max, z_max)
                 [7]: 顶面左后角 (x_min, y_max, z_max)

      Returns:
          result_shape
    """
    RADIUS = 0.03
    @property
    def properties(self):
        return {"strength": 10000,
                "dieralect": 200}

    @staticmethod
    def build_cylinder(points, edges, radius):
        """
        :param points: 顶点坐标
        :param edges: 边索引
        :param radius: 圆柱半径
        :return:
        根据points和edges生成圆柱杆
        """
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
    def add_sphere(points, radius, result_shape):
        """
        :param points: 顶点坐标
        :param radius: 球半径
        :param result_shape: 底面
        :return:
        根据points和radius添加顶点球体
        """
        pts = [gp_Pnt(x, y, z) for x, y, z in points]
        RADIUS = radius  #可调球半径
        for p in pts:
            sphere = BRepPrimAPI_MakeSphere(p, RADIUS).Shape()
            # 布尔加
            result_shape = BRepAlgoAPI_Fuse(result_shape, sphere).Shape()
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





