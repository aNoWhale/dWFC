from src.WFC.Tile import Tile
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopLoc import TopLoc_Location 
import numpy as np

def get_bounding_box(shape: TopoDS_Shape) -> tuple:
    """获取几何体的包围盒参数"""
    bbox = Bnd_Box()
    bbox.SetGap(1e-4)
    brepbndlib.Add(shape, bbox)
    
    # 获取包围盒的最小和最大点
    x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
    
    # 计算中心点
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2
    
    # 计算尺寸
    size_x = x_max - x_min
    size_y = y_max - y_min
    size_z = z_max - z_min
    
    return (x_min, y_min, z_min, x_max, y_max, z_max,
            center_x, center_y, center_z,
            size_x, size_y, size_z)

def transform_shape_to_bounding_box(shape: TopoDS_Shape, target_bbox: tuple ,shapebbox:tuple) -> TopoDS_Shape:
    """
    将几何体变换到目标包围盒中
    
    参数:
        shape: 输入的几何体
        target_bbox: 目标包围盒，格式为(x_min, y_min, z_min, x_max, y_max, z_max)
        shapebbox: 几何体包围盒，格式为(x_min, y_min, z_min, x_max, y_max, z_max, center_x, center_y, center_z, size_x, size_y, size_z)
    
    返回:
        变换后的几何体
    """
    # 获取原始几何体的包围盒信息
    # (x_min, y_min, z_min, x_max, y_max, z_max,
    #  center_x, center_y, center_z,
    #  size_x, size_y, size_z) = get_bounding_box(shape)
    (x_min, y_min, z_min, x_max, y_max, z_max,
     center_x, center_y, center_z,
     size_x, size_y, size_z) = shapebbox
    # 解析目标包围盒
    t_x_min, t_y_min, t_z_min, t_x_max, t_y_max, t_z_max = target_bbox
    
    # 计算目标包围盒的中心点和尺寸
    t_center_x = (t_x_min + t_x_max) / 2
    t_center_y = (t_y_min + t_y_max) / 2
    t_center_z = (t_z_min + t_z_max) / 2
    
    t_size_x = t_x_max - t_x_min
    t_size_y = t_y_max - t_y_min
    t_size_z = t_z_max - t_z_min
    
    # 计算缩放因子（取最小比例以确保完全放入）
    scale_x = t_size_x / size_x if size_x != 0 else 1.0
    scale_y = t_size_y / size_y if size_y != 0 else 1.0
    scale_z = t_size_z / size_z if size_z != 0 else 1.0
    scale_factor = min(scale_x, scale_y, scale_z)
    
    # 创建变换矩阵
    trsf = gp_Trsf()
    
    # 1. 平移到原点（以原始几何体中心为基准）
    trsf.SetTranslation(gp_Vec(-center_x, -center_y, -center_z))
    
    # 2. 缩放
    scale_trsf = gp_Trsf()
    scale_trsf.SetScale(gp_Pnt(0, 0, 0), scale_factor)
    trsf = scale_trsf * trsf
    
    # 3. 平移到目标包围盒中心
    translate_trsf = gp_Trsf()
    translate_trsf.SetTranslation(gp_Vec(t_center_x, t_center_y, t_center_z))
    trsf = translate_trsf * trsf
    
    # 应用变换
    transformed_shape = TopoDS_Shape(shape)
    transformed_shape.Checked(True)
    loc = TopLoc_Location(trsf) 
    transformed_shape.Move(loc)
    transformed_shape.Checked(True)
    return transformed_shape






class STPtile(Tile):
    """实例化以后调用实例方法即可，减少内存开支

    """
    def __init__(self, stp_file_path,bbox=None):
        try:
            self.shape = read_step_file(stp_file_path)
            self.bbox = get_bounding_box(self.shape) if bbox is None else bbox
            print(f"成功读取STP文件: {stp_file_path}")
        except Exception as e:
            print(f"读取STP文件失败: {e},{stp_file_path}")
            return
    def build(self,points,*args,**kwargs):
        """
        call this method to build the tile structure to stp.
        :param points:
        :param args:
        :param kwargs:
        :return:
        input:
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
        return:
             result_shape
        """
        x_min, y_min, z_min = np.min(points, axis=0)
        x_max, y_max, z_max = np.max(points, axis=0)
        box= (x_min, y_min, z_min, x_max, y_max, z_max)
        result_shape=transform_shape_to_bounding_box(self.shape, box, self.bbox)
        return result_shape